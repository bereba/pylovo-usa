import traceback
import warnings

import numpy as np
import pandas as pd

import src.database.database_client as dbc
from src import utils
from src.cable_installation.electrical_grid_builder import ElectricalGridBuilder
from src.config_loader import *
from src.electrical_backend.opendss_backend import OpenDSSBackend


class ResultExistsError(Exception):
    "Raised when the regional_identifier has already been created."


class GridGenerator:
    """
    Generates the grid for the given regional_identifier area
    """

    def __init__(self, regional_identifier=999999, **kwargs):
        self.regional_identifier = regional_identifier
        self.dbc = dbc.DatabaseClient()
        self.dbc.insert_version_if_not_exists()
        self.dbc.insert_parameter_tables(consumer_categories=CONSUMER_CATEGORIES)
        self.logger = utils.create_logger(name="GridGenerator", log_level=LOG_LEVEL, log_file=LOG_FILE)

    def __del__(self):
        self.dbc.__del__()

    # ------------------------------------------------------------
    # MAIN GRID GENERATION FUNCTIONS
    # ------------------------------------------------------------

    def generate_grid_for_single_regional_identifier(self, regional_identifier: str) -> None:
        """
        Generates the grid for a single regional_identifier.

        :param regional_identifier: Postal code for which the grid should be generated.
        :type regional_identifier: str
        """
        self.regional_identifier = regional_identifier
        print(
            "-------------------- start",
            self.regional_identifier,
            "---------------------------",
        )

        self.dbc.create_temp_tables()  # create temp tables for the grid generation

        try:
            self.generate_grid()
            # Save data from temporary tables to result tables
            self.dbc.save_tables(regional_identifier=self.regional_identifier)
        except ResultExistsError:
            self.dbc.logger.info(f"Grid for the postcode area {regional_identifier} has already been generated.")
        except Exception as e:
            self.logger.error(
                f"Error during grid generation for regional_identifier {
                    self.regional_identifier}: {e}"
            )
            self.logger.info(
                f"Skipped regional_identifier {
                    self.regional_identifier} due to generation error."
            )
            self.dbc.conn.rollback()  # rollback the transaction
            traceback.print_exc()
            return

        self.dbc.drop_temp_tables()  # drop temp tables
        self.dbc.commit_changes()  # commit the changes to the database

        print(
            "-------------------- end",
            self.regional_identifier,
            "-----------------------------",
        )

    def generate_grid_for_multiple_regional_identifier(self, df_regional_identifier: pd.DataFrame) -> None:
        """generates grid for all regional_identifier contained in the column 'regional_identifier' of df_samples

        :param df_regional_identifier: table that contains regional_identifier for grid generation
        :type df_regional_identifier: pd.DataFrame
        """
        self.dbc.create_temp_tables()  # create temp tables for the grid generation

        for _, row in df_regional_identifier.iterrows():
            self.regional_identifier = str(row["regional_identifier"])
            print(
                "-------------------- start",
                self.regional_identifier,
                "---------------------------",
            )
            try:
                self.generate_grid()
                # Save data from temporary tables to result tables
                self.dbc.save_tables(regional_identifier=self.regional_identifier)
                self.dbc.reset_tables()  # Reset temporary tables
            except ResultExistsError:
                self.dbc.logger.info(
                    f"Grid for the postcode area {
                        self.regional_identifier} has already been generated."
                )
            except Exception as e:
                self.logger.error(
                    f"Error during grid generation for regional_identifier {
                        self.regional_identifier}: {e}"
                )
                self.logger.info(
                    f"Skipped regional_identifier {
                        self.regional_identifier} due to generation error."
                )
                self.dbc.conn.rollback()  # rollback the transaction
                continue
            print(
                "-------------------- end",
                self.regional_identifier,
                "-----------------------------",
            )

        self.dbc.drop_temp_tables()  # drop temp tables
        self.dbc.commit_changes()  # commit the changes to the database

    def generate_grid(self):
        if self.dbc.is_grid_generated(self.regional_identifier):
            raise ResultExistsError(
                f"The grids for the postcode area {
                    self.regional_identifier} is already generated "
                f"for the version {VERSION_ID}."
            )
        self.prepare_postcodes()
        self.prepare_buildings()
        self.prepare_transformers()
        self.prepare_ways()

        self.apply_kmeans_clustering()

        # First position LV_Transformers for each bcid cluster and buildings
        # with grid_level_connection = LV
        self.position_distribution_transformers()

        # Now position MV_substations and connect them to the LV_Transformers
        # and the buildings which have grid_level_connection = MV
        self.position_mv_substations()

        # Install cables for each grid.
        self.install_cables_parallel(max_workers=1)

    # ------------------------------------------------------------
    # DATA PREPARATION SECTION
    # ------------------------------------------------------------

    def prepare_postcodes(self):
        """
        Caches postcode from raw data tables and stores in temporary tables.
        FROM: postcode
        INTO: postcode_result
        """
        self.dbc.copy_postcode_result_table(self.regional_identifier)
        self.logger.info(
            f"Working on regional_identifier {
                self.regional_identifier}"
        )

    def prepare_buildings(self):
        """
        Caches buildings from raw data tables and stores in temporary tables.
        FROM: res, oth
        INTO: buildings_tem
        """
        self.dbc.set_residential_buildings_table(self.regional_identifier)
        self.dbc.set_other_buildings_table(self.regional_identifier)
        self.logger.info("Buildings_tem table prepared")
        self.dbc.remove_duplicate_buildings()
        self.logger.info("Duplicate buildings removed from buildings_tem")

        unloadcount = self.dbc.set_building_peak_load()
        self.logger.info(
            f"Building peakload calculated in buildings_tem, {unloadcount} unloaded buildings are removed from buildings_tem"
        )
        # Update all buildings with peak load > TRESHHOLD to MV level
        self.dbc.assign_grid_level_connection_by_peak_load()

        self.dbc.set_regional_identifier_settlement_type(self.regional_identifier)
        self.logger.info("Load density and settlement_type in postcode_result")

        self.dbc.assign_close_buildings()

        self.dbc.remove_zero_peak_load_buildings()

        # for debugging purposes: keep only n buildings for LV and MV
        # self.dbc.keep_only_n_buildings_for_lv(n=10)
        # self.dbc.keep_only_n_buildings_for_mv(n=10)

    def prepare_transformers(self):
        """
        Cache transformers from raw data tables and stores in temporary tables.
        FROM: transformers
        INTO: buildings_tem
        """
        self.dbc.insert_transformers(self.regional_identifier)
        self.logger.info("Transformers inserted in to the buildings_tem table")
        self.dbc.count_indoor_transformers()
        self.dbc.drop_indoor_transformers()
        self.logger.info("Indoor transformers dropped from the buildings_tem table")

    def prepare_ways(self):
        """
        Cache ways, create network, connect buildings to the ways network
        FROM: ways, buildings_tem
        INTO: ways_tem, buildings_tem, ways_tem_vertices_pgr, ways_tem_
        """
        ways_count = self.dbc.set_ways_tem_table(self.regional_identifier)
        self.logger.info(f"The ways_tem table filled with {ways_count} ways")
        self.logger.info("Connecting road_network to the buildings, this might take a while...")
        self.dbc.build_pgr_network_topology(self.regional_identifier)
        self.logger.info("Building connection finished in ways_tem")

        self.dbc.update_ways_cost()
        unconn = self.dbc.set_vertice_id()
        self.logger.debug(f"vertice id set, {unconn} buildings with no vertice id")

    # ------------------------------------------------------------
    # CLUSTERING SECTION
    # ------------------------------------------------------------

    def apply_kmeans_clustering(self):
        """
        Find connected components (subgraphs) of an undirected street-graph applying the Depth-First Search algorithm
        to edges and vertices from ways_tem and (if necessary due to their size) apply k-means clustering to these
        street network components.

        FROM: ways_tem, buildings_tem
        INTO: ways_tem, vertices_pgr, buildings_tem
        """

        # Get connected components from the street network
        component, vertices = self.dbc.get_connected_component()
        component_ids = np.unique(component)

        if len(component_ids) > 0:
            # Handle components based on number
            if len(component_ids) > 1:
                # Process multiple connected components
                for i, component_id in enumerate(component_ids):
                    related_vertices = vertices[np.argwhere(component == component_id)]
                    self._process_component_to_kcid(related_vertices, i)
            else:
                # Process single connected component
                self._process_component_to_kcid(vertices)
        else:
            # No components found - issue warning
            warnings.warn("No connected components found in ways_tem table")

        # Verify clustering was successful for all buildings
        no_kmean_count = self.dbc.count_no_kmean_buildings()
        if no_kmean_count not in [0, None]:
            warnings.warn(f"K-means clustering issue: {no_kmean_count} buildings not assigned to clusters")

    def _process_component_to_kcid(self, vertices, component_index=None):
        """Helper method to process components to kcid groups"""
        conn_building_count = self.dbc.count_connected_buildings(vertices)

        if conn_building_count <= 1 or conn_building_count is None:
            # Remove isolated or empty components
            self.dbc.delete_ways(vertices)
            self.dbc.delete_transformers_from_buildings_tem(vertices)
            self.logger.debug("Empty/isolated component removed. Ways and transformers deleted from temporary tables.")
        elif conn_building_count >= LARGE_COMPONENT_LOWER_BOUND:
            # K-means applied to large component to define subgroups with
            # cluster ids
            cluster_count = int(conn_building_count / LARGE_COMPONENT_DIVIDER)
            self.dbc.update_large_kmeans_cluster(vertices, cluster_count)
            log_msg = (
                f"Large component {component_index} clustered into {cluster_count} groups"
                if component_index is not None
                else f"Large component clustered into {cluster_count} groups"
            )
            self.logger.debug(log_msg)
        else:
            # Allocate cluster id for connected component smaller than the
            # building threshold
            self.dbc.update_kmeans_cluster(vertices)

    # ------------------------------------------------------------
    # TRANSFORMER POSITIONING SECTION
    # ------------------------------------------------------------
    def position_distribution_transformers(self):
        """
        Positions all transformers for LVeach bcid cluster (brownfield with existing transformers and greenfield)
        FROM: buildings_tem, lv_grid_result
        INTO: buildings_tem, lv_grid_result
        """
        kcid_length = self.dbc.get_kcid_length()

        for _ in range(kcid_length):
            kcid = self.dbc.get_next_unfinished_kcid(self.regional_identifier, "lv_grid_result")
            self.logger.info(f"working on kcid {kcid}")

            self.logger.debug(f"kcid{kcid} has no included transformer")
            # Create greenfield transformer clusters
            self.create_bcid_for_kcid(self.regional_identifier, kcid)
            self.logger.debug(f"kcid{kcid} building clusters finished")

    def create_bcid_for_kcid(self, regional_identifier: int, kcid: int) -> None:
        """

        Steps:
        1) Use the unified infrastructure placement interface for LV transformer placement
        2) Convert InfrastructureCluster results to the database format expected by downstream processes
        """
        try:
            self.logger.info(f"Starting LV clustering for kcid {kcid}, regional_identifier {regional_identifier}")

            # Get settlement type for this region
            settlement_type = self.dbc.get_settlement_type_from_regional_identifier(regional_identifier)

            # Use the unified infrastructure placement interface for LV
            # clustering
            infrastructure_clusters = self.dbc.perform_infrastructure_placement(
                kcid=kcid,
                regional_identifier=regional_identifier,
                grid_level="LV",
                settlement_type=settlement_type,
            )

            if not infrastructure_clusters:
                self.logger.warning(f"No LV infrastructure clusters created for kcid {kcid}")
                return

            # Clear previous clustering results for this kcid
            self.dbc.clear_lv_grid_result_in_kmean_cluster(regional_identifier, kcid)

            # Convert InfrastructureCluster results to traditional bcid format
            # Each infrastructure cluster becomes a building cluster (bcid)
            self.logger.info("Converting infrastructure clusters to building clusters (bcids)")

            # Save infrastructure placement results to database
            grid_result_ids = self.dbc.save_infrastructure_placement_results(
                infrastructure_clusters=infrastructure_clusters,
                kcid=kcid,
                regional_identifier=self.regional_identifier,
                grid_level="LV",
            )

            # Create transformer position records
            self.dbc.create_transformer_positions(
                infrastructure_clusters=infrastructure_clusters,
                grid_level="LV",
                grid_result_ids=grid_result_ids,
            )

            self.logger.info(
                f"Successfully created {
                    len(infrastructure_clusters)} LV building clusters "
                f"for regional_identifier={regional_identifier}, kcid={kcid}"
            )

        except Exception as e:
            self.logger.error(
                f"Error in LV clustering for kcid {kcid}: {
                    str(e)}"
            )
            raise

    def position_mv_substations(self):
        """
            Positions all MV substations for each kcid cluster using the new infrastructure placement
        engine.
            FROM: buildings_tem, lv_grid_result
            INTO: grid_result, transformer_positions
        """
        kcid_length = self.dbc.get_kcid_length()

        # Get settlement type for this region
        settlement_type = self.dbc.get_settlement_type_from_regional_identifier(self.regional_identifier)

        for _ in range(kcid_length):
            try:
                # Get next unfinished kcid for MV processing
                kcid = self.dbc.get_next_unfinished_kcid(self.regional_identifier, "grid_result")
                self.logger.info(f"Working on MV substation placement for kcid {kcid}")

                # Use the new unified infrastructure placement interface
                infrastructure_clusters = self.dbc.perform_infrastructure_placement(
                    kcid=kcid,
                    regional_identifier=self.regional_identifier,
                    grid_level="MV",
                    settlement_type=settlement_type,
                )

                if not infrastructure_clusters:
                    self.logger.info(f"No MV infrastructure clusters created for kcid {kcid}")
                    continue

                # Save infrastructure placement results to database
                grid_result_ids = self.dbc.save_infrastructure_placement_results(
                    infrastructure_clusters=infrastructure_clusters,
                    kcid=kcid,
                    regional_identifier=self.regional_identifier,
                    grid_level="MV",
                )

                # Create transformer position records
                self.dbc.create_transformer_positions(
                    infrastructure_clusters=infrastructure_clusters,
                    grid_level="MV",
                    grid_result_ids=grid_result_ids,
                )

                # Update LV transformers and buildings with scid and parent
                # references
                self.dbc.update_lv_mv_links(
                    infrastructure_clusters=infrastructure_clusters,
                    grid_result_ids=grid_result_ids,
                    kcid=kcid,
                    regional_identifier=self.regional_identifier,
                )

                # Set model status to completed
                self.dbc.finalize_mv_substation_placement(grid_result_ids)

                self.logger.info(
                    f"Successfully positioned {
                        len(infrastructure_clusters)} MV substations "
                    f"for kcid {kcid}"
                )

            except Exception as e:
                self.logger.error(
                    f"Error positioning MV substations for kcid {kcid}: {
                        str(e)}"
                )
                # Continue with next kcid rather than failing completely
                continue

        self.logger.info("Completed MV substation positioning for all kcids")

    # ------------------------------------------------------------
    #  LINE INSTALLATION SECTION
    # ------------------------------------------------------------

    def install_cables_parallel(self, max_workers: int = 1):
        """
        Parallelized version of install_cables using multiprocessing.
        Installs electrical cables to connect buildings and transformers in power grid clusters.

        This method creates a grid network for each building cluster (kcid, bcid) in the
        postal code area and connects the buildings with appropriate electrical cables. It follows
        a branch-by-branch approach, starting from the furthest nodes and working inward toward
        the transformer.

        The algorithm works as follows:
        1. Retrieves all clusters (kcid, bcid) for the postal code area
        2. For each cluster:
           a. Prepares building and connection data
           b. Creates an electrical network with OpenDSS
           c. Adds buses, transformers, and loads to the network
           d. Installs cables using a greedy algorithm that:
              - Starts from the furthest nodes from the transformer
              - Creates branches with maximum possible load
              - Selects minimum size cables that can handle the current
              - Connects branches back to transformer
        3. Tracks progress and saves the network configurations

        The cable installation prioritizes cost efficiency while ensuring the electrical
        requirements are met for each branch of the distribution network.

        Returns:
            None

        Args:
            max_workers: Maximum number of worker processes. If None, uses CPU count.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # KCID, SCID pair
        cluster_list = self.dbc.get_list_from_regional_identifier(self.regional_identifier)
        if not cluster_list:
            self.logger.warning(
                f"No clusters to process for regional_identifier {
                    self.regional_identifier}"
            )
            return

        self.logger.info(
            f"Starting parallel cable installation for {
                len(cluster_list)} clusters using {max_workers} workers."
        )

        # Create batches of clusters to process
        def create_batches(items, batch_size):
            for i in range(0, len(items), batch_size):
                yield items[i : i + batch_size]

        # Calculate batch size to distribute work evenly
        batch_size = max(1, len(cluster_list) // max_workers)
        cluster_batches = list(create_batches(cluster_list, batch_size))

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=GridGenerator._init_worker,
            initargs=(self.regional_identifier,),
        ) as executor:
            future_to_batch = {
                executor.submit(GridGenerator._process_cluster_batch, batch): batch for batch in cluster_batches
            }

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    future.result()
                    self.logger.debug(
                        f"Successfully processed batch with {
                            len(batch)} clusters"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to process batch with {
                            len(batch)} clusters: {e}",
                        exc_info=True,
                    )

        self.logger.info(
            f"Parallel cable installation completed for regional_identifier {
                self.regional_identifier}"
        )

    @staticmethod
    def _init_worker(regional_identifier):
        """Initialize worker process with one GridGenerator per worker."""
        global _worker_grid_generator
        _worker_grid_generator = GridGenerator(regional_identifier=regional_identifier)

    @staticmethod
    def _process_cluster_batch(cluster_batch):
        """Process a batch of clusters using the worker's GridGenerator."""
        global _worker_grid_generator
        try:
            for kcid, scid in cluster_batch:
                _worker_grid_generator._install_cables_for_cluster(kcid, scid)
        except Exception as e:
            print(f"Error in worker batch processing: {e}")
            raise

    def _install_cables_for_cluster(self, kcid, scid):
        """
        NEW: Builds complete MV-LV hierarchical grid using ElectricalGridBuilder.

        This method is the integration point for the existing parallel processing workflow.
        It has been updated to use the new backend-agnostic architecture while preserving
        the same interface for parallel compatibility.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID (was bcid in old system)
        """
        try:
            self.logger.debug(f"Building hierarchical grid for kcid {kcid}, scid {scid}")

            # Create backend and unified builder
            backend = OpenDSSBackend(logger=self.logger)
            builder = ElectricalGridBuilder(backend=backend, dbc=self.dbc, logger=self.logger)

            # Build entire hierarchical grid using new architecture
            success = builder.build_complete_grid_for_cluster(kcid, scid, self.regional_identifier)

            if success:
                self.logger.info(f"✓ Successfully built hierarchical grid K{kcid}_S{scid}")
            else:
                self.logger.error(f"✗ Failed to build hierarchical grid K{kcid}_S{scid}")

            return success

        except Exception as e:
            self.logger.error(
                f"Grid construction failed for K{kcid}_S{scid}: {
                    str(e)}",
                exc_info=True,
            )
            # Re-raise to maintain compatibility with parallel error handling
            raise
