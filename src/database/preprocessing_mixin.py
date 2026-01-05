import warnings
from abc import ABC

from src.config_loader import *
from src.database.base_mixin import BaseMixin

warnings.simplefilter(action="ignore", category=UserWarning)


class PreprocessingMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def insert_parameter_tables(self, consumer_categories: pd.DataFrame):
        self.cur.execute("SELECT count(*) FROM consumer_categories")
        categories_exist = self.cur.fetchone()[0]
        with self.sqla_engine.begin() as conn:
            if not categories_exist:
                consumer_categories.to_sql(
                    name="consumer_categories",
                    con=conn,
                    if_exists="append",
                    index=False,
                )
                self.logger.debug("Parameter tables are inserted")

    def insert_version_if_not_exists(self):
        count_query = f"""SELECT COUNT(*)
            FROM version
            WHERE "version_id" = '{VERSION_ID}'"""
        self.cur.execute(count_query)
        version_exists = self.cur.fetchone()[0]
        if not version_exists:
            # create new version
            consumer_categories_str = CONSUMER_CATEGORIES.to_json().replace("'", "''")
            other_parameters_dict = {
                "LARGE_COMPONENT_LOWER_BOUND": LARGE_COMPONENT_LOWER_BOUND,
                "LARGE_COMPONENT_DIVIDER": LARGE_COMPONENT_DIVIDER,
                "VN": VN,
                "V_BAND_LOW": V_BAND_LOW,
                "V_BAND_HIGH": V_BAND_HIGH,
            }
            other_paramters_str = str(other_parameters_dict).replace("'", "''")

            insert_query = f"""INSERT INTO version (version_id, version_comment, consumer_categories, other_parameters) VALUES
                ('{VERSION_ID}', '{VERSION_COMMENT}', '{consumer_categories_str}', '{other_paramters_str}')"""
            self.cur.execute(insert_query)
            self.logger.info(f"Version: {VERSION_ID} (created for the first time)")

    def get_postcode_table_for_regional_identifier(self, regional_identifier: int) -> pd.DataFrame:
        """get postcode table for given regional_identifier"""
        query = """SELECT *
                   FROM postcode
                   WHERE regional_identifier = %(p)s;"""
        df_postcode = pd.read_sql_query(query, con=self.conn, params={"p": regional_identifier})
        if len(df_postcode) == 0:
            raise ValueError(f"No entry in postcode table found for regional_identifier: {regional_identifier}")
        return df_postcode

    def copy_postcode_result_table(self, regional_identifier: int) -> None:
        """
        Copies the given regional_identifier entry from postcode to the postcode_result table
        :param regional_identifier:
        :return:
        """
        query = """INSERT INTO postcode_result (version_id, postcode_result_regional_identifier, geom)
                   SELECT %(v)s as version_id, regional_identifier, geom
                   FROM postcode
                   WHERE regional_identifier = %(p)s
                   LIMIT 1
                   ON CONFLICT (version_id,postcode_result_regional_identifier) DO NOTHING;"""

        self.cur.execute(query, {"v": VERSION_ID, "p": regional_identifier})

    def set_residential_buildings_table(self, regional_identifier: int):
        """
        * Fills buildings_tem with residential buildings which are inside the regional_identifier area
        :param regional_identifier:
        :return:
        """

        # Fill table - keep grid_level_connection NULL initially
        query = """INSERT INTO buildings_tem (osm_id, area, type, geom, center, floors, grid_level_connection)
                   SELECT osm_id, area, build_type, geom, ST_Centroid(geom), floors::int, NULL::varchar
                   FROM res
                   WHERE ST_Contains((SELECT post.geom
                                      FROM postcode_result as post
                                      WHERE version_id = %(v)s
                                        AND postcode_result_regional_identifier = %(regional_identifier)s
                                      LIMIT 1), ST_Centroid(res.geom));
        UPDATE buildings_tem
        SET regional_identifier = %(regional_identifier)s
        WHERE regional_identifier ISNULL;"""
        self.cur.execute(query, {"v": VERSION_ID, "regional_identifier": regional_identifier})

    def set_other_buildings_table(self, regional_identifier: int):
        """
        * Fills buildings_tem with other buildings which are inside the regional_identifier area
        * Sets all floors to 1
        :param regional_identifier:
        :return:
        """

        # Fill table - keep grid_level_connection NULL initially
        query = """INSERT INTO buildings_tem(osm_id, area, type, geom, center, grid_level_connection)
                   SELECT osm_id, area, use, geom, ST_Centroid(geom), NULL::varchar
                   FROM oth AS o
                   WHERE o.use in ('Commercial', 'Public')
                     AND ST_Contains((SELECT post.geom
                                      FROM postcode_result as post
                                      WHERE version_id = %(v)s
                                        AND postcode_result_regional_identifier = %(regional_identifier)s), ST_Centroid(o.geom));;
        UPDATE buildings_tem
        SET regional_identifier = %(regional_identifier)s
        WHERE regional_identifier ISNULL;
        UPDATE buildings_tem
        SET floors = 1
        WHERE floors ISNULL;"""
        self.cur.execute(query, {"v": VERSION_ID, "regional_identifier": regional_identifier})

    def remove_duplicate_buildings(self):
        """
        * Remove buildings without geometry or osm_id
        * Remove buildings which are duplicates of other buildings and have a copied id
        :return:
        """
        remove_query = """DELETE
                          FROM buildings_tem
                          WHERE geom ISNULL;"""
        self.cur.execute(remove_query)

        remove_noid_building = """DELETE
                                  FROM buildings_tem
                                  WHERE osm_id ISNULL;"""
        self.cur.execute(remove_noid_building)

        query = """DELETE
                   FROM buildings_tem
                   WHERE geom IN
                         (SELECT geom FROM buildings_tem GROUP BY geom HAVING count(*) > 1)
                     AND osm_id LIKE '%copy%';"""
        self.cur.execute(query)

    def set_regional_identifier_settlement_type(self, regional_identifier: int) -> None:
        """
        Determine settlement_type in postcode_result table based on the load_density (MVA/km²) for the given regional_identifier
        :param regional_identifier: Regional identifier (FIPS code)
        :return: None
        """
        # Get total peak load from buildings_tem for the specific
        # regional_identifier and area from postcode table
        load_density_query = """
            SELECT
                COALESCE(SUM(bt.peak_load_in_kw), 0) as total_load_kw,
                p.qkm as area_km2
            FROM postcode p
            LEFT JOIN buildings_tem bt ON bt.regional_identifier = p.regional_identifier
            WHERE p.regional_identifier = %(p)s
            GROUP BY p.qkm
        """

        self.cur.execute(load_density_query, {"p": regional_identifier})
        result = self.cur.fetchone()

        if not result or result[1] is None:
            raise ValueError(f"No area data found in postcode table for regional_identifier: {regional_identifier}")

        total_load_kw, area_km2 = result

        total_load_kw = float(total_load_kw) if total_load_kw is not None else 0.0
        area_km2 = float(area_km2) if area_km2 is not None else 0.0

        if area_km2 <= 0:
            raise ValueError(f"Invalid area ({area_km2}) for regional_identifier: {regional_identifier}")

        # Calculate load density in MVA/km² using power factor
        load_density_mva_km2 = total_load_kw / (POWER_FACTOR * 1000 * area_km2)
        self.logger.info(f"Load density for: {regional_identifier} is {load_density_mva_km2} MVA/km²")

        # Update database with load density and set settlement types based on
        # thresholds
        query = """
                UPDATE postcode_result
                SET load_density = %(load_density)s,
                    settlement_type = CASE
                                          WHEN %(load_density)s < %(rural_threshold)s THEN 1
                                          WHEN %(load_density)s < %(urban_threshold)s THEN 2
                                          ELSE 3
                        END
                WHERE version_id = %(v)s
                  AND postcode_result_regional_identifier = %(p)s;"""

        self.cur.execute(
            query,
            {
                "v": VERSION_ID,
                "load_density": load_density_mva_km2,
                "p": regional_identifier,
                "rural_threshold": RURAL_LD,
                "urban_threshold": URBAN_LD,
            },
        )

    def set_building_peak_load(self) -> int:
        """
        * Sets the area, type and peak_load in the buildings_tem table
        * Removes buildings with zero load from the buildings_tem table
        :return: Number of removed unloaded buildings from buildings_tem
        """
        query = """
                UPDATE buildings_tem
                SET area = ST_Area(geom);
                UPDATE buildings_tem
                SET houses_per_building = (CASE
                                               WHEN type IN ('TH', 'Commercial', 'Public', 'Industrial') THEN 1
                                               WHEN type = 'SFH' AND area < 160 THEN 1
                                               WHEN type = 'SFH' AND area >= 160 THEN 2
                                               WHEN type IN ('MFH', 'AB') THEN floor(area / %(avg_apartment_area)s) * floors
                                               ELSE 0
                    END);
                UPDATE buildings_tem b
                SET peak_load_in_kw = (CASE
                                           WHEN b.type IN ('SFH', 'TH', 'MFH', 'AB') THEN b.houses_per_building *
                                                                                          (SELECT peak_load FROM consumer_categories WHERE definition = b.type)
                                           WHEN b.type IN ('Commercial', 'Public', 'Industrial') THEN b.area *
                                                                                                      (SELECT peak_load_per_m2
                                                                                                       FROM consumer_categories
                                                                                                       WHERE definition = b.type)
                                           ELSE 0
                    END);"""
        self.cur.execute(query, {"avg_apartment_area": AVG_APARTMENT_AREA})

        count_query = """SELECT COUNT(*)
                          FROM buildings_tem
                          WHERE peak_load_in_kw = 0;"""
        self.cur.execute(count_query)
        count = self.cur.fetchone()[0]

        delete_query = """DELETE
                          FROM buildings_tem
                          WHERE peak_load_in_kw = 0;"""
        self.cur.execute(delete_query)

        return count

    def assign_grid_level_connection_by_peak_load(self) -> int:
        """
        Assigns LV/MV grid level based on peak load thresholds and caps very large consumers.

        Rules:
        - 0 < peak_load_in_kw <= lower_threshold_kw  => grid_level_connection = 'LV'
        - lower_threshold_kw < peak_load_in_kw <= upper_threshold_kw => grid_level_connection = 'MV'
        - peak_load_in_kw > upper_threshold_kw => set peak_load_in_kw = 0 and clear grid_level_connection


        :return:
        """
        query = """
            WITH lv AS (
                UPDATE buildings_tem
                SET grid_level_connection = 'LV'
                WHERE peak_load_in_kw > 0
                  AND peak_load_in_kw <= %(lower)s
                RETURNING 1
            ),
            mv AS (
                UPDATE buildings_tem
                SET grid_level_connection = 'MV'
                WHERE peak_load_in_kw > %(lower)s
                  AND peak_load_in_kw <= %(upper)s
                RETURNING 1
            ),
            zeroed AS (
                UPDATE buildings_tem
                SET peak_load_in_kw = 0,
                    grid_level_connection = NULL
                WHERE peak_load_in_kw > %(upper)s
                RETURNING 1
            )
            SELECT
                (SELECT COUNT(*) FROM lv)  AS lv_count,
                (SELECT COUNT(*) FROM mv)  AS mv_count,
                (SELECT COUNT(*) FROM zeroed) AS zeroed_count;
        """
        self.cur.execute(query, {"lower": LV_THRESHOLD_KW, "upper": MV_THRESHOLD_KW})
        lv_count, mv_count, zeroed_count = self.cur.fetchone()
        self.logger.info(f"Grid-level assignment: LV={lv_count}, MV={mv_count}, Zeroed={zeroed_count}")
        return

    def remove_zero_peak_load_buildings(self) -> int:
        """
        * Remove buildings with peak load = 0
        :return: Number of removed buildings from buildings_tem
        """
        query = """DELETE
                  FROM buildings_tem
                  WHERE peak_load_in_kw = 0;"""
        self.cur.execute(query)
        self.logger.info(
            f"Buildings with peak load = 0 removed from buildings_tem, {
                self.cur.rowcount} buildings removed"
        )
        return

    def assign_close_buildings(self) -> None:
        """
        * Set peak load to zero, if a building is too near or touching to a too large customer?
        :return:
        """
        total_zeroed = 0
        while True:
            cascade_zero_query = """
                WITH close (un) AS (
                    SELECT ST_Union(geom)
                    FROM buildings_tem
                    WHERE peak_load_in_kw = 0
                ),
                upd AS (
                    UPDATE buildings_tem b
                    SET peak_load_in_kw = 0
                    FROM close AS c
                    WHERE ST_Touches(b.geom, c.un)
                      AND b.type IN ('Commercial', 'Public', 'Industrial')
                      AND b.peak_load_in_kw != 0
                    RETURNING 1
                )
                SELECT COUNT(*) FROM upd;
            """
            self.cur.execute(cascade_zero_query)
            updated = self.cur.fetchone()[0]
            total_zeroed += int(updated or 0)
            if not updated:
                break

        self.logger.info(f"Close-building zeroed (touching cascades): {total_zeroed}")
        return None

    def insert_transformers(self, regional_identifier: int) -> None:
        """
        Add up the existing transformers from transformers table to the buildings_tem table
        :param regional_identifier:
        :return:
        """
        insert_query = """
                       --UPDATE transformers SET geom = ST_Centroid(geom) WHERE ST_GeometryType(geom) =  'ST_Polygon';
                       INSERT INTO buildings_tem (osm_id, geom)--(osm_id,center)
                       SELECT osm_id, geom
                       --FROM transformers WHERE ST_Within(geom, (SELECT geom FROM postcode_result LIMIT 1)) IS FALSE;
                       FROM transformers as t
                       WHERE ST_Within(t.geom, (SELECT geom
                                                FROM postcode_result
                                                WHERE postcode_result_regional_identifier = %(p)s
                                                  AND version_id = %(v)s)); --IS FALSE;
                       UPDATE buildings_tem
                       SET regional_identifier = %(p)s
                       WHERE regional_identifier ISNULL;
                       UPDATE buildings_tem
                       SET center = ST_Centroid(geom)
                       WHERE center ISNULL;
                       UPDATE buildings_tem
                       SET type = 'Transformer'
                       WHERE type ISNULL;
                       UPDATE buildings_tem
                       SET peak_load_in_kw = -1
                       WHERE peak_load_in_kw ISNULL;"""
        self.cur.execute(insert_query, {"p": regional_identifier, "v": VERSION_ID})

    def count_indoor_transformers(self) -> None:
        """counts indoor transformers before deleting them"""
        query = """WITH union_table (ungeom) AS
                                (SELECT ST_Union(geom) FROM buildings_tem WHERE peak_load_in_kw = 0)
                   SELECT COUNT(*)
                   FROM buildings_tem
                   WHERE ST_Within(center, (SELECT ungeom FROM union_table))
                     AND type = 'Transformer';"""
        self.cur.execute(query)
        count = self.cur.fetchone()[0]
        self.logger.debug(f"{count} indoor transformers will be deleted")

    def drop_indoor_transformers(self) -> None:
        """
        Drop transformer if it is inside a building with zero load
        :return:
        """
        query = """WITH union_table (ungeom) AS
                                (SELECT ST_Union(geom) FROM buildings_tem WHERE peak_load_in_kw = 0)
                   DELETE
                   FROM buildings_tem
                   WHERE ST_Within(center, (SELECT ungeom FROM union_table))
                     AND type = 'Transformer';"""
        self.cur.execute(query)

    def set_ways_tem_table(self, regional_identifier: int) -> int:
        """
        * Inserts ways inside the regional_identifier area (with buffer) to the ways_tem table
        :param regional_identifier:
        :return: number of ways in ways_tem
        """
        query = """INSERT INTO ways_tem (clazz, source, target, cost, reverse_cost, way_id, regional_identifier, geom)
                   SELECT clazz, source, target, cost, reverse_cost, way_id, %(p)s, geom
                   FROM ways AS w
                   WHERE ST_Intersects(w.geom, (SELECT ST_Buffer(geom, 50)
                                                FROM postcode_result
                                                WHERE version_id = %(v)s
                                                  AND postcode_result_regional_identifier = %(p)s));
        SELECT COUNT(*)
        FROM ways_tem;"""
        self.cur.execute(query, {"v": VERSION_ID, "p": regional_identifier})
        count = self.cur.fetchone()[0]

        if count == 0:
            raise ValueError(f"Ways table is empty for the given regional_identifier: {regional_identifier}")

        return count

    def build_pgr_network_topology(self, plz: int) -> None:
        """Builds the pgRouting-compatible network topology from the updated `ways_tem` table.

        This method uses the pgRouting 3.8+ workflow:
        1. pgr_extractVertices(): Extracts unique vertices from edge geometries
        2. UPDATE source: Links start points of edges to vertex IDs
        3. UPDATE target: Links end points of edges to vertex IDs

        This replaces the deprecated pgr_createTopology() function.
        """

        connection_query = """ SELECT setup_connection_infrastructure(); """
        self.cur.execute(connection_query)

        connection_query = """ SELECT draw_home_connections(); """
        self.cur.execute(connection_query)

        edge_table = f"ways_tem"
        vertices_table = f"{edge_table}_vertices_pgr"

        # Ensure source and target columns exist on the edge table
        # (required before pgr_extractVertices can work)
        self.cur.execute(f"""
            ALTER TABLE {edge_table} ADD COLUMN IF NOT EXISTS source integer;
            ALTER TABLE {edge_table} ADD COLUMN IF NOT EXISTS target integer;
        """)

        # Drop existing vertices table if it exists
        self.cur.execute(f"DROP TABLE IF EXISTS {vertices_table} CASCADE;")

        # Step 1: Create the vertices table using pgr_extractVertices
        self.cur.execute(f"""
            CREATE TABLE {vertices_table} AS
            SELECT id, geom
            FROM pgr_extractVertices('SELECT way_id AS id, geom FROM {edge_table} ORDER BY way_id');
        """)

        # Add primary key for performance
        self.cur.execute(f"""
            ALTER TABLE {vertices_table} ADD PRIMARY KEY (id);
        """)

        # Create spatial index on vertices for faster lookups
        self.cur.execute(f"""
            CREATE INDEX {vertices_table}_geom_idx ON {vertices_table} USING GIST (geom);
        """)

        # Step 2: Update source nodes - link start of each edge to matching vertex
        self.cur.execute(f"""
            UPDATE {edge_table} AS e
            SET source = v.id
            FROM {vertices_table} AS v
            WHERE ST_StartPoint(e.geom) = v.geom;
        """)

        # Step 3: Update target nodes - link end of each edge to matching vertex
        self.cur.execute(f"""
            UPDATE {edge_table} AS e
            SET target = v.id
            FROM {vertices_table} AS v
            WHERE ST_EndPoint(e.geom) = v.geom;
        """)

        # Create indexes on source and target for routing performance
        self.cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {edge_table}_source_idx ON {edge_table} (source);
            CREATE INDEX IF NOT EXISTS {edge_table}_target_idx ON {edge_table} (target);
        """)

        # Expose vertices table through a session-local view for easier downstream queries
        self.cur.execute(
            f"CREATE TEMP VIEW ways_tem_vertices_pgr AS SELECT * FROM {vertices_table}"
        )

    def update_ways_cost(self) -> None:
        """
        Calculates the length of each way and stores in ways_tem.cost as meter
        """
        query = """UPDATE ways_tem
                   SET cost = ST_Length(geom);
        UPDATE ways_tem
        SET reverse_cost = cost;"""
        self.cur.execute(query)

    def set_vertice_id(self) -> int:
        """
        Updates buildings_tem with the vertice_id s from ways_tem_vertices_pgr
        :return:
        """
        query = """UPDATE buildings_tem b
                   SET vertice_id = (SELECT id
                                     FROM ways_tem_vertices_pgr AS v
                                     WHERE ST_Equals(v.geom, b.center));"""
        self.cur.execute(query)

        query2 = """UPDATE buildings_tem b
                    SET connection_point = (SELECT target FROM ways_tem WHERE source = b.vertice_id LIMIT 1)
                    WHERE vertice_id IS NOT NULL
                      AND connection_point IS NULL;"""
        self.cur.execute(query2)

        count_query = """ SELECT COUNT(*)
                          FROM buildings_tem
                          WHERE connection_point IS NULL
                            AND peak_load_in_kw != 0;"""
        self.cur.execute(count_query)
        count = self.cur.fetchone()[0]

        delete_query = """DELETE
                          FROM buildings_tem
                          WHERE connection_point IS NULL
                            AND peak_load_in_kw != 0;"""
        self.cur.execute(delete_query)

        return count

    def get_fips_log(self) -> pd.DataFrame:
        """get fips log: the fips code of the region of interest of which the buildings have already been imported to the database"""
        query = """SELECT *
                   FROM fips_log;"""
        df_query = pd.read_sql_query(
            query,
            con=self.conn,
        )
        return df_query

    def keep_only_n_buildings_for_lv(self, n: int = 2) -> None:
        """
        Debug function: Keep only n LV buildings in buildings_tem table
        :param n: Number of LV buildings to keep
        """
        query = """
        WITH lv_buildings AS (
            SELECT osm_id
            FROM buildings_tem
            WHERE grid_level_connection = 'LV'
              AND peak_load_in_kw > 0
            ORDER BY osm_id
            LIMIT %(n)s
        )
        DELETE FROM buildings_tem
        WHERE grid_level_connection = 'LV'
          AND osm_id NOT IN (SELECT osm_id FROM lv_buildings);
        """
        self.cur.execute(query, {"n": n})
        self.logger.info(f"Kept only {n} LV buildings for debugging")

    def keep_only_n_buildings_for_mv(self, n: int = 2) -> None:
        """
        Debug function: Keep only n MV buildings in buildings_tem table
        :param n: Number of MV buildings to keep
        """
        query = """
        WITH mv_buildings AS (
            SELECT osm_id
            FROM buildings_tem
            WHERE grid_level_connection = 'MV'
              AND peak_load_in_kw > 0
            ORDER BY osm_id
            LIMIT %(n)s
        )
        DELETE FROM buildings_tem
        WHERE grid_level_connection = 'MV'
          AND osm_id NOT IN (SELECT osm_id FROM mv_buildings);
        """
        self.cur.execute(query, {"n": n})
        self.logger.info(f"Kept only {n} MV buildings for debugging")
