import warnings
from abc import ABC
from typing import Any

from shapely.geometry import LineString

from src.config_loader import *
from src.database.base_mixin import BaseMixin
from src.equipment_schema import TransformerEquipment, create_equipment_from_database_row

warnings.simplefilter(action="ignore", category=UserWarning)


class GridMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def get_vertices_from_bcid(self, regional_identifier: int, kcid: int, bcid: int, scid) -> tuple[dict, int]:
        """
        Get vertices and distance mapping for LV cable placement algorithm.

        This method retrieves all vertices (buildings and connection points) for a given
        LV cluster (bcid) and computes routing distances from each vertex to the
        distribution transformer using Dijkstra's algorithm.

        Args:
            regional_identifier: Regional identifier (postcode)
            kcid: K-means cluster ID
            bcid: Building cluster ID (LV transformer cluster)
            scid: Substation cluster ID

        Returns:
            Tuple of (vertex_distance_mapping, transformer_vertex_id) where:
            - vertex_distance_mapping: Dict mapping vertex_id -> distance_meters from transformer
            - transformer_vertex_id: Vertex ID of the distribution transformer
        """
        # get Transformer_vertice_ids from lv_grid_result table (for LV transformers)
        # If scid is provided, query the hierarchical table
        transformer_query = """SELECT dist_transformer_vertice_id
                                FROM lv_grid_result
                                WHERE version_id = %s
                                AND regional_identifier = %s
                                AND kcid = %s
                                AND scid = %s
                                AND bcid = %s"""
        self.cur.execute(transformer_query, (VERSION_ID, regional_identifier, kcid, scid, bcid))
        result = self.cur.fetchone()
        transformer = result[0] if result else None

        consumer_query = """SELECT vertice_id
                            FROM buildings_tem
                            WHERE regional_identifier = %(p)s
                              AND kcid = %(k)s
                              AND bcid = %(b)s AND scid = %(s)s;"""

        params = {"p": regional_identifier, "k": kcid, "b": bcid, "s": scid}

        self.cur.execute(consumer_query, params)
        consumer = [t[0] for t in self.cur.fetchall()]

        connection_query = """SELECT DISTINCT connection_point
                              FROM buildings_tem
                              WHERE regional_identifier = %(p)s
                                AND kcid = %(k)s
                                AND bcid = %(b)s AND scid = %(s)s;"""
        self.cur.execute(connection_query, params)
        connection = [t[0] for t in self.cur.fetchall()]

        vertices_query = """ SELECT DISTINCT node, agg_cost
                             FROM pgr_dijkstra(
                                     'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem'::text,
                                     %(o)s, %(c)s::integer[], false)
                             ORDER BY agg_cost;"""

        self.cur.execute(vertices_query, {"o": transformer, "c": consumer})
        data = self.cur.fetchall()
        # data contains tuples of (vertex_id, routing_distance_meters) from the Dijkstra query
        # t[0] = vertex ID, t[1] = routing distance in meters
        vertex_distance_mapping = {t[0]: t[1] for t in data if t[0] in consumer or t[0] in connection}

        return vertex_distance_mapping, transformer

    def get_node_geom(self, vid: int):
        query = """SELECT ST_X(ST_Transform(geom, 4326)), ST_Y(ST_Transform(geom, 4326))
                   FROM ways_tem_vertices_pgr
                   WHERE id = %(id)s;"""
        self.cur.execute(query, {"id": vid})
        geo = self.cur.fetchone()

        return geo

    def get_vertices_from_connection_points(self, connection: list) -> list:
        query = """SELECT vertice_id
                   FROM buildings_tem
                   WHERE connection_point IN %(c)s
                     AND type != 'Transformer';"""
        self.cur.execute(query, {"c": tuple(connection)})
        data = self.cur.fetchall()
        return [t[0] for t in data]

    def get_path_to_bus(self, vertice: int, transformer: int) -> list:
        """routing problem: find the shortest path from vertice to the transformer of the cluster"""
        query = """SELECT node
                   FROM pgr_Dijkstra(
                           'SELECT way_id as id, source, target, cost, reverse_cost FROM ways_tem', %(v)s, %(o)s,
                           false);"""
        """query = WITH
                    dijkstra AS(
                        SELECT * FROM pgr_Dijkstra(
                                        'SELECT way_id, source, target, cost, reverse_cost FROM ways_tem', %(v)s, %(o)s, false)
                    ),
                        get_geom AS(
                            SELECT dijkstra. *,
                            -- adjusting directionality
                                CASE
                                    WHEN dijkstra.node = ways.source THEN geom
                                    ELSE ST_Reverse(geom)
                                END AS route_geom
                            FROM dijkstra JOIN ways ON(edge=way_id)
                            ORDER BY seq)
                        SELECT seq, cost,
                        degrees(ST_azimuth(ST_StartPoint(route_geom), ST_EndPoint(route_geom))) AS azimuth,
                        ST_AsText(route_geom),
                        route_geom
                    FROM get_geom
                    ORDER BY seq;"""
        self.cur.execute(query, {"o": transformer, "v": vertice})
        data = self.cur.fetchall()
        way_list = [t[0] for t in data]

        return way_list

    def get_path_to_bus_with_length(
        self,
        load_vertex: int,
        substation_vertex_id: int,
        geom_col: str = "geom",  # set to "geom" if that's your column
    ) -> tuple[list[int], float]:
        """
        Returns (path_nodes, length_meters) for the shortest path between two vertices.
        Length is computed on WGS84 geography (meters). Works even if ways_tem is projected.
        """
        sql = f"""
        WITH route AS (
        SELECT seq, node, edge
        FROM pgr_dijkstra(
            'SELECT way_id AS id, source, target, cost, reverse_cost FROM ways_tem',
            %(start)s, %(goal)s, false
        )
        ORDER BY seq
        ),
        nodes AS (
        SELECT array_agg(node ORDER BY seq) AS nodes
        FROM route
        ),
        edges AS (
        SELECT
            CASE
            WHEN r.node = w.source THEN w.{geom_col}
            ELSE ST_Reverse(w.{geom_col})
            END AS geom
        FROM route r
        JOIN ways_tem w ON w.way_id = r.edge
        WHERE r.edge <> -1
        )
        SELECT
        (SELECT nodes FROM nodes) AS nodes,
        COALESCE(
            (SELECT SUM(ST_Length(ST_Transform(geom, 4326)::geography)) FROM edges),
            0.0
        ) AS meters;
        """
        self.cur.execute(sql, {"start": load_vertex, "goal": substation_vertex_id})
        row = self.cur.fetchone()
        if not row:
            return [], 0.0

        nodes, meters = row
        return (nodes or []), float(meters)

    def insert_mv_line(
        self,
        geom: list,
        kcid: int,
        scid: int,
        line_name: str,
        equipment_id: str,
        from_bus: int,
        to_bus: int,
        length_km: float,
    ) -> None:
        """Insert MV line (20kV) into the database."""
        query = """
        INSERT INTO lines_result (
            grid_result_id, lv_grid_result_id, grid_level,
            line_name, std_type, equipment_id,
            from_bus, to_bus, length_km, geom
        )
        VALUES (
            (SELECT grid_result_id FROM grid_result
             WHERE version_id = %(v)s AND kcid = %(kcid)s AND scid = %(scid)s),
            NULL,
            'MV',
            %(line_name)s,
            %(equipment_id)s,
            %(equipment_id)s,
            %(from_bus)s,
            %(to_bus)s,
            %(length_km)s,
            ST_SetSRID(%(geom)s::geometry, %(epsg)s)
        )
        """
        self.cur.execute(
            query,
            {
                "v": VERSION_ID,
                "kcid": int(kcid),
                "scid": int(scid),
                "line_name": line_name,
                "equipment_id": equipment_id,
                "from_bus": int(from_bus),
                "to_bus": int(to_bus),
                "length_km": float(length_km),
                "geom": LineString(geom).wkb_hex,
                "epsg": EPSG,
            },
        )

    def insert_lv_line(
        self,
        geom: list,
        kcid: int,
        scid: int,
        bcid: int,
        line_name: str,
        equipment_id: str,
        from_bus: int,
        to_bus: int,
        length_km: float,
    ) -> None:
        """Insert LV line (400V) into the database."""
        query = """
        INSERT INTO lines_result (
            grid_result_id, lv_grid_result_id, grid_level,
            line_name, std_type, equipment_id,
            from_bus, to_bus, length_km, geom
        )
        VALUES (
            (SELECT grid_result_id FROM grid_result
             WHERE version_id = %(v)s AND kcid = %(kcid)s AND scid = %(scid)s),
            (SELECT lv_grid_result_id FROM lv_grid_result
             WHERE version_id = %(v)s AND kcid = %(kcid)s
               AND scid = %(scid)s AND bcid = %(bcid)s),
            'LV',
            %(line_name)s,
            %(equipment_id)s,
            %(equipment_id)s,
            %(from_bus)s,
            %(to_bus)s,
            %(length_km)s,
            ST_SetSRID(%(geom)s::geometry, %(epsg)s)
        )
        """
        self.cur.execute(
            query,
            {
                "v": VERSION_ID,
                "kcid": int(kcid),
                "scid": int(scid),
                "bcid": int(bcid),
                "line_name": line_name,
                "equipment_id": equipment_id,
                "from_bus": int(from_bus),
                "to_bus": int(to_bus),
                "length_km": float(length_km),
                "geom": LineString(geom).wkb_hex,
                "epsg": EPSG,
            },
        )

    def is_grid_generated(self, regional_identifier: int):
        """
        Check if grid exists.

        Args:
            regional_identifier: Postal code to be checked

        Returns:
            bool: True if record exists, False otherwise
        """
        query = """
            SELECT 1
            FROM postcode_result
            WHERE version_id = %(version_id)s AND postcode_result_regional_identifier = %(regional_identifier)s
            LIMIT 1;
        """

        self.cur.execute(
            query,
            {"version_id": VERSION_ID, "regional_identifier": regional_identifier},
        )
        result = self.cur.fetchone()
        return result is not None

    def get_substation_for_scid(self, kcid: int, scid: int) -> tuple[TransformerEquipment, int]:
        """
        Get substation data with full equipment specifications for MV network construction.

        Returns:
            Dictionary with substation_vertice_id, substation_rated_power, and equipment object
        """
        # First get the substation data
        query = """
        SELECT substation_vertice_id, equipment_id
        FROM grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid))
        result = self.cur.fetchone()

        if result:
            substation_vertice_id = result[0]
            equipment_id = result[1]

            # Now fetch the equipment data
            equipment_query = """
            SELECT * FROM equipment_data
            WHERE name = %s
            """
            self.cur.execute(equipment_query, (equipment_id,))
            equipment_row = self.cur.fetchone()

            if not equipment_row:
                raise ValueError(f"Equipment not found: {equipment_id}")

            # Convert to dict and create equipment object
            columns = [desc[0] for desc in self.cur.description]
            equipment_dict = dict(zip(columns, equipment_row, strict=False))
            equipment = create_equipment_from_database_row(equipment_dict)

            return (
                equipment,
                substation_vertice_id,
            )

        return None

    def get_lv_transformers_for_scid(self, kcid: int, scid: int) -> list[dict]:
        """
        Get all LV transformers with full equipment specifications for a given scid.

        Returns:
            List of dicts with bcid, transformer vertex, power rating, and equipment object
        """
        # First get all transformer data
        query = """
        SELECT bcid, dist_transformer_vertice_id,
               dist_transformer_rated_power, equipment_id
        FROM lv_grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid))
        results = self.cur.fetchall()

        transformer_list = []
        for row in results:
            bcid = row[0]
            transformer_vertice_id = row[1]
            transformer_rated_power = row[2]
            equipment_id = row[3]

            # Fetch the equipment data for each transformer
            equipment_query = """
            SELECT * FROM equipment_data
            WHERE name = %s
            """
            self.cur.execute(equipment_query, (equipment_id,))
            equipment_row = self.cur.fetchone()

            if not equipment_row:
                raise ValueError(f"Equipment not found: {equipment_id}")

            # Convert to dict and create equipment object
            columns = [desc[0] for desc in self.cur.description]
            equipment_dict = dict(zip(columns, equipment_row, strict=False))
            equipment = create_equipment_from_database_row(equipment_dict)

            transformer_list.append(
                {
                    "bcid": bcid,
                    "transformer_vertice_id": transformer_vertice_id,
                    "transformer_rated_power": transformer_rated_power,
                    "equipment": equipment,  # Full equipment object
                }
            )

        return transformer_list

    def get_mv_buildings_for_scid(self, kcid: int, scid: int) -> list[dict]:
        """
        Get MV buildings (high load buildings that connect directly to MV) for a given scid.

        Returns:
            List of dicts with building information for direct MV connections
        """
        query = """
        SELECT osm_id, connection_point, peak_load_in_kw, type, vertice_id
        FROM buildings_tem
        WHERE kcid = %s
          AND scid = %s
          AND grid_level_connection = 'MV'
        ORDER BY peak_load_in_kw DESC
        """
        self.cur.execute(query, (kcid, scid))
        results = self.cur.fetchall()

        return [
            {
                "osm_id": row[0],
                "connection_point": row[1],
                "peak_load_kw": row[2],
                "building_type": row[3],
                "vertice_id": row[4],  # Building centroid vertex ID
            }
            for row in results
        ]

    def get_bcids_for_scid(self, kcid: int, scid: int) -> list[int]:
        """
        Get all bcid clusters under a given substation cluster (scid).

        Returns:
            List of bcid integers
        """
        query = """
        SELECT DISTINCT bcid
        FROM lv_grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
        ORDER BY bcid
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid))
        results = self.cur.fetchall()

        return [row[0] for row in results]

    def get_lv_transformer_for_bcid(self, kcid: int, scid: int, bcid: int) -> dict:
        """
        Get LV transformer data with full equipment specifications for a specific bcid cluster.

        Returns:
            Dictionary with transformer vertex, power rating, and equipment object
        """
        # First get the transformer data
        query = """
        SELECT dist_transformer_vertice_id, dist_transformer_rated_power, equipment_id
        FROM lv_grid_result
        WHERE version_id = %s
          AND kcid = %s
          AND scid = %s
          AND bcid = %s
        """
        self.cur.execute(query, (VERSION_ID, kcid, scid, bcid))
        result = self.cur.fetchone()

        if result:
            transformer_vertice_id = result[0]
            transformer_rated_power = result[1]
            equipment_id = result[2]

            # Now fetch the equipment data
            equipment_query = """
            SELECT * FROM equipment_data
            WHERE name = %s
            """
            self.cur.execute(equipment_query, (equipment_id,))
            equipment_row = self.cur.fetchone()

            if not equipment_row:
                raise ValueError(f"Equipment not found: {equipment_id}")

            # Convert to dict and create equipment object
            columns = [desc[0] for desc in self.cur.description]
            equipment_dict = dict(zip(columns, equipment_row, strict=False))
            equipment = create_equipment_from_database_row(equipment_dict)

            return {
                "transformer_vertice_id": transformer_vertice_id,
                "transformer_rated_power": transformer_rated_power,
                "equipment": equipment,  # Full equipment object
            }
        return None

    def save_grid_cluster(self, regional_identifier: int, kcid: int, scid: int, grid_data: dict[str, Any]) -> None:
        """
        Save grid construction results for a single cluster to database.

        Args:
            kcid: K-means cluster ID
            scid: Substation cluster ID
            grid_data: Complete grid data from backend export
        """
        import json

        # Convert grid_data to JSON string if needed
        grid_json = json.dumps(grid_data) if not isinstance(grid_data, str) else grid_data

        # Use UPSERT to ensure grid_result row exists
        upsert_query = """
        INSERT INTO grid_result (
            version_id, regional_identifier, kcid, scid,
            substation_rated_power, substation_vertice_id,
            equipment_id, model_status, grid
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (version_id, regional_identifier, kcid, scid)
        DO UPDATE SET
            grid = EXCLUDED.grid,
            model_status = COALESCE(EXCLUDED.model_status, grid_result.model_status)
        """

        # Set default values for missing fields (will be updated during
        # clustering if needed)
        self.cur.execute(
            upsert_query,
            (
                str(VERSION_ID),
                int(regional_identifier),
                int(kcid),
                int(scid),
                0,  # default substation_rated_power
                0,  # default substation_vertice_id
                None,  # equipment_id = NULL (avoid FK constraint)
                1,  # model_status = 1 (completed)
                grid_json,
            ),
        )
