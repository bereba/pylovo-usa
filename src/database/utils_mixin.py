import warnings
from abc import ABC

from config.config_table_structure import *
from src.config_loader import *
from src.database.base_mixin import BaseMixin

warnings.simplefilter(action="ignore", category=UserWarning)


class UtilsMixin(BaseMixin, ABC):
    def __init__(self):
        super().__init__()

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def create_temp_tables(self) -> None:
        for query in TEMP_CREATE_QUERIES.values():
            self.cur.execute(query, {"epsg": EPSG})

    def drop_temp_tables(self) -> None:
        for table_name in TEMP_CREATE_QUERIES.keys():
            self.cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.cur.execute("DROP TABLE IF EXISTS ways_tem_vertices_pgr")

    def commit_changes(self):
        self.conn.commit()

    def get_list_from_regional_identifier(self, regional_identifier: int) -> list:
        query = """SELECT DISTINCT kcid, scid
                   FROM grid_result
                   WHERE version_id = %(v)s
                     AND regional_identifier = %(p)s
                   ORDER BY kcid, scid;"""
        self.cur.execute(query, {"p": regional_identifier, "v": VERSION_ID})
        cluster_list = self.cur.fetchall()

        return cluster_list

    def get_regional_identifier_from_region(self, region: dict[str, str]) -> int:
        query = """SELECT regional_identifier
                   FROM postcode
                   WHERE state_abbr = %(st)s
                    AND county_name = %(c)s
                    AND subdivision_name = %(s)s;"""
        self.cur.execute(query, {"st": region["STATE"], "c": region["COUNTY"], "s": region["COUNTY_SUBDIVISION"]})
        regional_identifier = self.cur.fetchone()[0]
        return regional_identifier

    def delete_transformers_from_buildings_tem(self, vertices: list) -> None:
        """
        Deletes selected transformers from buildings_tem
        :param vertices:
        :return:
        """
        query = """
                DELETE
                FROM buildings_tem
                WHERE vertice_id IN %(v)s;"""
        self.cur.execute(query, {"v": tuple(map(int, vertices[:,0]))})

    def get_consumer_categories(self):
        """
        Returns: A dataframe with self-defined consumer categories and typical values
        """
        query = """SELECT *
                   FROM consumer_categories"""
        cc_df = pd.read_sql_query(query, self.conn)
        cc_df.set_index("definition", drop=False, inplace=True)
        cc_df.sort_index(inplace=True)
        self.logger.debug("Consumer categories fetched.")
        return cc_df
