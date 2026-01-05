import os
import subprocess
import time
import warnings
from pathlib import Path

import pandas as pd
import psycopg2 as psy
import sqlparse

import src.database.database_client as dbc
from config.config_table_structure import *
from src.config_loader import *


class DatabaseConstructor:
    """
    Constructs a ready to use src database. Be careful about overwriting the tables.
    It uses databaseClient to connect to the database and create tables and import data.
    """

    def __init__(self, dbc_obj=None):
        self.extensions_added = False

        if dbc_obj:
            self.dbc = dbc_obj
        else:
            self.dbc = dbc.DatabaseClient()

    def get_table_name_list(self):
        with self.dbc.conn.cursor() as cur:
            cur.execute(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema = %s""",
                (TARGET_SCHEMA,),
            )
            table_name_list = [tup[0] for tup in cur.fetchall()]

        return table_name_list

    def table_exists(self, table_name):
        if table_name in self.get_table_name_list():
            warnings.warn(f"{table_name} table is overwritten!")
            return True
        else:
            return False

    def create_table(self, table_name):
        # create extension if not exists for recognition of geom datatypes
        if not self.extensions_added:
            with self.dbc.conn.cursor() as cur:
                # create schema if not exists
                cur.execute(f"CREATE SCHEMA IF NOT EXISTS {TARGET_SCHEMA};")
                print(f"CREATE SCHEMA {TARGET_SCHEMA}")
                self.dbc.conn.commit()
                # create extension if not exists for recognition of geom
                # datatypes
                cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                print("CREATE EXTENSION postgis")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgRouting;")
                print("CREATE EXTENSION pgRouting")
                self.dbc.conn.commit()
                self.extensions_added = True

        if table_name == "all":
            try:
                with self.dbc.conn.cursor() as cur:
                    for table_name, query in CREATE_QUERIES.items():
                        cur.execute(query, {"epsg": EPSG})
                        print(f"CREATE TABLE {table_name}")
                self.dbc.conn.commit()
            except (Exception, psy.DatabaseError) as error:
                raise error
        elif table_name in CREATE_QUERIES:
            try:
                with self.dbc.conn.cursor() as cur:
                    cur.execute(CREATE_QUERIES[table_name], {"epsg": EPSG})
                    print(f"CREATE TABLE {table_name}")
                self.dbc.conn.commit()
            except (Exception, psy.DatabaseError) as error:
                raise error
        else:
            raise ValueError(
                f"Table name {table_name} is not a valid parameter value for the function create_table. See config.py"
            )

    def ogr_to_db(self, ogr_file_list, skip_failures: bool = False):
        """
        OGR/GDAL is a translator library for raster and vector geospatial data formats
        inserts building data specified into database
        """

        for file_dict in ogr_file_list:
            st = time.time()
            file_path = Path(file_dict["path"])
            assert file_path.exists(), file_path
            file_name = file_path.stem
            table_name = file_dict.get("table_name", file_name)

            table_exists = self.table_exists(table_name=table_name)
            command = [
                "ogr2ogr",
                "-append" if table_exists else "-overwrite",
                "-progress",
                "-f",
                "PostgreSQL",
                f"PG:dbname={DBNAME} user={DBUSER} password={PASSWORD} host={HOST} port={PORT}",
                str(file_path),
                "-nln",
                # explicitly tells ogr2ogr where to append (for the case of
                # table already existing)
                f"{TARGET_SCHEMA}.{table_name}",
                "-nlt",
                # "MULTIPOLYGON",
                "PROMOTE_TO_MULTI",
                "-t_srs",
                f"EPSG:{EPSG}",
                "-lco",
                "geometry_name=geom",
                # ensures creation happens in correct schema
                "-lco",
                f"SCHEMA={TARGET_SCHEMA}",
            ]
            if skip_failures:
                command.append("-skipfailures")


            if skip_failures:
                # Capture stderr for error processing, but show stdout in real-time
                result = subprocess.run(
                    command,
                    check=True,
                    shell=False,
                    stdout=None,  # Let stdout pass through for progress
                    stderr=subprocess.PIPE,
                )

                error_list = result.stderr.decode().replace("\r", "").split("\n")
                error_list = [e[e.find("ERROR: ") : e.find("DETAIL: ")] for e in error_list]
                error_list = [e.strip("\n") for e in error_list if "ERROR: " in e]
                error_set = set(error_list)

                if error_set:
                    print(f"Warning: Error(s) occurred while processing {file_name}:")
                    for error in error_set:
                        print("\t" + error)
                        if "duplicate key value violates unique constraint" in error:
                            print("\tThis is likely due to importing already existing data.")
            else:
                # No error capture needed, let all output pass through
                result = subprocess.run(command, check=True, shell=False, stdout=None, stderr=None)

            et = time.time()
            print(f"{file_name} is successfully imported to db in {int(et - st)} s")

    def transformers_to_db(self, geojson_path: Path):
        """Load transformer data from local GeoJSON file and populate the transformers table.

        Simply loads the power.geojson file using ogr2ogr which handles CRS transformation.
        Uses -overwrite flag to upsert data.

        Args:
            geojson_path: Path to the power.geojson file. If not provided, uses the default
                         location based on REGION configuration.
        """

        # Check if the file exists
        if not os.path.isfile(geojson_path):
            print(f"Warning: Transformer file not found: {geojson_path}")
            return

        # Simply use ogr_to_db with the original file - it handles everything
        trafo_dict = [{"path": geojson_path, "table_name": "transformers"}]

        # ogr_to_db will handle CRS transformation and upsert
        self.ogr_to_db(trafo_dict, skip_failures=True)

    def csv_to_db(self, file_dict, overwrite=False):
        """
        Import a single CSV file to database.

        Args:
            file_dict: Dict with 'path' and optional 'table_name'
            overwrite: If True, delete existing data first. If False, just append.
        """
        st = time.time()
        file_path = Path(file_dict["path"])
        assert file_path.exists(), file_path
        file_name = file_path.stem
        table_name = file_dict.get("table_name", file_name)

        # Read CSV data
        df = pd.read_csv(file_path, index_col=False)

        if overwrite and self.table_exists(table_name=table_name):
            # Delete all existing data if overwrite is True
            with self.dbc.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name}")
                self.dbc.conn.commit()
            print(f"Cleared existing data from {table_name}")

        df.to_sql(
            name=table_name,
            con=self.dbc.sqla_engine,
            if_exists="append",
            index=False,
        )

        et = time.time()
        print(f"{file_name}: {len(df)} records processed in {int(et - st)}s")

    def create_public_2po_table(self):
        """
        Reads the large SQL file in 10% chunks, executes complete statements on-the-fly,
        and defers incomplete statements until the next chunk.
        """
        cur = self.dbc.conn.cursor()

        # Path to your SQL file, which includes creation of the table
        sc_path = os.path.join(
            os.getcwd(),
            "raw_data",
            "imports",
            REGION["STATE"].replace(" ", "_"),
            REGION["COUNTY"].replace(" ", "_"),
            REGION["COUNTY_SUBDIVISION"].replace(" ", "_"),
            "ROAD_NETWORK",
            "road_network.sql",
        )

        file_size = os.path.getsize(sc_path)

        # We read 10% at a time.  (Or pick a chunk size in bytes that works for
        # your environment.)
        chunk_size = max(1, file_size // 100)
        chars_read = 0

        leftover = ""  # Holds any partial statement that didn't end with a semicolon

        print("\nStart inserting ways into road_network table.")
        with open(sc_path, encoding="utf-8") as sc_file:
            while True:
                # Read next chunk
                data = sc_file.read(chunk_size)
                if not data:
                    # No more data to read
                    break

                chars_read += len(data)
                progress = round(chars_read * 100 / file_size)
                print(f"\rProgress: {progress}%", end="", flush=True)

                # Combine leftover from previous read with current chunk
                combined = leftover + data

                # Use sqlparse to split out complete statements
                statements = sqlparse.split(combined)

                # If sqlparse.split() returns multiple statements, the last one
                # might be incomplete. We’ll keep it as leftover if needed.
                if len(statements) > 1:
                    # Execute all statements except possibly the last
                    for stmt in statements[:-1]:
                        stmt = stmt.strip()
                        if stmt and not stmt.startswith("--"):
                            cur.execute(stmt)
                            self.dbc.conn.commit()

                    # Check if the last statement ends with a semicolon or not
                    last_stmt = statements[-1].strip()
                    if last_stmt.endswith(";") and not last_stmt.startswith("--"):
                        # It's a complete statement
                        cur.execute(last_stmt)
                        self.dbc.conn.commit()
                        leftover = ""
                    else:
                        leftover = last_stmt
                else:
                    # 0 or 1 statements from sqlparse
                    if len(statements) == 1:
                        # Could be complete or incomplete
                        stmt = statements[0].strip()
                        if stmt.endswith(";") and not stmt.startswith("--"):
                            # It's complete, execute it
                            cur.execute(stmt)
                            self.dbc.conn.commit()
                            leftover = ""
                        else:
                            # It's incomplete, keep it
                            leftover = stmt
                    else:
                        # No statements found. This can happen if combined was empty or whitespace.
                        # Just continue reading next chunk
                        pass
        print("\nInserted all ways into road_network table.")

    def ways_to_db(self):
        """This function transform the output of osm2po to the ways table, refer to the issue
        https://github.com/TongYe1997/Connector-syn-grid/issues/19"""

        st = time.time()

        cur = self.dbc.conn.cursor()

        # Transform to ways table
        query = """INSERT INTO ways
            SELECT  clazz,
                    source,
                    target,
                    cost,
                    reverse_cost,
                    id AS way_id,
                    ST_Transform(geom_way, %(epsg)s) as geom
            FROM road_network"""
        cur.execute(query, {"epsg": EPSG})

        # Drop road_network table, as it is not needed anymore
        query = "DROP TABLE road_network"
        cur.execute(query)

        self.dbc.conn.commit()

        et = time.time()
        print(f"Ways are successfully imported to db in {int(et - st)} s")

    def dump_functions(self):
        """
        Creates the SQL functions that are needed for the app to operate
        """
        cur = self.dbc.conn.cursor()

        # List of SQL function files to execute
        sql_files = ["postgres_dump_functions.sql"]

        for sql_file in sql_files:
            sc_path = os.path.join(os.getcwd(), "src", sql_file)

            # Check if file exists before trying to execute
            if os.path.exists(sc_path):
                with open(sc_path) as sc_file:
                    print(f"Executing {sql_file} script with schema '{TARGET_SCHEMA}'.")
                    sql = sc_file.read()
                    cur.execute(sql)
                    self.dbc.conn.commit()
                    print(f"Successfully executed {sql_file}")
            else:
                print(f"Warning: {sql_file} not found at {sc_path}, skipping...")

    def drop_all_tables(self):
        """
        Drops all tables in the database
        """
        cur = self.dbc.conn.cursor()

        cur.execute("DROP EXTENSION IF EXISTS pgRouting CASCADE;")
        print("Dropped pgRouting extension.")
        cur.execute("DROP EXTENSION IF EXISTS postgis CASCADE;")
        print("Dropped postgis extension.")

        cur.execute(f"DROP SCHEMA {TARGET_SCHEMA} CASCADE")
        self.dbc.conn.commit()
