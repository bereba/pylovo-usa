import glob
import os

from src.config_loader import *
from src.database.database_constructor import DatabaseConstructor
from src.grid_generator import GridGenerator


def import_buildings_for_single_regional_identifier(gg: GridGenerator):
    """
    Imports building data to the database for a given FIPS code specified in the GridGenerator object.
    FIPS code is added to fips_log table to avoid importing the same building data again.

    :param gg: Grid generator object for querying relevant FIPS code data
    """
    # Retrieve regional_identifier from GridGenerator object
    dbc_client = gg.dbc
    regional_identifier = gg.regional_identifier

    # Check wether building data for this FIPS code is already in the database
    postcode_entry = dbc_client.get_postcode_table_for_regional_identifier(regional_identifier)
    gg.logger.info(
        f"LV grids will be generated for {
            postcode_entry.iloc[0]['regional_identifier']} "
        f"{postcode_entry.iloc[0]['county_name']}"
    )

    logs = dbc_client.get_fips_log()
    if regional_identifier in logs["fips_code"].values:
        gg.logger.info(f"Buildings for FIPS code {regional_identifier} have already been added to the database.")
        return
    else:
        gg.logger.info(f"Buildings for FIPS code {regional_identifier} will be added to the database.")

    # Define the path for building shapefiles
    data_path = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "raw_data",
            "imports",
            REGION["STATE"].replace(" ", "_"),
            REGION["COUNTY"].replace(" ", "_"),
            REGION["COUNTY_SUBDIVISION"].replace(" ", "_"),
            "BUILDINGS_OUTPUT",
            "SHP",
        )
    )
    shapefiles_pattern = os.path.join(data_path, "*.shp")

    # Retrieve all matching shapefiles
    files_to_add = glob.glob(shapefiles_pattern, recursive=True)

    # Handle cases where no matching files are found
    if not files_to_add:
        raise FileNotFoundError(f"No shapefiles found for regional_identifier {regional_identifier} in {data_path}")

    # Create a list of dictionaries for ogr_to_db()
    ogr_ls_dict = create_list_of_shp_files(files_to_add)

    # Add building data to the database
    sgc = DatabaseConstructor(dbc_obj=dbc_client)
    sgc.ogr_to_db(ogr_ls_dict, skip_failures=True)

    # adding the added fips_code to the log file
    dbc_client.write_fips_log(int(regional_identifier))

    gg.logger.info(f"Buildings for FIPS code {regional_identifier} have been successfully added to the database.")


def import_buildings_for_multiple_regional_identifier(regional_identifier_list: list[int]):
    """
    imports building data to db for multiple regional_identifier
    """
    # Define the path for building shapefiles
    data_path = os.path.abspath(os.path.join(PROJECT_ROOT, "raw_data", "buildings"))
    shapefiles_pattern = os.path.join(data_path, "*.shp")  # Pattern for shapefiles

    # retrieving all shape files
    files_list = glob.glob(shapefiles_pattern, recursive=True)

    # get all regional_identifiers that need to be imported for the
    # classification
    regional_identifier_to_add = list(set(regional_identifier_list))  # dropping duplicates

    # check in fips_log if any regional_identifier are already on the database
    gg = GridGenerator(regional_identifier=999999)
    dbc_client = gg.dbc
    df_log = dbc_client.get_fips_log()
    log_regional_identifier_list = df_log["fips_code"].values.tolist()
    regional_identifier_to_add = list(
        set(regional_identifier_to_add).difference(log_regional_identifier_list)
    )  # dropping already imported regional_identifier
    regional_identifier_to_add = list(map(str, regional_identifier_to_add))

    # creating a list that only contains the files to add
    files_to_add = []
    for file in files_list:
        for regional_identifier in regional_identifier_to_add:
            if regional_identifier in file:
                files_to_add.append(file)
    files_to_add = list(set(files_to_add))  # dropping duplicates

    if files_to_add:
        # define a list of required shapefiles to add to the database for the
        # function scg.ogr_to_db()
        ogr_ls_dict = create_list_of_shp_files(files_to_add)

        # adding the buildings to the database
        sgc = DatabaseConstructor()
        sgc.ogr_to_db(ogr_ls_dict)

        # adding the fips_code to the log file
        for regional_identifier in regional_identifier_to_add:
            dbc_client.write_fips_log(int(regional_identifier))


def create_list_of_shp_files(files_to_add):
    """
    Creates a list of dictionaries required for the scg.ogr_to_db() function.

    :param files_to_add: List of shapefile paths to add.
    :return: A list of dictionaries with keys "path" and "table_name".
    """
    ogr_ls_dict = []

    # Process each file path
    for file_path in files_to_add:
        # Determine table_name based on file naming convention
        if "Oth" in file_path:
            table_name = "oth"
        elif "Res" in file_path:
            table_name = "res"
        else:
            raise ValueError(f"Shapefile '{file_path}' cannot be assigned to 'res' or 'oth'.")

        ogr_ls_dict.append({"path": file_path, "table_name": table_name})

    # Ensure the list is not empty
    if ogr_ls_dict:
        return ogr_ls_dict
    else:
        raise Exception("No valid shapefiles found for the requested regional_identifier.")
