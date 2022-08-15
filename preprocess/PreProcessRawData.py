import geojson
import pandas as pd
import pickle
import re
import sys

from bs4 import BeautifulSoup
from copy import deepcopy
from constants.Settings import ADJACENCY_MATRIX, BASEMAP_FILE, TRAVEL_TIME_MATRIX, GRIDS_MEDOID, GRIDS_NODES_COUNT, \
    HQ_ID,  USE_REAL_TRAVEL_TIME
from entity.Dataset import *
from entity.PatrolArea import *
from entity.Sector import *
from entity.SubSector import *
from util.utils import calculate_haversine_distance


def extract_info(description):
    """
    :param
    description:  a string describing a single patrol location
    :return: a tuple containing information about a patrol area;
    """
    soup = BeautifulSoup(description, 'html.parser')
    info_list = []

    for td in soup.find_all('td'):
        td = str(td)
        info_list.append(re.search("<td>(.+?)</td>", td).group(1))

    return info_list[0], info_list[1], info_list[2]


def process_patrol_areas_files():

    # Read all input files
    with open(BASEMAP_FILE) as f:
        gj = geojson.load(f)

    with open(TRAVEL_TIME_MATRIX, "rb") as fp:
        global_time_matrix = pickle.load(fp)

    with open(GRIDS_MEDOID, "rb") as fp:
        grids_medoid = pickle.load(fp)

    with open(GRIDS_NODES_COUNT, "rb") as fp:
        grids_nodes_count = pickle.load(fp)

    patrol_areas = gj['features']

    npc_mapping_table = {}
    div_mapping_table = {}
    sub_sectors_dict = {}
    id_no = 0
    npc_no = 0

    for patrol_area in patrol_areas:
        # Temporary Id to identify HQ
        Id = patrol_area['properties']['id']

        # Ignore the hexagon grids that do not contain any road
        if Id not in grids_medoid.keys():
            continue

        # Identify the HQ where all patrol team will start from
        # if Id == HQ_ID:
        #     hq_base = PatrolArea(Id=Id, name="HQ", coordinates=grids_medoid[Id])
        #     continue
        #
        # if Id in EXCLUSION_LIST:
        #     continue

        npc, div, div_code = extract_info(patrol_area['properties']['Description'])

        coord = grids_medoid[Id]
        demands = grids_nodes_count[Id]

        # Map the npc and division
        # id_no += 1
        # Id = id_no

        # if npc not in npc_mapping_table.keys():
        #     npc_mapping_table[npc] = npc_no
        #     npc_no += 1

        # npc_code = npc_mapping_table[npc]
        npc_kml_code = patrol_area['properties']['Name']

        if npc not in npc_mapping_table.keys():
            npc_mapping_table[npc] = []

        if npc_kml_code not in npc_mapping_table[npc]:
            npc_mapping_table[npc].append(npc_kml_code)

        if div_code not in div_mapping_table.keys():
            div_mapping_table[div_code] = []
        div_mapping_table[div_code].append(npc)

        name = str(div_code) + "_" + str(npc_kml_code) + "-" + str(Id)
        if npc not in sub_sectors_dict.keys():
            sub_sectors_dict[npc] = []
        sub_sectors_dict[npc].append(PatrolArea(Id, name, npc_kml_code, npc, div_code, div, coord, demands))

    sub_sectors = []

    for sub_sector_name in sub_sectors_dict.keys():
        sub_sectors.append(
            SubSector(str(npc_mapping_table[sub_sector_name][0]), sub_sector_name, sub_sectors_dict[sub_sector_name]))

    sectors_dict = {}

    for div_code in div_mapping_table.keys():
        if div_code not in sectors_dict.keys():
            sectors_dict[div_code] = []
        for sub_sector in sub_sectors:
            if sub_sector.get_name() in div_mapping_table[div_code]:
                sectors_dict[div_code].append(sub_sector)

    sectors = []

    for sector_id in sectors_dict.keys():
        # Exclude sectors with no. of sub-sector < 2
        if len(sectors_dict[sector_id]) < 2:
            continue

        sector_time_matrix = {}

        all_patrol_areas = []

        for sub_sector in sectors_dict[sector_id]:
            all_patrol_areas += sub_sector.get_patrol_areas()

        # orig_patrol_areas = deepcopy(all_patrol_areas)
        # Insert HQ into the beginning of the list of all patrol areas
        # all_patrol_areas.insert(0, hq_base)

        # Populate Distance Matrix
        print("Retrieving Distance Matrix...\n")
        if USE_REAL_TRAVEL_TIME:
            for patrol_area_src in all_patrol_areas:
                sector_time_matrix[patrol_area_src.get_id()] = {}
                for patrol_area_dest in all_patrol_areas:
                    if patrol_area_src.get_id() == 0 and patrol_area_dest.get_id() == 0:
                        sector_time_matrix[patrol_area_src.get_id()][patrol_area_dest.get_id()] = \
                            global_time_matrix[HQ_ID][HQ_ID]
                    elif patrol_area_src.get_id() == 0:
                        sector_time_matrix[patrol_area_src.get_id()][patrol_area_dest.get_id()] = \
                            global_time_matrix[HQ_ID][patrol_area_dest.get_id()]
                    elif patrol_area_dest.get_id() == 0:
                        sector_time_matrix[patrol_area_src.get_id()][patrol_area_dest.get_id()] = \
                            global_time_matrix[patrol_area_src.get_id()][HQ_ID]
                    else:
                        sector_time_matrix[patrol_area_src.get_id()][patrol_area_dest.get_id()] = \
                            global_time_matrix[patrol_area_src.get_id()][patrol_area_dest.get_id()]

        else:
            for patrol_area_src in all_patrol_areas:
                sector_time_matrix[patrol_area_src.get_id()] = {}
                for patrol_area_dest in all_patrol_areas:

                    if patrol_area_src.get_id() != patrol_area_dest.get_id():
                        sector_time_matrix[patrol_area_src.get_id()][patrol_area_dest.get_id()] = \
                            calculate_haversine_distance((patrol_area_src.get_lat(), patrol_area_src.get_lon()),
                                                         (patrol_area_dest.get_lat(), patrol_area_dest.get_lon()))
                    else:
                        sector_time_matrix[patrol_area_src.get_id()][patrol_area_dest.get_id()] = 0

        sectors.append(Sector(sector_id, sectors_dict[sector_id], all_patrol_areas, sector_time_matrix))

        # Manually add 2 agents to Sectors E and F
        if sector_id in ["E", "F"]:
            agents = sectors[-1].get_agents()
            for idx in range(2):
                new_agent_id = agents[idx] + str(idx)
                sectors[-1].add_agent(new_agent_id)

    # This matrix will be used to represent the relationships amongst the sector as graph and used as masking function
    # in the attention model in the communication network
    adj_matrix = pd.read_csv(ADJACENCY_MATRIX, index_col=0)

    return Dataset(sectors, global_time_matrix, adj_matrix)
