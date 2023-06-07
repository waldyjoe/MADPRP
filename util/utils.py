import datetime
import haversine as hs
import json
import math
import numpy as np
import os
import sys
import requests

from collections import Counter

from constants.Settings import BETA_P, BETA_R, START_SHIFT, TIME_UNIT, T, TAU_TARGET


def calculate_haversine_distance(coord1, coord2):
    """
    Use to generate travel time matrix if real data is not available
    :param coord1: (lat, lon) in degrees
    :param coord2: (lat, lon) in degrees
    :return: haversine distance in km
    """
    return hs.haversine(coord1, coord2)


def convert_to_time_intervals(time_table, k=0):
    """
    Convert a single time-table into a list of tuple [<patrol area id>, [start_time_idx, end_time_idx]]

    :param time_table:
    :param k:
    :return:
    """
    prev_elem = 1e7
    output = []
    temp_schedule = []

    time_table = time_table[k:]

    for idx in range(len(time_table)):

        if idx == 0:
            if time_table[idx] != prev_elem:
                temp_schedule.append(time_table[idx])
                temp_schedule.append([T[idx + k]])
        else:
            if time_table[idx] != prev_elem:
                #                 temp_schedule.append(input[idx])
                temp_schedule[1].append(T[idx + k])
                #                 print(temp_schedule)
                output.append(temp_schedule)
                temp_schedule = [time_table[idx], [T[idx + k]]]
                if idx == len(time_table) - 1:
                    temp_schedule[1].append(T[idx + k] + TIME_UNIT)
                    output.append(temp_schedule)
            else:
                if idx == len(time_table) - 1:
                    temp_schedule[1].append(T[idx + k] + TIME_UNIT)
                    output.append(temp_schedule)
                else:
                    prev_elem = time_table[idx]
                    continue

        prev_elem = time_table[idx]

    return output


def convert_to_util_intervals(array):
    """
    Convert an array of number of agents at every time step to
    a list of list [<no. of agents>, [start_time_idx, end_time_idx]]
    :param array:
    :return:
    """

    prev_elem = 1e7
    output = []
    temp = []

    for idx in range(len(array)):

        if idx == 0:
            if array[idx] != prev_elem:
                temp.append(array[idx])
                temp.append([idx])
        else:
            if array[idx] != prev_elem:
                temp[1].append(idx)
                output.append(temp)
                temp = [array[idx], [idx]]
            else:
                if idx == len(array) - 1:
                    temp[1].append(idx)
                    output.append(temp)
                else:
                    prev_elem = array[idx]
                    continue

        prev_elem = array[idx]

    return output


def query_osrm(_original, _destination):
    """
    Query the open street map data for travel time between two points.
    :param _original:
    :param _destination:
    :return:
    """
    url = 'http://router.project-osrm.org/route/v1/driving/'
    url += ';'.join([','.join(_original), ','.join(_destination)])
    response = requests.get(url)
    response_json = response.json()

    return response_json["routes"][0]["duration"] / 60


def calculate_real_travel_time(coord1, coord2):
    origin = [str(coord1[0]), str(coord1[1])]
    destination = [str(coord2[0]), str(coord2[1])]

    return query_osrm(origin, destination)


def to_real_time(sim_time):
    """
    Convert simulation time to real time
    :param sim_time:
    :return:
    """
    hours = int(sim_time / 60)
    mins = int(sim_time % 60)

    return str(START_SHIFT + hours) + ":" + (str(mins) if mins > 0 else "00")


def get_time_index(sim_time):
    return int(sim_time / TIME_UNIT)


def to_sim_time(time_index):
    return int(time_index * TIME_UNIT)


def round_to_nearest(n, m):
    return n + (m - n) % m


def response_utility_fn(tau_k):
    """
    Response utility reduces with lateness
    :param tau_k: response time in minutes
    :return:
    """
    return math.exp(-1 / BETA_R * max(0, tau_k - TAU_TARGET))


def presence_utility_fn(real, required):
    """
    All patrol time within the patrol requirement has an utility of 1 while each additional patrol time unit beyond
    what is required has exponential decreasing value
    :param real: total patrol time (in time period)
    :param required: total patrol requirement (in time period)
    :return:
    """

    additional_time = int(max(0, real - required))

    if additional_time == 0:
        return real
    else:
        base_utility = required
        additional_utility = 0

        for additional_time_unit in range(1, additional_time + 1):
            additional_utility += additional_time_unit * math.exp(-1 / BETA_P * additional_time_unit)

        return base_utility + additional_utility


def merge_dict(dict1, dict2):
    dict1_orig = Counter(dict1)
    dict2_orig = Counter(dict2)

    return dict1_orig + dict2_orig


def add_tuple_key_to_dict(orig_dict, add_key):
    new_dict = {}

    for key in orig_dict.keys():
        new_dict[(key, add_key)] = orig_dict[key]

    return new_dict


def extract_matrix(df_matrix, name_list):
    """

    :param df_matrix: dataframe
    :param name_list: subset of the sector ids
    :return: a tuple of 2D np array and a dictionary to map the sector id to the column index
    """
    column_names = list(df_matrix.columns)
    mapping_table = {}

    for column_name in column_names:
        mapping_table[column_name] = column_names.index(column_name)

    np_matrix = df_matrix.to_numpy()
    new_len = len(name_list)
    new_np_matrix = np.empty(shape=(new_len, new_len), dtype='int')
    new_mapping_table = {}

    i = 0
    for name in name_list:
        idx = mapping_table[name]
        if name not in new_mapping_table.keys():
            new_mapping_table[name] = i
        new_idx = new_mapping_table[name]
        for other_name in name_list:
            if name != other_name:
                if other_name not in new_mapping_table.keys():
                    i += 1
                    new_mapping_table[other_name] = i
                old_other_idx = mapping_table[other_name]
                new_other_idx = new_mapping_table[other_name]
                new_np_matrix[new_idx][new_other_idx] = np_matrix[idx][old_other_idx]
            else:
                new_np_matrix[new_idx][new_idx] = 0

    return new_np_matrix, new_mapping_table


def one_hot_encode(x, n_classes):
    return np.eye(n_classes)[x]


def get_input_parameters(sectors, sector_ids, encoding_size):
    input_parameters = {}

    all_patrol_areas = []  # a list of all patrol area ids across the sectors
    for sector_id in sectors.keys():
        all_patrol_areas += [area.get_id() for area in sectors[sector_id].get_all_patrol_areas()]
    all_patrol_areas = sorted(all_patrol_areas)
    subagent_dim = max([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
    area_size = len(all_patrol_areas) + 2  # add 0 and -1 in the list of areas
    state_size = subagent_dim + area_size + 1  # 1 additional dimension corresponds to a list of encoded schedule
    input_parameters["n_agents"] = len(sectors.keys())
    input_parameters["state_size"] = state_size
    input_parameters["area_size"] = area_size
    input_parameters["subagent_dim"] = subagent_dim
    input_parameters["encoding_size"] = encoding_size
    input_parameters["sector_ids"] = sector_ids

    return input_parameters


def get_input_parameters_madqn(sectors, encoding_size):
    input_parameters = {}

    all_patrol_areas = []  # a list of all patrol area ids across the sectors
    for sector_id in sectors.keys():
        all_patrol_areas += [area.get_id() for area in sectors[sector_id].get_all_patrol_areas()]
    all_patrol_areas = sorted(all_patrol_areas)
    subagents_count = np.sum([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
    area_size = len(all_patrol_areas) + 2  # add 0 and -1 in the list of areas
    state_size = 1 + subagents_count + area_size + 2 # 1 additional dimension corresponds to a list of encoded schedule
    action_size = len(sectors.keys())
    input_parameters["state_size"] = state_size
    input_parameters["area_size"] = area_size
    input_parameters["action_size"] = action_size
    input_parameters["subagents_count"] = subagents_count
    input_parameters["encoding_size"] = encoding_size


    return input_parameters


def str_to_bool(input_str):
    if input_str.lower() in ["true", "yes", "t", "y", "1"]:
        return True
    else:
        return False


# NOT IN USE
def export_problem(sector, scenarios):
    with open(os.path.join(sys.path[0], 'problem.txt'), 'w') as f:
        f.write("Sector Info")
        f.write(sector.get_sector().show_summary())
        f.write("\n\n")
        f.write("ID mapping")
        f.write(str(sector.get_map_to()))
        f.write("\n\n")
        f.write("Patrol areas assigned to each agent")
        f.write(str(sector.get_table_by_sub_sector()))
        f.write("\n\n")
        f.write("Neighbouring information")
        f.write(str(sector.get_table_by_petrol_area()))
        f.write("\n\n")
        f.write("Scenarios")
        f.write(str(scenarios))
        f.write("\n\n")
        for sub_sector_id in sector.get_sector().get_master_table().keys():
            sub_sector = sector.get_sector().get_master_table()[sub_sector_id]
            f.write(str(sub_sector_id) + "_" + sub_sector.get_name() + "\n")
            patrol_area_list = []
            for patrol_area in sub_sector.get_patrol_areas():
                patrol_area_list.append((patrol_area.get_id(), patrol_area.get_name()))
            f.write(str(patrol_area_list) + "\n")


def reverse_mapping(table):
    reversed_table = {}

    for key in table.keys():
        reversed_table[table[key]] = key

    return reversed_table


def mapped_back(input, map_table):
    output = []

    # Add one more key to represent unavailability / travelling
    map_table[-1] = -1

    for i in input:
        output.append(map_table[int(i)])

    return output

#
# def calculate_real_travel_time(coord1, coord2):
#     # URL to the API
#     baseURL = "https://developers.onemap.sg/privateapi/routingsvc/route?"
#     startURL = "start="
#     endURL = "&end="
#     suffixURL = "&routeType=drive&token="
#
#     origin_coor = str(coord1[0]) + "," + str(coord1[1])
#     dest_coor = str(coord2[0]) + "," + str(coord2[1])
#
#     # give a name to .json file
#     fo1 = 'temp.json'
#
#     expiry_time = float(1617084143)
#     token = TOKEN
#
#     d = datetime.datetime.now()
#     unixtime = float(time.mktime(d.timetuple()))
#
#     if unixtime >= expiry_time:
#         token, expiry_time = get_onemap_token()
#
#     URL = '%s%s%s%s%s%s%s' % (baseURL, startURL, origin_coor, endURL, dest_coor, suffixURL, token)
#     # query the api
#     try:
#         urllib.request.urlretrieve(URL, fo1)
#         # open the output .json file and retrieve the planning area
#         with open(fo1) as data_file:
#             data = json.load(data_file)
#             # save the planning area and the name of the location into a temporary dataframe
#             travel_time = data["route_summary"]["total_time"] / 60
#             data_file.close()
#             os.remove(fo1)
#     except:
#         travel_time = 1e7
#
#     return travel_time

# def get_onemap_token():
#
#     url = "https://developers.onemap.sg/privateapi/auth/post/getToken"
#
#     payload = "{\"email\":\"waldy.joe.2018@phdcs.smu.edu.sg\",\"password\":\"Galaxy_s8\"}"
#     headers = {
#         'Content-Type': 'application/json',
#         'Cookie': 'Domain=developers.onemap.sg; onemap2=CgAACmBhSdaD+AXGBTcNAg==; _toffuid=rB8E8GBhSdYl0jCHAyM0Ag=='
#     }
#
#     response = requests.request("POST", url, headers=headers, data=payload)
#
#     return str(response.json()['access_token']), float(response.json()['expiry_timestamp'])
