"""
Preprocess geofile to compute grid medoid, grid node count (road nodes per grid) and travel time matrix from OSRM
"""

import datetime
import geojson
import requests
import pickle
import numpy as np
import time

from util.utils import calculate_real_travel_time
from sklearn_extra.cluster import KMedoids

import sys

READ_ROAD_NODES = True
START_AFRESH = False


def pre_process_road_nodes(filename):

    with open(filename) as f:
        gj = geojson.load(f)

    road_nodes = gj['features']

    hexagon_grids = {}

    for node in road_nodes:

        grid_id = node['properties']['id']  # Id for the hexagon grid

        if grid_id not in hexagon_grids.keys():
            hexagon_grids[grid_id] = []

        coord = node['geometry']['coordinates']  # [lon,lat]

        hexagon_grids[grid_id].append(coord)

    return hexagon_grids

def find_medoid(coord_list):

    X = np.asarray(coord_list)
    kmedoids = KMedoids(n_clusters=1, random_state=0).fit(X)
    medoid = kmedoids.cluster_centers_[0]

    return medoid

if __name__ == "__main__":

    # Save the start time of the programme running
    start_run_time = datetime.datetime.now()

    # Read the raw geojson file containing the road nodes attributes
    filename = "../data/road_name_grid0.02_npc2.geojson"
    # Pre process the road nodes data into a dictionary with grid id as key and
    # the list of coordinates belonging to the grid
    hexagon_grids = pre_process_road_nodes(filencompame)

    # Find the medoid of each hexagon grid
    grids_medoid = {}
    grids_nodes_count = {}

    for grid_id in hexagon_grids.keys():
        if grid_id is not None:
            medoid = find_medoid(hexagon_grids[grid_id])
            nodes_count = len(hexagon_grids[grid_id])
            grids_medoid[grid_id] = medoid
            grids_nodes_count[grid_id] = nodes_count

    with open("../data/grids_medoid.pkl", "wb") as fp:
        pickle.dump(grids_medoid, fp)

    with open("../data/grids_nodes_count.pkl", "wb") as fp:
        pickle.dump(grids_nodes_count, fp)

    # Compute real travel time matrix for the grids based on their medoids
    print("Retrieving travel time matrix")

    if START_AFRESH:
        time_matrix = {}
    else:
        with open("../data/travel_time_matrix.pkl", "rb") as fp:
            time_matrix = pickle.load(fp)

    counter = 0
    total_grids = pow(len(grids_medoid.keys()), 2)
    for grid_src in grids_medoid.keys():
        if grid_src not in time_matrix.keys():
            time_matrix[grid_src] = {}
        for grid_dest in grids_medoid.keys():
            counter += 1
            print("Progress: {} out of {}".format(counter, total_grids))
            if grid_src != grid_dest:
                if grid_dest not in time_matrix[grid_src].keys():
                    time_matrix[grid_src][grid_dest] = \
                        calculate_real_travel_time(grids_medoid[grid_src], grids_medoid[grid_dest])

            else:
                time_matrix[grid_src][grid_dest] = 0

            if counter % 1000 == 0:
                with open("../data/travel_time_matrix.pkl", "wb") as fp:
                    pickle.dump(time_matrix, fp)

    with open("../data/travel_time_matrix.pkl", "wb") as fp:
        pickle.dump(time_matrix, fp)

    # Compute the end time
    end_run_time = datetime.datetime.now()
    run_duration = (end_run_time - start_run_time).total_seconds()
    print("Total computation time: " + str(run_duration) + 's')