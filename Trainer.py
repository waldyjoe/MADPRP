import argparse
import datetime
import json
import os
import pandas as pd
import pickle
import sys
import torch
# from actor_critic import A2CTrain
from constants.Settings import CPU_UTIL
from mavfa import MAVFATrain
from dqn import DQNTrain
from vfa import VFATrain

from util.utils import extract_matrix, str_to_bool



if __name__ == "__main__":
    # Save the start time of the programme running
    start_run_time = datetime.datetime.now()

    # Parse the argument
    parser = argparse.ArgumentParser(description="Training Policy")
    parser.add_argument("--sectors", default='EFL', type=str)
    parser.add_argument("--model", default="VFA", type=str)
    parser.add_argument("--pre_trained", default="False", type=str)
    parser.add_argument("--episode", default=10000, type=int)
    parser.add_argument("--poisson_mean", default=2, type=int)
    parser.add_argument("--encoding_size", default=5, type=int)
    parser.add_argument("--checkpoint", default="False", type=str)
    parser.add_argument("--single_agent", default=None, type=str)  # To run a single agent problem
    parser.add_argument("--replicate", default="False", type=str)
    parser.add_argument("--save", default="False", type=str)
    parser.add_argument("--parallel_train", default="True", type=str)
    parser.add_argument("--parallel_heuristic", default="False", type=str)
    parser.add_argument("--comms_net", default="True", type=str)
    parser.add_argument("--best_response", default="True", type=str)
    parser.add_argument("--cpu_count", default=0, type=int)

    args = parser.parse_args()

    print("Load input data..." + "\n")
    # Load a preprocessed data
    with open("./data/processed_data.pkl", "rb") as fp:
        data = pickle.load(fp)

    global_time_matrix = data.get_time_matrix()
    # Select the multiple sectors to train
    # data.show_summary(0)
    sectors = {}
    for element in args.sectors:
        sectors[element] = data.get_master_table()[element]

    for sector_id in sectors:
        sectors[sector_id].update_proximity_table(sectors, global_time_matrix)

    adj_matrix = extract_matrix(data.get_adj_matrix(), [sector_id for sector_id in sectors])
    neighbours_table = data.get_neighbours_table()

    print("Start training " + str(args.model) + " for Multi-Sectors " + str(args.sectors))

    if "vfa" in args.model.lower():

        comms = "Comms/" if str_to_bool(args.comms_net) else "NoComms/"
        br = "BR" if str_to_bool(args.best_response) else "NoBR"

        subfolder_name = comms + br

        if str_to_bool(args.parallel_train):
            # Train in parallel
            # Check the number of available CPU
            cpu_count = int(CPU_UTIL * os.cpu_count()) if args.cpu_count == 0 else args.cpu_count
            vfa_agent = MAVFATrain.train_parallel(sectors, global_time_matrix, adj_matrix, neighbours_table, cpu_count,
                                                  args, subfolder_name)
        else:
            # Train sequentially
            vfa_agent = MAVFATrain.train(sectors, global_time_matrix, adj_matrix, neighbours_table, args, subfolder_name)


    if "dqn" in args.model.lower():
        if str_to_bool(args.parallel_train):
            subfolder_name = "BR" if str_to_bool(args.best_response) else "NoBR"
            cpu_count = int(CPU_UTIL * os.cpu_count()) if args.cpu_count == 0 else args.cpu_count
            dqn_agent = DQNTrain.train_parallel(sectors, global_time_matrix, adj_matrix, neighbours_table, cpu_count,
                                                args, subfolder_name)


    # if args.model == "A2C":
    #     actor_critic = A2CTrain.train(sector, args)

    end_run_time = datetime.datetime.now()
    run_duration = (end_run_time - start_run_time).total_seconds() / 60
    print("Training Completed!")
    print("Total computation time: " + str(run_duration) + ' mins')
