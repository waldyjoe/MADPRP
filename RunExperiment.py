import argparse
import datetime
import glob
import os
import pickle
import random
import ray
import torch

from constants.Settings import *
from data.ScenarioGenerator import *
# from dqn.DQNAgent import DQNAgent
from mavfa.MAVFAAgent import *
from model.Simulator import *
from model.Simulator_parallel import *
from util.utils import extract_matrix, get_input_parameters, str_to_bool
from vfa.VFAAgent import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Run one experiment (defined as having same initial schedules) for x number scenario
# (defined as different occurrences of dynamic events) for the selected model
def run_experiment(sectors, time_matrix, adj_matrix, neighbours_table, exp_idx, model, input_parameters, trained_parameters,
                   results_raw, computation_times_raw, time_trackers, filename, args):
    """
    :param sectors: A dictionary consisting of sector id as key and sector object as values
    :param time_matrix: A global time matrix for all patrol areas across all sectors defined in sectors
    :param exp_idx: Experiment index for tracking purposes
    :param model: Model used (for e.g. VFA, DQN, myopic, greedy, etc)
    :param args: Input args specified by users
    :return: results_raw (a list of tuple <response, success, presence>) and computation_time_raw (a list of computational time per incident per scenario)
    """
    # print(datetime.datetime.now())

    # Initialize parameters
    if args.single_agent:
        is_multi_agent = False
        sectors_file_id = args.single_agent
        sectors = {k: v for k, v in sectors.items() if k == args.single_agent}
        parameter_file_dir = "./vfa/parameter/vfa_parameters_" + args.single_agent + ".pth"
        folder_name = "vfa"
    else:
        is_multi_agent = True
        sectors_file_id = args.sectors
        parameter_file_dir = "./mavfa/parameter/vfa_parameters_" + args.sectors + ".pth"
        folder_name = "mavfa"

    if str_to_bool(args.replicate):

        # Load the test scenario
        with open("./experiment/test_cases/test_case_" + str(args.sectors) + "_" + str(exp_idx) + ".pkl", "rb") as fp:
            test_case = pickle.load(fp)

        initial_schedules_dict = test_case[0]
        test_scenarios = test_case[1]

    else:
        test_scenarios = {}
        initial_schedules_dict = {}
        for sector_id in sectors.keys():
            test_data_folder = glob.glob("./data/Testing/" + str(args.poisson_mean) + "/Sector_"
                                         + str(sector_id) + "/*.pkl")
            # Generate one test case (initial schedule and several test scenarios)
            # Select the initial schedule from the testing data folder
            selected_file = random.choice(test_data_folder)
            print(selected_file)
            with open(selected_file, "rb") as fp:
                initial_schedule = pickle.load(fp)

            # Update the objective value of the initial schedule
            # (currently the objective value is based on the SetCover model) / NOT APPLICABLE
            initial_schedule.update_objective_value(get_objective_value(initial_schedule, sectors[sector_id]))
            initial_schedules_dict[sector_id] = initial_schedule

            # Generate test scenarios. The number of scenarios determined by args.scenario
            scenarios = generate_scenario(sectors[sector_id], args.scenario, args.poisson_mean)

            # Add the scenario of each sector into the combined list of scenarios across all the sectors
            for s in range(args.scenario):
                if s in test_scenarios.keys():
                    test_scenarios[s] += scenarios[s]
                else:
                    test_scenarios[s] = scenarios[s]

        # Sort the incidents across the sectors in a chronological order
        for s in range(args.scenario):
            test_scenarios[s] = sorted(test_scenarios[s], key=lambda x: x.get_start_time())

        # Save the test case for future replication
        test_case = (initial_schedules_dict, test_scenarios)
        if str_to_bool(args.save):
            with open("./experiment/test_cases/test_case_" + str(args.sectors) + "_" + str(exp_idx) + ".pkl",
                      "wb") as fp:
                pickle.dump(test_case, fp)

    # For single-agent problem, filter only relevant input data
    if not is_multi_agent:
        initial_schedules_dict = {k: v for k, v in initial_schedules_dict.items() if k == args.single_agent}
        for s in range(len(test_scenarios)):
            scenarios = [incident for incident in test_scenarios[s] if
                         incident.get_sector() == args.single_agent]
            test_scenarios[s] = scenarios

    # For each model, run the experiment for all the test scenarios
    print("Running model " + args.model + "...")

    if str_to_bool(args.parallel_heuristic) and model.lower != "greedy":
        result_raw, computation_time_raw, time_tracker = run_simulation(sectors, test_scenarios, initial_schedules_dict,
                                                                        time_matrix, adj_matrix, neighbours_table,
                                                                        model, input_parameters, trained_parameters)
        results_raw.append(result_raw)
        computation_times_raw.append(computation_time_raw)
        time_trackers.append(time_tracker)
    else:
        simulator = Simulator(sectors, test_scenarios, initial_schedules_dict, time_matrix, adj_matrix, neighbours_table,
                              best_response=str_to_bool(args.best_response))

        learning_agent = None
        if str_to_bool(args.pre_trained):
            n_agents = input_parameters["n_agents"]
            state_size = input_parameters["state_size"]
            area_size = input_parameters["area_size"]
            subagent_dim = input_parameters["subagent_dim"]
            encoding_size = input_parameters["encoding_size"]
            sector_ids = input_parameters["sector_ids"]

            if len(sectors.keys()) > 1:
                learning_agent = MAVFAAgent("False", sector_ids, n_agents, state_size, ACTION_SIZE, SEED, area_size,
                                       subagent_dim, encoding_size, HIDDEN_DIM, ATTN_OUT_DIM,
                                            trained_parameters=trained_parameters, imported_memory=[],
                                            comms_net=str_to_bool(args.comms_net))
            else:
                # Single agent problem
                learning_agent = VFAAgent(state_size, ACTION_SIZE, SEED, "False", sector_ids, area_size, subagent_dim,
                                     encoding_size, trained_parameters=trained_parameters)

        result_raw, computation_time_raw = simulator.run(model, learning_agent)  # Input policy (in string) and learning agent
        # Return the raw results and raw computation times for each scenario in a given experiment
        results_raw.append(result_raw)
        computation_times_raw.append(computation_time_raw)
        time_trackers.append(simulator.get_time_tracker())

    if str_to_bool(args.record_result):
        # Save the raw results
        with open("./output/experiment_results_raw_" + filename + ".pkl", "wb") as fp:
            pickle.dump(results_raw, fp)

        # Save the raw computational timings
        with open("./output/computation_times_raw_" + filename + ".pkl", "wb") as fp:
            pickle.dump(computation_times_raw, fp)

        # Save time tracker
        with open("./output/time_tracker_" + filename + ".pkl", "wb") as fp:
            pickle.dump(time_trackers, fp)


@ray.remote
def run_experiment_parallel(sectors, time_matrix, adj_matrix, neighbours_table, exp_idx, model, input_parameters,
                            trained_parameters, args):

    print("Running experiment no: " + str(exp_idx + 1))

    # Initialize parameters
    if args.single_agent:
        is_multi_agent = False
        sectors_file_id = args.single_agent
        sectors = {k: v for k, v in sectors.items() if k == args.single_agent}
        parameter_file_dir = "./vfa/parameter/vfa_parameters_" + args.single_agent + ".pth"
        folder_name = "vfa"
    else:
        is_multi_agent = True
        sectors_file_id = args.sectors
        parameter_file_dir = "./mavfa/parameter/vfa_parameters_" + args.sectors + ".pth"
        folder_name = "mavfa"

    if str_to_bool(args.replicate):

        # Load the test scenario
        with open("./experiment/test_cases/test_case_" + str(args.sectors) + "_" + str(exp_idx) + ".pkl", "rb") as fp:
            test_case = pickle.load(fp)

        initial_schedules_dict = test_case[0]
        test_scenarios = test_case[1]

    else:
        test_scenarios = {}
        initial_schedules_dict = {}
        for sector_id in sectors.keys():
            test_data_folder = glob.glob("./data/Testing/" + str(args.poisson_mean) + "/Sector_"
                                         + str(sector_id) + "/*.pkl")
            # Generate one test case (initial schedule and several test scenarios)
            # Select the initial schedule from the testing data folder
            selected_file = random.choice(test_data_folder)
            print(selected_file)
            with open(selected_file, "rb") as fp:
                initial_schedule = pickle.load(fp)

            # Update the objective value of the initial schedule
            # (currently the objective value is based on the SetCover model) / NOT APPLICABLE
            initial_schedule.update_objective_value(get_objective_value(initial_schedule, sectors[sector_id]))
            initial_schedules_dict[sector_id] = initial_schedule

            # Generate test scenarios. The number of scenarios determined by args.scenario
            scenarios = generate_scenario(sectors[sector_id], args.scenario, args.poisson_mean)

            # Add the scenario of each sector into the combined list of scenarios across all the sectors
            for s in range(args.scenario):
                if s in test_scenarios.keys():
                    test_scenarios[s] += scenarios[s]
                else:
                    test_scenarios[s] = scenarios[s]

        # Sort the incidents across the sectors in a chronological order
        for s in range(args.scenario):
            test_scenarios[s] = sorted(test_scenarios[s], key=lambda x: x.get_start_time())

        # Save the test case for future replication
        test_case = (initial_schedules_dict, test_scenarios)
        if str_to_bool(args.save):
            with open("./experiment/test_cases/test_case_" + str(args.sectors) + "_" + str(exp_idx) + ".pkl",
                      "wb") as fp:
                pickle.dump(test_case, fp)

    if not is_multi_agent:
        initial_schedules_dict = {k: v for k, v in initial_schedules_dict.items() if k == args.single_agent}
        for s in range(len(test_scenarios)):
            scenarios = [incident for incident in test_scenarios[s] if
                         incident.get_sector() == args.single_agent]
            test_scenarios[s] = scenarios

    # For each model, run the experiment for all the test scenarios
    print("Running model " + args.model + "...")

    if str_to_bool(args.parallel_heuristic) and model.lower != "greedy":
        result_raw, computation_time_raw, time_tracker = run_simulation(sectors, test_scenarios, initial_schedules_dict,
                                                                        time_matrix, adj_matrix, neighbours_table,
                                                                        model, input_parameters, trained_parameters)

        return exp_idx, result_raw, computation_time_raw, time_tracker

    else:
        simulator = Simulator(sectors, test_scenarios, initial_schedules_dict, time_matrix, adj_matrix,
                              neighbours_table, best_response=str_to_bool(args.best_response))

        learning_agent = None
        if str_to_bool(args.pre_trained):
            n_agents = input_parameters["n_agents"]
            state_size = input_parameters["state_size"]
            area_size = input_parameters["area_size"]
            subagent_dim = input_parameters["subagent_dim"]
            encoding_size = input_parameters["encoding_size"]
            sector_ids = input_parameters["sector_ids"]

            if len(sectors.keys()) > 1:
                learning_agent = MAVFAAgent("False", sector_ids, n_agents, state_size, ACTION_SIZE, SEED, area_size,
                                       subagent_dim, encoding_size, HIDDEN_DIM, ATTN_OUT_DIM,
                                            trained_parameters=trained_parameters, imported_memory=[],
                                            comms_net=str_to_bool(args.comms_net))
            else:
                # Single agent problem
                learning_agent = VFAAgent(state_size, ACTION_SIZE, SEED, "False", sector_ids, area_size, subagent_dim,
                                     encoding_size, trained_parameters=trained_parameters)

        result_raw, computation_time_raw = simulator.run(model, learning_agent)  # Input policy (in string) and learning agent
        # Return the raw results and raw computation times for each scenario in a given experiment
        return exp_idx, result_raw, computation_time_raw, simulator.get_time_tracker()

if __name__ == "__main__":
    # Save the start time of the programme running
    start_run_time = datetime.datetime.now()

    # Parse the argument
    parser = argparse.ArgumentParser(description="Running Experiments")
    parser.add_argument("--sectors", default='EFL', type=str)
    parser.add_argument("--experiment", default=30, type=int)
    parser.add_argument("--scenario", default=20, type=int)
    parser.add_argument("--replicate", default="False", type=str)
    parser.add_argument("--save", default="False", type=str)
    parser.add_argument("--model", default="Myopic", type=str)
    parser.add_argument("--poisson_mean", default=2, type=int)
    parser.add_argument("--pre_trained", default="False", type=str)
    parser.add_argument("--encoding_size", default=5, type=int)
    parser.add_argument("--parallel_exp", default="True", type=str)
    parser.add_argument("--parallel_heuristic", default="False", type=str)
    parser.add_argument("--record_result", default="True", type=str)
    parser.add_argument("--single_agent", default=None, type=str)  # To run a single agent problem
    parser.add_argument("--comms_net", default="True", type=str)
    parser.add_argument("--best_response", default="True", type=str)
    parser.add_argument("--cpu_count", default=0, type=int)

    args = parser.parse_args()

    # Default filename to standardize the naming of the output files
    if args.single_agent:
        default_filename = str(args.sectors) + "_" + str(args.single_agent) + "_" + str(args.model.lower()) + "_" + str(args.experiment) + "_" + str(
            args.scenario)
    else:
        default_filename = str(args.sectors) + "_" + str(args.model.lower()) + "_" + str(args.experiment) + "_" + str(args.scenario)

    # Load a preprocessed data
    print("Load input data..." + "\n")
    with open("./data/processed_data.pkl", "rb") as fp:
        data = pickle.load(fp)
    # data.show_summary(1)

    # Extract the global time travel matrix
    global_time_matrix = data.get_time_matrix()

    # Create a dictionary of sectors
    sectors = {}
    for element in args.sectors:
        sectors[element] = data.get_master_table()[element]

    for sector_id in sectors:
        sectors[sector_id].update_proximity_table(sectors, global_time_matrix)

    adj_matrix = extract_matrix(data.get_adj_matrix(), [sector_id for sector_id in sectors])
    neighbours_table = data.get_neighbours_table()
    input_parameters = {}  # Input parameters to build the value function agent
    trained_parameters = None

    print("Start experiments for Multi-Sectors " + str(args.sectors) + " with " + str(args.model.lower()))

    # Load the pre-trained model (optional)
    if str_to_bool(args.pre_trained) and args.model.lower() not in ["myopic", "greedy"] :

        # all_patrol_areas = []  # a list of all patrol area ids across the sectors
        # for sector_id in sectors.keys():
        #     all_patrol_areas += [area.get_id() for area in sectors[sector_id].get_all_patrol_areas()]
        # all_patrol_areas = sorted(all_patrol_areas)
        # subagent_dim = max([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
        # area_size = len(all_patrol_areas) + 2  # add 0 and -1 in the list of areas
        # state_size = subagent_dim + area_size + 1  # 1 additional dimension corresponds to a list of encoded schedule
        # input_parameters["n_agents"] = len(sectors.keys())
        # input_parameters["state_size"] = state_size
        # input_parameters["area_size"] = area_size
        # input_parameters["subagent_dim"] = subagent_dim
        # input_parameters["encoding_size"]= args.encoding_size
        # input_parameters["sector_ids"] = args.sectors
        if args.single_agent:
            sectors = {k: v for k, v in sectors.items() if k == args.single_agent}
            input_parameters = get_input_parameters(sectors, args.single_agent, args.encoding_size)
            trained_parameters = torch.load("./vfa/parameter/vfa_parameters_" + args.single_agent + ".pth",
                                            map_location=device)
        else:
            comms = "Comms/" if str_to_bool(args.comms_net) else "NoComms/"
            br = "BR" if str_to_bool(args.best_response) else "NoBR"

            subfolder_name = comms + br

            default_filename = str(args.sectors) + "_" + str(args.model.lower()) + "_" + comms[:-1] + "_" + br + "_" + str(
                args.experiment) + "_" + str(args.scenario)

            input_parameters = get_input_parameters(sectors, args.sectors, args.encoding_size)
            trained_parameters = torch.load("./mavfa/" + subfolder_name + "/parameter/vfa_parameters_" +
                                            args.sectors + "_final.pth", map_location=device)


    # Output files in the form of a list of a list
    results_raw = []
    computation_times_raw = []
    time_trackers = []

    if str_to_bool(args.parallel_exp):

        cpu_count = int(CPU_UTIL * os.cpu_count()) if args.cpu_count == 0 else args.cpu_count
        i = 0
        num_exp_remain = args.experiment

        while num_exp_remain > 0:

            num_parallel_run = min(num_exp_remain, cpu_count)
            parallel_results = []
            for idx in range(num_parallel_run):
                parallel_results.append(run_experiment_parallel.remote(sectors, global_time_matrix, adj_matrix,
                                                                       neighbours_table, i + idx, args.model,
                                                                       input_parameters, trained_parameters, args))
            results = ray.get(parallel_results)
            # Result format (episode_idx, response_count, success_count, presence_score, experiences)
            results = sorted(results, key=lambda x: x[0])

            for idx in range(num_parallel_run):
                results_raw.append(results[idx][1])
                computation_times_raw.append(results[idx][2])
                time_trackers.append(results[idx][3])

            if str_to_bool(args.record_result):
                # Save the raw results
                with open("./output/experiment_results_raw_" + default_filename + ".pkl", "wb") as fp:
                    pickle.dump(results_raw, fp)

                # Save the raw computational timings
                with open("./output/computation_times_raw_" + default_filename + ".pkl", "wb") as fp:
                    pickle.dump(computation_times_raw, fp)

                # Save time tracker
                with open("./output/time_tracker_" + default_filename + ".pkl", "wb") as fp:
                    pickle.dump(time_trackers, fp)

            num_exp_remain -= num_parallel_run
            print(str(args.experiment - num_exp_remain) + " out of " + str(args.experiment) + " episodes completed")
            i += num_parallel_run

    else:
        # Run experiments sequentially
        for i in range(args.experiment):
            print("Running experiment no: " + str(i + 1))
            run_experiment(sectors, global_time_matrix, adj_matrix, neighbours_table, i, args.model, input_parameters,
                           trained_parameters, results_raw, computation_times_raw, time_trackers, default_filename, args)
            print(str(i + 1) + " out of " + str(args.experiment) + " episodes completed")
        # results_raw.append(result_raw)
        # computation_times_raw.append(computation_time_raw)





    end_run_time = datetime.datetime.now()
    run_duration = (end_run_time - start_run_time).total_seconds()
    print("Experiments Completed!")
    print("Total computation time: " + str(run_duration) + 's')
