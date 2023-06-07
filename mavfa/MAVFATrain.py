import datetime
import numpy as np
import random
import math
import torch.multiprocessing as mp
import glob
import ray
import sys
import os
import pickle
import torch

from copy import deepcopy
from ray.util import inspect_serializability

from constants.Settings import ACTION_SIZE, ATTN_OUT_DIM, BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START, HIDDEN_DIM, SEED, T
from data.ScenarioGenerator import generate_scenario
from mavfa.MAVFAAgent import MAVFAAgent
from model.Simulator import Simulator
from model.Simulator_parallel import *
from util.ScheduleUtil import compute_hamming_distance_joint, get_global_Q_j, get_objective_value, \
    get_objective_value_MA, get_patrol_presence_status_MA, get_patrol_count_table_MA, get_post_joint_state, \
    get_post_state
from util.utils import get_time_index, str_to_bool
from vfa.VFAAgent import VFAAgent

# ray.init(address=os.environ["ip_head"])
# print("Nodes in the Ray cluster:")
# print(ray.nodes())


def train(sectors, time_matrix, adj_matrix, neighbours_table, args, subfolder_name):
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
        parameter_file_dir = "./mavfa/" + subfolder_name + "/parameter/vfa_parameters_" + args.sectors + ".pth"
        folder_name = "mavfa/" + subfolder_name

    D = time_matrix  # Time travel matrix for all patrol areas across sectors
    Q_j = get_global_Q_j(sectors)
    all_patrol_areas = []  # a list of all patrol area ids across the sectors
    patrol_area_to_sector_map = {}
    for sector_id in sectors.keys():
        for area in sectors[sector_id].get_all_patrol_areas():
            patrol_area_to_sector_map[area.get_id()] = sector_id
            all_patrol_areas.append(area.get_id())
    all_patrol_areas = sorted(all_patrol_areas)
    mask = np.stack([adj_matrix[0] for n in range(BATCH_SIZE)])
    mask_nobatch = np.array([adj_matrix[0]])

    # Declare VFA agent
    n_agents = len(sectors.keys())
    subagent_dim = max([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
    area_size = len(all_patrol_areas) + 2  # add 0 and -1 in the list of areas
    state_size = subagent_dim + area_size + 1  # 1 additional dimension corresponds to a list of encoded schedule

    input_parameters = {"n_agents": n_agents, "state_size": state_size, "area_size": area_size,
                        "subagent_dim": subagent_dim, "encoding_size": args.encoding_size, "sector_ids": args.sectors}

    # Load existing trained parameters
    trained_parameters = None
    imported_memory = None

    if str_to_bool(args.pre_trained):
        trained_parameters = torch.load(parameter_file_dir, map_location=device)
        try:
            with open("./" + folder_name + "/parameter/vfa_replaybuffer_" + sectors_file_id + ".pkl", "rb") as fp:
                imported_memory = pickle.load(fp)
        except:
            pass

    if is_multi_agent:
        # Multi agent problem
        vfa_agent = MAVFAAgent(args.pre_trained, args.sectors, n_agents, state_size, ACTION_SIZE, SEED, area_size,
                               subagent_dim,
                               args.encoding_size, HIDDEN_DIM, ATTN_OUT_DIM, trained_parameters=trained_parameters,
                               imported_memory=imported_memory, comms_net=str_to_bool(args.comms_net))
    else:
        # Single agent problem
        sector = sectors[args.single_agent]
        vfa_agent = VFAAgent(state_size, ACTION_SIZE, SEED, args.pre_trained, sector, area_size, subagent_dim,
                             args.encoding_size, trained_parameters=trained_parameters,
                             imported_memory=imported_memory)

    i = 0  # Iterator for number of episodes
    steps_done = 0

    # For plotting
    loss_by_step = []
    improve_response_by_episode = []
    improve_success_by_episode = []
    improve_presence_by_episode = []

    if str_to_bool(args.checkpoint):
        with open("./" + folder_name + "/output/vfa_loss_by_step_" + sectors_file_id + ".pkl", "rb") as fp:
            loss_by_step = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_response_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_response_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_success_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_success_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_presence_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_presence_by_episode = pickle.load(fp)




    training_instances = []

    training_folder = "./data/Training/instances/" + str(args.poisson_mean)
    training_file = "/" + folder_name + "/training_instances_" + str(args.sectors) + ".pkl"

    num_episode = args.episode
    if str_to_bool(args.replicate):
        with open(training_folder + training_file, "rb") as fp:
            training_instances = pickle.load(fp)
            num_episode = len(training_instances)

        if str_to_bool(args.checkpoint):
            checkpoint = len(improve_response_by_episode)
            num_episode -= checkpoint
            i = checkpoint
            steps_done = loss_by_step[-1][0]

    while i < num_episode:

        start_episode = datetime.datetime.now()

        if str_to_bool(args.replicate):
            initial_schedules_dict = training_instances[i][0]
            training_scenarios = training_instances[i][1]
        else:
            scenarios = []
            initial_schedules_dict = {}
            for sector_id in sectors.keys():
                training_data_folder = glob.glob("./data/Training/" + str(args.poisson_mean) + "/Sector_"
                                                 + str(sector_id) + "/*.pkl")
                selected_file = random.choice(training_data_folder)
                print(selected_file)
                with open(selected_file, "rb") as fp:
                    initial_schedule = pickle.load(fp)

                # Update the objective value of the initial schedule
                # (currently the objective value is based on the SetCover model)
                initial_schedule.update_objective_value(get_objective_value(initial_schedule, sectors[sector_id]))
                initial_schedules_dict[sector_id] = initial_schedule

                # Only 1 scenario per training episode, assume each sector has the same incident rate
                training_scenarios = generate_scenario(sectors[sector_id], 1, args.poisson_mean)
                scenario = training_scenarios[0]
                # Compile the incidents across sectors into one single list
                scenarios += scenario

            # Sort the incidents across the sectors in chronological order
            scenarios = sorted(scenarios, key=lambda x: x.get_start_time())
            # Update the training scenarios
            training_scenarios = {0: scenarios}

        # For single-agent problem, filter only relevant input data
        if not is_multi_agent:
            initial_schedules_dict = {k: v for k, v in initial_schedules_dict.items() if k == args.single_agent}
            scenarios = [incident for incident in training_scenarios[0] if incident.get_sector() == args.single_agent]
            training_scenarios = {0: scenarios}

        # Create simulator object
        simulator = Simulator(sectors, training_scenarios, initial_schedules_dict, time_matrix, adj_matrix,
                              neighbours_table, best_response=str_to_bool(args.best_response))
        scenarios = training_scenarios[0]

        # print(len(sectors.keys()))
        # sys.exit()

        schedules_dict = deepcopy(initial_schedules_dict)

        # Run the scenario
        incident_count = len(scenarios)
        print("Number of incidents: " + str(incident_count))
        for k in range(incident_count):

            start_incident = datetime.datetime.now()
            incident = scenarios[k]
            # print(incident.to_string())
            # convert each agent's schedule to a local state

            # Time and patrol area statuses
            if is_multi_agent:
                global_state_pre = [get_time_index(incident.get_start_time()) / len(T)] + \
                                   get_patrol_presence_status_MA(get_patrol_count_table_MA(schedules_dict), Q_j)
                joint_state_pre = get_post_joint_state(schedules_dict, Q_j, all_patrol_areas,
                                                       get_time_index(incident.get_start_time()), subagent_dim)
            else:
                state_pre = get_post_state(schedules_dict[args.single_agent],
                                           sectors[args.single_agent],
                                           get_time_index(incident.get_start_time()))
            # with open("./output/joint_state.pkl", "wb") as fp:
            #     pickle.dump(joint_states_pre, fp)
            #
            # sys.exit()


            f_p_pre = get_objective_value_MA(schedules_dict, sectors)

            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            # eps_threshold = 0
            if sample > eps_threshold:
                # Execute action based on learned policy
                if str_to_bool(args.parallel_heuristic):
                    assigned_agent, schedules_dict, response_utility = parallel_this_process(schedules_dict,
                                                                                             initial_schedules_dict,
                                                                                             sectors, 0, incident,
                                                                                             neighbours_table,
                                                                                             args.model,
                                                                                             is_multi_agent,
                                                                                             patrol_area_to_sector_map,
                                                                                             all_patrol_areas,
                                                                                             D, mask_nobatch, Q_j,
                                                                                             input_parameters,
                                                                                             trained_parameters)
                else:
                    assigned_agent, schedules_dict, response_utility = simulator.find_optimal_decision(schedules_dict,
                                                                                                       0,
                                                                                                       incident,
                                                                                                       args.model,
                                                                                                       vfa_agent)





            else:
                # Execute random action
                if str_to_bool(args.parallel_heuristic):
                    assigned_agent, schedules_dict, response_utility = parallel_this_process(schedules_dict,
                                                                                             initial_schedules_dict,
                                                                                             sectors, 0, incident,
                                                                                             neighbours_table, "random",
                                                                                             is_multi_agent,
                                                                                             patrol_area_to_sector_map,
                                                                                             all_patrol_areas,
                                                                                             D, mask_nobatch, Q_j)
                else:
                    assigned_agent, schedules_dict, response_utility = simulator.find_optimal_decision(schedules_dict,
                                                                                                       0,
                                                                                                       incident,
                                                                                                       "random")

            # print(schedule.get_time_tables())
            # defects = check_defects(schedule, sector)
            # if len(defects) > 0:
            #     for defect in defects:
            #         print(defect.to_string())
            #     sys.exit("During Training")
            # convert the new schedule to state
            # joint_states_post = np.array([local_states_post_dict[sector_id] for sector_id in sectors.keys()])
            if is_multi_agent:
                global_state_post = [get_time_index(incident.get_start_time()) / len(T)] + \
                                    get_patrol_presence_status_MA(get_patrol_count_table_MA(schedules_dict), Q_j)
                joint_state_post = get_post_joint_state(schedules_dict, Q_j, all_patrol_areas,
                                                        get_time_index(incident.get_start_time()), subagent_dim)
            else:
                state_post = get_post_state(schedules_dict[args.single_agent],
                                            sectors[args.single_agent],
                                            get_time_index(incident.get_start_time()))

            f_p_post = get_objective_value_MA(schedules_dict, sectors)

            # post_state = get_state(schedule, sector, get_time_index(incident.get_start_time()))
            reward = response_utility * f_p_post - f_p_pre  # Implicit reward

            # Update the z table
            simulator.z[0].append(response_utility)

            # To indicate if the current incident is the final incident of the scenario
            done = 0
            if k == incident_count - 1:
                done = 1

            # Dummy variable
            action = 0

            steps_done += 1

            # print(str(steps_done) + " steps done")
            if is_multi_agent:
                loss = vfa_agent.step(joint_state_pre, global_state_pre, action, reward, joint_state_post,
                                      global_state_post, done, mask)
            else:
                loss = vfa_agent.step(state_pre, action, reward, state_post, done)

            if loss:
                print("No of steps: " + str(steps_done) + " with loss: " + str(loss))
                loss_by_step.append((steps_done, loss))
                with open("./" + folder_name + "/output/vfa_loss_by_step_" + sectors_file_id + ".pkl", "wb") as fp:
                    pickle.dump(loss_by_step, fp)

            # Computation time for each incident
            end_incident = datetime.datetime.now()
            run_duration_incident = (end_incident - start_incident).total_seconds()
            print("Decision Time per incident: " + str(run_duration_incident))
            print(str(k + 1) + " out of " + str(incident_count) + " incidents completed")

        # For each episode, the final objective value (% of successful respond + patrol utilisation)
        response_count = simulator.get_response_count(0)
        success_count = simulator.get_success_count(0)
        presence_score = get_objective_value_MA(schedules_dict, sectors)

        # Timestamp for end of training
        end_training_episode = datetime.datetime.now()

        # Run myopic model
        if not str_to_bool(args.rerun_myopic):
            results_myopic = training_instances[i][2]
        else:
            # simulator_myopic = Simulator(sector, training_scenarios, initial_schedule)
            simulator_myopic = Simulator(sectors, training_scenarios, initial_schedules_dict, time_matrix, adj_matrix,
                                         neighbours_table)
            results_myopic, _ = simulator_myopic.run("myopic")
            results_myopic = results_myopic[0]["Total"]


        response_count_my, success_count_my, presence_score_my, _ = results_myopic


        # Compute % improvement over myopic
        improve_response = (response_count - response_count_my) / max(1, response_count_my) * 100
        improve_success = (success_count - success_count_my) / max(1, success_count_my) * 100
        improve_presence = (presence_score - presence_score_my) / presence_score_my * 100
        # measure = (final_score - myopic_score) / myopic_score * 100

        improve_response_by_episode.append((i, steps_done, improve_response))
        improve_success_by_episode.append((i, steps_done, improve_success))
        improve_presence_by_episode.append((i, steps_done, improve_presence))
        with open("./" + folder_name + "/output/vfa_response_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_response_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_success_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_success_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_presence_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_presence_by_episode, fp)

        # print(response_rate*incident_count, simulator.z[0])
        # print(response_rate_my*incident_count, simulator_myopic.z[0])
        print(str(response_count) + " out of " + str(incident_count) +
              " incidents attended using VFA with " + str(success_count) + " incidents responded on time/earlier.")
        print(str(response_count_my) + " out of " + str(incident_count) +
              " incidents attended myopically with " + str(success_count_my) + " incidents responded on time/earlier.")

        # print(final_obj_value)

        end_episode = datetime.datetime.now()
        episode_training_duration = (end_training_episode - start_episode).total_seconds()
        episode_myopic_duration = (end_episode - end_training_episode).total_seconds()
        print("Total computation time for 1 training episode: " + str(episode_training_duration) + 's')
        print("Total computation time for 1 myopic run: " + str(episode_myopic_duration) + 's')
        print(str(i + 1) + " out of " + str(args.episode) + " episodes completed")

        # Export the parameters learnt so far
        if (i + 1) % 1000 == 0:
            torch.save(vfa_agent.get_network().state_dict(), "./" + folder_name + "/parameter/vfa_parameters_" +
                       sectors_file_id + "_" + str(i + 1) + ".pth")
            with open("./" + folder_name + "/parameter/vfa_replay_buffer_" + sectors_file_id + "_" + str(i + 1) + ".pkl", "wb") as fp:
                pickle.dump(vfa_agent.get_memory().get_memory_list(), fp)



        if str_to_bool(args.save) or str_to_bool(args.checkpoint):
            training_instances.append((initial_schedules_dict, training_scenarios, results_myopic))

            if not os.path.exists(training_folder):
                os.makedirs(training_folder)

            with open(training_folder + training_file, "wb") as fp:
                pickle.dump(training_instances, fp)



        # Increment the episode number
        i += 1



    # Export the final learnt parameters
    torch.save(vfa_agent.get_network().state_dict(),  parameter_file_dir)
    with open("./" + folder_name + "/parameter/vfa_replaybuffer_" + sectors_file_id + ".pkl", "wb") as fp:
        pickle.dump(vfa_agent.get_memory().get_memory_list(), fp)

    return vfa_agent


def train_parallel(sectors, time_matrix, adj_matrix, neighbours_table, cpu_count, args, subfolder_name):
    start_run_time = datetime.datetime.now()
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
        parameter_file_dir = "./mavfa/" + subfolder_name + "/parameter/vfa_parameters_" + args.sectors + ".pth"
        folder_name = "mavfa/" + subfolder_name

    all_patrol_areas = []  # a list of all patrol area ids across the sectors
    patrol_area_to_sector_map = {}
    for sector_id in sectors.keys():
        for area in sectors[sector_id].get_all_patrol_areas():
            patrol_area_to_sector_map[area.get_id()] = sector_id
            all_patrol_areas.append(area.get_id())
    all_patrol_areas = sorted(all_patrol_areas)
    mask = np.stack([adj_matrix[0] for n in range(BATCH_SIZE)])

    # Declare VFA agent
    n_agents = len(sectors.keys())
    subagent_dim = max([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
    area_size = len(all_patrol_areas) + 2  # add 0 and -1 in the list of areas
    state_size = subagent_dim + area_size + 1  # 1 additional dimension corresponds to a list of encoded schedule

    input_parameters = {"n_agents": n_agents, "state_size": state_size, "area_size": area_size,
                        "subagent_dim": subagent_dim, "encoding_size": args.encoding_size, "sector_ids": args.sectors}

    # Load existing trained parameters
    trained_parameters = None
    imported_memory = []
    if str_to_bool(args.pre_trained):
        try:
            trained_parameters = torch.load(parameter_file_dir, map_location=device)
        except:
            pass

        try:
            with open("./" + folder_name + "/parameter/vfa_replaybuffer_" + sectors_file_id + ".pkl", "rb") as fp:
                imported_memory = pickle.load(fp)
        except:
            pass

    if is_multi_agent:
        # Multi agent problem
        vfa_agent = MAVFAAgent(args.pre_trained, args.sectors, n_agents, state_size, ACTION_SIZE, SEED, area_size,
                               subagent_dim,
                               args.encoding_size, HIDDEN_DIM, ATTN_OUT_DIM, trained_parameters=trained_parameters,
                               imported_memory=imported_memory, comms_net=str_to_bool(args.comms_net))
    else:
        # Single agent problem
        sector = sectors[args.single_agent]
        vfa_agent = VFAAgent(state_size, ACTION_SIZE, SEED, args.pre_trained, sector, area_size, subagent_dim,
                             args.encoding_size, trained_parameters=trained_parameters, imported_memory=imported_memory)

    training_instances = []

    training_folder = "./data/Training/instances/" + str(args.poisson_mean)
    training_file = "/" + folder_name + "/training_instances_" + str(args.sectors) + ".pkl"

    # Populate the scenarios for all episodes
    if str_to_bool(args.replicate):
        with open(training_folder + training_file, "rb") as fp:
            training_instances = pickle.load(fp)
        num_episode = len(training_instances)
        # re_run_myopic = False

        # if str_to_bool(args.checkpoint):
        #     orig_training_instances = deepcopy(training_instances[:args.checkpoint])
        #     training_instances = training_instances[args.checkpoint:]
        #     num_episode -= args.checkpoint
    else:
        i = 0  # Iterator for number of episodes
        num_episode = args.episode
        # re_run_myopic = True

        while i < num_episode:
            # print(i)
            scenarios = []
            initial_schedules_dict = {}
            for sector_id in sectors.keys():
                training_data_folder = glob.glob("./data/Training/" + str(args.poisson_mean) + "/Sector_"
                                                 + str(sector_id) + "/*.pkl")
                selected_file = random.choice(training_data_folder)
                # print(selected_file)
                with open(selected_file, "rb") as fp:
                    initial_schedule = pickle.load(fp)

                # Update the objective value of the initial schedule
                # (currently the objective value is based on the SetCover model)
                initial_schedule.update_objective_value(get_objective_value(initial_schedule, sectors[sector_id]))
                initial_schedules_dict[sector_id] = initial_schedule

                # Only 1 scenario per training episode, assume each sector has the same incident rate
                training_scenarios = generate_scenario(sectors[sector_id], 1, args.poisson_mean)
                scenario = training_scenarios[0]
                # Compile the incidents across sectors into one single list
                scenarios += scenario

            # Sort the incidents across the sectors in chronological order
            scenarios = sorted(scenarios, key=lambda x: x.get_start_time())
            # Update the training scenarios
            training_scenarios = {0: scenarios}
            # Initialise results_myopic as None
            training_instances.append([initial_schedules_dict, training_scenarios, None])
            # Run training and myopic concurrently per episode
            i += 1

    # Save the training instances
    if str_to_bool(args.save):

        if not os.path.exists(training_folder):
            os.makedirs(training_folder)

        with open(training_folder + training_file, "wb") as fp:
            pickle.dump(training_instances, fp)

    # Training starts here
    i = 0  # Iterator for number of episodes
    num_episode_remain = num_episode
    steps_done = 0
    num_steps_since_last_learning = 0
    trained_parameters = vfa_agent.get_network().state_dict()  # Initial NN parameters

    orig_save_param_freq = 1000  # every 1000 episodes
    curr_save_param_freq = orig_save_param_freq


    # For plotting
    loss_by_step = []
    improve_response_by_episode = []
    improve_success_by_episode = []
    improve_presence_by_episode = []
    response_by_episode = []
    success_by_episode = []
    presence_by_episode = []
    hamming_by_episode = []


    if str_to_bool(args.checkpoint):
        with open("./" + folder_name + "/output/vfa_loss_by_step_" + sectors_file_id + ".pkl", "rb") as fp:
            loss_by_step = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_response_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_response_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_success_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_success_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_presence_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_presence_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_response_rate_" + sectors_file_id + ".pkl", "rb") as fp:
            response_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_success_rate_" + sectors_file_id + ".pkl", "rb") as fp:
            success_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_presence_score_" + sectors_file_id + ".pkl", "rb") as fp:
            presence_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/vfa_hamming_score_" + sectors_file_id + ".pkl", "rb") as fp:
            hamming_by_episode = pickle.load(fp)

        checkpoint = len(improve_response_by_episode)
        if checkpoint > orig_save_param_freq:
            curr_save_param_freq = int(checkpoint / orig_save_param_freq) * orig_save_param_freq
        num_episode_remain -= checkpoint
        i = checkpoint
        steps_done = loss_by_step[-1][0]

    # update_counter = 0
    print("Total No. of Training Episodes:", num_episode)
    # For each training step i, <cpu_count> of episodes are running in parallel
    # print("Model's state_dict:")
    # for param_tensor in vfa_agent.get_network().state_dict():
    #     print(param_tensor, "\t",vfa_agent.get_network().state_dict()[param_tensor].size())
    # #
    # total_params = sum(p.numel() for p in vfa_agent.get_network().parameters() if p.requires_grad)
    # print(total_params)
    # sys.exit()
    while num_episode_remain > 0:
        # while i < num_episode:
        start_episode = datetime.datetime.now()

        num_parallel_run = min(num_episode_remain, cpu_count)
        # num_parallel_run = 10
        # initial_schedules_dict = training_instances[i - 1][0]
        # training_scenarios = training_instances[i - 1][1]
        # incident_count = len(training_scenarios)

        # Run training episodes in parallel
        parallel_results = []

        for idx in range(num_parallel_run):
            parallel_results.append(mavfa_train.remote(sectors, time_matrix, adj_matrix, neighbours_table, args,
                                                       is_multi_agent, training_instances[i + idx][0],
                                                       training_instances[i + idx][1],
                                                       args.model, steps_done, patrol_area_to_sector_map,
                                                       all_patrol_areas, i + idx, input_parameters,
                                                       trained_parameters, show_details=True))

        results = ray.get(parallel_results)
        # Sort the results by the episode index.
        # Result format (episode_idx, response_count, success_count, presence_score, experiences)
        results = sorted(results, key=lambda x: x[0])

        # Update the replay buffer
        experiences = []
        for result in results:
            experiences += result[5]

        for experience in experiences:
            if is_multi_agent:
                state, global_state, action, reward, next_state, next_global_state, done = experience
                vfa_agent.get_memory().add(state, global_state, action, reward, next_state, next_global_state, done)
            else:
                state, action, reward, next_state, done = experience
                vfa_agent.get_memory().add(state, action, reward, next_state, done)

        loss = None
        steps_done_for_recording = steps_done
        num_steps_since_last_learning += len(experiences)

        num_learning_cycles = int(num_steps_since_last_learning / LEARN_EVERY)
        # To be counted in the next learning steps
        buffer_steps = num_steps_since_last_learning % LEARN_EVERY

        for learning_cycle in range(num_learning_cycles):
            if len(vfa_agent.get_memory()) > BATCH_SIZE:
                steps_done_for_recording += LEARN_EVERY
                setattr(vfa_agent, 't_step', vfa_agent.get_t_step() + LEARN_EVERY)
                sampled_experiences = vfa_agent.get_memory().sample()
                if is_multi_agent:
                    loss = vfa_agent.learn(sampled_experiences, GAMMA, mask)
                else:
                    loss = vfa_agent.learn(sampled_experiences, GAMMA)
            if loss:
                print("No of steps: " + str(steps_done_for_recording) + " with loss: " + str(loss))
                loss_by_step.append((steps_done_for_recording, loss))

        # Update parameters
        steps_done += len(experiences)
        # dummy update to ensure that the network parameters are updated at every iteration
        # if len(vfa_agent.get_memory()) > BATCH_SIZE:
        #     setattr(vfa_agent, 't_step', update_counter)
        #     sampled_experiences = vfa_agent.get_memory().sample()
        #     if is_multi_agent:
        #         loss = vfa_agent.learn(sampled_experiences, GAMMA, mask)
        #     else:
        #         loss = vfa_agent.learn(sampled_experiences, GAMMA)
        #     if loss:
        #         print("No of steps: " + str(steps_done) + " with loss: " + str(loss))
        #         loss_by_step.append((steps_done, loss))

        # Export the new trained parameters to be passed to workers for subsequent training episodes
        torch.save(vfa_agent.get_network().state_dict(), parameter_file_dir)
        trained_parameters = torch.load(parameter_file_dir, map_location=device)
        # Update steps counting
        num_steps_since_last_learning = buffer_steps

        episode_idxs = [i + idx for idx in range(num_parallel_run)]

        # Get results for myopic run
        # if str_to_bool(args.replicate) and not str_to_bool(args.checkpoint) and re_run_myopic == False:
        if str_to_bool(args.rerun_myopic) == False:
            # If replicate the training instances without any checkpoints, read the results directly from the file
            results_my = []
            episode_idxs = [i + idx for idx in range(num_parallel_run)]
            for episode_idx in episode_idxs:
                result_myopic = training_instances[episode_idx][2]
                results_my.append((episode_idx, result_myopic))
        else:
            # If not a replicate run or is a replicate with checkpoint
            # Run myopic in parallel
            # print("I am not supposed to be here")
            # sys.exit()
            parallel_results_my = []
            for idx in range(num_parallel_run):
                parallel_results_my.append(mavfa_train.remote(sectors, time_matrix, adj_matrix, neighbours_table, args,
                                                              is_multi_agent, training_instances[i + idx][0],
                                                              training_instances[i + idx][1],
                                                              "myopic", steps_done, patrol_area_to_sector_map,
                                                              all_patrol_areas, i + idx, show_details=True))

            results_my = ray.get(parallel_results_my)
            # Sort the results by the episode index
            # Result format (episode_idx, (response_count, success_count, presence_score, incident_count))
            results_my = sorted(results_my, key=lambda x: x[0])
            # re_run_myopic = True

        # Compute % improvement over myopic
        for idx in range(num_parallel_run):
            episode_idx, response_count, success_count, presence_score, hamming_score, _ = results[idx]
            episode_idx_my, result_myopic = results_my[idx]
            response_count_my, success_count_my, presence_score_my, hamming_score_my, incident_count_dummy = \
                result_myopic
            # sys.exit()

            if episode_idx != episode_idx_my:
                sys.exit("Mistmatched episode idx!")

            if str_to_bool(args.save) or str_to_bool(args.checkpoint):
                # Update the training_instances with myopic results
                training_instances[episode_idx][2] = result_myopic

            incident_count = len(training_instances[episode_idx][1][0])

            if incident_count != incident_count_dummy:
                sys.exit("Mismatched incident count")

            improve_response = (response_count - response_count_my) / max(1, response_count_my) * 100
            improve_success = (success_count - success_count_my) / max(1, success_count_my) * 100
            improve_presence = (presence_score - presence_score_my) / presence_score_my * 100

            improve_response_by_episode.append((episode_idx, steps_done, improve_response))
            improve_success_by_episode.append((episode_idx, steps_done, improve_success))
            improve_presence_by_episode.append((episode_idx, steps_done, improve_presence))

            response_by_episode.append(response_count / incident_count)
            success_by_episode.append(success_count / incident_count)
            presence_by_episode.append(presence_score)
            hamming_by_episode.append(hamming_score)

            print(str(response_count) + " out of " + str(incident_count) +
                  " incidents attended using VFA with " + str(success_count) + " incidents responded on time/earlier.")
            print(str(response_count_my) + " out of " + str(incident_count) +
                  " incidents attended myopically with " + str(
                success_count_my) + " incidents responded on time/earlier.")

        # Save files
        with open("./" + folder_name + "/output/vfa_loss_by_step_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(loss_by_step, fp)
        with open("./" + folder_name + "/output/vfa_response_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_response_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_success_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_success_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_presence_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_presence_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_response_rate_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(response_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_success_rate_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(success_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_presence_score_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(presence_by_episode, fp)
        with open("./" + folder_name + "/output/vfa_hamming_score_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(hamming_by_episode, fp)
        with open("./" + folder_name + "/parameter/vfa_replaybuffer_" + sectors_file_id + ".pkl",
                  "wb") as fp:
            pickle.dump(vfa_agent.get_memory().get_memory_list(), fp)

        # if str_to_bool(args.save) and not str_to_bool(args.replicate):
        if str_to_bool(args.save) or str_to_bool(args.checkpoint):

            if not os.path.exists(training_folder):
                os.makedirs(training_folder)

            with open(training_folder + training_file, "wb") as fp:
                pickle.dump(training_instances, fp)

        num_episode_remain -= num_parallel_run
        print(str(num_episode - num_episode_remain) + " out of " + str(num_episode) + " episodes completed")
        i += num_parallel_run
        # update_counter += 1

        # Save trained parameters
        if len(improve_response_by_episode) > curr_save_param_freq:

            torch.save(vfa_agent.get_network().state_dict(), "./" + folder_name + "/parameter/vfa_parameters_" +
                       sectors_file_id + "_" + str(curr_save_param_freq) + ".pth")

            curr_save_param_freq = curr_save_param_freq + orig_save_param_freq



    # Export the final learnt parameters and replay buffer
    torch.save(vfa_agent.get_network().state_dict(), parameter_file_dir)
    with open("./" + folder_name + "/parameter/vfa_replaybuffer_" + sectors_file_id + ".pkl", "wb") as fp:
        pickle.dump(vfa_agent.get_memory().get_memory_list(), fp)

    return vfa_agent


@ray.remote
def mavfa_train(sectors, time_matrix, adj_matrix, neighbours_table, args, is_multi_agent, initial_schedules_dict,
                training_scenarios, policy, steps_done, patrol_area_to_sector_map, all_patrol_areas,
                episode_idx, input_parameters={}, trained_parameters=None,
                show_details=False):
    print("Running Episode " + str(episode_idx + 1))
    start_episode = datetime.datetime.now()

    if policy.lower() == "myopic":
        print("Running myopic")
        if str_to_bool(args.parallel_heuristic):
            results_myopic, _, _ = run_simulation(sectors, training_scenarios, initial_schedules_dict,
                                                  time_matrix, adj_matrix, neighbours_table,
                                                  policy, show_details=show_details)
        else:
            simulator_myopic = Simulator(sectors, training_scenarios, initial_schedules_dict, time_matrix, adj_matrix,
                                         neighbours_table)
            results_myopic, _ = simulator_myopic.run("myopic")

        end_episode = datetime.datetime.now()
        episode_duration = (end_episode - start_episode).total_seconds()
        print("Total computation time for 1 episode: " + str(episode_duration) + 's')
        print("Episode " + str(episode_idx + 1) + " completed")

        return episode_idx, results_myopic[0]["Total"]

    else:
        print("VFA Training")

        D = time_matrix  # Time travel matrix for all patrol areas across sectors
        Q_j = get_global_Q_j(sectors)
        mask = np.array([adj_matrix[0]])
        z = {0: []}
        experiences = []

        n_agents = input_parameters["n_agents"]
        state_size = input_parameters["state_size"]
        area_size = input_parameters["area_size"]
        subagent_dim = input_parameters["subagent_dim"]

        if is_multi_agent:
            # Multi agent problem
            vfa_agent = MAVFAAgent(args.pre_trained, args.sectors, n_agents, state_size, ACTION_SIZE, SEED, area_size,
                                   subagent_dim,
                                   args.encoding_size, HIDDEN_DIM, ATTN_OUT_DIM, trained_parameters=trained_parameters,
                                   imported_memory=[], comms_net=str_to_bool(args.comms_net))
        else:
            # Single agent problem
            sector = sectors[args.single_agent]
            vfa_agent = VFAAgent(state_size, ACTION_SIZE, SEED, args.pre_trained, sector, area_size, subagent_dim,
                                 args.encoding_size, trained_parameters=trained_parameters)

        # For single-agent problem, filter only relevant input data
        if not is_multi_agent:
            initial_schedules_dict = {k: v for k, v in initial_schedules_dict.items() if k == args.single_agent}
            scenarios = [incident for incident in training_scenarios[0] if incident.get_sector() == args.single_agent]
            training_scenarios = {0: scenarios}

        # Create simulator object
        simulator = Simulator(sectors, training_scenarios, initial_schedules_dict, time_matrix, adj_matrix,
                              neighbours_table, best_response=str_to_bool(args.best_response))
        scenarios = training_scenarios[0]

        schedules_dict = deepcopy(initial_schedules_dict)

        # Run the scenario
        incident_count = len(scenarios)
        print("Number of incidents: " + str(incident_count))
        for k in range(incident_count):

            start_incident = datetime.datetime.now()
            incident = scenarios[k]
            # print(incident.to_string())
            # convert each agent's schedule to a local state

            # Time and patrol area statuses
            if is_multi_agent:
                global_state_pre = [get_time_index(incident.get_start_time()) / len(T)] + \
                                   get_patrol_presence_status_MA(get_patrol_count_table_MA(schedules_dict), Q_j)
                joint_state_pre = get_post_joint_state(schedules_dict, Q_j, all_patrol_areas,
                                                       get_time_index(incident.get_start_time()), subagent_dim)
            else:
                state_pre = get_post_state(schedules_dict[args.single_agent],
                                           sectors[args.single_agent],
                                           get_time_index(incident.get_start_time()))

            f_p_pre = get_objective_value_MA(schedules_dict, sectors)

            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
            # eps_threshold = 100
            if sample > eps_threshold:
                # Execute action based on learned policy
                # print("Use Model")
                if str_to_bool(args.parallel_heuristic):
                    assigned_agent, schedules_dict, response_utility = parallel_this_process(schedules_dict,
                                                                                             initial_schedules_dict,
                                                                                             sectors, 0, incident,
                                                                                             neighbours_table,
                                                                                             args.model,
                                                                                             is_multi_agent,
                                                                                             patrol_area_to_sector_map,
                                                                                             all_patrol_areas,
                                                                                             D, mask, Q_j,
                                                                                             input_parameters,
                                                                                             trained_parameters)
                else:

                    assigned_agent, schedules_dict, response_utility = simulator.find_optimal_decision(schedules_dict,
                                                                                                       0,
                                                                                                       incident,
                                                                                                       args.model,
                                                                                                       vfa_agent)


            else:
                # Execute random action
                # print("Use Random")
                if str_to_bool(args.parallel_heuristic):
                    assigned_agent, schedules_dict, response_utility = parallel_this_process(schedules_dict,
                                                                                             initial_schedules_dict,
                                                                                             sectors, 0, incident,
                                                                                             neighbours_table, "random",
                                                                                             is_multi_agent,
                                                                                             patrol_area_to_sector_map,
                                                                                             all_patrol_areas,
                                                                                             D, mask, Q_j,
                                                                                             input_parameters,
                                                                                             trained_parameters)
                else:
                    assigned_agent, schedules_dict, response_utility = simulator.find_optimal_decision(schedules_dict,
                                                                                                       0,
                                                                                                       incident,
                                                                                                       "random")

            if is_multi_agent:
                global_state_post = [get_time_index(incident.get_start_time()) / len(T)] + \
                                    get_patrol_presence_status_MA(get_patrol_count_table_MA(schedules_dict), Q_j)
                joint_state_post = get_post_joint_state(schedules_dict, Q_j, all_patrol_areas,
                                                        get_time_index(incident.get_start_time()), subagent_dim)
            else:
                state_post = get_post_state(schedules_dict[args.single_agent],
                                            sectors[args.single_agent],
                                            get_time_index(incident.get_start_time()))

            f_p_post = get_objective_value_MA(schedules_dict, sectors)

            reward = response_utility * f_p_post - f_p_pre  # Implicit reward

            # Update the z table
            z[0].append(response_utility)

            # To indicate if the current incident is the final incident of the scenario
            done = 0
            if k == incident_count - 1:
                done = 1

            steps_done += 1

            # Dummy variable
            action = 0

            # print(str(steps_done) + " steps done")
            if is_multi_agent:
                experience = joint_state_pre, global_state_pre, action, reward, joint_state_post, global_state_post, \
                             done
            else:
                experience = state_pre, action, reward, state_post, done

            experiences.append(experience)

            # Computation time for each incident
            end_incident = datetime.datetime.now()
            run_duration_incident = (end_incident - start_incident).total_seconds()
            if show_details:
                print("Decision Time per incident: " + str(run_duration_incident))
                print(str(k + 1) + " out of " + str(incident_count) + " incidents completed")

        # For each episode, the final objective value (% of successful respond + patrol utilisation)
        response_count = get_response_count(z, 0)
        success_count = get_success_count(z, 0)
        presence_score = get_objective_value_MA(schedules_dict, sectors)
        hamming_score = compute_hamming_distance_joint(schedules_dict, initial_schedules_dict)

        end_episode = datetime.datetime.now()
        episode_duration = (end_episode - start_episode).total_seconds()
        print("Total computation time for 1 episode: " + str(episode_duration) + 's')
        print("Episode " + str(episode_idx + 1) + " completed")

        return episode_idx, response_count, success_count, presence_score, hamming_score, experiences
