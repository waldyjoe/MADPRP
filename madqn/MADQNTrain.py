import datetime
import numpy as np
import random
import math
import copy
import glob
import ray
import sys
import os
import pickle
import torch

from copy import deepcopy

from data.ScenarioGenerator import generate_scenario
from madqn.MADQNAgent import MADQNAgent
from model.ReschedulerMA import *
from model.Simulator import Simulator
from model.Simulator_parallel import *
from util.ScheduleUtil import get_global_Q_j, get_objective_value, get_objective_value_MA, \
    get_patrol_presence_status_MA, get_patrol_count_table_MA, get_pre_state_MA
from util.utils import get_time_index, round_to_nearest, str_to_bool


def train_parallel(sectors, time_matrix, adj_matrix, neighbours_table, cpu_count, args, subfolder_name):

    sectors_file_id = args.sectors
    parameter_file_dir = "./madqn/" + subfolder_name + "/parameter/madqn_parameters_" + args.sectors + ".pth"
    folder_name = "madqn/" + subfolder_name

    all_patrol_areas = []  # a list of all patrol area ids across the sectors
    patrol_area_to_sector_map = {}
    for sector_id in sectors.keys():
        for area in sectors[sector_id].get_all_patrol_areas():
            patrol_area_to_sector_map[area.get_id()] = sector_id
            all_patrol_areas.append(area.get_id())
    all_patrol_areas = sorted(all_patrol_areas)

    # Declare DQN agent
    subagents_count = np.sum([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
    area_size = len(all_patrol_areas) + 2  # add 0 and -1 in the list of areas
    # time account for 1 element, 2 elements accounted for incident location and schedules of all subagents
    state_size = 1 + subagents_count + area_size + 2

    action_size = len(sectors.keys()) # Action refers to the sector to assign the incident to
    input_parameters = {"state_size": state_size, "area_size": area_size, "action_size": action_size,
                        "subagents_count" : subagents_count, "encoding_size": args.encoding_size,
                        "sector_ids": args.sectors}

    # Load existing trained parameters
    trained_parameters = None
    imported_memory = []
    if str_to_bool(args.pre_trained):
        try:
            trained_parameters = torch.load(parameter_file_dir, map_location=device)
        except:
            pass

        try:
            with open("./" + folder_name + "/parameter/madqn_replaybuffer_" + sectors_file_id + ".pkl", "rb") as fp:
                imported_memory = pickle.load(fp)
        except:
            pass

    dqn_agent = MADQNAgent(args.pre_trained, state_size, action_size, SEED, area_size, subagents_count, args.encoding_size,
                         trained_parameters=trained_parameters, imported_memory=imported_memory)

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
    trained_parameters = dqn_agent.get_network().state_dict()  # Initial NN parameters

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
        with open("./" + folder_name + "/output/madqn_loss_by_step_" + sectors_file_id + ".pkl", "rb") as fp:
            loss_by_step = pickle.load(fp)
        with open("./" + folder_name + "/output/madqn_response_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_response_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/madqn_success_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_success_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/madqn_presence_" + sectors_file_id + ".pkl", "rb") as fp:
            improve_presence_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/madqn_response_rate_" + sectors_file_id + ".pkl", "rb") as fp:
            response_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/madqn_success_rate_" + sectors_file_id + ".pkl", "rb") as fp:
            success_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/madqn_presence_score_" + sectors_file_id + ".pkl", "rb") as fp:
            presence_by_episode = pickle.load(fp)
        with open("./" + folder_name + "/output/madqn_hamming_score_" + sectors_file_id + ".pkl", "rb") as fp:
            hamming_by_episode = pickle.load(fp)

        checkpoint = len(improve_response_by_episode)
        if checkpoint > orig_save_param_freq:
            curr_save_param_freq = int(checkpoint / orig_save_param_freq) * orig_save_param_freq
        num_episode_remain -= checkpoint
        i = checkpoint
        steps_done = loss_by_step[-1][0]

    print("Total No. of Training Episodes:", num_episode)
    # For each training step i, <cpu_count> of episodes are running in parallel
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
            parallel_results.append(dqn_train.remote(sectors, time_matrix, adj_matrix, neighbours_table, args,
                                                     training_instances[i + idx][0], training_instances[i + idx][1],
                                                     args.model, steps_done, patrol_area_to_sector_map,
                                                     all_patrol_areas, i + idx, input_parameters, trained_parameters,
                                                     show_details=True))

        results = ray.get(parallel_results)
        # Sort the results by the episode index.
        # Result format (episode_idx, response_count, success_count, presence_score, experiences)
        results = sorted(results, key=lambda x: x[0])

        # Update the replay buffer
        experiences = []
        for result in results:
            experiences += result[5]

        for experience in experiences:
            state, action, reward, next_state, done = experience
            dqn_agent.get_memory().add(state, action, reward, next_state, done)

        loss = None
        steps_done_for_recording = steps_done
        num_steps_since_last_learning += len(experiences)

        num_learning_cycles = int(num_steps_since_last_learning / LEARN_EVERY)
        # To be counted in the next learning steps
        buffer_steps = num_steps_since_last_learning % LEARN_EVERY

        for learning_cycle in range(num_learning_cycles):
            if len(dqn_agent.get_memory()) > BATCH_SIZE:
                steps_done_for_recording += LEARN_EVERY
                setattr(dqn_agent, 't_step', dqn_agent.get_t_step() + LEARN_EVERY)
                sampled_experiences = dqn_agent.get_memory().sample()
                loss = dqn_agent.learn(sampled_experiences, GAMMA)

            if loss:
                print("No of steps: " + str(steps_done_for_recording) + " with loss: " + str(loss))
                loss_by_step.append((steps_done_for_recording, loss))

        # Export the new trained parameters to be passed to workers for subsequent training episodes
        torch.save(dqn_agent.get_network().state_dict(), parameter_file_dir)
        trained_parameters = torch.load(parameter_file_dir, map_location=device)
        # Update steps counting
        num_steps_since_last_learning = buffer_steps
        steps_done += len(experiences)

        episode_idxs = [i + idx for idx in range(num_parallel_run)]

        # Get results for myopic run
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
            parallel_results_my = []
            for idx in range(num_parallel_run):
                parallel_results_my.append(dqn_train.remote(sectors, time_matrix, adj_matrix, neighbours_table, args,
                                                         training_instances[i + idx][0], training_instances[i + idx][1],
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
            response_count_my, success_count_my, presence_score_my, incident_count_dummy = result_myopic

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
                  " incidents attended using DQN with " + str(
                success_count) + " incidents responded on time/earlier.")
            print(str(response_count_my) + " out of " + str(incident_count) +
                  " incidents attended myopically with " + str(
                success_count_my) + " incidents responded on time/earlier.")

        # Save files
        with open("./" + folder_name + "/output/madqn_loss_by_step_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(loss_by_step, fp)
        with open("./" + folder_name + "/output/madqn_response_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_response_by_episode, fp)
        with open("./" + folder_name + "/output/madqn_success_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_success_by_episode, fp)
        with open("./" + folder_name + "/output/madqn_presence_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(improve_presence_by_episode, fp)
        with open("./" + folder_name + "/output/madqn_response_rate_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(response_by_episode, fp)
        with open("./" + folder_name + "/output/madqn_success_rate_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(success_by_episode, fp)
        with open("./" + folder_name + "/output/madqn_presence_score_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(presence_by_episode, fp)
        with open("./" + folder_name + "/output/madqn_hamming_score_" + sectors_file_id + ".pkl", "wb") as fp:
            pickle.dump(hamming_by_episode, fp)
        with open("./" + folder_name + "/parameter/madqn_replaybuffer_" + sectors_file_id + ".pkl",
                  "wb") as fp:
            pickle.dump(dqn_agent.get_memory().get_memory_list(), fp)

        # if str_to_bool(args.save) and not str_to_bool(args.replicate):
        if str_to_bool(args.save) or str_to_bool(args.checkpoint):

            if not os.path.exists(training_folder):
                os.makedirs(training_folder)

            with open(training_folder + training_file, "wb") as fp:
                pickle.dump(training_instances, fp)

        num_episode_remain -= num_parallel_run
        print(str(num_episode - num_episode_remain) + " out of " + str(num_episode) + " episodes completed")
        i += num_parallel_run


        # Save trained parameters
        if len(improve_response_by_episode) > curr_save_param_freq:
            torch.save(dqn_agent.get_network().state_dict(), "./" + folder_name + "/parameter/madqn_parameters_" +
                       sectors_file_id + "_" + str(curr_save_param_freq) + ".pth")

            curr_save_param_freq = curr_save_param_freq + orig_save_param_freq

    # Export the final learnt parameters and replay buffer
    torch.save(dqn_agent.get_network().state_dict(), parameter_file_dir)
    with open("./" + folder_name + "/parameter/madqn_replaybuffer_" + sectors_file_id + ".pkl", "wb") as fp:
        pickle.dump(dqn_agent.get_memory().get_memory_list(), fp)

    return dqn_agent

@ray.remote
def dqn_train(sectors, time_matrix, adj_matrix, neighbours_table, args, initial_schedules_dict,
                training_scenarios, policy, steps_done, patrol_area_to_sector_map, all_patrol_areas,
                episode_idx, input_parameters={}, trained_parameters=None,
                show_details=False):

    # print("Running Episode " + str(episode_idx + 1))
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

        print("MADQN Training")

        D = time_matrix  # Time travel matrix for all patrol areas across sectors
        Q_j = get_global_Q_j(sectors)
        z = {0: []}
        experiences = []

        state_size = input_parameters["state_size"]
        area_size = input_parameters["area_size"]
        action_size = input_parameters["action_size"]
        subagents_count = input_parameters["subagents_count"]

        dqn_agent = MADQNAgent(args.pre_trained, state_size, action_size, SEED, area_size, subagents_count, args.encoding_size,
                             trained_parameters=trained_parameters, imported_memory=[])

        # Create simulator object
        simulator = Simulator(sectors, training_scenarios, initial_schedules_dict, time_matrix, adj_matrix,
                              neighbours_table, best_response=str_to_bool(args.best_response))
        scenarios = training_scenarios[0]
        Z =[]
        schedules_dict = deepcopy(initial_schedules_dict)

        # Run the scenario
        incident_count = len(scenarios)
        print("Number of incidents: " + str(incident_count))
        for k in range(incident_count):
            start_incident = datetime.datetime.now()
            incident = scenarios[k]
            incident_sector = incident.get_sector()
            # print(incident.to_string())

            incident_time_index = get_time_index(incident.get_start_time())
            # incident_location = incident.get_location().get_id()

            state_pre = get_pre_state_MA(schedules_dict, Q_j, all_patrol_areas, incident)
            f_p_pre = get_objective_value_MA(schedules_dict, sectors)

            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

            action_space = []  # All possible action
            action_space_eff = []  # All feasible action

            # Compile all possible actions
            for sector_id in schedules_dict.keys():
                action_space.append(sector_id)

                if not(sector_id != incident_sector and sector_id not in neighbours_table[incident_sector]):
                    action_space_eff.append(sector_id)

            if sample > eps_threshold:
                action_idx = dqn_agent.act(state_pre)
                action = action_space[action_idx]

            else:
                # Execute random action
                action = random.choice(action_space)
                action_idx = action_space.index(action)

            if action in action_space_eff:

                # Original schedules_dict is kept intact
                temp_schedules_dict = deepcopy(schedules_dict)
                # Insert the incident into the sector id and reschedule it
                schedule, response_utility = simulator.incident_response_procedure(temp_schedules_dict, action, 0,
                                                                                   incident, "myopic")

                if schedule:
                    schedules_dict[action] = schedule
                    Z.append(response_utility)

                    if str_to_bool(args.best_response):
                        schedules_dict = simulator.best_response_procedure(schedules_dict, incident_time_index, "myopic")

                else:

                    Z.append(0)

            else:

                Z.append(0)

            if k == incident_count - 1:
                state_post = get_pre_state_MA(schedules_dict, Q_j, all_patrol_areas, incident)
            else:
                state_post = get_pre_state_MA(schedules_dict, Q_j, all_patrol_areas, scenarios[k+1])

            f_p_post = get_objective_value_MA(schedules_dict, sectors)

            reward = Z[-1] * f_p_post - f_p_pre  # Implicit reward

            # To indicate if the current incident is the final incident of the scenario
            done = 0
            if k == incident_count - 1:
                done = 1

            steps_done += 1

            experience = state_pre, action_idx, reward, state_post, done

            experiences.append(experience)

            # Computation time for each incident
            end_incident = datetime.datetime.now()
            run_duration_incident = (end_incident - start_incident).total_seconds()
            if show_details:
                print("Decision Time per incident: " + str(run_duration_incident))
                print(str(k + 1) + " out of " + str(incident_count) + " incidents completed")

        # For each episode, the final objective value (% of successful respond + patrol utilisation)
        response_count = sum([x for x in Z if x > 0])
        success_count = sum([x for x in Z if x == 1])
        presence_score = get_objective_value_MA(schedules_dict, sectors)
        hamming_score = compute_hamming_distance_joint(schedules_dict, initial_schedules_dict)

        end_episode = datetime.datetime.now()
        episode_duration = (end_episode - start_episode).total_seconds()
        print("Total computation time for 1 episode: " + str(episode_duration) + 's')
        print("Episode " + str(episode_idx + 1) + " completed")

        return episode_idx, response_count, success_count, presence_score, hamming_score, experiences


def get_response_time(schedule, agent, action_time, incident, D):

    time_tables = schedule.get_time_tables()
    # Time from current time to incident location
    time_to_incident = round_to_nearest(
        D[time_tables[agent][action_time]][incident.get_location().get_id()],
        TIME_UNIT)

    # Time periods from current time to incident location
    travel_time_slots = get_time_index(time_to_incident)

    response_time = action_time - get_time_index(incident.get_start_time()) + travel_time_slots

    return response_time
