import datetime
import numpy as np
import random
import math
import copy
import glob
import sys
import os
import pickle
import torch

from copy import deepcopy

from constants.Settings import ACTION_SIZE, EPS_DECAY, EPS_END, EPS_START, SEED
from data.ScenarioGenerator import generate_scenario
from vfa.VFAAgent import VFAAgent
from model.Simulator import Simulator
from util.ScheduleUtil import get_objective_value, get_post_state, check_defects
from util.utils import get_time_index


def train(sector, args):

    # training_data_folder = glob.glob("./data/Training/" + str(args.poisson_mean) + "/Sector_"
    #                                  + str(sector.get_id()) + "/*.pkl")

    # state_size = 1 + len(sector.get_agents()) + sector.get_patrol_areas_count()  # State_1
    # state_size = 2 + len(sector.get_agents())    # State_2
    state_size = 2 + len(sector.get_agents()) + sector.get_patrol_areas_count() + 1 # State 3
    # state_size = 2 + len(sector.get_agents()) + 1  # State 5/6
    vfa_agent = VFAAgent(state_size, ACTION_SIZE, SEED, args.new_run, args.sector, sector.get_patrol_areas_count()+2,
                         len(sector.get_agents()), args.encoding_size)

    i = 1 # Iterator for number of episodes
    steps_done = 0

    # For plotting
    loss_by_step = []
    obj_value_by_episode = []

    improve_response_by_episode = []
    improve_success_by_episode = []
    improve_presence_by_episode = []

    training_instances = []

    training_folder = "./data/Training/instances/" + str(args.poisson_mean) + "/Sector_" + str(sector.get_id())
    training_file = "/training_instances.pkl"

    if args.replicate == "True":
        with open(training_folder + training_file, "rb") as fp:
            training_instances = pickle.load(fp)
    else:
        training_data_folder = glob.glob("./data/Training/" + str(args.poisson_mean) + "/Sector_"
                                         + str(sector.get_id()) + "/*.pkl")

    while i <= args.episode:

        # if args.replicate == "True":
        #     with open(training_folder + training_file, "rb") as fp:
        #         training_instances = pickle.load(fp)
        # else:
        #     training_data_folder = glob.glob("./data/Training/" + str(args.poisson_mean) + "/Sector_"
        #                                      + str(sector.get_id()) + "/*.pkl")

        start_episode = datetime.datetime.now()

        if args.replicate == "True":
            initial_schedule = training_instances[i-1][0]
            training_scenarios = training_instances[i-1][1]
        else:
            # Generate multiple scenarios of the initial schedule based on set cover model
            selected_file = random.choice(training_data_folder)
            print(selected_file)
            with open(selected_file, "rb") as fp:
                initial_schedule = pickle.load(fp)

            # Only 1 scenario per training episode
            training_scenarios = generate_scenario(sector, 1, args.poisson_mean)

        scenario = training_scenarios[0]
        # Update the objective value of the initial schedule
        # (currently the objective value is based on the SetCover model)
        initial_schedule.update_objective_value(get_objective_value(initial_schedule, sector))

        # Create simulator object
        simulator = Simulator(sector, training_scenarios, initial_schedule)

        schedule = deepcopy(initial_schedule)

        # Run the scenario
        incident_count = len(scenario)
        print("Number of incidents: " + str(incident_count))
        for k in range(incident_count):

            incident = scenario[k]
            # print(incident.to_string())

            # convert schedule to a state
            pre_schedule = deepcopy(schedule)
            pre_state = get_state(schedule, sector, get_time_index(incident.get_start_time()))
            fitness_pre = schedule.get_objective_value()

            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

            if sample > eps_threshold:
                # Execute action based on learned policy
                # print("Use Model")
                schedule, response, fitness_post = simulator.incident_response_procedure(schedule, 0, incident,
                                                                                         args.model, vfa_agent)

            else:
                # Execute random action
                # print("Use Random")
                schedule, response, fitness_post = simulator.incident_response_procedure(schedule, 0, incident, "random_lite")

            # print(schedule.get_time_tables())
            # defects = check_defects(schedule, sector)
            # if len(defects) > 0:
            #     for defect in defects:
            #         print(defect.to_string())
            #     sys.exit("During Training")
            # convert the new schedule to state
            post_state = get_state(schedule, sector, get_time_index(incident.get_start_time()))

            # Fitness value of new_schedule - fitness value of current schedule
            # reward = BETA_1*response + BETA_2*(fitness_post - fitness_pre)  # weighted reward
            reward = response * fitness_post - fitness_pre  # Implicit reward

            # To indicate if the current incident is the final incident of the scenario
            done = 0
            if k == incident_count - 1:
                done = 1

            # Dummy variable
            action = 0

            steps_done += 1

            # print(str(steps_done) + " steps done")

            loss = vfa_agent.step(pre_state, action, reward, post_state, done)
            if loss:
                print("No of steps: " + str(steps_done) + " with loss: " + str(loss))
                loss_by_step.append((steps_done, loss))
                with open("./output/vfa_loss_by_step_" + sector.get_id() + ".pkl", "wb") as fp:
                    pickle.dump(loss_by_step, fp)

        # For each episode, the final objective value (% of successful respond + patrol utilisation)
        response_count = simulator.get_respond_count(0)
        success_count = simulator.get_success_count(0)
        presence_score = get_objective_value(schedule, sector)
        # final_score = BETA_1*response_rate + BETA_2*get_objective_value(schedule, sector)

        # Run myopic model
        if args.replicate == "True":
            results_myopic = training_instances[i-1][2]
        else:
            simulator_myopic = Simulator(sector, training_scenarios, initial_schedule)
            if "lite" in args.model.lower():
                results_myopic, _ = simulator_myopic.run("myopic_lite")
            else:
                results_myopic, _ = simulator_myopic.run("myopic")

        # myopic_score = results_myopic[0]
        response_count_my, success_count_my, presence_score_my = results_myopic[0]
        # myopic_score = BETA_1*response_rate_my + BETA_2*presence_score_my
        
        # Compute % improvement over myopic
        # improve_response = (response_rate*incident_count - response_rate_my*incident_count) / max(1, response_rate_my*incident_count) * 100
        improve_response = (response_count - response_count_my) / max(1, response_count_my) * 100
        improve_success = (success_count - success_count_my) / max(1, success_count_my) * 100
        improve_presence = (presence_score - presence_score_my) / presence_score_my * 100
        # measure = (final_score - myopic_score) / myopic_score * 100

        improve_response_by_episode.append((i, steps_done, improve_response))
        improve_success_by_episode.append((i, steps_done, improve_success))
        improve_presence_by_episode.append((i, steps_done, improve_presence))
        with open("./output/vfa_response_" + sector.get_id() + ".pkl", "wb") as fp:
            pickle.dump(improve_response_by_episode, fp)
        with open("./output/vfa_success_" + sector.get_id() + ".pkl", "wb") as fp:
            pickle.dump(improve_success_by_episode, fp)
        with open("./output/vfa_presence_" + sector.get_id() + ".pkl", "wb") as fp:
            pickle.dump(improve_presence_by_episode, fp)
        
        # print(response_rate*incident_count, simulator.z[0])
        # print(response_rate_my*incident_count, simulator_myopic.z[0])
        print(str(response_count) + " out of " + str(incident_count) +
              " incidents attended using VFA with " + str(success_count) + " incidents responded on time/earlier.")
        print(str(response_count_my) + " out of " + str(incident_count) +
              " incidents attended myopically with " + str(success_count_my) + " incidents responded on time/earlier.")

        # print(final_obj_value)

        end_episode = datetime.datetime.now()
        episode_duration = (end_episode - start_episode).total_seconds()
        print("Total computation time for 1 episode: " + str(episode_duration) + 's')
        print(str(i) + " out of " + str(args.episode) + " episodes completed")

        # Export the parameters learnt so far
        if i % 1000 == 0:
            torch.save(vfa_agent.get_local_network().state_dict(), "./vfa/parameter/vfa_parameters_" +
                       sector.get_id() + "_" + str(i) + ".pth")
            # with open("./vfa/parameter/vfa_replaybuffer_" + sector.get_id() + "_" + str(i) + ".pkl", "wb") as fp:
            #     pickle.dump(vfa_agent.memory, fp)

        if args.save == "True" and args.replicate != "True":
            training_instances.append((initial_schedule, training_scenarios, results_myopic))

            if not os.path.exists(training_folder):
                os.makedirs(training_folder)

            with open(training_folder + training_file, "wb") as fp:
                pickle.dump(training_instances, fp)

        # Increment the episode number
        i += 1

    # Export the final learnt parameters
    torch.save(vfa_agent.get_local_network().state_dict(), "./vfa/parameter/vfa_parameters_" + sector.get_id() + ".pth")

    return vfa_agent








