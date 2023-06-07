import datetime
import multiprocessing as mp
import pickle
import random
import sys

from copy import deepcopy

from constants.Settings import BATCH_SIZE, EPSILON_B, MAX_ITER, TAU_TARGET, RESPONSE_PERIOD, TIME_LIMIT_B
from mavfa.MAVFAAgent import MAVFAAgent
from model.PatrolPresence import *
from model.ReschedulerMA import *
from util.ScheduleUtil import compute_hamming_distance_joint, get_global_Q_j, get_objective_value_MA, get_post_state
from util.utils import response_utility_fn, round_to_nearest
from vfa.VFAAgent import VFAAgent


class Simulator(object):

    def __init__(self, sectors=None, scenarios=None, initial_schedules_dict=None, time_matrix=None, adj_matrix=None,
                 neighbours_table=None, best_response=True):

        self.sectors = sectors  # Dictionary of sectors
        self.scenarios = scenarios  # Dictionary of scenarios
        self.initial_schedules_dict = initial_schedules_dict  # A dictionary of {sector id : schedule}

        self.S = list(self.scenarios.keys())
        self.R = {}
        self.z = {}
        for s in self.scenarios:
            self.R[s] = list(range(len(self.scenarios[s])))  # List of incidents at each scenario
            self.z[s] = []  # To contain the response utility factor for each incident in a scenario

        self.D = time_matrix  # Time travel matrix for all patrol areas across sectors
        self.Q_j = get_global_Q_j(sectors)  # A dictionary of
        # {patrol area id: min patrol time requirement in time period}
        all_patrol_areas = []  # a list of all patrol area ids across the sectors
        self.patrol_area_to_sector_map = {}
        for sector_id in sectors.keys():
            for area in sectors[sector_id].get_all_patrol_areas():
                self.patrol_area_to_sector_map[area.get_id()] = sector_id
                all_patrol_areas.append(area.get_id())

        self.all_patrol_areas = sorted(all_patrol_areas)
        self.subagent_dim = max([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
        self.mask = np.array([adj_matrix[0]])
        self.neighbours_table = neighbours_table
        self.is_multi_agent = True if len(sectors.keys()) > 1 else False

        self.time_tracker = {"Decision": [], "Assignment": [], "Best Response": []}  # To track computational times
        self.best_response = best_response # Indicator to turn on/off best response

    def run(self, policy, learning_agent=None, show_details=False):

        results = []  # a list of tuple (response_count, success_count, presence_utility)
        timings = []  # a list of list of list of computation time per incident per scenario
        # logs = []

        for s in self.S:
            print("Run scenario: " + str(s + 1))
            # start_run = datetime.datetime.now()

            timings_per_scenario = []
            # To record the detailed results per incident per scenario
            results_per_scenario = []

            schedules_dict = deepcopy(self.initial_schedules_dict)

            incident_counter = 0
            print("Number of incidents: " + str(len(self.scenarios[s])))
            for incident in self.scenarios[s]:
                start_incident = datetime.datetime.now()
                # print(incident.to_string())

                # Solution is a tuple of (sector_id, schedules dict, response utility))
                if "greedy" in policy.lower():
                    solution = self.find_greedy_solution(schedules_dict, s, incident, policy)
                elif policy.lower() == "dqn" or "central" in policy.lower():
                    solution = self.find_optimal_decision_central(schedules_dict, s, incident, policy, learning_agent)
                elif policy.lower() == "madqn":
                    solution = self.find_optimal_decision_madqn(schedules_dict, s, incident, policy, learning_agent)
                else:
                    # Find optimal incident assignment and rescheduling decisions
                    solution = self.find_optimal_decision(schedules_dict, s, incident, policy, learning_agent)

                schedules_dict = solution[1]
                # update self.z[s].append()
                self.z[s].append(solution[2])

                end_incident = datetime.datetime.now()
                # Computation time for each incident
                run_duration_incident = (end_incident - start_incident).total_seconds()
                timings_per_scenario.append(run_duration_incident)
                # For each incident in a scenario, record the incident sector, the responding sector and
                # the response utility). For analysis
                results_per_scenario.append((incident.get_sector(), solution[0], solution[2]))

                self.time_tracker["Decision"].append(run_duration_incident)

                if show_details:
                    print("Decision Time per incident: " + str(run_duration_incident))
                    print(str(incident_counter + 1) + " out of " + str(len(self.scenarios[s])) + " incidents completed")
                incident_counter += 1

            response_count = self.get_response_count(s)
            success_count = self.get_success_count(s)
            presence_utility = get_objective_value_MA(schedules_dict, self.sectors)
            hamming_dist = compute_hamming_distance_joint(schedules_dict, self.initial_schedules_dict)
            # print(presence_utility)
            # print(response_count)
            # print(success_count)
            # print(incident_counter)
            # for sector in schedules_dict.keys():
            #     # print(schedules_dict[sector].get_time_tables())
            #     # print(schedules_dict[sector].get_avail_indicators())
            #     print(get_effective_time_tables(schedules_dict[sector]))
            # sys.exit()
            results.append({"Breakdown": results_per_scenario, "Total": (response_count, success_count,
                                                                         presence_utility, hamming_dist,
                                                                         len(self.scenarios[s]))})
            timings.append(timings_per_scenario)

        return results, timings

    def find_greedy_solution(self, schedules_dict, s, incident, policy):
        """

        :param schedules_dict:
        :param s:
        :param incident:
        :param policy:
        :return:
        """

        incident_time_index = get_time_index(incident.get_start_time())
        incident_location = incident.get_location().get_id()
        # response_time_kpi = get_time_index(round_to_nearest(TAU_TARGET, TIME_UNIT))
        incident_sector = incident.get_sector()


        action_space = []  # All possible action
        action_space_eff = []  # All feasible action

        # Save the previous obj values of the joint schedules
        # prev_obj_value = get_objective_value_MA(schedules_dict, self.sectors)
        # Compile all possible actions
        for sector_id in schedules_dict.keys():

            if not (sector_id in self.neighbours_table[incident.get_sector()] or sector_id == incident.get_sector()):
                continue

            schedule = schedules_dict[sector_id]

            # Find feasible action i.e. agent and corresponding time to respond
            for agent in schedule.get_agents():
                for k in range(RESPONSE_PERIOD):
                    if incident_time_index + k <= len(T) - 1:
                        # agent must be available/ not travelling
                        if schedule.get_time_tables()[agent][incident_time_index + k] > 0:
                            # ignore anything beyond the shift / T
                            if incident_time_index + k > len(T) - 1:
                                continue
                            # Only include feasible actions
                            if schedule.get_avail_indicators()[agent][incident_time_index + k] >= 1:
                                action_space_eff.append((sector_id, agent, incident_time_index + k))
                    # Include all possible actions
                    action_space.append((sector_id, agent, incident_time_index + k))

        # if effective action space is empty, incident cannot be assigned
        if len(action_space_eff) == 0:
            return None, schedules_dict, 0

        priority_actions = []

        # for k in range(get_time_index(TAU_TARGET)):
        #     # each response time period consists of list of action
        #     # priority_actions.append([])
        #     actions = [x for x in action_space_eff if x[2] == incident_time_index + k]
        #     # min_distance = 1000
        #     for action in actions:
        #         distance = self.D[schedules_dict[action[0]].get_time_tables()[action[1]][action[2]]][incident_location]
        #         arrival_time = incident_time_index + k + get_time_index(round_to_nearest(distance, TIME_UNIT))
        #         priority_actions.append((action, distance, arrival_time))

        for action in action_space_eff:
            distance = self.D[schedules_dict[action[0]].get_time_tables()[action[1]][action[2]]][incident_location]
            arrival_time = action[2] + get_time_index(round_to_nearest(distance, TIME_UNIT))
            priority_actions.append((action, distance, arrival_time))

        # Shuffle the list first before sorting
        random.shuffle(priority_actions)
        priority_actions = sorted(priority_actions, key=lambda x: x[2])

        for priority_action in priority_actions:

            temp_schedules_dict = deepcopy(schedules_dict)
            action = priority_action[0]
            response_time = self.get_response_time(temp_schedules_dict[action[0]], action[1], action[2], incident)
            rescheduler = ReschedulerMA(temp_schedules_dict, self.initial_schedules_dict, action[0],
                                        self.sectors, self.D, self.mask, self.Q_j, t_k=incident_time_index)
            chosen_schedule = rescheduler.reschedule_with_incident(incident, action[1:])

            if chosen_schedule:
                response_utility = response_utility_fn(response_time * TIME_UNIT)
                temp_schedules_dict[action[0]] = chosen_schedule

                if len(temp_schedules_dict.keys()) > 1 and self.best_response:
                    temp_schedules_dict = self.best_response_procedure(temp_schedules_dict, incident_time_index, policy)

                return action[0], temp_schedules_dict, response_utility

        # No Feasible action
        return None, schedules_dict, 0

    def find_optimal_decision_central(self, schedules_dict, s, incident, policy, learning_agent=None):

        incident_time_index = get_time_index(incident.get_start_time())
        # incident_location = incident.get_location().get_id()
        incident_sector = incident.get_sector()


        action_space = []  # All possible action
        action_space_eff = []  # All feasible action

        # Compile all possible actions
        for sector_id in schedules_dict.keys():
            # if sector_id != incident_sector and sector_id not in self.neighbours_table[incident_sector]:
            #     continue

            schedule = schedules_dict[sector_id]

            # Find feasible action i.e. agent and corresponding time to respond
            for agent in schedule.get_agents():
                for k in range(RESPONSE_PERIOD):
                    if incident_time_index + k <= len(T) - 1:
                        # agent must be available/ not travelling
                        if schedule.get_time_tables()[agent][incident_time_index + k] > 0:
                            # ignore anything beyond the shift / T
                            if incident_time_index + k > len(T) - 1:
                                continue
                            # Only include feasible actions
                            if schedule.get_avail_indicators()[agent][incident_time_index + k] >= 1:
                                action_space_eff.append((sector_id, agent, incident_time_index + k))
                    # Include all possible actions
                    action_space.append((sector_id, agent, incident_time_index + k))

        # if effective action space is empty, incident cannot be assigned
        if len(action_space_eff) == 0:
            return None, schedules_dict, 0

        action_idx = learning_agent.act(get_pre_state_MA(schedules_dict, self.Q_j, self.all_patrol_areas, incident))
        action = action_space[action_idx]

        if action in action_space_eff:
            response_time = self.get_response_time(schedules_dict[action[0]], action[1], action[2], incident)
            rescheduler = ReschedulerMA(schedules_dict, self.initial_schedules_dict, action[0],
                                        self.sectors, self.D, self.mask, self.Q_j, t_k=incident_time_index)
            chosen_schedule = rescheduler.reschedule_with_incident(incident, action[1:])

            if chosen_schedule:
                response_utility = response_utility_fn(response_time * TIME_UNIT)
                schedules_dict[action[0]] = chosen_schedule
                return action[0], schedules_dict, response_utility

        else:

            return None, schedules_dict, 0

    def find_optimal_decision_madqn(self, schedules_dict, s, incident, policy, learning_agent=None):

        incident_sector = incident.get_sector()
        incident_time_index = get_time_index(incident.get_start_time())

        action_space = []  # All possible action
        action_space_eff = []  # All feasible action


        # Compile all possible actions
        for sector_id in schedules_dict.keys():
            action_space.append(sector_id)

            if not (sector_id != incident_sector and sector_id not in self.neighbours_table[incident_sector]):
                action_space_eff.append(sector_id)

        state_pre = get_pre_state_MA(schedules_dict, self.Q_j, self.all_patrol_areas, incident)
        action_idx = learning_agent.act(state_pre)

        action = action_space[action_idx]

        if action in action_space_eff:

            # Original schedules_dict is kept intact
            temp_schedules_dict = deepcopy(schedules_dict)
            # Insert the incident into the sector id and reschedule it
            schedule, response_utility = self.incident_response_procedure(temp_schedules_dict, action, 0,
                                                                               incident, "myopic")

            if schedule:
                schedules_dict[action] = schedule

                if self.best_response:
                    schedules_dict = self.best_response_procedure(schedules_dict, incident_time_index, "myopic")

                return action, schedules_dict, response_utility

            else:

                return None, schedules_dict, 0

        else:

            return None, schedules_dict, 0



    def find_optimal_decision(self, schedules_dict, s, incident, policy, learning_agent=None):
        """

        :param schedules_dict: A dictionary of {sector id : schedule}
        :param s: index of scenario
        :param incident: an incident instance
        :param policy: to determine the action taken
        :param learning_agent: if any
        :return: a tuple of <assigned sector, updated schedules_dict and the response utility value>
        """
        candidate_solns = []
        # Initialise best solution
        best_solution = (None, schedules_dict, 0)
        incident_sector = incident.get_sector()
        t_k = get_time_index(incident.get_start_time())

        # For each sector, insert the incident -> reschedule its own schedule -> Do best response across all sectors
        for sector_id in schedules_dict.keys():
            # If the responding agent is not adjacent to the incident sector, skip
            if sector_id != incident_sector and sector_id not in self.neighbours_table[incident_sector]:
                continue
            # print("Assign to " + str(sector_id))
            start_assignment = datetime.datetime.now()
            # Original schedules_dict is kept intact
            temp_schedules_dict = deepcopy(schedules_dict)
            # Insert the incident into the sector id and reschedule it
            schedule, response_utility = self.incident_response_procedure(temp_schedules_dict, sector_id,
                                                                          s, incident, policy,
                                                                          learning_agent)

            end_assignment = datetime.datetime.now()
            # Computation time for each assignment
            run_duration_assignment = (end_assignment - start_assignment).total_seconds()
            self.time_tracker["Assignment"].append(run_duration_assignment)

            # If agent sector_id cannot respond to the incident, try assigning the incident to other sector_id
            if response_utility == 0:
                continue

            temp_schedules_dict[sector_id] = schedule

            # initial_obj_value = get_objective_value_MA(temp_schedules_dict, self.sectors)

            # do best response if is a multi agent/sectors problem
            start_best_response = datetime.datetime.now()
            if len(temp_schedules_dict.keys()) > 1 and self.best_response:
                temp_schedules_dict = self.best_response_procedure(temp_schedules_dict, t_k, policy,
                                                                   learning_agent)

                # final_obj_value = get_objective_value_MA(temp_schedules_dict, self.sectors)

            # increment = final_obj_value - initial_obj_value
            # if increment < 0:
            #     print("Best response procedure  do not improve solution")
            #     sys.exit()

            end_best_response = datetime.datetime.now()
            # Computation time for each best response
            run_duration_best_response = (end_best_response - start_best_response).total_seconds()
            self.time_tracker["Best Response"].append(run_duration_best_response)

            candidate_solns.append((sector_id, temp_schedules_dict, response_utility))

        # If incident cannot be assigned to any sectors, return the initial best solution value
        if not candidate_solns:
            return best_solution

        # If policy is random
        if "random" in policy.lower():
            return random.choice(candidate_solns)

        best_reward = -1e7

        for solution in candidate_solns:
            reward = solution[2] * get_objective_value_MA(solution[1], self.sectors) - \
                     get_objective_value_MA(schedules_dict, self.sectors) + self.get_value(solution[1], t_k,
                                                                                           learning_agent)
            if reward > best_reward:
                best_solution = solution
                best_reward = reward

        return best_solution

    def best_response_procedure(self, schedules_dict, t_k, policy, learning_agent=None):

        sector_ids = list(schedules_dict.keys())
        initial_schedules_dict = deepcopy(schedules_dict)
        best_schedules_dict = initial_schedules_dict
        prev_schedules_dict = None
        best_obj_value = get_objective_value_MA(initial_schedules_dict, self.sectors) + \
                         self.get_value(initial_schedules_dict, t_k, learning_agent)

        # Add the initial joint schedule into the sampling pool
        sampling_pool = [initial_schedules_dict]
        random_sampling_pool = [initial_schedules_dict]

        i = 0
        t_start = datetime.datetime.now()
        t = 0
        while i < MAX_ITER and t < TIME_LIMIT_B / len(self.sectors):

            random_no = random.random()
            # Randomly sample schedules from the pool
            if random_no > EPSILON_B:
                random.shuffle(sampling_pool)
            # Select the best schedules so far
            else:
                if prev_schedules_dict:
                    sampling_pool.insert(0, prev_schedules_dict)
                    random_sampling_pool.insert(0, prev_schedules_dict)

            # If sampling pool is empty, exit the best response procedure
            if len(sampling_pool) == 0:
                break

            sampled_schedules_dict = deepcopy(sampling_pool[0])
            sampling_pool.pop(0)

            count_table = get_patrol_count_table_MA(sampled_schedules_dict)

            new_schedules_dicts = []

            current_obj_value = get_objective_value_MA(sampled_schedules_dict, self.sectors) + \
                                self.get_value(sampled_schedules_dict, t_k, learning_agent)

            global_defects = check_min_patrol_presence_MA(count_table, self.Q_j)

            for sector_id in sector_ids:
                # print("Computing best response for Sector " + str(sector_id))

                # Compile list of external defects i.e patrol locations that are not patrolled minimally
                # across all sectors

                if len(global_defects) > 0:
                    # print("before filtering")
                    # for defect in global_defects:
                    #     print(defect.to_string())
                    #     print(self.patrol_area_to_sector_map[defect.get_location()])
                    #
                    # before_filter = len(global_defects)

                    # Filter only defects that belong to adjacent sectors
                    global_defects = [defect for defect in global_defects if
                                      self.patrol_area_to_sector_map[defect.get_location()] == sector_id or
                                      self.patrol_area_to_sector_map[defect.get_location()] in
                                      self.neighbours_table[sector_id]]
                    # print(sector_id, self.neighbours_table[sector_id])
                    # print("After filtering", sector_id)
                    # after_filter = len(global_defects)
                    # for defect in global_defects:
                    #     print(defect.to_string())
                    #     print(self.patrol_area_to_sector_map[defect.get_location()])

                # Reschedule Sector sector_id schedule with the external defects
                rescheduler = ReschedulerMA(sampled_schedules_dict, self.initial_schedules_dict, sector_id,
                                            self.sectors, self.D, self.mask, self.Q_j, t_k,
                                            learning_agent=learning_agent)

                new_schedules_dict = rescheduler.reschedule_without_incident(global_defects)

                if isinstance(new_schedules_dict, dict):
                    new_schedules_dicts.append(new_schedules_dict)

            # For each sector's best response, select those which improves
            for new_schedules_dict in new_schedules_dicts:
                new_obj_value = get_objective_value_MA(new_schedules_dict, self.sectors) + \
                                self.get_value(new_schedules_dict, t_k, learning_agent)
                if new_obj_value > current_obj_value:
                    if new_obj_value > best_obj_value:
                        prev_schedules_dict = new_schedules_dict
                        best_schedules_dict = new_schedules_dict
                        best_obj_value = new_obj_value
                    else:
                        if new_obj_value > current_obj_value:
                            sampling_pool.append(new_schedules_dict)
                            random_sampling_pool.append(new_schedules_dict)

            i += 1
            t = (datetime.datetime.now() - t_start).total_seconds()

        if "random" in policy.lower():
            return random.choice(random_sampling_pool)

        # print("Best response", t)

        return best_schedules_dict

    def incident_response_procedure(self, schedules_dict, sector_id, s, incident, policy, learning_agent=None):
        """
        For single agent problem
        :param schedules_dict:
        :param sector_id:
        :param s:
        :param incident:
        :param policy:
        :param learning_agent:
        :param time_matrix:
        :return:
        """

        # print(incident.to_string())
        # print(schedule.get_time_tables())
        incident_time_index = get_time_index(incident.get_start_time())
        incident_location = incident.get_location().get_id()
        response_time_kpi = get_time_index(round_to_nearest(TAU_TARGET, TIME_UNIT))

        # Save the previous obj values of the joint schedules
        prev_obj_value = get_objective_value_MA(schedules_dict, self.sectors)

        schedule = deepcopy(schedules_dict[sector_id])

        action_space = []  # All possible action
        action_space_eff = []  # All feasible action
        q_table = {}
        z_table = {}
        re_schedules = {}

        random_action = False

        # Find feasible action i.e. agent and corresponding time to respond
        for agent in schedule.get_agents():
            for k in range(RESPONSE_PERIOD):
                if incident_time_index + k <= len(T) - 1:
                    # agent must be available/ not travelling
                    if schedule.get_time_tables()[agent][incident_time_index + k] > 0:
                        # ignore anything beyond the shift / T
                        if incident_time_index + k > len(T) - 1:
                            continue
                        # Only include feasible actions
                        if schedule.get_avail_indicators()[agent][incident_time_index + k] >= 1:
                            action_space_eff.append((agent, incident_time_index + k))
                # Include all possible actions
                action_space.append((agent, incident_time_index + k))

        # if effective action space is empty, incident cannot be assigned
        if len(action_space_eff) == 0:
            # print("No feasible rescheduling action")
            # self.z[s].append(0)  # 0 means incident is not attended to
            return schedule, 0
            # return schedule, 0

        # If policy used is DQN
        # if "dqn" in policy.lower():
        #
        #     temp_schedules_dict = deepcopy(schedules_dict)
        #
        #     action_idx = learning_agent.act(get_pre_state(schedule, self.sectors[sector_id], incident))
        #     action = action_space[action_idx]
        #
        #     # If the agent is not available
        #     if action in action_space_eff:
        #
        #         # Time required to reach the incident upon responding action
        #         response_time = self.get_response_time(schedule, action[0], action[1], incident)
        #
        #         rescheduler = ReschedulerMA(schedules_dict, self.initial_schedules_dict, sector_id, self.sectors,
        #                                     self.D, self.mask, self.Q_j, t_k=action[1])
        #         re_schedules[action] = rescheduler.reschedule_with_incident(incident, action)
        #         if re_schedules[action]:
        #             z_table[action] = response_utility_fn(response_time * TIME_UNIT)
        #         else:
        #             re_schedules[action] = schedule
        #             z_table[action] = 0
        #     else:
        #         re_schedules[action] = schedule
        #         z_table[action] = 0
        #
        #     # Update objective value of a schedule
        #     # re_schedules[action].update_objective_value(
        #     #     get_objective_value(re_schedules[action], self.sectors[sector_id]))
        #     temp_schedules_dict[sector_id] = re_schedules[action]
        #
        #     q_table[action] = z_table[action] * get_objective_value_MA(temp_schedules_dict, self.sectors) - \
        #                       prev_obj_value + self.get_value(temp_schedules_dict, action[1], learning_agent)

        # Always dispatch the nearest available agent immediately
        elif "greedy" in policy.lower():

            priority_actions = []

            for k in range(get_time_index(TAU_TARGET)):
                # each response time period consists of list of action
                actions = [x for x in action_space_eff if x[1] == incident_time_index + k]
                for action in actions:
                    distance = self.D[schedule.get_time_tables()[action[0]][action[1]]][incident_location]
                    arrival_time = incident_time_index + k + get_time_index(round_to_nearest(distance, TIME_UNIT))
                    priority_actions.append((action, distance, arrival_time))

            # Shuffle the list first before sorting
            random.shuffle(priority_actions)
            priority_actions = sorted(priority_actions, key=lambda x: x[2])

            for priority_action in priority_actions:
                action = priority_action[0]
                response_time = self.get_response_time(schedule, D, action[0], action[1], incident)
                rescheduler = ReschedulerMA(schedules_dict, self.initial_schedules_dict, sector_id, self.sectors,
                             self.D, self.mask, self.Q_j, t_k=incident_time_index)
                chosen_schedule = rescheduler.reschedule_with_incident(incident, action)

                if chosen_schedule:
                    response_utility = response_utility_fn(response_time * TIME_UNIT)
                    return chosen_schedule, response_utility

            return schedule, 0

        # For policies other than DQN and greedy
        else:
            # print(action_space)

            # print(action_space_eff)
            for action in action_space_eff:

                temp_schedules_dict = deepcopy(schedules_dict)

                response_time = self.get_response_time(schedule, action[0], action[1], incident)

                start_run_action = datetime.datetime.now()

                if policy.lower() in ["heuristics", "myopic", "random", "vfa", "vfa_lite", "myopic_lite"]:

                    # print(action)
                    start_run_heur = datetime.datetime.now()

                    if "random" in policy.lower():
                        random_action = True

                    rescheduler = ReschedulerMA(schedules_dict, self.initial_schedules_dict, sector_id, self.sectors,
                                                self.D, self.mask, self.Q_j, t_k=incident_time_index,
                                                learning_agent=learning_agent)
                    re_schedules[action] = rescheduler.reschedule_with_incident(incident, action)

                    end_run_heur = datetime.datetime.now()
                    run_duration_heur = (end_run_heur - start_run_heur).total_seconds()
                    # print("Total computation time for 1 rescheduling action: " + str(run_duration_heur) + 's')

                    if re_schedules[action]:
                        # defects = check_defects(re_schedules[action], self.sector, action[1])
                        # if len(defects) > 0:
                        #     sys.exit("Defective schedule")

                        # For checking purposes
                        if compute_hamming_distance(re_schedules[action], self.initial_schedules_dict[sector_id]) > \
                                MAX_PERTURBATION_DISTANCE:
                            sys.exit("Too much perturbation (MAIN)")

                        z_table[action] = response_utility_fn(response_time * TIME_UNIT)

                    else:
                        # No feasible rescheduling
                        re_schedules[action] = schedule
                        z_table[action] = 0

                    temp_schedules_dict[sector_id] = re_schedules[action]

                    q_table[action] = z_table[action] * get_objective_value_MA(temp_schedules_dict, self.sectors) - \
                                      prev_obj_value + self.get_value(temp_schedules_dict, incident_time_index,
                                                                      learning_agent)

                end_run_action = datetime.datetime.now()
                run_duration_action = (end_run_action - start_run_action).total_seconds()
                # print("Total computation time for 1 action: " + str(run_duration_action) + 's')

        # print(q_table)

        # Choose action with highest q_value
        _, max_q = max(q_table.items(), key=lambda x: x[1])
        # print(max_q)
        # Compile a list of action with the same q_value
        candidates = []

        for action in action_space:

            if action in q_table.keys():
                if q_table[action] == max_q:
                    candidates.append(action)

        # If there are more than one candidates, randomly select one
        max_action = random.choice(candidates)
        # print(max_action, max_q)

        # print(re_schedules[max_action].get_time_tables())

        # If policy is random, randomly choose a feasible action
        if random_action:
            max_action = random.choice(list(q_table.keys()))

        # self.z[s].append(z_table[max_action])

        # incident_level_log.append((max_action, z_table[max_action]))

        return re_schedules[max_action], z_table[max_action]

    # def get_success_rate(self, s):
    #
    #     return sum(self.z[s]) / len(self.z[s])

    def get_response_count(self, s):

        response_count = 0

        for x in self.z[s]:
            if x > 0:
                response_count += 1

        return response_count

        # return sum(self.z[s])

    def get_success_count(self, s):

        success_count = 0

        for x in self.z[s]:
            if x == 1:
                success_count += 1

        return success_count

    def get_value(self, schedules_dict, decision_time, learning_agent=None):
        """

        :param schedules_dict:
        :param decision_time: time index when incident occurs / decision needs to be made
        :param learning_agent:
        :return:
        """
        score = 0
        # Add value function
        if isinstance(learning_agent, VFAAgent) or isinstance(learning_agent, MAVFAAgent) :
            if self.is_multi_agent:
                global_state = [decision_time / len(T)] + \
                               get_patrol_presence_status_MA(get_patrol_count_table_MA(schedules_dict), self.Q_j)
                score += GAMMA * learning_agent.return_value(get_post_joint_state(schedules_dict, self.Q_j,
                                                                                  self.all_patrol_areas, decision_time,
                                                                                  self.subagent_dim), global_state,
                                                             self.mask)
            else:
                score += GAMMA * learning_agent.return_value(get_post_state(schedules_dict[next(iter(schedules_dict))],
                                                                            self.sectors[next(iter(schedules_dict))],
                                                                            decision_time))

        return score

    def get_response_time(self, schedule, agent, action_time, incident):
        """

        :param schedule:
        :param agent:
        :param action_time: time index when the agent acts upon the incident call
        :param incident:
        :return:
        """

        time_tables = schedule.get_time_tables()
        # Time from current time to incident location
        time_to_incident = round_to_nearest(
            self.D[time_tables[agent][action_time]][incident.get_location().get_id()],
            TIME_UNIT)

        # Time periods from current time to incident location
        travel_time_slots = get_time_index(time_to_incident)

        response_time = action_time - get_time_index(incident.get_start_time()) + travel_time_slots

        return response_time

    def get_time_tracker(self):
        return self.time_tracker
