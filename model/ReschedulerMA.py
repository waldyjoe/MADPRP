import datetime
import gc
import random
import numpy as np
import sys

from copy import deepcopy
from constants.Settings import ACTION_SIZE, ATTN_OUT_DIM, EPSILON_H, GAMMA, HIDDEN_DIM, MAX_PERTURBATION_DISTANCE, \
    TIME_LIMIT_B, TIME_LIMIT_H, SEED
from entity.Schedule import *
from mavfa.MAVFAAgent import *
from util.ScheduleUtil import *
from util.utils import get_time_index, round_to_nearest
from vfa.VFAAgent import VFAAgent



class ReschedulerMA(object):

    def __init__(self, schedules_dict=None, original_schedules_dict=None, sector_id=None, sectors=None,
                 time_matrix=None, adj_matrix=None, Q_j=None, t_k=None, random=False, learning_agent=None):

        self.schedules_dict = schedules_dict  # Schedule prior to rescheduling
        self.prev_score = get_objective_value_MA(schedules_dict, sectors)
        self.original_schedules_dict = original_schedules_dict
        self.sector_id = sector_id
        self.sectors = sectors
        self.t_k = t_k  # Refers to current time
        self.D = time_matrix
        self.Q_j = Q_j

        self.tabu_list = []
        # If random is True, selection of repair action will be random instead of choosing repair action with the best score
        self.random = random

        self.learning_agent = learning_agent

        all_patrol_areas = []  # a list of all patrol area ids across the sectors
        for sector_id in sectors.keys():
            all_patrol_areas += [area.get_id() for area in sectors[sector_id].get_all_patrol_areas()]

        self.all_patrol_areas = sorted(all_patrol_areas)
        self.subagent_dim = max([sectors[sector_id].get_agents_count() for sector_id in sectors.keys()])
        self.mask = np.array([adj_matrix[0]])

        self.start_time = datetime.datetime.now()
        self.is_multi_agent = True if len(sectors.keys()) > 1 else False

    def reschedule_with_incident(self, incident, action):
        curr_schedule = deepcopy(self.schedules_dict[self.sector_id])
        # print("Initial Schedule")
        # print(initial_schedule.get_time_tables())

        # Insert incident
        # print("Inserting Incident")
        if not self.insert_incident(curr_schedule, incident, action):
            return None
        # print("After Inserting Incident")
        # print(curr_schedule.get_time_tables())
        # best_score = get_objective_value(curr_schedule, self.sector)
        # print(best_score)
        # sys.exit()

        return self.check_repair(curr_schedule)

    def insert_incident(self, schedule, incident, action):
        """

        :param schedule:
        :param incident:
        :param action: (<agent>, <time_index>)
        :return: No need to return any schedule since schedule is dynamically updated
        """

        agent = action[0]
        action_time = action[1]
        # print("Insert Incident")
        time_tables = schedule.get_time_tables()
        # print(time_tables)
        incident_loc = incident.get_location().get_id()
        # incident_time_index = get_time_index(incident.get_start_time())
        resolution_time_index = get_time_index(incident.get_resolution_time())

        # Time from current time to incident location
        time_to_incident = round_to_nearest(
            self.D[time_tables[agent][action_time]][incident.get_location().get_id()],
            TIME_UNIT)
        # Time periods from current time to incident location
        travel_time_slots = get_time_index(time_to_incident)

        for buffer in range(travel_time_slots):
            # Action or schedule beyond T are ignored
            if action_time + buffer <= len(T) - 1:
                time_tables[agent][action_time + buffer] = -1

        # Agent must be at the incident location after travelling
        if action_time + travel_time_slots <= len(T) - 1:
            time_tables[agent][action_time + travel_time_slots] = incident_loc
        else:
            # Agent can only arrive at incident location beyond the shift time
            # print("Incident can only be attended to beyond the shift time")
            return False

        # Agent must be unavailable when responding to the incident
        for responding_period in range(resolution_time_index):
            if action_time + travel_time_slots + responding_period <= len(T) - 1:
                time_tables[agent][action_time + travel_time_slots + responding_period] = incident_loc

        schedule.update_time_intervals(time_tables)

        # Agent will be unavailable during incident response
        new_unavail = (agent, (incident.get_location().get_id(), action_time + travel_time_slots,
                                    action_time + travel_time_slots + resolution_time_index))

        schedule.update_unavailability_table(new_unavail[0], new_unavail[1])

        # Update availability indicators
        for idx in range(action_time, action_time + travel_time_slots + resolution_time_index):

            # Action or schedule beyond T are ignored
            if idx > len(T) - 1:
                continue
            schedule.update_avail_indicators(agent, idx, 0)

        # Update objective value and patrol presence
        schedule.update_objective_value(get_objective_value(schedule, self.sectors[self.sector_id]))

        return True

    def reschedule_without_incident(self, defects):

        candidate_solns = []
        best_score = self.prev_score
        best_schedules_dict = self.schedules_dict

        t_start = datetime.datetime.now()
        # for each defect, fix it and do an ejection chain process to get final obj value
        for defect in defects:
            initial_schedules_dict = deepcopy(self.schedules_dict)
            initial_schedules_dict[self.sector_id] = self.check_repair(self.repair(
                initial_schedules_dict[self.sector_id], defect))

            if isinstance(initial_schedules_dict[self.sector_id], Schedule):
                candidate_solns.append(initial_schedules_dict)

            # To limit the processing time
            t = (datetime.datetime.now() - t_start).total_seconds()
            # print("Reschedule w/o incident", t)
            if t > TIME_LIMIT_B / len(self.sectors):
                break



        # find the move that return the best obj value
        for solution in candidate_solns:
            # Only evaluate schedule type objects
            if isinstance(solution, dict):
                # score = get_objective_value_MA(solution, self.sectors)
                score = self.get_score(solution)
                if score > best_score:
                    best_score = score
                    best_schedules_dict = solution

        return best_schedules_dict

    def check_repair(self, schedule):

        temp_schedules_dict = deepcopy(self.schedules_dict)

        if not isinstance(schedule, Schedule):
            return None

        is_defective = True
        t_start = datetime.datetime.now()
        last_acceptable_schedule = None

        while is_defective:

            temp_schedules_dict[self.sector_id] = schedule

            # check tabu list
            if self.is_tabu(schedule):

                if self.is_acceptable(temp_schedules_dict):
                    return schedule
                else:
                    return None

            # insert the schedule into tabu list
            self.tabu_list.append(schedule)

            temp_schedules_dict[self.sector_id] = schedule
            defects = check_defects_MA(temp_schedules_dict, self.sector_id, self.sectors, self.D, self.Q_j, self.t_k)

            # sort defects by type. priority given to type 1
            defects = sorted(defects, key=lambda x: x.type, reverse=False)

            # for defect in defects:
            #     print(defect.to_string())
            # sys.exit()

            # If the resulting schedule is very different from the initial schedule, do not explore this path
            if compute_hamming_distance(schedule, self.original_schedules_dict[self.sector_id]) > \
                    MAX_PERTURBATION_DISTANCE:
                # print("Too much perturbation")
                if last_acceptable_schedule:
                    return last_acceptable_schedule
                else:
                    return None

            if self.is_acceptable(temp_schedules_dict):
                last_acceptable_schedule = schedule

            # # To check whether time limit has been reached
            # current_time = datetime.datetime.now()
            # run_duration = (current_time - self.start_time).total_seconds()
            # # print(run_duration)

            if len(defects) == 0:
                is_defective = False
            # # If time limit is reached and no type 1 defect
            # elif run_duration > TIME_LIMIT and is_acceptable(schedule, self.sector):
            #     return schedule
            # elif run_duration > 5*TIME_LIMIT:
            #     return None
            else:
                # Fix the first defect
                schedule = self.repair(schedule, defects[0])

                if not schedule:
                    # print("Defect cannot be repaired")
                    return None

            # To limit the processing time
            t = (datetime.datetime.now() - t_start).total_seconds()
            if t > TIME_LIMIT_H / len(self.sectors):
                return last_acceptable_schedule

        return schedule

    def repair(self, original_schedule, defect):
        """

        :param original_schedule:
        :param defect:
        :return: schedule
        """
        # print("repairing")

        schedule = deepcopy(original_schedule)
        # if compute_hamming_distance(schedule, self.original_schedule) > MAX_PERTURBATION_DISTANCE:
        #     return None

        time_tables = schedule.get_time_tables()

        # print(self.schedule.get_time_tables())
        # print("before repair")
        # print(time_tables)
        # print(compute_hamming_distance(schedule, self.original_schedule))
        #
        # print(defect.to_string())



        # To contain all possible candidate solutions
        candidate_solns = []

        value = defect.get_value()

        if defect.get_type() == 1:
            # Insert and eject
            agent = defect.get_agents()[0]
            # start, end = defect.get_interval()
            # dest = time_tables[agent][end]
            # origin = time_tables[agent][start]
            candidate_solns += self.eject_operator(schedule, defect)
            candidate_solns += self.replace_operator(schedule, defect, self.sectors[self.sector_id].get_proximity_table())

        # For defect type 2
        else:
            # Update time tables and time intervals with availability

            # get the time intervals with availability of all agents in all sectors
            temp_schedules_dict = deepcopy(self.schedules_dict)
            temp_schedules_dict[self.sector_id] = schedule

            global_count_table = get_patrol_count_table_MA(temp_schedules_dict)

            time_tables_avail = get_effective_time_tables(schedule)
            time_intervals_avail = {}
            for agent in time_tables_avail:
                time_intervals_avail[agent] = convert_to_time_intervals(time_tables_avail[agent])
            # Check
            local_patrol_areas = list(self.sectors[self.sector_id].get_presence_table().keys())
            external_patrol_areas = []

            # Tables for schedule before the action time
            count_table_bf = {}
            time_intervals_by_loc_bf = {}

            # Tables for schedule after the action time
            count_table_aft = {}
            time_intervals_by_loc_aft = {}

            for agent in time_tables_avail.keys():
                for idx in range(len(T)):
                    location = time_tables_avail[agent][idx]
                    # Sieve out external patrol areas
                    if location not in local_patrol_areas:
                        external_patrol_areas.append(location)
                    if idx < self.t_k:
                        if location not in count_table_bf:
                            count_table_bf[location] = 0
                        count_table_bf[location] += 1
                    else:
                        if location not in count_table_aft:
                            count_table_aft[location] = 0
                        count_table_aft[location] += 1

            # print(count_table_bf)
            # print(count_table_aft)
            # Compute the surplus patrol times for each location minus the time spent attending to incident
            avail_surplus = {}
            for key in self.Q_j:
                # min_patrol_presence = round_to_nearest(Q_j[key], TIME_UNIT) / TIME_UNIT
                if key not in local_patrol_areas + external_patrol_areas:
                    continue

                if key not in avail_surplus:
                    avail_surplus[key] = 0
                if key not in count_table_bf:
                    count_table_bf[key] = 0
                if key not in count_table_aft:
                    count_table_aft[key] = 0

                external_contribution = global_count_table[key] - count_table_aft[key] - count_table_bf[key]
                avail_surplus[key] = count_table_aft[key] - (self.Q_j[key] - external_contribution -
                                                             count_table_bf[key])



            # Shortlist patrol areas with surpluses to be replaced by the patrol area with defect
            # Greedy choice = choose the patrol area with surpluses which result in the least defect

            # Filter patrol areas that have surpluses >= the defect magnitude
            # Splitting of patrol area is not done because it may cause more defect
            filtered_dict = {k: v for (k, v) in avail_surplus.items() if v >= value}
            # print(filtered_dict)

            if not filtered_dict:
                # print("No repair can be done")
                # Assign a penalty cost?
                return original_schedule

            # Find the available time intervals by location
            for agent in time_tables_avail.keys():
                for item in time_intervals_avail[agent]:
                    if item[0] <= 0:
                        continue
                    if get_time_index(item[1][1]) <= self.t_k:
                        if item[0] not in time_intervals_by_loc_bf:
                            time_intervals_by_loc_bf[item[0]] = []
                        time_intervals_by_loc_bf[item[0]].append([agent, item[1]])
                    else:
                        if item[0] not in time_intervals_by_loc_bf:
                            time_intervals_by_loc_bf[item[0]] = []
                        if item[0] not in time_intervals_by_loc_aft:
                            time_intervals_by_loc_aft[item[0]] = []

                        # If the action time is in between the time interval , ignore
                        if get_time_index(item[1][0]) < self.t_k:
                            continue
                            # time_intervals_by_loc_bf[item[0]].append([agent, [item[1][0], T[self.action_time]]])
                            # time_intervals_by_loc_aft[item[0]].append([agent, [T[self.action_time], item[1][1]]])
                        else:
                            time_intervals_by_loc_aft[item[0]].append([agent, item[1]])

            # print(filtered_dict)
            # print(unavail_by_loc_r)
            # print(time_intervals_by_loc_r)

            possible_new_schedules = []

            # Replace the surpluses patrol area with the defect patrol area
            for key in filtered_dict:
                if key not in time_intervals_by_loc_aft:
                    continue
                for interval in time_intervals_by_loc_aft[key]:
                    # print(interval)
                    temp_time_tables = deepcopy(time_tables)
                    # Replace the interval with defect patrol areas

                    # Check if the interval can accommodate the defect value
                    interval_length = int((interval[1][1] - interval[1][0]) / TIME_UNIT)
                    # if interval_length >= value:
                    if interval_length >= 0:
                        # print(key, interval)
                        temp_schedule = deepcopy(schedule)
                        # for i in range(defect.get_value()):
                        for i in range(interval_length):
                            temp_time_tables[interval[0]][get_time_index(interval[1][0]) + i] = \
                                defect.get_location()
                        # defects = check_defects(temp_schedule, self.sector)
                        temp_schedule.update_time_tables(temp_time_tables)
                        candidate_solns.append(temp_schedule)
                        # print(temp_time_tables)
                    else:
                        continue

        # schedule.update_time_tables(time_tables)

        # Choose the candidate solution with best obj value
        best_score = -1e7
        best_schedule = None
        for solution in candidate_solns:
            # print("evaluating solutions")
            # Only evaluate schedule type objects
            if isinstance(solution, Schedule):
                # Don't consider future value when fixing a defect because value fn determines
                # a feasible schedule's suitability in anticipating future incident.
                # After repair, a schedule may still be infeasible
                # defects = check_defects(solution, self.sectors[self.sector_id])
                temp_schedules_dict = deepcopy(self.schedules_dict)
                temp_schedules_dict[self.sector_id] = solution
                defects = check_defects_MA(temp_schedules_dict, self.sector_id, self.sectors, self.D, self.Q_j,
                                           self.t_k)
                # sort defects by type. priority given to type 1
                defects = sorted(defects, key=lambda x: x.type, reverse=False)
                defect_penalty = 0
                # Impose a defect penalty if the resulting schedule has type 1 defect
                for defect in defects:
                    if defect.get_type() == 1:
                        defect_penalty += abs(defect.get_value())
                # score = get_objective_value(solution, self.sector)
                score = self.get_score(temp_schedules_dict) - defect_penalty
                # print(solution.get_time_tables()[agent])

                # print(solution.get_time_tables())
                # print(score)
                if score > best_score:
                    best_score = score
                    best_schedule = solution
                    # best_schedule.update_objective_value(get_objective_value(best_schedule, self.sectors[self.sector_id]))
                    # print("Better solution found")
                    # print(best_schedule.get_time_tables())

        # With certain probability, choose random repair action
        random_number = random.random()
        if self.random or random_number < EPSILON_H:
            candidate_solns = [x for x in candidate_solns if isinstance(x, Schedule)]
            if len(candidate_solns) > 0:
                best_schedule = random.choice(candidate_solns)
                # best_schedule.update_objective_value(get_objective_value(best_schedule, self.sector))

        if not best_schedule:
            # print("Repair cannot be done")
            if defect.get_type() != 1:
                return original_schedule
            else:
                return None

        # print("After repair")
        # print(best_schedule.get_time_tables())
        # print(time_tables)
        # return best_schedule
        # sys.exit()
        return best_schedule

    def eject_operator(self, schedule, defect):

        candidate_solns = []

        time_tables = schedule.get_time_tables()
        # Update time tables and time intervals with availability
        time_tables_avail = get_effective_time_tables(schedule)
        time_intervals_avail = {}
        for agent in time_tables_avail:
            time_intervals_avail[agent] = convert_to_time_intervals(time_tables_avail[agent])

        # Insert and eject
        value = defect.get_value()
        agent = defect.get_agents()[0]
        start, end = defect.get_interval()
        dest = time_tables[agent][end]
        origin = time_tables[agent][start]

        # For insufficient case
        if value > 0:

            # Choose to eject either origin, dest, or a combination of both based on least cost

            # if both origin and destination are incident response action then repairing is not possible
            if time_tables_avail[agent][start] == 0 and time_tables_avail[agent][end] == 0:
                # print("No repair can be done")
                return candidate_solns

            # unrepairable = False

            # Check every possible insertion and replacement and its corresponding score
            # split_idx refers to the index of split. If the defect magnitude is 3,the splits can be done as
            # (0,3), (1,2), (2,1) and (3,0). (1,2) means eject one time period from start point and 2 from the end point
            for split_idx in range(value + 1):

                temp_time_tables = deepcopy(time_tables)

                # Backward check only if there is at least 1 ejection from the start point
                if split_idx > 0:
                    infeasible = False
                    for i in range(split_idx + 1):
                        # If need to eject beyond the start of shift or start of action time
                        if start - i < 0 or start - i < self.t_k:
                            infeasible = True
                            continue
                        item = time_tables_avail[agent][start - i]
                        # If the patrol area to eject is an incident response, cannot repair
                        if item == 0:
                            infeasible = True
                            continue

                        # Insert travelling time
                        if i == split_idx:
                            temp_time_tables[agent][start - i] = origin
                        else:
                            temp_time_tables[agent][start - i] = -1

                    # If there is no feasible insertion and ejection, this move is infeasible
                    if infeasible:
                        candidate_solns.append(None)
                        continue

                # Forward check only if there is at least 1 ejection from the end point
                if (value - split_idx) > 0:

                    infeasible = False

                    for j in range(value - split_idx + 1):
                        # Ignore time index beyond the shift hour
                        if end + j > len(T) - 1:
                            # If the final slot is not a destination, the move is infeasible
                            if temp_time_tables[agent][len(T) - 1] == -1:
                                infeasible = True
                            continue
                        item = time_tables_avail[agent][end + j]
                        # If the patrol area to eject is an incident response, cannot repair
                        if item == 0:
                            infeasible = True
                            continue

                        # Insert travelling time
                        if split_idx + j == value:
                            temp_time_tables[agent][end + j] = dest
                        else:
                            temp_time_tables[agent][end + j] = -1

                    # If there is no feasible insertion and ejection, this move is infeasible
                    if infeasible:
                        candidate_solns.append(None)
                        continue

                # Create a new schedule
                temp_schedule = deepcopy(schedule)

                temp_schedule.update_time_tables(temp_time_tables)
                # print(temp_schedule.get_time_tables())
                candidate_solns.append(temp_schedule)

            # sys.exit()

        # For excess case (If the time gap is too long)
        else:
            # print(time_tables)
            abs_value = abs(value)

            if origin != dest:
                # Check every possible insertion and replacement and its corresponding score
                # split_idx refers to the index of split. If the defect magnitude is 3,the splits can be done as
                # (0,3), (1,2), (2,1) and (3,0). (1,2) means replace one time period of travelling with start point and
                # 2 from the end point

                for split_idx in range(abs_value + 1):
                    # print(split_idx)
                    is_fixed = False

                    temp_time_tables = deepcopy(time_tables)

                    if split_idx > 0:
                        for i in range(1, split_idx + 1):
                            temp_time_tables[agent][start + i] = origin
                        is_fixed = True

                    if (abs_value - split_idx) > 0:
                        for j in range(1, abs_value - split_idx + 1):
                            temp_time_tables[agent][end - j] = dest
                        is_fixed = True

                    if is_fixed:
                        # Create a new schedule
                        temp_schedule = deepcopy(schedule)
                        # print(temp_time_tables)
                        temp_schedule.update_time_tables(temp_time_tables)
                        # print(temp_time_tables)

                        # print(temp_schedule.get_time_tables()[agent])
                        candidate_solns.append(temp_schedule)

            else:

                temp_time_tables = deepcopy(time_tables)
                for i in range(1, abs_value + 1):
                    temp_time_tables[agent][start + i] = origin
                temp_schedule = deepcopy(schedule)
                # print(temp_time_tables)
                temp_schedule.update_time_tables(temp_time_tables)
                # print(temp_time_tables)

                # print(temp_schedule.get_time_tables()[agent])
                candidate_solns.append(temp_schedule)
            # sys.exit("type 1 excess case")

        return candidate_solns

    def replace_operator(self, schedule, defect, proximity_table):
        # print("I am here")
        candidate_solns = []

        time_tables = schedule.get_time_tables()
        # Update time tables and time intervals with availability
        time_tables_avail = get_effective_time_tables(schedule)
        time_intervals_avail = {}
        for agent in time_tables_avail:
            time_intervals_avail[agent] = convert_to_time_intervals(time_tables_avail[agent])

        # Insert and eject
        value = defect.get_value()
        agent = defect.get_agents()[0]
        start, end = defect.get_interval()
        dest = time_tables[agent][end]
        origin = time_tables[agent][start]

        # Compute the number of patrol time unit at the destination to be replaced
        isSame = True
        end_dest = end
        while isSame:

            end_dest += 1
            if end_dest <= len(T) - 1:
                if time_tables[agent][end_dest] != dest:
                    isSame = False
            else:
                isSame = False

        replace_slot = end_dest - end

        # For insufficient case (not enough gap between 2 patrol areas), replace the next location to be within the acceptable distance
        if value > 0:
            time_gap = end - start - 1

            # if the existing gap is at least 1 time unit
            if time_gap > 0:
                # replace the next destination with location that is within that gap
                # print("Replace with the location that is within that gap")
                # orig_idx = station_ids.index(origin)

                for neighbour in proximity_table[origin][time_gap]:
                    infeasible = False
                    # if orig_idx - time_gap >= 0:
                    # Replace with the location before
                    temp_time_tables = deepcopy(time_tables)
                    # neighbour = station_ids[orig_idx - time_gap]
                    for i in range(replace_slot):
                        # Cannot replace if agent is responding to incident
                        if time_tables_avail[agent][end + i] == 0:
                            infeasible = True
                            continue
                        temp_time_tables[agent][end + i] = neighbour.get_id()

                    if not infeasible:
                        temp_schedule = deepcopy(schedule)
                        temp_schedule.update_time_tables(temp_time_tables)
                        candidate_solns.append(temp_schedule)
                        # print(temp_time_tables[agent])

                # replace the next destination with location that is reasonably nearer
                # print("Replace with the location that is reasonably nearer")
                if replace_slot > 1:

                    for j in range(1, replace_slot):
                        time_gap_ext = time_gap + j
                        # print(time_gap_ext)
                        for neighbour in proximity_table[origin][time_gap_ext]:
                            infeasible = False
                            # if orig_idx - time_gap_ext >= 0:
                            # Replace with the location before
                            temp_time_tables = deepcopy(time_tables)
                            # neighbour = station_ids[orig_idx - time_gap_ext]
                            for k in range(j):
                                # Cannot replace if agent is responding to incident
                                if time_tables_avail[agent][end + k] == 0:
                                    infeasible = True
                                    continue
                                temp_time_tables[agent][end + k] = -1
                            for l in range(replace_slot - j):
                                # Cannot replace if agent is responding to incident
                                if time_tables_avail[agent][end + j + l] == 0:
                                    infeasible = True
                                    continue
                                temp_time_tables[agent][end + j + l] = neighbour.get_id()

                            if not infeasible:
                                temp_schedule = deepcopy(schedule)
                                temp_schedule.update_time_tables(temp_time_tables)
                                candidate_solns.append(temp_schedule)
                                # print(temp_time_tables[agent])
        # For excess case, replace the next location to be within the acceptable distance
        elif value < 0:

            abs_value = abs(value)
            time_gap = end - start - 1

            for split_idx in range(time_gap - abs_value):

                temp_time_tables = deepcopy(time_tables)
                time_gap_ext = time_gap

                if split_idx > 0:

                    for i in range(1, split_idx + 1):
                        temp_time_tables[agent][start + i] = origin

                    time_gap_ext = time_gap - split_idx

                for neighbour in proximity_table[origin][time_gap_ext]:
                    infeasible = False

                    temp_time_tables_1 = deepcopy(temp_time_tables)
                    # neighbour = station_ids[orig_idx - time_gap]
                    for i in range(replace_slot):
                        # Cannot replace if agent is responding to incident
                        if time_tables_avail[agent][end + i] == 0:
                            infeasible = True
                            continue
                        temp_time_tables_1[agent][end + i] = neighbour.get_id()

                    if not infeasible:
                        temp_schedule = deepcopy(schedule)
                        temp_schedule.update_time_tables(temp_time_tables_1)
                        candidate_solns.append(temp_schedule)
                        # print(temp_time_tables[agent])

        return candidate_solns

    def is_tabu(self, schedule):

        # Only check against tabu list if schedule is not None
        if not isinstance(schedule, Schedule):
            return False

        for tabu_item in self.tabu_list:
            if schedule.get_time_tables() == tabu_item.get_time_tables():
                return True

        return False

    def get_score(self, schedules_dict):

        # Compute reward, the idea is to select schedule that has minimal decrease in obj value
        # score = get_objective_value(schedule, self.sector) - self.prev_score
        score = get_objective_value_MA(schedules_dict, self.sectors) - self.prev_score
        # Add value function
        if isinstance(self.learning_agent, VFAAgent) or isinstance(self.learning_agent, MAVFAAgent):
            if self.is_multi_agent:
                global_state = [self.t_k / len(T)] + \
                               get_patrol_presence_status_MA(get_patrol_count_table_MA(schedules_dict), self.Q_j)
                score += GAMMA * self.learning_agent.return_value(get_post_joint_state(schedules_dict, self.Q_j,
                                                                                       self.all_patrol_areas,
                                                                                       self.t_k,
                                                                                       self.subagent_dim), global_state,
                                                                  self.mask)
            else:
                score += GAMMA * self.learning_agent.return_value(get_post_state(
                    schedules_dict[next(iter(schedules_dict))], self.sectors[next(iter(schedules_dict))],
                    self.t_k))

        # For parallel runs
        elif isinstance(self.learning_agent, tuple) and self.learning_agent[0]:
            input_parameters = self.learning_agent[0]
            trained_parameters = self.learning_agent[1]
            n_agents = input_parameters["n_agents"]
            state_size = input_parameters["state_size"]
            area_size = input_parameters["area_size"]
            subagent_dim = input_parameters["subagent_dim"]
            encoding_size = input_parameters["encoding_size"]
            sector_ids = input_parameters["sector_ids"]

            if self.is_multi_agent:
                global_state = [self.t_k / len(T)] + \
                               get_patrol_presence_status_MA(get_patrol_count_table_MA(schedules_dict), self.Q_j)

                learning_agent = MAVFAAgent("False", sector_ids, n_agents, state_size, ACTION_SIZE, SEED, area_size,
                                            subagent_dim, encoding_size, HIDDEN_DIM, ATTN_OUT_DIM, trained_parameters=trained_parameters)

                score += GAMMA * learning_agent.return_value(get_post_joint_state(schedules_dict, self.Q_j,
                                                                                       self.all_patrol_areas,
                                                                                       self.t_k,
                                                                                       self.subagent_dim), global_state,
                                                                  self.mask)
            else:
                learning_agent = VFAAgent(state_size, ACTION_SIZE, SEED, "False", sector_ids, area_size, subagent_dim,
                                          encoding_size)
                score += GAMMA * learning_agent.return_value(get_post_state(
                    schedules_dict[next(iter(schedules_dict))], self.sectors[next(iter(schedules_dict))],
                    self.t_k))
            # To Free up memory
            del learning_agent
            gc.collect()

        return score

    def is_acceptable(self, schedules_dict):
        """
        Check if a schedule is acceptable i.e only contain defect type 2
        :param schedule:
        :param sector:
        :return:
        """
        # defects = check_defects(schedule, sector)

        defects = check_defects_MA(schedules_dict, self.sector_id, self.sectors, self.D, self.Q_j, self.t_k)

        if len(defects) > 0:
            for defect in defects:
                if defect.get_type() != 2:
                    return False

        return True
