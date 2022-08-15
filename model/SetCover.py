import re
import sys

from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.util.environment import get_environment

from constants.Settings import TIME_UNIT, T
from entity.Schedule import *
from util.utils import round_to_nearest
from util.ScheduleUtil import convert_to_time_intervals


class SetCover(object):

    def __init__(self, sector=None, scenarios=None, schedule=None):
        self.sector = sector
        self.scenarios = scenarios
        self.schedule = schedule

    def build_model(self, **kwargs):

        model_name = "Initial Patrol SetCover Scheduler for Sector %s" % self.sector.get_id()
        mdl = Model(model_name, **kwargs)
        mdl.agents = self.sector.get_agents()

        # Parameters
        J = [j.get_id() for j in self.sector.get_all_patrol_areas()] # List of patrol areas
        D = self.sector.get_distance_matrix()

        # A dictionary with patrol area ID as key and a list of patrol area IDs that are within the response time
        N_j = self.sector.get_neighbours_table()
        # A dictionary with patrol area ID as key and demand as value
        Q_j = self.sector.get_presence_table()

        # Print Parameters
        # print("Agents: {}".format(mdl.agents))
        # print("Patrol areas by agents: {}".format(L_i))
        # print("Demands by patrol areas: {}".format(Q_l))
        # print()

        S = list(self.scenarios.keys())  # List of scenarios
        R = {}

        for s in S:
            R[s] = list(range(len(self.scenarios[s])))  # List of incidents at each scenario

        big_m = 1e7

        # Decision variables

        # Binary var if agent a is at location l at time t
        y = mdl.binary_var_cube(mdl.agents, J, T, "y_i_j_t")
        z = []

        # "Binary var if the response time for request r in scenario s is met"
        for s in S:
            z.append(mdl.binary_var_matrix([s], R[s], "z_s_r"))

        # Objective Function
        # Maximise the success rate + patrolling presence
        mdl.maximize(mdl.sum(mdl.sum(z[s][s, r] for r in R[s]) for s in S) / sum([len(R[s]) for s in S]) +
                     mdl.sum(mdl.sum(y[i, j, t] for j in J for t in T) for i in mdl.agents)
                     / len(T) / len(mdl.agents))

        # Constraints
        for i in mdl.agents:
            for t in T:
                # Each agent at most can only be at one location at one time or on the way to somewhere
                mdl.add_constraint(mdl.sum(y[i, j, t] for j in J) <= 1,
                                   "Agent can at most be at one location or unavailable")

        for j in J:
            # Each patrol areas must be visited for at least a minimum number of period
            mdl.add_constraint(mdl.sum(y[i, j, t] for i in mdl.agents for t in T) >=
                               round_to_nearest(Q_j[j], TIME_UNIT) / TIME_UNIT,
                               "Each patrol area must be patrolled for a period depending on "
                               "the size of its road network")

        # Patrol consecutiveness
        for i in mdl.agents:
            for j1 in J:
                for j2 in J:
                    for t1 in T[:len(T) - 1]:
                        for t2 in T[1:]:
                            if t1 < t2:
                                mdl.add_constraint(t1 + D[j1][j2] + TIME_UNIT - t2 <=
                                                   big_m * (2 - y[i, j1, t1] - y[i, j2, t2]))

        for s in S:
            for r in R[s]:
                mdl.add_constraint(
                    mdl.sum(y[i, n_j.get_id(), self.scenarios[s][r].get_start_time()]
                            for i in mdl.agents
                            for n_j in N_j[self.scenarios[s][r].get_location().get_id()]) >= z[s][s, r],
                    "A request is covered if there is at least one agent within the response time")

        return mdl

    def optimize(self, show=False):
        """

        :return: True (if feasible schedule exists) or False
        """
        print("Building model...")
        model = self.build_model()
        model.parameters.timelimit.set(60)
        # model.parameters.mip.tolerances.mipgap = 0.05
        if show:
            model.print_information()
        # Export the model to .lp file
        # model.export_as_lp("patrol_" + str(self.sector.get_id()) + ".lp")

        print("Solving...")

        sol = model.solve()

        if sol:
            print("Problem solved")
            return self.output_schedule(sol)
        else:
            print("Problem could not be solved: " + str(model.solve_details.status_code))
            return None

    def output_schedule(self, mp_sol):
        """

        :param mp_sol - solution from CPLEX
        :return: Always true since feasible Schedule exists
        """

        time_tables = {}

        # Extract the routes from the output json file
        for variable in mp_sol.as_dict().keys():

            # var_name = variable['name']
            y = re.search('y_i_j_t_kml_(.+)', str(variable))

            if y:
                agent, loc, time = y.group(1).split('_')
                agent = 'kml_' + agent
                if agent not in time_tables.keys():
                    time_tables[agent] = [-1] * len(T)

                time_tables[agent][int(int(time) / TIME_UNIT)] = int(loc)

        time_intervals = {}

        for agent in time_tables.keys():
            # Convert the time table into a time interval format
            output = convert_to_time_intervals(time_tables[agent])
            time_intervals[agent] = output

        # patrol_presence = compute_patrol_presence(time_tables)
        objective_value = mp_sol.get_objective_value()

        self.schedule = Schedule(time_tables, time_intervals, objective_value)

        return True

    def get_schedule(self):
        return self.schedule

