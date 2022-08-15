import argparse
import datetime
import os
import pickle
import sys

from constants.Settings import *
from data.ScenarioGenerator import *
from model.SetCover import *
from model.DispatchRules import *


def generate_sample_initial_schedule(sector, sample_no, scenario_num, poisson_mean, start_num, solver):
    folder_name = "./data/SampleInitialSchedule/" + str(poisson_mean) + f"/{solver}" + "/Sector_" + str(sector.get_id())
    file_name = "/initial_schedule_"

    # Compute the location probability distribution for incident occurences
    if not os.path.isfile("./data/location_pdf_" + sector.get_id() + ".pkl"):
        compute_location_pdf(sector)

    for i in range(sample_no):

        # Save the start time of the programme running
        start_solve_time = datetime.datetime.now()

        # To indicate whether initial schedule is successfully created
        successful = False

        while not successful:

            # Generate multiple scenarios of the initial schedule based on set cover model
            initial_scenarios = generate_scenario(sector, scenario_num, poisson_mean)
            # If no scenario generated, proceed to generate another scenario. The value of iterator i remains
            if not initial_scenarios:
                continue

            # Build and solve the set cover model to produce initial schedule
            if solver == 'cplex':
                initial_model = SetCover(sector, initial_scenarios)
            elif solver == 'dispatch_rules':
                initial_model = DispatchRules(sector, initial_scenarios)

            if initial_model.optimize():
                initial_schedule = initial_model.get_schedule()
                successful = True

                # Save the sample initial schedule
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                with open(folder_name + file_name + str(i + start_num) + ".pkl", "wb") as fp:
                    pickle.dump(initial_schedule, fp)

                # print(initial_schedule.get_time_tables())
                # sys.exit()

                print(str(i + 1) + " out of " + str(sample_no) + " samples have been generated")

        end_solve_time = datetime.datetime.now()
        solve_duration = (end_solve_time - start_solve_time).total_seconds()
        print("Total computation time for a sample: " + str(solve_duration) + 's')
        print()

    print("All sample initial schedules have been generated.")
    print()


if __name__ == "__main__":

    # Save the start time of the programme running
    start_run_time = datetime.datetime.now()

    # Parse the argument
    parser = argparse.ArgumentParser(description="Generate Initial Schedule Samples")
    parser.add_argument("--sector", default='E', type=str)
    parser.add_argument("--sample", default="100", type=int)
    parser.add_argument("--scenario", default="20", type=int)
    parser.add_argument("--poisson_mean", default=2, type=int)
    parser.add_argument("--start", default=0, type=int)  # start > 0 if there are existing samples
    parser.add_argument("--add_agent", default=0, type=int)
    parser.add_argument("--solver", default="cplex", choices=["cplex", "dispatch_rules", "sa"])

    args = parser.parse_args()

    print("Load input data..." + "\n")

    with open("./data/processed_data.pkl", "rb") as fp:
        data = pickle.load(fp)

    # data.show_summary(0)

    # Sort in ascending order of the sector ID
    sorted_sectors = sorted(data.get_sectors(), key=lambda x: x.id)

    for sector in sorted_sectors:

        # Ignore other sectors
        if sector.get_id() != args.sector:
            continue

        agents = sector.get_agents()
        if args.add_agent > 0:
            for idx in range(args.add_agent):
                new_agent_id = agents[idx] + str(idx)

                sector.add_agent(new_agent_id)

        print("Generating sample initial schedules...")
        print()
        generate_sample_initial_schedule(sector, args.sample, args.scenario, args.poisson_mean, args.start, args.solver)

    end_run_time = datetime.datetime.now()
    run_duration = (end_run_time - start_run_time).total_seconds()
    print("Total computation time: " + str(run_duration) + 's')
