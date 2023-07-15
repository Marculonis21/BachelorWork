#!/usr/bin/env python

from experiment_setter import Experiments

import roboEvo
import lzma

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--open"             , default=False , const=True          , nargs='?'                               , type=str , help="Open saved individual")
parser.add_argument("--experiment"       , default=[]    ,                       nargs='*'                               , type=str , help="Enter name of 1 or more experiments to run")
parser.add_argument("--experiment_names" , default=False , action="store_true" ,                                                      help="Show all created experiment names")
parser.add_argument("--batch"            , default=False , const=True          , nargs='?'                               , type=int , help="Number of iterations of the whole evolutionary algorithm")
parser.add_argument("--debug"            , default=False , action="store_true" ,                                                      help="Run env in debug mode")
parser.add_argument("--no_graph"         , default=False , action="store_true" ,                                                      help="Hide graph of the algorithm showing fitness values in generations")

def __experiments_valid(args, experiments):
    """Function for validating experiment names.

    This function checks the validity of selected experiment names before
    running them.

    Raises: 
        AssertionError : ``Unknown experiment name `<name>``` followed by list of valid experiment names. 
            Error on invalid experiment name. 
    """

    for exp in args.experiment:
        try: 
            experiments.get_experiment(exp)
        except AssertionError as e:
            print(e)
            sys.exit()

def main(args):
    """Main TUI function

    Function that receives all input arguments and performs operations based on
    input arguments.
    """
    experiments = Experiments()

    # Run selected saved individual
    if args.open: 
        try:
            agent, robot, individual = roboEvo.gaAgents.BaseAgent.load(args.open)
            run_reward = roboEvo.render_run(agent, robot, individual)
            print("Run reward: ", run_reward)
        except FileNotFoundError as e:
            print("Problem while loading save file - File not found")
            print(e)
        except lzma.LZMAError as e:
            print("Problem while loading save file - Incorrect file")
            print("TUI only supports visualizing of best individuals (.save files)")
            print("To visualize other individual than the best one, use visualization in GUI")
        sys.exit()

    # Show names of all created experiments
    if args.experiment_names: 
        print("List of created experiments: ")
        for name in experiments.get_experiment_names():
            print(" -", name)
        sys.exit()

    # Start experiment in batch mode
    if args.batch:
        # test if experiment was set
        if not args.experiment:
            print("Error - At least one --experiment must be selected for batch run")
            sys.exit()

        __experiments_valid(args, experiments)

        for exp in args.experiment:
            print(f"Starting experiment - {exp}")
            for run in range(args.batch):
                print(f"Starting batch run - {run+1}/{args.batch}")

                params = experiments.get_experiment(exp)

                if params.note == "":
                    params.note = f"run{run+1}" 
                else:
                    params.note = f"run{run+1}_{params.note}" 

                params.show_graph = not args.no_graph
                roboEvo.run_experiment(params)

    # Start single run of experiment
    else:
        if args.experiment: # starting experiment by experiment name
            __experiments_valid(args, experiments)

            for exp in args.experiment:
                print(f"Starting experiment - {exp}")
                params = experiments.get_experiment(exp)
                params.show_graph = not args.no_graph
                roboEvo.run_experiment(params, args.debug)

        else: # starting experiment from code
            robot = roboEvo.robots.StickAnt()
            agent = roboEvo.gaAgents.TFSAgent(robot, [(1,2),False,False,(2,3)])

            params = roboEvo.ExperimentParams(robot=robot,
                                              agent=agent,
                                              note="")

            params.show_graph = not args.no_graph
            roboEvo.run_experiment(params, args.debug)

    print("Exiting ...")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
