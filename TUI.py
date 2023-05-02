#!/usr/bin/env python

from experiment_setter import Experiments

import roboEvo

import argparse
import sys
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--open"             , default=False , const=True          , nargs='?'                               , type=str , help="Open saved individual")
parser.add_argument("--experiment"       , default=False ,                       nargs='?'                               , type=str , help="Enter name of 1 or more experiments to run")
parser.add_argument("--experiment_names" , default=False , action="store_true" ,                                                      help="Show all created experiment names")
parser.add_argument("--batch"            , default=False , const=True          , nargs='?'                               , type=int , help="Number of iterations of the whole evolutionary algorithm")
parser.add_argument("--debug"            , default=False , action="store_true" ,                                                      help="Run env in debug mode")
parser.add_argument("--no_graph"         , default=False , action="store_true" ,                                                      help="Hide graph of the algorithm showing fitness values in generations")

def main(args):
    experiments = Experiments()

    # Run selected saved individual
    if args.open: 
        try:
            agent, robot, individual = roboEvo.gaAgents.BaseAgent.load(args.open)
            run_reward = roboEvo.render_run(agent, robot, individual)
            print("Run reward: ", run_reward)
        except Exception as e:
            print("Problem occured while loading save file\n")
            print(e)
        sys.exit()

    # Show names of all created experiments
    if args.experiment_names: 
        print("List of created experiments: ")
        for name in experiments.get_experiment_names():
            print(" -", name)
        sys.exit()

    # Start experiment in batch mode
    if args.batch:
        for i in range(args.batch):
            print(f"STARTING BATCH RUN - {i+1}/{args.batch}")

            if args.experiment:
                print(f"Starting experiment {args.experiment}")
                params = experiments.get_experiment(args.experiment)
            else:
                params = experiments.exp11_TFS_spotlike()

            params.note = f"run{i+1}_{params.note}" 
            params.show_graph = not args.no_graph
            roboEvo.run_experiment(params)

    # Start single experiment
    else:
        if args.experiment:
            params = experiments.get_experiment(args.experiment)
        else:
            params = experiments.exp12_SineFull_spotlike()
            params.note = "motors_test_" + params.note

        params.show_graph = not args.no_graph
        roboEvo.run_experiment(params, args.debug)
    print("Exiting ...")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
