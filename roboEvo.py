#!/usr/bin/env python

# creating and registering custom environment
import resources.gymnasiumCustomEnv as _

import resources.gaAgents as gaAgents
import resources.robots.robots as robots

import argparse

# import gym
import gymnasium as gym
from gymnasium.wrappers import time_limit
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time
import tempfile
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument("--open", default=False, const=True, nargs='?', type=str, help="Open saved individual")
parser.add_argument("--debug", default=False, action="store_true", help="Run env in debug mode")
parser.add_argument("--batch", default=False, const=True, nargs='?', type=int, help="Number of iterations in batch")
parser.add_argument("--batch_note", default=False, const=True, nargs='?', type=str, help="Batch run note")

GUI_FLAG = False
GUI_FITNESS = []
GUI_GEN_NUMBER = 0

GUI_PREVIEW = False
GUI_ABORT = False 

GRAPH_VALUES = [[],[],[]]
EPISODE_HISTORY = []

def RaisePreview():
    global GUI_PREVIEW
    GUI_PREVIEW = True

def RaiseAbort():
    global GUI_ABORT
    GUI_ABORT = True

def simulationRun(env, agent, actions, render=False):
    steps = -1
    individual_reward = 0
    terminated = False
    truncated = False

    env.reset()
    while True:
        steps += 1
        action = agent.get_action(actions, steps).squeeze()
        
        # obs, reward, terminated, truncated, info
        _, _, terminated, truncated, info = env.step(action)

        if render:
            env.render()

        if truncated:
            print("$$$$$$$$$$ truncated")

        if terminated or truncated: # calculate reward after finishing last step
            individual_reward = (info["x_position"]-0.5*abs(info["y_position"]))
            break

    # Optionally - return REWARD + some colected information ??? 
    return individual_reward

def evolution(robot, agent, client, generation_count, population_size, debug=False):
    global GRAPH_VALUES, EPISODE_HISTORY, GUI_GEN_NUMBER, GUI_FITNESS, GUI_PREVIEW
    GUI_FITNESS = []
    GRAPH_VALUES = [[],[],[]]
    EPISODE_HISTORY = []

    population = agent.generate_population(population_size)
    robot_source_files = []
        
    environments = []

    # create individual source files for different bodies
    for i in range(len(population)):
        file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")

        robot.create(file, agent.body_part_mask, population[i][1])

        robot_source_files.append(file)

        env = gym.make('custom/CustomEnv-v0',
                       xml_file=file.name,
                       reset_noise_scale=0.0,
                       disable_env_checker=True,
                       render_mode=None,
                       )
        env = time_limit.TimeLimit(env, max_episode_steps=500)

        environments.append(env)
        if not agent.use_body_parts:
            break

    best_individual = tuple()
    for generations in range(generation_count+1):
        if GUI_ABORT:
            break

        GUI_GEN_NUMBER = generations

        if GUI_PREVIEW: # GUI FORCED PREVIEW
            simulationRun(best_individual[1], agent, best_individual[0], render=True)
            GUI_PREVIEW = False
        # else: # ITERATION FORCED PREVIEW
        #     if generations != 0 and generations % 50 == 0:
        #         print("Generation: ", generations)
        #         simulationRun(best_individual[1], agent, best_individual[0], render=True)

        print("FUTURES STARTING")
        # Get fitness values
        fitness_values = []
        futures = []
        if debug:
            for index, individual in enumerate(population):
                fitness_values.append(simulationRun(environments[index if agent.use_body_parts else 0], agent, individual, generations % 25 == 0))
        else:
            for index, individual in enumerate(population):
                futures.append(client.submit(simulationRun, environments[index if agent.use_body_parts else 0], agent, individual))

            print("FUTURES GATHERING")
            fitness_values = client.gather(futures)

        if GUI_FLAG:
            GUI_FITNESS = [np.mean(fitness_values), 
                           min(fitness_values),
                           max(fitness_values)]

        if generations % 5 == 0: # frequency of info update
            print("GRAPH")
            GRAPH_VALUES[0].append(np.mean(fitness_values))
            GRAPH_VALUES[1].append(min(fitness_values))
            GRAPH_VALUES[2].append(max(fitness_values))
            # for later statistical reconstruction
            EPISODE_HISTORY.append(fitness_values)

            if not GUI_FLAG:
                print("Best fitness: ", max(fitness_values))

                plt.cla()
                plt.title('Training')
                plt.xlabel('5 Generations')
                plt.ylabel('Fitness')
                plt.plot(GRAPH_VALUES[0], label='Mean')
                plt.plot(GRAPH_VALUES[1], label='Min')
                plt.plot(GRAPH_VALUES[2], label='Max')
                plt.legend(loc='upper left', fontsize=9)
                plt.tight_layout()
                plt.pause(0.1)

        # get index of the best individuals for elitism
        # elitism_count = int(population_size*0.05) # 5% of pop size

        # fitness_values = np.array(fitness_values)
        # top_indices = np.argpartition(fitness_values, -elitism_count)
        # sorted_top_indices = top_indices[np.argsort(fitness_values[top_indices])[::-1]]

        # elite_individuals = []
        # for top_i in sorted_top_indices:
        #     elite_individuals.append(copy.deepcopy((population[top_i], environments[top_i if agent.use_body_parts else 0])))
        # best_individual = elite_individuals[0]
        # best_individual = (copy.deepcopy(population[best_index]), environments[best_index if agent.use_body_parts else 0])

        print("BEST INDEX")
        best_index = np.argmax(fitness_values)
        best_individual = (copy.deepcopy(population[best_index]), environments[best_index if agent.use_body_parts else 0])

        print("OPS")
        # Genetic operators - selection, crossover, mutation
        parents          = agent.selection(population,fitness_values)
        children         = agent.crossover(parents)
        mutated_children = agent.mutation(children)
        population = mutated_children

        print("ELITE")
        # apply elitism
        # for i, elite in enumerate(elite_individuals):
        #     population[i] = elite[0] # elite:(genetic info, env)
        population[0] = best_individual[0]

        print(f"finished gen {generations}")
        # change environments based on possible robot morphology changes
        for i in range(len(population)):
            environments[i].close()
            robot.create(robot_source_files[i], agent.body_part_mask, population[i][1])

            if generations != generation_count:
                env = gym.make('custom/CustomEnv-v0',
                            xml_file=robot_source_files[i].name,
                            reset_noise_scale=0.0,
                            disable_env_checker=True,
                            render_mode=None,
                            )
                env = time_limit.TimeLimit(env, max_episode_steps=500)
                environments[i] = env

            if not agent.use_body_parts:
                break

        print("ENV TRANSFER DONE")

    # after evo close tmp files
    for file in robot_source_files:
        file.close()

    if not GUI_FLAG:
        plt.close()

    return best_individual

################################################################################
################################################################################

class RunParams:
    def __init__(self, 
                 robot:robots.BaseRobot, 
                 agent:gaAgents.AgentType,
                 ga_population_size=100,
                 ga_generation_count=150,
                 show_best=False,
                 save_best=False,
                 save_dir='./saves/individuals',
                 note=""):

        self.robot               = robot
        self.agent               = agent
        self.ga_population_size  = ga_population_size
        self.ga_generation_count = ga_generation_count
        self.show_best           = show_best
        self.save_best           = save_best
        self.save_dir            = save_dir
        self.note                = note


def Run(gui, params:RunParams, args=None):
    global GUI_FLAG
    GUI_FLAG = gui

    # run multithreaded
    # client = Client(n_workers=11,threads_per_worker=1,scheduler_port=0, silence_logs=logging.ERROR)
    client = Client(n_workers=11,threads_per_worker=1,scheduler_port=0)

    try:
        print("RUNNING")
        best_individual_actions, best_individual_env = evolution(params.robot, 
                                                                 params.agent,
                                                                 client,
                                                                 generation_count=params.ga_generation_count, 
                                                                 population_size=params.ga_population_size, 
                                                                 debug=args.debug if args != None else False)
        print("DONE")
    finally:
        client.close()

    best_reward = simulationRun(best_individual_env, agent, best_individual_actions, params.show_best)

    current_time = time.time()
    if params.save_best:
        gaAgents.AgentType.save(agent, robot, best_individual_actions, params.save_dir + f"/{params.note}_individual_run{current_time}_rew{best_reward}.save")

    episode_history = np.array(EPISODE_HISTORY)
    np.save(params.save_dir + f"/{params.note}_episode_history{current_time}", episode_history)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Run selected saved individual
    if args.open:
        agent, individual = (None, None)
        file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")
        try:
            agent, robot, individual = gaAgents.AgentType.load(args.open)

            robot.create(file, agent.body_part_mask, individual[1])

            env = gym.make('custom/CustomEnv-v0',
                           xml_file=file.name,
                           reset_noise_scale=0.0,
                           disable_env_checker=True
                           )
            env = time_limit.TimeLimit(env, max_episode_steps=500)

            reward = simulationRun(env, agent, individual, render=True)
            print("Run reward: ", reward)
            print()
            # env.close() # TODO: NO NEED TO CLOSE?

        except Exception as e:
            print("Problem occured while loading save file\n")
            print(e)

        finally:
            file.close()
        
        quit()

    if args.batch: # BATCH RUN
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts))
        # agent = gaAgents.SineFuncFullAgent(robot, [False for _ in range(len(robot.body_parts))])

        args.batch_note = "TFS(p4_s3_cr1)"

        batch_dir = "./saves/batch_runs/" + f"run_{type(robot).__name__}_{type(agent).__name__}_{args.batch_note}_{time.time()}/"
        os.makedirs(batch_dir)
        params = RunParams(robot, 
                           agent, 
                           ga_population_size=150,
                           ga_generation_count=200, 
                           show_best=False, 
                           save_best=True,
                           save_dir=batch_dir,
                           note="")

        for i in range(args.batch):
            print(f"STARTING BATCH RUN - {i+1}/{args.batch}")
            params.note = f"run{i+1}"
            Run(False, params)

    else: # SINGLE RUN
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts))
        Run(False, RunParams(robot, 
                             agent, 
                             ga_population_size=150,
                             ga_generation_count=10, 
                             show_best=False, 
                             save_best=True,
                             save_dir="./saves/individuals",
                             note="TFS(4_3_1)"), args)

    print("DONE")
