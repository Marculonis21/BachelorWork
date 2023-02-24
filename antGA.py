#!/usr/bin/env python

import argparse
import gaAgent
from robots.robots import *

import gym
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time
import tempfile
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--open", default=False, const=True, nargs='?', type=str, help="Open saved individual")
parser.add_argument("--debug", default=False, action="store_true", help="Run env in debug mode")

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
    done = False
    finishline_bonus_reached = False

    body_heights = []
    facing_directions = []

    _ = env.reset()
    while not done:
        steps += 1
        action = agent.get_action(actions, steps)

        _, _, done, info = env.step(action)
        body_heights.append(info["y_position"])
        facing_directions.append(info["facing_direction"])

        if render:
            env.render(start_paused=True)
        if info["x_position"] > 39.5 and finishline_bonus_reached: # if able to get to the end of the map 
            finishline_bonus_reached = True
            individual_reward += (env._max_episode_steps - steps)

        if done:
            # sim end

            # individual_reward = 0
            # individual_reward -= np.var(body_heights)*10

            desired_directions = np.array([np.array([1,0]) for _ in range(steps+1)])
            facing_directions =  np.array(facing_directions)
             
            dir_diff = np.linalg.norm(desired_directions - facing_directions, axis=1)
            max_diff = np.max(dir_diff)
            # individual_reward -= 10*max_diff

            # individual_reward = (info["x_position"]-0.5*abs(info["y_position"]))*(2-max_diff) # = x distance from start

            # distance in reward=(desired_dir - 0.5*distance_offaxis) - 10*body_height_variance
            individual_reward = (info["x_position"]-0.5*abs(info["y_position"])) - np.var(body_heights)*10  

    # REWARD + INFO
    # return individual_reward, (body_heights, facing_directions)
    return individual_reward

def evolution(robot, agent, client, generation_count, population_size, debug=False):
    global GRAPH_VALUES, EPISODE_HISTORY, GUI_GEN_NUMBER, GUI_FITNESS, GUI_PREVIEW

    population = agent.generate_population(population_size)
    robot_source_files = []
        
    environments = []

    # create individual source files for different bodies
    for i in range(len(population)):
        file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")

        robot.create(file, agent.body_part_mask, population[i][1])

        robot_source_files.append(file)

        env = gym.make('CustomAntLike-v1',
                       xml_file=file.name,
                       reset_noise_scale=0.0)
        env._max_episode_steps = 500

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
        else: # ITERATION FORCED PREVIEW
            if generations != 0 and generations % 50 == 0:
                print("Generation: ", generations)
                simulationRun(best_individual[1], agent, best_individual[0], render=True)

        # Get fitness values
        fitness_values = []
        futures = []
        if debug:
            for index, individual in enumerate(population):
                fitness_values.append(simulationRun(environments[index if agent.use_body_parts else 0], agent, individual, generations % 25 == 0))
        else:
            for index, individual in enumerate(population):
                futures.append(client.submit(simulationRun, environments[index if agent.use_body_parts else 0], agent, individual))

            fitness_values = client.gather(futures)

        if GUI_FLAG:
            GUI_FITNESS = [np.mean(fitness_values), 
                           min(fitness_values),
                           max(fitness_values)]

        if generations % 5 == 0: # frequency of info update
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

        # copy needed
        best_index = np.argmax(fitness_values)
        best_individual = (copy.deepcopy(population[best_index]), environments[best_index if agent.use_body_parts else 0])

        # selection, crossover, mutation
        parents          = agent.selection(population,fitness_values)
        children         = agent.crossover(parents)
        mutated_children = agent.mutation(children)
        population = mutated_children

        # elitism
        population[0] = best_individual[0] # b_i[0] = best individual gen values, b_i[1] = best individual env

        # change environments based on possible robot body changes
        for i in range(len(population)):
            environments[i].close()
            robot.create(robot_source_files[i], agent.body_part_mask, population[i][1])

            env = gym.make('CustomAntLike-v1',
                           xml_file=robot_source_files[i].name,
                           reset_noise_scale=0.0)
            env._max_episode_steps = 500
            environments[i] = env

            if not agent.use_body_parts:
                break

    # close tmp files
    for file in robot_source_files:
        file.close()

    return best_individual

###################################################
###################################################
class RunParams:
    def __init__(self, 
                 robot:BaseRobot, 
                 agent:gaAgent.AgentType,
                 ga_population_size=500,
                 ga_generation_count=100,
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

    # RUN multithreaded
    client = Client(n_workers=12,threads_per_worker=1,scheduler_port=0, silence_logs=logging.ERROR)

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

    if params.save_best:
        current_time = time.time()
        gaAgent.AgentType.save(agent, robot, best_individual_actions, params.save_dir + f"/{params.note}_individual_run{current_time}_rew{best_reward}.save")
        # graph_data = np.array(GRAPH_VALUES)
        # np.save(params.save_dir + f"/{params.note}_graph_data{current_time}", graph_data)
        episode_history = np.array(EPISODE_HISTORY)
        np.save(params.save_dir + f"/{params.note}_episode_history{current_time}", episode_history)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Run selected saved individual
    if args.open:
        agent, individual = (None, None)
        file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")
        try:
            agent, robot, individual = gaAgent.AgentType.load(args.open)

            robot.create(file, agent.body_part_mask, individual[1])

            env = gym.make('CustomAntLike-v1',
                           xml_file=file.name,
                           reset_noise_scale=0.0)
            env._max_episode_steps = 500
            reward = simulationRun(env, agent, individual, render=True)
            env.close()

            print("Run reward: ", reward)

        except Exception as e:
            print("Problem occured while loading save file\n")

        finally:
            file.close()

        quit()

    robot = SpotLike()
    # agent = gaAgent.FullRandomAgent(robot, [False for _ in range(len(robot.body_parts))], 40)
    # agent = gaAgent.SineFuncFullAgent(robot, [False for _ in range(len(robot.body_parts))])
    agent = gaAgent.TFSAgent(robot, [False for _ in range(len(robot.body_parts))])
    Run(False, RunParams(robot, 
                         agent, 
                         ga_population_size=200,
                         ga_generation_count=200, 
                         show_best=True, 
                         save_best=True,
                         save_dir="./saves/individuals",
                         note="TFS_p3_s3_cr4"))

    print("DONE")
