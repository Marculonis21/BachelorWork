#!/usr/bin/env python

import gaAgent
from robots.robots import *

import gym
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time
from mujoco_py import GlfwContext
import tempfile

import sys

GUIFLAG = False
GUI_GRAPH_VALUES = [[],[],[]]
GUI_FITNESS = []
GUI_GEN_NUMBER = 0

GUI_PREVIEW = False
GUI_ABORT = False 

def simulationRun(env, agent, actions, render=False, render_start_paused=False):
    steps = -1
    individual_reward = 0
    done = False
    finishline_bonus_reached = False

    observation = env.reset()
    while not done:
        steps += 1
        action = agent.get_action(actions, steps)

        observation, reward, done, info = env.step(action)

        if render:
            env.render(start_paused=render_start_paused)

        if info["x_position"] > 39.5 and finishline_bonus_reached: # if able to get to the end of the map 
            finishline_bonus_reached = True
            individual_reward += (env._max_episode_steps - steps)

        if done:
            # sim end
            individual_reward = info["x_position"] # = x distance from start

    return individual_reward

def evolution(robot, agent, client, generation_count, population_size, debug=False):
    population = agent.generate_population(population_size)

    robot_source_files = []

    environments = []

    for i in range(len(population)):
        file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")

        if agent.use_body_parts:
            robot.create(file, agent.body_part_mask, population[i][1])
        else:
            robot.create(file, agent.body_part_mask)

        robot_source_files.append(file)

        env = gym.make('CustomAntLike-v1',
                       xml_file=file.name,
                       reset_noise_scale=0.0)
        env._max_episode_steps = 500

        environments.append(env)

    max_fitnesses = []
    mean_fitnesses = []
    min_fitnesses = []

    global GUI_GRAPH_VALUES, GUI_GEN_NUMBER, GUI_FITNESS, GUI_PREVIEW
    if GUIFLAG:
        GUI_GRAPH_VALUES = [[],[],[]]

    best_individual = tuple()
    for generations in range(generation_count+1):
        if GUI_ABORT:
            break

        GUI_GEN_NUMBER = generations

        if GUI_PREVIEW:
            # simulationRun(0, env, agentType, best_individual, render=True, render_start_paused=True);
            # DO PREVIEWY STUFF
            GUI_PREVIEW = False
        else:
            if generations % 25 == 0:
                print("Generation: ", generations)

        # Get fitness values
        fitness_values = []
        futures = []
        if debug:
            for index, individual in enumerate(population):
                fitness_values.append(simulationRun(environments[index], agent, individual, generations % 25 == 0, True))
        else:
            for index, individual in enumerate(population):
                futures.append(client.submit(simulationRun, environments[index], agent, individual))

            fitness_values = client.gather(futures)

        if GUIFLAG:
            GUI_FITNESS = [np.mean(fitness_values), 
                           min(fitness_values),
                           max(fitness_values)]

        if generations % 5 == 0: # frequency of GUI graph update
            if GUIFLAG:
                GUI_GRAPH_VALUES[0].append(np.mean(fitness_values))
                GUI_GRAPH_VALUES[1].append(min(fitness_values))
                GUI_GRAPH_VALUES[2].append(max(fitness_values))
            else:
                print("Best fitness: ", max(fitness_values))

                mean_fitnesses.append(np.mean(fitness_values))
                min_fitnesses.append(min(fitness_values))
                max_fitnesses.append(max(fitness_values))

                plt.cla()
                plt.title('Training')
                plt.xlabel('Episode')
                plt.ylabel('Fitness')
                plt.plot(mean_fitnesses, label='Mean')
                plt.plot(min_fitnesses, label='Min')
                plt.plot(max_fitnesses, label='Max')
                plt.legend(loc='upper left', fontsize=9)
                plt.tight_layout()
                plt.pause(0.1)

        # need copy so it doesn't get changed by other genetic ops
        best_individual = (copy.deepcopy(population[np.argmax(fitness_values)]), environments[np.argmax(fitness_values)])

        # selection, crossover, mutation
        parents = agent.selection(population,fitness_values)
        children = agent.crossover(parents)
        mutated_children = agent.mutation(children)
        population = mutated_children

        # elitism
        population[0] = best_individual[0] # b_i[0] = best individual gen values, b_i[1] = best individual env

        # change environments based on possible robot body changes
        for i in range(len(population)):
            if agent.use_body_parts:
                robot.create(robot_source_files[i], agent.body_part_mask, population[i][1])
            else:
                robot.create(robot_source_files[i], agent.body_part_mask)

            env = gym.make('CustomAntLike-v1',
                           xml_file=robot_source_files[i].name,
                           reset_noise_scale=0.0)
            env._max_episode_steps = 500
            environments[i] = env

    # close tmp files
    for file in robot_source_files:
        file.close()

    return best_individual

###################################################
###################################################
def printHelp():
    print("-h --help       ... Print help")
    print("-o <individual> ... Select input file to play")

def RunFromGui(robot : BaseRobot, agent : gaAgent.AgentType, population_size=50, generation_count=250, show_best=False, save_best=False, save_dir='./saves/individuals'):
    global GUIFLAG
    GUIFLAG = True

    client = Client(n_workers=10,threads_per_worker=1,scheduler_port=0)

    try:
        print("RUNNING")
        best_individual_actions, best_individual_env = evolution(robot, agent, client, generation_count=generation_count, population_size=population_size, debug=False)
        print("DONE")
    finally:
        client.close()

    if GUI_ABORT:
        return

    best_reward = simulationRun(best_individual_env, agent, best_individual_actions, show_best, render_start_paused=True)

    if save_best:
        current_time = time.time()
        agent.save(best_individual_actions, save_dir + f"/individual_run{current_time}_rew{best_reward}")
        graph_data = np.array(GUI_GRAPH_VALUES)
        np.save(save_dir + f"/graph_data{current_time}", graph_data)

def RaisePreview():
    global GUI_PREVIEW
    GUI_PREVIEW = True

def RaiseAbort():
    global GUI_ABORT
    GUI_ABORT = True

if __name__ == "__main__":

    if ("-h" in sys.argv or "--help" in sys.argv):
        printHelp()
        quit()

    # if ("-o" in sys.argv):
    #     agent, individual = (None, None)
    #     try:
    #         # agent = gaAgent.SineFuncFullAgent(4)
    #         # agent = gaAgent.FullRandomAgent(500, 8)
    #         # agent = gaAgent.SineFuncHalfAgent(4)
    #         individual = np.load(sys.argv[sys.argv.index("-o") + 1])
    #     except Exception as e:
    #         print("Problem occured - loading file\n")
    #         printHelp()

    #     reward = simulationRun(0, envs, agent, individual, render=True, render_start_paused=True)
    #     print("Run reward: ", reward)
    #     quit()

    robot = AntV3()
    agent = gaAgent.SineFuncFullAgent(robot, [False for _ in range(len(robot.body_parts))])
    RunFromGui(robot, agent)

    print("DONE")
