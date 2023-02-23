#!/usr/bin/env python

from distributed.nanny import silence_logging
from numpy.typing import NDArray
import gaAgent
from robots.robots import *

import gym
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time
# from mujoco_py import GlfwContext
import tempfile

import sys

GUI_FLAG = False
GUI_FITNESS = []
GUI_GEN_NUMBER = 0

GUI_PREVIEW = False
GUI_ABORT = False 

GRAPH_VALUES = [[],[],[]]

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

            individual_reward = (info["x_position"]-0.5*abs(info["y_position"]))*(2-max_diff) # = x distance from start

    return individual_reward

def evolution(robot, agent, client, generation_count, population_size, debug=False):
    global GRAPH_VALUES, GUI_GEN_NUMBER, GUI_FITNESS, GUI_PREVIEW

    population = agent.generate_population(population_size)
    robot_source_files = []
        
    environments = []

    # create individual source files for different bodies
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
            if agent.use_body_parts:
                robot.create(robot_source_files[i], agent.body_part_mask, population[i][1])
            else:
                robot.create(robot_source_files[i], agent.body_part_mask)

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
def printHelp():
    print("-h --help       ... Print help")
    print("-o <individual> ... Select input file to play")

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

        

def Run(gui, params:RunParams):
    global GUI_FLAG
    GUI_FLAG = gui

    import logging
    client = Client(n_workers=12,threads_per_worker=1,scheduler_port=0,
                    silence_logs=logging.ERROR)

    try:
        print("RUNNING")
        best_individual_actions, best_individual_env = evolution(params.robot, 
                                                                 params.agent,
                                                                 client,
                                                                 generation_count=params.ga_generation_count, 
                                                                 population_size=params.ga_population_size, 
                                                                 debug=False)
        print("DONE")
    finally:
        client.close()

    print("HEY")

    best_reward = simulationRun(best_individual_env, agent, best_individual_actions, params.show_best)

    if params.save_best:
        current_time = time.time()
        agent.save(best_individual_actions, params.save_dir + f"/{params.note}_individual_run{current_time}_rew{best_reward}")
        graph_data = np.array(GRAPH_VALUES)
        np.save(params.save_dir + f"/{params.note}_graph_data{current_time}", graph_data)

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

    #     reward = simulationRun(0, envs, agent, individual, render=True)
    #     print("Run reward: ", reward)
    #     quit()

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
    # LAST CHAGNE - actuator kp from 150 to 100
    # LAST IDEA - pls change env so it can find if it's flipped or not
