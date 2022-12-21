#!/usr/bin/env python

from multiprocessing import current_process
import gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time
import gaAgent
from mujoco_py import GlfwContext

import sys

GUIFLAG = False
GUI_GRAPH_VALUES = [[],[],[]]
GUI_FITNESS = []
GUI_GEN_NUMBER = 0

GUI_PREVIEW = False
GUI_ABORT = False 

def simulationRun(env, agent, actions, render=False, render_start_paused=False):
    global step_cycle

    steps = -1
    individual_reward = 0
    done = False
    finishline_bonus = False
    observation = env.reset()
    while not done:
        steps += 1

        action = agent.get_action(actions, steps)

        observation, reward, done, info = env.step(action)
        if render:
            env.render(start_paused=render_start_paused)

        if info["x_position"] > 39.5 and finishline_bonus: # if able to get to the end of the map 
            finishline_bonus = True
            individual_reward += (env._max_episode_steps - steps)

        if done:
            # sim end
            individual_reward = info["x_position"] # x-distance
            # if info["is_flipped"]:
            #     individual_reward = 0

    return individual_reward

def evolution(env, agentType, client, generation_count, population_size, step_cycle=0, debug=False):
    agent = agentType
    population = agent.generate_population(population_size)

    max_fitnesses = []
    mean_fitnesses = []
    min_fitnesses = []

    global GUI_GRAPH_VALUES, GUI_GEN_NUMBER, GUI_FITNESS, GUI_PREVIEW
    if GUIFLAG:
        GUI_GRAPH_VALUES = [[],[],[]]

    best_individual = None
    for generations in range(generation_count+1):
        if GUI_ABORT:
            break

        GUI_GEN_NUMBER = generations

        if GUI_PREVIEW:
            simulationRun(env, agentType, best_individual, render=True, render_start_paused=True);
            GUI_PREVIEW = False
        else:
            if generations % 25 == 0:
                print("Generation: ",generations)

        # Get fitness values
        fitness_values = []
        futures = []
        if debug:
            for individual in population:
                fitness_values.append(simulationRun(env, agent, individual, generations % 25 == 0))
        else:
            for individual in population:
                futures.append(client.submit(simulationRun, env, agent,individual))

            fitness_values = client.gather(futures)

        if GUIFLAG:
            GUI_FITNESS = [np.mean(fitness_values), 
                            min(fitness_values),
                            max(fitness_values)]

        if generations % 10 == 0:
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
        best_individual = copy.deepcopy(population[np.argmax(fitness_values)])

        # selection, crossover, mutation
        parents = agent.selection(population,fitness_values)
        children = agent.crossover(parents)
        mutated_children = agent.mutation(children)
        population = mutated_children

        # elitism
        population[0] = best_individual

    return best_individual

###################################################
###################################################
def printHelp():
    print("-h --help       ... Print help")
    print("-o <individual> ... Select input file to play")

def RunFromGui(robot='Ant-v3', max_steps=500, agent='', show_best=False, save_best=False, save_dir='./saves/individuals'):
    # GlfwContext(offscreen=True)  # Create a window to init GLFW.
    global GUIFLAG
    GUIFLAG = True

    env = gym.make(robot,
                   reset_noise_scale=0.0)

    env._max_episode_steps = max_steps

    client = Client(n_workers=10,threads_per_worker=1,scheduler_port=0)

    control_agent = None
    if agent == "Full Random":
        # create classes for each robot - contains possible actions, ...
        control_agent = gaAgent.FullRandomAgent(25, env.action_space.shape[0])
    elif agent == "Sine Function Full":
        control_agent = gaAgent.SineFuncFullAgent(env.action_space.shape[0])
    elif agent == "Sine Function Half":
        control_agent = gaAgent.SineFuncHalfAgent(env.action_space.shape[0]//2)
    elif agent == "Step Cycle Half":
        control_agent = gaAgent.StepCycleHalfAgent(20, env.action_space.shape[0])
    else:
        raise AttributeError("Unknown control agent type - " + agent)

    best_individual = evolution(env, control_agent, client, generation_count=10, population_size=50, debug=False)

    client.close()
    if GUI_ABORT:
        env.close()
        return

    best_reward = simulationRun(env, control_agent, best_individual, render=show_best, render_start_paused=True)
    env.close()

    if save_best:
        current_time = time.time()
        control_agent.save(best_individual, save_dir + f"/individual_run{current_time}_rew{best_reward}")
        graph_data = np.array(GUI_GRAPH_VALUES)
        np.save(save_dir + f"/graph_data{current_time}", graph_data)

def RaisePreview():
    global GUI_PREVIEW
    GUI_PREVIEW = True

def RaiseAbort():
    global GUI_ABORT
    GUI_ABORT = True

if __name__ == "__main__":
    # env = gym.make('CustomAntLike-v1',
    env = gym.make('Ant-v3',
                   reset_noise_scale=0.0)

    env._max_episode_steps = 500

    if ("-h" in sys.argv or "--help" in sys.argv):
        printHelp()
        quit()

    if ("-o" in sys.argv):
        agent, individual = (None, None)
        try:
            # agent = gaAgent.SineFuncFullAgent(4)
            agent = gaAgent.FullRandomAgent(500, 8)
            # agent = gaAgent.SineFuncHalfAgent(4)
            individual = np.load(sys.argv[sys.argv.index("-o") + 1])
        except Exception as e:
            print("Problem occured - loading file\n")
            printHelp()

        reward = simulationRun(env, agent, individual, render=True, render_start_paused=True)
        print("Run reward: ", reward)
        quit()

    client = Client(n_workers=12,threads_per_worker=1,scheduler_port=0)
    print(client)

    # step_cycle = 25
    # agent = gaAgent.SineFuncFullAgent(4)
    agent = gaAgent.SineFuncHalfAgent(4)
    best = evolution(env, agent, client, generation_count=250, population_size=50, debug=False)

    print("LAST RUN")
    best_reward = simulationRun(env, agent, best, render=True)
    print("last best\n",best)
    env.close()

    print("Last run - Best reward: ", best_reward)

    agent.save(best, f"./saves/individuals/individual_run{time.time()}_rew{best_reward}")
    client.close()
    print("DONE")
