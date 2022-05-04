#!/usr/bin/env python

import gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time
import gaAgent

import sys

def simulationRun(agent, actions, render=False, render_start_paused=False):
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
            if info["is_flipped"]:
                individual_reward = 0

    return individual_reward

def evolution(agentType, client, population_size, step_cycle=0, debug=False):
    agent = agentType
    population = agent.generate_population(population_size)

    max_fitnesses = []
    mean_fitnesses = []
    min_fitnesses = []

    best_individual = None
    for generations in range(250):
        if generations % 25 == 0:
            print("Generation: ",generations)

        # Get fitness values
        fitness_values = []
        futures = []
        if debug:
            for individual in population:
                fitness_values.append(simulationRun(agent, individual, generations % 25 == 0))
        else:
            for individual in population:
                futures.append(client.submit(simulationRun, agent,individual))

            fitness_values = client.gather(futures)

        if generations % 10 == 0:
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
            agent = gaAgent.SineFuncHalfAgent()
            individual = np.load(sys.argv[sys.argv.index("-o") + 1])
        except Exception as e:
            print("Problem occured - loading file\n")
            printHelp()

        reward = simulationRun(agent, individual, render=True, render_start_paused=True)
        print("Run reward: ", reward)
        quit()

    client = Client(n_workers=12,threads_per_worker=1,scheduler_port=0)
    print(client)

    # step_cycle = 25
    agent = gaAgent.SineFuncHalfAgent()
    best = evolution(agent, client, population_size=50, debug=False)

    print("LAST RUN")
    best_reward = simulationRun(agent, best, render=True)

    print("Last run - Best reward: ", best_reward)

    agent.save(best, f"./saves/individuals/individual_run{time.time()}_rew{best_reward}")
    client.shutdown()
    print("DONE")
