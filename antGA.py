#!/usr/bin/env python

import gym
from gym.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time

import gaAgent

def simulationRun(agent, actions, render=False):
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
            env.render()

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

if __name__ == "__main__":
    client = Client(n_workers=12,threads_per_worker=1,scheduler_port=0)
    print(client)
    env = gym.make('Ant-v3',
                   reset_noise_scale=0.0)

    env._max_episode_steps = 500

    # step_cycle = 25
    agent = gaAgent.SineFuncHalfAgent()
    # best = evolution(gaAgent.StepCycleHalfAgent(step_cycle, 8), client, population_size=50, step_cycle=step_cycle)
    best = evolution(agent, client, population_size=50, debug=False)

    print("LAST RUN")
    best_reward = simulationRun(agent, best, render=True)

    print("Last run - Best reward: ", best_reward)

    agent.save(best, f"./saves/individuals/individual_{best_reward}")
    client.shutdown()
