#!/usr/bin/env python

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import time

def random_population(population_size, action_size, action_count):
    population = []

    for i in range(population_size):
        individual = 2*np.random.random(size=(action_count, action_size,)) - 1
        population.append(individual)

    return population

def tournament_selection(population, fitness_values, k=5): # TOURNAMENT
    new_population = []
    for i in range(0,len(population)):
        individuals = []
        fitnesses = []

        for _ in range(0,k):
            idx = random.randint(0,len(population)-1)
            individuals.append(population[idx])
            fitnesses.append(fitness_values[idx])

        new_population.append(individuals[np.argmax(fitnesses)])

    return new_population

def crossover(population,cross_prob=1):
    new_population = []

    for i in range(0,len(population)//2):
        indiv1 = population[2*i]
        indiv2 = population[2*i+1]

        if random.random()<cross_prob:
            # zvolime index krizeni nahodne
            crossover_point = random.randint(0, len(indiv1))
            end2 =  copy.deepcopy(indiv2[:crossover_point])
            indiv2[:crossover_point] = indiv1[:crossover_point]
            indiv1[:crossover_point] = end2

        new_population.append(indiv1)
        new_population.append(indiv2)

    return new_population

def crossover_uniform(population):
    new_population = []

    for i in range(0,len(population)//2):
        indiv1 = np.copy(population[2*i])
        indiv2 = np.copy(population[2*i+1])
        for x in range(len(indiv1)):
            if random.random() <= 0.5:
                indiv1[x], indiv2[x] = indiv2[x], indiv1[x]

        new_population.append(indiv1)
        new_population.append(indiv2)

    return new_population

def mutation(population,indiv_mutation_prob=0.3,action_mutation_prob=0.05):
    new_population = []

    for i in range(0,len(population)):
        individual = population[i]
        if random.random() < indiv_mutation_prob:
            for j in range(0,len(individual)):
                if random.random() < action_mutation_prob:
                    individual[i] = 2*np.random.random(size=(8,)) - 1

        new_population.append(individual)

    return new_population

env = gym.make('Ant-v3', reset_noise_scale=0.0)
env._max_episode_steps = 500

# defaults for rewards
# "forward_reward_weight = 1" - missing
# ctrl_cost_weight = 0.5,
# contact_cost_weight = 5e-4,
# healthy_reward = 1.0,

# Sim steps = 5*max_episode_steps 
def evolution():
    population = random_population(50, 8, env._max_episode_steps)

    for generations in range(10):
        if generations % 25 == 0:
            print(generations)

        # Get fitness values
        fitness_values = []
        for ID,individual in enumerate(population):
            steps = -1
            individual_reward = 0
            done = False
            observation = env.reset()
            while not done:
                steps += 1

                # INFO contains useful reward info
                observation, reward, done, info = env.step(individual[steps])
                if ID == 0 and generations % 10 == 0:
                    env.render()

                individual_reward += reward

            fitness_values.append(individual_reward)

        if generations % 10 == 0:
            print(max(fitness_values))

        best_individual = population[np.argmax(fitness_values)]

        # selection, crossover, mutation
        parents = tournament_selection(population,fitness_values)
        children = crossover_uniform(parents)
        mutated_children = mutation(children)
        population = mutated_children

        # elitism
        population[0] = best_individual

    return best_individual

best = evolution()

env = gym.wrappers.Monitor(env, 'video/test', force=True)

print("LAST RUN")

steps = -1
individual_reward = 0
done = False
observation = env.reset()
while not done:
    steps += 1
    observation, reward, done, _ = env.step(best[steps])
    env.render()

    individual_reward += reward

print("Last run - Best reward: ", individual_reward)
