#!/usr/bin/env python
import copy
import numpy as np
import random

class GA:
    @staticmethod
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

    @staticmethod
    def crossover_single_point(population):
        new_population = []

        for i in range(0,len(population)//2):
            indiv1 = copy.deepcopy(population[2*i])
            indiv2 = copy.deepcopy(population[2*i+1])

            crossover_point = random.randint(1, len(indiv1))
            end2 = copy.deepcopy(indiv2[:crossover_point])
            indiv2[:crossover_point] = indiv1[:crossover_point]
            indiv1[:crossover_point] = end2

            new_population.append(indiv1)
            new_population.append(indiv2)

        return new_population

    @staticmethod
    def crossover_uniform(population):
        new_population = []

        for i in range(len(population)//2):
            child1 = copy.deepcopy(population[2*i])
            child2 = copy.deepcopy(population[2*i+1])
            for x in range(len(child1)):
                if random.random() <= 0.5:
                    child1[x] = population[2*i+1][x]
                    child2[x] = population[2*i][x]

            new_population.append(child1)
            new_population.append(child2)

        return new_population

    @staticmethod
    def mutation(population, action_size, indiv_mutation_prob=0.3, action_mutation_prob=0.05):
        new_population = []

        for individual in population:
            if random.random() < indiv_mutation_prob:
                for j in range(len(individual)):
                    if random.random() < action_mutation_prob:
                        individual[j] = 2*np.random.random(size=(action_size,)) - 1

            new_population.append(individual)

        return new_population
