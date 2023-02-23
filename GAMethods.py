#!/usr/bin/env python
import copy
import numpy as np
import random

class GA:
    @staticmethod
    def roulette_selection(population, fitness_values): # TOURNAMENT
        fitness_values = [x if x > 0 else 0 for x in fitness_values]
        sum = np.sum(fitness_values)
        if sum == 0:
            return tournament_selection(population, fitness_values, 5)
        selection_probability = fitness_values/sum

        new_population_indexes = np.random.choice(len(population), size=len(population), p=selection_probability)
        population = np.array(population, dtype=object)

        return population[new_population_indexes]

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
    def crossover_single_point(population, use_body_parts):
        new_population = []

        for i in range(0,len(population)//2):
            indiv1 = copy.deepcopy(population[2*i])
            indiv2 = copy.deepcopy(population[2*i+1])

            indiv1_actions, indiv1_body = indiv1
            indiv2_actions, indiv2_body = indiv2

            # crossover for actions
            crossover_point_actions = random.randint(1, len(indiv1_actions))
            end2 = copy.deepcopy(indiv2_actions[:crossover_point_actions])

            indiv2_actions[:crossover_point_actions] = indiv1_actions[:crossover_point_actions]
            indiv1_actions[:crossover_point_actions] = end2

            # crossover for body
            if use_body_parts:
                crossover_point_body = random.randint(1, len(indiv1_body))
                end2 = copy.deepcopy(indiv2_body[:crossover_point_body])

                indiv2_body[:crossover_point_body] = indiv1_body[:crossover_point_body]
                indiv1_body[:crossover_point_body] = end2

            new_population.append([indiv1_actions, indiv1_body])
            new_population.append([indiv2_actions, indiv2_body])

        return new_population

    @staticmethod
    def crossover_uniform(population, use_body_parts):
        new_population = []

        for i in range(len(population)//2):
            child1 = copy.deepcopy(population[2*i])
            child2 = copy.deepcopy(population[2*i+1])

            child1_actions, child1_body = child1
            child2_actions, child2_body = child2
            for a in range(len(child1_actions)):
                if random.random() <= 0.5:
                    child1_actions[a] = population[2*i+1][0][a]
                    child2_actions[a] = population[2*i][0][a]

            if use_body_parts:
                for b in range(len(child1_body)):
                    if random.random() <= 0.5:
                        child1_body[b] = population[2*i+1][1][b]
                        child2_body[b] = population[2*i][1][b]

            new_population.append([child1_actions, child1_body])
            new_population.append([child2_actions, child2_body])

        return new_population

    @staticmethod
    def mutation(population, action_size, use_body_parts, indiv_mutation_prob=0.3, action_mutation_prob=0.05, body_mutation_prob=0.2):
        new_population = []

        for individual in population:
            if random.random() < indiv_mutation_prob:
                actions = []
                body = []

                actions, body = individual

                for a in range(len(actions)):
                    if random.random() < action_mutation_prob:
                        actions[a] = 2*np.random.random(size=(action_size,)) - 1

                if use_body_parts:
                    for b in range(len(body)):
                        if random.random() < body_mutation_prob:
                            body[b] = 1.5*np.random.random(size=(1))

                individual = [actions, body]

            new_population.append(individual)

        return new_population
