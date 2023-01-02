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
    def crossover_single_point(population, use_body_parts):
        new_population = []

        for i in range(0,len(population)//2):
            indiv1 = copy.deepcopy(population[2*i])
            indiv2 = copy.deepcopy(population[2*i+1])

            if not use_body_parts:
                crossover_point = random.randint(1, len(indiv1))
                end2 = copy.deepcopy(indiv2[:crossover_point])
                indiv2[:crossover_point] = indiv1[:crossover_point]
                indiv1[:crossover_point] = end2

                new_population.append(indiv1)
                new_population.append(indiv2)

            else:
                indiv1_actions, indiv1_body = indiv1
                indiv2_actions, indiv2_body = indiv2

                # crossover for actions
                crossover_point_actions = random.randint(1, len(indiv1_actions))
                end2 = copy.deepcopy(indiv2_actions[:crossover_point_actions])

                indiv2_actions[:crossover_point_actions] = indiv1_actions[:crossover_point_actions]
                indiv1_actions[:crossover_point_actions] = end2

                # crossover for body
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

            if not use_body_parts:
                for x in range(len(child1)):
                    if random.random() <= 0.5:
                        child1[x] = population[2*i+1][x]
                        child2[x] = population[2*i][x]

                new_population.append(child1)
                new_population.append(child2)

            else:
                child1_actions, child1_body = child1
                child2_actions, child2_body = child2
                for x in range(len(child1_actions)):
                    if random.random() <= 0.5:
                        child1_actions[x] = population[2*i+1][0][x]
                        child2_actions[x] = population[2*i][0][x]

                for x in range(len(child1_body)):
                    if random.random() <= 0.5:
                        child1_body[x] = population[2*i+1][1][x]
                        child2_body[x] = population[2*i][1][x]

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

                if use_body_parts:
                    actions, body = individual
                else:
                    actions = individual

                for i in range(len(actions)):
                    if random.random() < action_mutation_prob:
                        actions[i] = 2*np.random.random(size=(action_size,)) - 1

                for i in range(len(body)):
                    if random.random() < body_mutation_prob:
                        body[i] = 1.5*np.random.random(size=(1))

                if use_body_parts:
                    individual = [actions, body]
                else:
                    individual = actions

            new_population.append(individual)

        return new_population
