#!/usr/bin/env python
import copy
from distributed.client import PooledRPCCall
import numpy as np
import random

class Operators:
    @staticmethod
    def roulette_selection(population, fitness_values): # TOURNAMENT
        num_positive = np.sum([1 if x > 0 else 0 for x in fitness_values])
        if num_positive < len(population)/3:
            return Operators.tournament_selection(population, fitness_values, int(len(population)*0.1))

        fitness_values = [x if x > 0 else 0 for x in fitness_values]
        sum = np.sum(fitness_values)

        selection_probability = fitness_values/sum

        new_population_indexes = np.random.choice(len(population), size=len(population), p=selection_probability)

        population = np.array(population, dtype=object)
        
        return population[new_population_indexes]

    @staticmethod
    def tournament_selection(population, fitness_values, k=5): # TOURNAMENT
        """
        Runs tournamnets between randomly chosen individuals and selects the best from each tournament.
        """

        fitness_values = np.array(fitness_values)

        new_population = []
        for _ in range(len(population)):
            idx = np.random.choice(len(population), size=k)
            fitnesses = fitness_values[idx]
            new_population.append(population[np.argmax(fitnesses)])

        return new_population

    @staticmethod
    def tournament_prob_selection(population, fitness_values, probability=0.9, k=5): # TOURNAMENT
        """
        Runs tournamnets between randomly chosen individuals and selects one acording to probability based on their results

        p-selection = p*((1-p)^indiv_tournament_result)
        """
        population = np.array(population, dtype=object)
        fitness_values = np.array(fitness_values)

        new_population = []
        for _ in range(len(population)):
            idx = np.random.choice(len(population), size=k)

            individuals = population[idx]
            fitnesses = fitness_values[idx]

            indiv_fitness = zip(individuals, fitnesses)
            sorted_indivs = [indiv for (indiv, _) in sorted(indiv_fitness, key=lambda x: x[1], reverse=True)]

            selection_probability = probability*((1-probability)**np.arange(k))

            selected = np.random.choice(len(sorted_indivs), p=selection_probability)
            new_population.append(sorted_indivs[selected])

        return new_population

    @staticmethod
    def crossover_single_point(population, evolve_control, evolve_body):
        new_population = []

        for i in range(0,len(population)//2):
            indiv1 = copy.deepcopy(population[2*i])
            indiv2 = copy.deepcopy(population[2*i+1])

            indiv1_actions, indiv1_body = indiv1
            indiv2_actions, indiv2_body = indiv2

            # crossover for actions
            if evolve_control:
                crossover_point_actions = random.randint(1, len(indiv1_actions))
                end2 = copy.deepcopy(indiv2_actions[:crossover_point_actions])

                indiv2_actions[:crossover_point_actions] = indiv1_actions[:crossover_point_actions]
                indiv1_actions[:crossover_point_actions] = end2

            # crossover for body
            if evolve_body:
                crossover_point_body = random.randint(1, len(indiv1_body))
                end2 = copy.deepcopy(indiv2_body[:crossover_point_body])

                indiv2_body[:crossover_point_body] = indiv1_body[:crossover_point_body]
                indiv1_body[:crossover_point_body] = end2

            new_population.append([indiv1_actions, indiv1_body])
            new_population.append([indiv2_actions, indiv2_body])

        return new_population

    @staticmethod
    def crossover_uniform(population, evolve_control, evolve_body):
        new_population = []

        for i in range(len(population)//2):
            child1 = copy.deepcopy(population[2*i])
            child2 = copy.deepcopy(population[2*i+1])

            child1_actions, child1_body_parts = child1
            child2_actions, child2_body_parts = child2

            if evolve_control:
                for a in range(len(child1_actions)):
                    if random.random() <= 0.5:
                        child1_actions[a] = population[2*i+1][0][a]
                        child2_actions[a] = population[2*i][0][a]

            if evolve_body:
                for b in range(len(child1_body_parts)):
                    if random.random() <= 0.5:
                        child1_body_parts[b] = population[2*i+1][1][b]
                        child2_body_parts[b] = population[2*i][1][b]

            new_population.append([child1_actions, child1_body_parts])
            new_population.append([child2_actions, child2_body_parts])

        return new_population

    @staticmethod
    def uniform_mutation(population, agent):
        new_population = []

        for individual in population:
            if random.random() < agent.individual_mutation_prob:
                actions, body = individual

                if agent.evolve_control:
                    for a in range(len(actions)):
                        if random.random() < agent.action_mutation_prob:
                            new_action = np.array([])

                            for part in range(len(agent.action_range)):
                                new_action = np.concatenate([new_action, np.random.uniform(agent.action_range[part][0], agent.action_range[part][1], size=agent.action_range[part][2])])
                                    
                            actions[a] = new_action

                if agent.evolve_body:
                    for b in range(len(body)):
                        if random.random() < agent.body_mutation_prob:
                            body[b] = np.random.uniform(agent.body_range[b][0], agent.body_range[b][1])

                individual = [actions, body]

            new_population.append(individual)

        return new_population

    @staticmethod
    def uniform_shift_mutation(population, agent):
        new_population = []

        for individual in population:
            if np.random.random() < agent.individual_mutation_prob:
                actions, body = individual

                if agent.evolve_control:
                    for a in range(len(actions)):
                        if np.random.random() < agent.action_mutation_prob:
                            action_shift = np.array([])

                            for part in range(len(agent.action_range)):
                                # max 5% shift
                                action_shift = np.concatenate([action_shift, np.random.uniform(agent.action_range[part][0]*0.05, agent.action_range[part][1]*0.05, size=agent.action_range[part][2])])

                            actions[a] += action_shift

                if agent.evolve_body:
                    for b in range(len(body)):
                        if random.random() < agent.body_mutation_prob:
                            body[b] = np.random.uniform(agent.body_range[b][0], agent.body_range[b][1])

                individual = [actions, body]

            new_population.append(individual)

        return new_population
