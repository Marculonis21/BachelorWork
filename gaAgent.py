#!/usr/bin/env python
from abc import ABC, abstractmethod
import numpy as np

import GAMethods
GA = GAMethods.GA()

class AgentType(ABC):
    @abstractmethod
    def get_action(self, individual, step): pass

    @abstractmethod
    def generate_population(self, population_size): pass

    @abstractmethod
    def selection(self, population, fitness_values): pass

    @abstractmethod
    def crossover(self, population): pass

    @abstractmethod
    def mutation(self, population): pass

class StepCycleHalfAgent(AgentType):
    def __init__(self, action_count, true_action_size):
        self.action_count = action_count
        self.action_size = true_action_size

    def get_action(self, individual, step):
        action = individual[step % self.action_count]

        full_action = np.array([ action[0], action[1], action[2], action[3],
                                -action[0],-action[1],-action[2],-action[3]])

        return full_action

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            individual = 2*np.random.random(size=(self.action_count, self.action_size//2,)) - 1
            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        return GA.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return GA.crossover_uniform(population)

    def mutation(self, population):
        return GA.mutation(population, self.action_size//2, indiv_mutation_prob=0.25, action_mutation_prob=0.03)
