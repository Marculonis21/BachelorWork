#!/usr/bin/env python
from abc import ABC, abstractmethod
import numpy as np
import random
import math

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

    def save(self, individual, path):
        np.save(path, individual)

    def load(self, path):
        return np.load(path)

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

class SineFuncHalfAgent(AgentType):
    # individual = amplitude, frequency, shift-x, shift-y for each leg
    def get_action(self, individual, step):
        actions = []
        for i in range(len(individual)//4):
            amp    = individual[4*i]
            freq   = individual[4*i+1]
            shiftx = individual[4*i+2]
            shifty = individual[4*i+3]

            value = amp*math.sin(freq*(step/10) + shiftx) + shifty
            if value > 1:
                value = 1
            if value < -1:
                value = -1

            actions.append(value)

        full_action = np.array([ actions[0], actions[1], actions[2], actions[3],
                                -actions[0],-actions[1],-actions[2],-actions[3]])
        # full_action = np.array([ actions[0], actions[1], actions[2], actions[3],
        #                          actions[4], actions[5], actions[6], actions[7]])

        return np.array(full_action)

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            individual = []
            for i in range(4):
                # individual = [[random.uniform(0,1), random.uniform(0,10), random.uniform(0, 2*math.pi)] for _ in range(4)]
                individual.append(random.uniform(0.2,1))         # amplitude
                individual.append(random.uniform(0.5,10))        # frequency
                individual.append(random.uniform(0,2*math.pi)) # shift-x
                individual.append(random.uniform(-0.5,0.5))        # shift-y

            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        return GA.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return GA.crossover_uniform(population)

    def mutation(self, population):
        new_population = []

        individual_mutation_rate = 0.75
        gene_mutation_rate = 0.10

        for individual in population:
            if random.random() < individual_mutation_rate:
                category = -1
                for i in range(len(individual)):
                    category = (category + 1) % 4
                    if random.random() < gene_mutation_rate:
                        if category == 0:
                            individual[i] = random.uniform(0.2,1)
                        if category == 1:
                            individual[i] = random.uniform(0.5,10)
                        if category == 2:
                            individual[i] = random.uniform(0, 2*math.pi)
                        if category == 3:
                            individual[i] = random.uniform(-0.5,0.5)


            new_population.append(individual)

        return new_population

if __name__ == "__main__":
    agent = SineFuncHalfAgent()
    indiv = agent.generate_population(1)[0]

    print(indiv)
    for i in range(10):
        print(i, agent.get_action(indiv, i))

    agent.save(indiv, "./saves/individuals/test")

