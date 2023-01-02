#!/usr/bin/env python
from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import numpy as np
import random
import math
import gym

import GAMethods
GA = GAMethods.GA()

class AgentType(ABC):
    def  __init__(self, robot, body_part_mask):
        self.use_body_parts = any(body_part_mask)
        self.body_part_mask = np.array(body_part_mask)

        file = robot.create_default()
        default_env = gym.make('CustomAntLike-v1', xml_file=file.name)
        file.close()

        self.action_size = default_env.action_space.shape[0]

    @abstractclassmethod
    def ForGUI(cls): pass

    @abstractproperty
    def description(self): pass

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
    def __init__(self, robot, body_part_mask, action_repeat, GUI=False):
        if not GUI:
            super(StepCycleHalfAgent, self).__init__(robot, body_part_mask)
            self.action_size = self.action_size//2
            self.action_repeat = action_repeat 

    @classmethod
    def ForGUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, None, True)

    @property
    def description(self):
        return "Step Cycle Half agent\n    Combination of random and half agent. STEPCOUNT long sequences of random actions for half of the motors are created and and then by symmetry transfered to opposing motors. During runtime, sequences of actions are repeatedly performed"

    def get_action(self, individual, step):
        if self.use_body_parts:
            actions, _ = individual
        else:
            actions = individual

        action = actions[step % self.action_repeat]

        full_action = np.concatenate([action,-action]) 

        return full_action

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            actions = 2*np.random.random(size=(self.action_repeat, self.action_size,)) - 1

            individual = []
            if self.use_body_parts:
                body_parts = 1.5*np.random.random(size=(np.sum(self.body_part_mask)))
                individual = [actions, body_parts]
            else:
                individual = actions

            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        return GA.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return GA.crossover_uniform(population, self.use_body_parts)

    def mutation(self, population):
        return GA.mutation(population, self.action_size, False, indiv_mutation_prob=0.25, action_mutation_prob=0.03)

class SineFuncFullAgent(AgentType):
    # individual = amplitude, frequency, shift-x, shift-y for each leg
    def __init__(self, robot, body_part_mask, GUI=False):
        if not GUI:
            super(SineFuncFullAgent, self).__init__(robot, body_part_mask)

    @classmethod
    def ForGUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, True)

    @property
    def description(self):
        return "Sine Function Full agent\n    Each motor of the robot is controlled by sine wave. Values of these agents are made of only 4 parameters (amplitude, frequency, shiftX, shiftY) for each motor."

    def get_action(self, individual, step):
        if self.use_body_parts:
            values, _ = individual
        else:
            values = individual

        actions = []
        for i in range(len(values)//4):
            amp    = values[4*i]
            freq   = values[4*i+1]
            shiftx = values[4*i+2]
            shifty = values[4*i+3]

            result = amp*math.sin(freq*(step/10) + shiftx) + shifty
            if result > 1:
                result = 1
            if result < -1:
                result = -1

            actions.append(result)

        actions = np.array(actions)
        full_action = np.concatenate([actions, -actions])

        return np.array(full_action)

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            actions = []
            for i in range(self.action_size):
                actions.append(random.uniform(0.2,1))         # amplitude
                actions.append(random.uniform(0.5,10))        # frequency
                actions.append(random.uniform(0,2*math.pi))   # shift-x
                actions.append(random.uniform(-0.5,0.5))      # shift-y

            individual = []
            if self.use_body_parts:
                body_parts = 1.5*np.random.random(size=(np.sum(self.body_part_mask)))
                individual = [actions, body_parts]
            else:
                individual = actions

            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        return GA.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return GA.crossover_uniform(population, self.use_body_parts)

    def mutation(self, population):
        new_population = []

        individual_mutation_prob = 0.75
        action_mutation_prob = 0.10
        body_mutation_prob = 0.05

        for individual in population:
            if random.random() < individual_mutation_prob:
                actions = []
                body = []

                if self.use_body_parts:
                    actions, body = individual
                else:
                    actions = individual

                for category in range(len(actions)):
                    if random.random() < action_mutation_prob:
                        if category % 4 == 0:
                            actions[category] = random.uniform(0.2,1)
                        if category % 4 == 1:
                            actions[category] = random.uniform(0.5,10)
                        if category % 4 == 2:
                            actions[category] = random.uniform(0, 2*math.pi)
                        if category % 4 == 3:
                            actions[category] = random.uniform(-0.5,0.5)

                for i in range(len(body)):
                    if random.random() < body_mutation_prob:
                        body[i] = 1.5*np.random.random(size=(1))

                if self.use_body_parts:
                    individual = [actions, body]
                else:
                    individual = actions

            new_population.append(individual)

        return new_population

class SineFuncHalfAgent(AgentType):
    # individual = amplitude, frequency, shift-x, shift-y for each leg
    def __init__(self, robot, body_part_mask, GUI=False):
        if not GUI:
            super(SineFuncHalfAgent, self).__init__(robot, body_part_mask)
            self.action_size = self.action_size//2

    @classmethod
    def ForGUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, True)

    @property
    def description(self):
        return "Sine Function Half agent\n    Similar to Sine Function Full agent, however only half of robot's motors are controlled by sine waves. The other half is symmetrical (point symmetry through center of the body)."

    def get_action(self, individual, step):
        if self.use_body_parts:
            values, _ = individual
        else:
            values = individual

        actions = []
        for i in range(len(values)//4):
            amp    = values[4*i]
            freq   = values[4*i+1]
            shiftx = values[4*i+2]
            shifty = values[4*i+3]

            result = amp*math.sin(freq*(step/10) + shiftx) + shifty
            if result > 1:
                result = 1
            if result < -1:
                result = -1

            actions.append(result)

        actions = np.array(actions)
        full_action = np.concatenate([actions,-actions]) 

        return full_action

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            actions = []
            for i in range(self.action_size):
                actions.append(random.uniform(0.2,1))         # amplitude
                actions.append(random.uniform(0.5,10))        # frequency
                actions.append(random.uniform(0,2*math.pi))   # shift-x
                actions.append(random.uniform(-0.5,0.5))      # shift-y

            individual = []
            if self.use_body_parts:
                body_parts = 1.5*np.random.random(size=(np.sum(self.body_part_mask)))
                individual = [actions, body_parts]
            else:
                individual = actions

            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        return GA.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return GA.crossover_uniform(population, self.use_body_parts)

    def mutation(self, population):
        new_population = []

        individual_mutation_prob = 0.75
        action_mutation_prob = 0.10
        body_mutation_prob = 0.05

        for individual in population:
            if random.random() < individual_mutation_prob:
                actions = []
                body = []

                if self.use_body_parts:
                    actions, body = individual
                else:
                    actions = individual

                for category in range(len(actions)):
                    if random.random() < action_mutation_prob:
                        if category % 4 == 0:
                            actions[category] = random.uniform(0.2,1)
                        if category % 4 == 1:
                            actions[category] = random.uniform(0.5,10)
                        if category % 4 == 2:
                            actions[category] = random.uniform(0, 2*math.pi)
                        if category % 4 == 3:
                            actions[category] = random.uniform(-0.5,0.5)

                for i in range(len(body)):
                    if random.random() < body_mutation_prob:
                        body[i] = 1.5*np.random.random(size=(1))

                if self.use_body_parts:
                    individual = [actions, body]
                else:
                    individual = actions

            new_population.append(individual)

        return new_population

class FullRandomAgent(AgentType):
    def __init__(self, robot, body_part_mask, action_repeat, GUI=False):
        if not GUI:
            super(FullRandomAgent, self).__init__(robot, body_part_mask)
            self.action_repeat = action_repeat

    @classmethod
    def ForGUI(cls):
        "Return default agent for GUI needs"
        return cls(None,None,None,True)

    @property
    def description(self):
        return "Full Random agent\n    Starts off as a sequence of random actions for each motor for chosen amount of steps. Behavior of the agent is then made by repeating this sequence till end state is reached periodically."

    def get_action(self, individual, step):
        actions = []
        if self.use_body_parts:
            actions, _ = individual
        else:
            actions = individual

        action = actions[step % self.action_repeat]

        return action

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            # gen actions
            actions = 2*np.random.random(size=(self.action_repeat, self.action_size,)) - 1

            individual = []
            if self.use_body_parts:
                body_parts = 1.5*np.random.random(size=(np.sum(self.body_part_mask)))
                individual = [actions, body_parts]
            else:
                individual = actions

            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        return GA.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return GA.crossover_uniform(population, self.use_body_parts)

    def mutation(self, population):
        return GA.mutation(population, self.action_size, self.use_body_parts, indiv_mutation_prob=0.25, action_mutation_prob=0.03)


if __name__ == "__main__":
    pass
    # agent = SineFuncHalfAgent(4)
    # indiv = agent.generate_population(1)[0]

    # print(indiv)
    # for i in range(10):
    #     print(i, agent.get_action(indiv, i))

    # agent.save(indiv, "./saves/individuals/test")
