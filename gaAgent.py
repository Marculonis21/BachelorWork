#!/usr/bin/env python
from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import numpy as np
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

        self.arguments = {}

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
    def __init__(self, robot, body_part_mask, cycle_repeat, GUI=False):
        if not GUI:
            super(StepCycleHalfAgent, self).__init__(robot, body_part_mask)
            self.action_size = self.action_size//2

            self.arguments = {"cycle_repeat":cycle_repeat}



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

        action = actions[step % self.arguments["cycle_repeat"]]

        full_action = np.concatenate([action,-action]) 

        return full_action

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            actions = 2*np.random.random(size=(self.arguments["cycle_repeat"], self.action_size,)) - 1

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

class SineFuncFullAgent(AgentType):
    # individual = amplitude, frequency, shift-x, shift-y for each leg
    def __init__(self, robot, body_part_mask, GUI=False):
        if not GUI:
            super(SineFuncFullAgent, self).__init__(robot, body_part_mask)

        self.arguments = {}
        self.arguments["amplitude_range"] = {"MIN":0.5, "MAX":5}
        self.arguments["frequency_range"] = {"MIN":0.5, "MAX":5}
        self.arguments["shift_x_range"]   = {"MIN":0,   "MAX":2*math.pi}
        self.arguments["shift_y_range"]   = {"MIN": self.arguments["frequency_range"]["MIN"]/2,
                                             "MAX": self.arguments["frequency_range"]["MAX"]/2}
        # self.amplitude_range = 
        # self.frequency_range = 
        # self.shift_x_range   = 
        # self.shift_y_range   = 

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

        full_action = np.array(actions)

        return np.array(full_action)

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            actions = []
            for i in range(self.action_size):
                actions.append(np.random.uniform(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"])) # amplitude
                actions.append(np.random.uniform(self.arguments["frequency_range"]["MIN"], self.arguments["frequency_range"]["MAX"])) # frequency
                actions.append(np.random.uniform(self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"]))     # shift-x
                actions.append(np.random.uniform(self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"]))     # shift-y

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
            if np.random.random() < individual_mutation_prob:
                actions = []
                body = []

                if self.use_body_parts:
                    actions, body = individual
                else:
                    actions = individual

                for category in range(len(actions)):
                    if np.random.random() < action_mutation_prob:
                        if category % 4 == 0:
                            actions[category] = np.random.uniform(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"])
                        if category % 4 == 1:
                            actions[category] = np.random.uniform(self.arguments["frequency_range"]["MIN"], self.arguments["frequency_range"]["MAX"])
                        if category % 4 == 2:
                            actions[category] = np.random.uniform(self.arguments["shift_x_range"]["MIN"], self.arguments["shift_x_range"]["MAX"])
                        if category % 4 == 3:
                            actions[category] = np.random.uniform(self.arguments["shift_y_range"]["MIN"], self.arguments["shift_y_range"]["MAX"])

                for i in range(len(body)):
                    if np.random.random() < body_mutation_prob:
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

        self.arguments = {}
        self.arguments["amplitude_range"] = {"MIN":0.5, "MAX":5}
        self.arguments["frequency_range"] = {"MIN":0.5, "MAX":5}
        self.arguments["shift_x_range"]   = {"MIN":0,   "MAX":2*math.pi}
        self.arguments["shift_y_range"]   = {"MIN": self.arguments["frequency_range"]["MIN"]/2,
                                             "MAX": self.arguments["frequency_range"]["MAX"]/2}

        # self.amplitude_range = {"MIN":0.5, "MAX":5}
        # self.frequency_range = {"MIN":0.5, "MAX":5}
        # self.shift_x_range   = {"MIN":0,   "MAX":2*math.pi}
        # self.shift_y_range   = {"MIN":self.frequency_range["MIN"]/2, "MAX":self.frequency_range["MAX"]/2} # probably set to not allow no movement

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
                actions.append(np.random.uniform(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"])) # amplitude
                actions.append(np.random.uniform(self.arguments["frequency_range"]["MIN"], self.arguments["frequency_range"]["MAX"])) # frequency
                actions.append(np.random.uniform(self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"]))   # shift-x
                actions.append(np.random.uniform(self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"]))   # shift-y

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
            if np.random.random() < individual_mutation_prob:
                actions = []
                body = []

                if self.use_body_parts:
                    actions, body = individual
                else:
                    actions = individual

                for category in range(len(actions)):
                    if np.random.random() < action_mutation_prob:
                        if category % 4 == 0:
                            actions[category] = np.random.uniform(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"])
                        if category % 4 == 1:
                            actions[category] = np.random.uniform(self.arguments["frequency_range"]["MIN"], self.arguments["frequency_range"]["MAX"])
                        if category % 4 == 2:
                            actions[category] = np.random.uniform(self.arguments["shift_x_range"]["MIN"], self.arguments["shift_x_range"]["MAX"])
                        if category % 4 == 3:
                            actions[category] = np.random.uniform(self.arguments["shift_y_range"]["MIN"], self.arguments["shift_y_range"]["MAX"])

                for i in range(len(body)):
                    if np.random.random() < body_mutation_prob:
                        body[i] = 1.5*np.random.random(size=(1))

                if self.use_body_parts:
                    individual = [actions, body]
                else:
                    individual = actions

            new_population.append(individual)

        return new_population

class FullRandomAgent(AgentType):
    def __init__(self, robot, body_part_mask, cycle_repeat, GUI=False):
        if not GUI:
            super(FullRandomAgent, self).__init__(robot, body_part_mask)

        self.arguments = {"cycle_repeat":cycle_repeat}

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

        action = actions[step % self.arguments["cycle_repeat"]]

        return action

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            # gen actions
            actions = 2*np.random.random(size=(self.arguments["cycle_repeat"], self.action_size,)) - 1

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
