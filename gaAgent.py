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
        actions, _ = individual

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
                individual = [actions, None]

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

    @classmethod
    def ForGUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, True)

    @property
    def description(self):
        return "Sine Function Full agent\n    Each motor of the robot is controlled by sine wave. Values of these agents are made of only 4 parameters (amplitude, frequency, shiftX, shiftY) for each motor."

    def get_action(self, individual, step):
        values, _ = individual

        step = step/10

        actions = []
        for i in range(len(values)//4):
            amp    = values[4*i]
            freq   = values[4*i+1]
            shiftx = values[4*i+2]
            shifty = values[4*i+3]

            result = amp*math.sin(freq*step + shiftx) + shifty
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
                individual = [actions, None]

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

                actions, body = individual

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

                if self.use_body_parts:
                    for i in range(len(body)):
                        if np.random.random() < body_mutation_prob:
                            body[i] = 1.5*np.random.random(size=(1))

                individual = [actions, body]

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

    @classmethod
    def ForGUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, True)

    @property
    def description(self):
        return "Sine Function Half agent\n    Similar to Sine Function Full agent, however only half of robot's motors are controlled by sine waves. The other half is symmetrical (point symmetry through center of the body)."

    def get_action(self, individual, step):
        values, _ = individual

        step = step/10

        actions = []
        for i in range(len(values)//4):
            amp    = values[4*i]
            freq   = values[4*i+1]
            shiftx = values[4*i+2]
            shifty = values[4*i+3]

            result = amp*math.sin(freq*step + shiftx) + shifty
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
                individual = [actions, None]

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

                actions, body = individual

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

                if self.use_body_parts:
                    for i in range(len(body)):
                        if np.random.random() < body_mutation_prob:
                            body[i] = 1.5*np.random.random(size=(1))

                individual = [actions, body]

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
        actions, _ = individual

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
                individual = [actions, None]

            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        return GA.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return GA.crossover_uniform(population, self.use_body_parts)

    def mutation(self, population):
        return GA.mutation(population, self.action_size, self.use_body_parts, indiv_mutation_prob=0.25, action_mutation_prob=0.03)

# https://ic.unicamp.br/~reltech/PFG/2017/PFG-17-07.pdf
# https://web.fe.up.pt/~pro09025/papers/Shafii%20N.%20-%202009%20-%20A%20truncated%20fourier%20series%20with%20genetic%20algorithm%20for%20the%20control%20of%20biped%20locomotion.pdf
class TFSAgent(AgentType):
    def __init__(self, robot, body_part_mask, GUI=False):
        if not GUI:
            super(TFSAgent, self).__init__(robot, body_part_mask)

        self.arguments = {"period":3,
                          "series_length":3,
                          "coeficient_range":4}

    @classmethod
    def ForGUI(cls):
        "Return default agent for GUI needs"
        return cls(None,None,True)

    @property
    def description(self):
        return "TFSAgent\n    TFSAgent uses Truncated Fourier Series for each motor with potential of developing more complex periodical sequences with comparison to simpler sine wave agents."

    def get_action(self, individual, step):
        values, _ = individual

        def remap(x, in_min, in_max, out_min, out_max):
            return (((x - in_min)*(out_max-out_min))/(in_max-in_min)) + out_min

        step = step/5

        action = []

        N = np.arange(self.arguments["series_length"]) + 1
        for i in range(self.action_size):
            a = values[i][:self.arguments["series_length"]]
            b = values[i][-self.arguments["series_length"]:]

            _out = np.sum(a*np.cos((N*np.pi* step)/self.arguments["period"]) + b*np.sin((N*np.pi* step)/self.arguments["period"]))

            action.append(remap(_out, self.min, self.max, -1, 1))

        return action

    def generate_population(self, population_size):
        population = []

        N = np.arange(self.arguments["series_length"]) + 1

        _a = np.ones([self.arguments["series_length"]]) * self.arguments["coeficient_range"]
        _b = np.ones([self.arguments["series_length"]]) * self.arguments["coeficient_range"]
        _step = np.linspace(0, 2*self.arguments["period"], 100000).reshape(-1,1)
        _min_max_search = np.sum(_a*np.cos(((N*np.pi)*_step)/self.arguments["period"]) + _b*np.sin(((N*np.pi)*_step)/self.arguments["period"]), axis=1)
        self.max = np.array([np.max(_min_max_search)])
        self.min = -self.max

        for _ in range(population_size):
            values = []

            # for each motor gen TFS coeficients
            for _ in range(self.action_size):

                # don't get a_0 ... only shifts
                a = np.random.uniform(-self.arguments["coeficient_range"], self.arguments["coeficient_range"], size=self.arguments["series_length"])
                b = np.random.uniform(-self.arguments["coeficient_range"], self.arguments["coeficient_range"], size=self.arguments["series_length"])

                values.append(np.concatenate([a,b]))

            individual = []

            if self.use_body_parts:
                body_parts = 1.5*np.random.random(size=(np.sum(self.body_part_mask)))
                individual = [values, body_parts]
            else:
                individual = [values, None]

            population.append(individual)

        return population

    def selection(self, population, fitness_values):
        # return GA.tournament_selection(population, fitness_values, 5)
        return GA.roulette_selection(population, fitness_values)

    def crossover(self, population):
        return GA.crossover_uniform(population, self.use_body_parts)

    def mutation(self, population):
        new_population = []

        individual_mutation_prob = 0.5
        action_mutation_prob = 0.2
        body_mutation_prob = 0.05

        for individual in population:
            if np.random.random() < individual_mutation_prob:
                actions = []
                body = []

                actions, body = individual

                for i in range(len(actions)):
                    if np.random.random() < action_mutation_prob:
                        a = np.random.uniform(-self.arguments["coeficient_range"], self.arguments["coeficient_range"], size=self.arguments["series_length"])
                        b = np.random.uniform(-self.arguments["coeficient_range"], self.arguments["coeficient_range"], size=self.arguments["series_length"])
                        actions[i] = np.concatenate([a,b])

                if self.use_body_parts:
                    for i in range(len(body)):
                        if np.random.random() < body_mutation_prob:
                            body[i] = 1.5*np.random.random(size=(1))

                individual = [actions, body]

            new_population.append(individual)

        return new_population
