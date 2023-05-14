#!/usr/bin/env python
from abc import ABC, abstractmethod, abstractclassmethod, abstractproperty
import numpy as np
import math
# import gym
import gymnasium as gym
import pickle
import lzma
from enum import Enum

import resources.gaOperators as gaOperators
Operators = gaOperators.Operators

class EvoType(Enum):
    CONTROL = 0
    BODY = 1
    CONTROL_BODY_PARALEL = 2
    CONTROL_BODY_SERIAL = 3

class BaseAgent(ABC):
    def  __init__(self, robot, body_part_mask, evo_type=EvoType.CONTROL):

        # find out if evolution should continue after first evolution ended - (control evolution
        # changes to body evolution)
        self.continue_evo = False
        if evo_type == EvoType.CONTROL_BODY_SERIAL:
            self.continue_evo = True
            evo_type = EvoType.CONTROL

        self.evolve_control = True if evo_type == EvoType.CONTROL or \
                                      evo_type == EvoType.CONTROL_BODY_PARALEL \
                              else False
        self.evolve_body = True if evo_type == EvoType.BODY or \
                                   evo_type == EvoType.CONTROL_BODY_PARALEL \
                           else False

        # apply body_mask only if we actually want to evolve body
        # when not evolving body - robot stays with its default body values
        self.orig_body_part_mask = body_part_mask
        if self.evolve_body:
            self.body_part_mask = body_part_mask
        else:
            self.body_part_mask = [False]*len(body_part_mask)

        self.action_range = [] 
        self.body_range   = [] 
       
        file = robot.create_default()
        default_env = gym.make(robot.environment_id, xml_file=file.name)
        file.close()

        assert default_env.action_space.shape is not None
        self.action_size = default_env.action_space.shape[0]
        default_env.close()

        self.arguments = {}

        # to be rewritten in child classes
        self.individual_mutation_prob = 0 
        self.action_mutation_prob     = 0
        self.body_mutation_prob       = 0

    def switch_evo_phase(self):
        self.continue_evo = False
        self.body_part_mask = self.orig_body_part_mask

        self.evolve_control = False
        self.evolve_body = True

        self.action_range = [] 
        self.body_range   = [] 

    @abstractclassmethod
    def for_GUI(cls): pass

    @abstractproperty
    def description(self): pass

    @abstractmethod
    def get_action(self, individual, step): pass

    @abstractmethod
    def generate_population(self, population_size, load_dir=""): pass

    @abstractmethod
    def selection(self, population, fitness_values): pass

    @abstractmethod
    def crossover(self, population): pass

    @abstractmethod
    def mutation(self, population): pass

    @staticmethod
    def save(agent, robot, individual, path):
        with lzma.open(path, "wb") as save_file:
            pickle.dump((agent, robot, individual), save_file)

    @staticmethod
    def load(path):
        with lzma.open(path, "rb") as save_file:
            agent, robot, individual = pickle.load(save_file)

        return agent, robot, individual

class StepCycleHalfAgent(BaseAgent):
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, GUI=False):
        if not GUI:
            super(StepCycleHalfAgent, self).__init__(robot, body_parts, evo_type)
            self.action_size = self.action_size//2

        self.arguments = {"cycle_repeat": 25}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

    @classmethod
    def for_GUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, GUI=True)

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
            actions = []
            body_parts = []

            actions = 2*np.random.random(size=(self.arguments["cycle_repeat"], self.action_size,)) - 1

            if self.evolve_body:
                body_parts = np.array([])
                for value in self.body_part_mask:
                    if value:
                        assert isinstance(value, tuple) or isinstance(value, list)
                        _part = np.random.uniform(value[0], value[1],size=1)
                        body_parts = np.concatenate([body_parts,_part])
            body_parts = np.array([body_parts])

            individual = [actions, body_parts]
            population.append(individual)

        return np.array(population, dtype=object)

    def selection(self, population, fitness_values):
        return Operators.tournament_selection(population, fitness_values, 5)

    def crossover(self, population):
        return Operators.crossover_uniform(population, self.evolve_control, self.evolve_body)

    def mutation(self, population):
        if self.action_range == []:
            self.action_range = [(-1.0,1.0,self.action_size)] 
            self.body_range    = [] 

            for value in self.body_part_mask:
                if value:
                    assert isinstance(value, tuple)
                    self.body_range.append((value[0], value[1]))

        return Operators.uniform_mutation(population, self)

class SineFuncFullAgent(BaseAgent):
    # individual = amplitude, period, shift-x, shift-y for each leg
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, GUI=False):
        if not GUI:
            super(SineFuncFullAgent, self).__init__(robot, body_parts, evo_type)

        self.arguments = {}
        self.arguments = {"amplitude_range" : {"MIN":0.1,  "MAX":2},
                          "period_range"    : {"MIN":1,    "MAX":2},
                          "shift_x_range"   : {"MIN":0,    "MAX":2*math.pi}}
        self.arguments["shift_y_range"] = {"MIN": -self.arguments["amplitude_range"]["MAX"]/4, 
                                           "MAX":  self.arguments["amplitude_range"]["MAX"]/4}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

    @classmethod
    def for_GUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, GUI=True)

    @property
    def description(self):
        return "Sine Function Full agent\n    Each motor of the robot is controlled by sine wave. Values of these agents are made of only 4 parameters (amplitude, frequency, shiftX, shiftY) for each motor."

    def get_action(self, individual, step):
        values, _ = individual

        step = step/5

        values = np.array(values)
        amp    = values[:,0]
        period = values[:,1]
        shiftX = values[:,2]
        shiftY = values[:,3]
        full_action = amp*np.sin((2*np.pi*step)/period + shiftX) + shiftY

        def __remap(x, in_min, in_max, out_min, out_max):
            return (((x - in_min)*(out_max-out_min))/(in_max-in_min)) + out_min

        for i in range(len(full_action)):
            full_action[i] = __remap(full_action[i], self.min, self.max, -1, 1)

        return np.array(full_action)

    def generate_population(self, population_size):
        population = []

        _amp   = self.arguments["amplitude_range"]["MAX"]
        _period = self.arguments["period_range"]["MAX"]
        _shiftX = 0
        _shiftY = self.arguments["shift_y_range"]["MAX"]

        _step = np.linspace(0, _period, 100000).reshape(-1,1)

        _min_max_search = _amp*np.sin((2*np.pi*_step)/_period + _shiftX) + _shiftY

        self.max = np.array([np.max(_min_max_search)])
        self.min = -self.max

        for _ in range(population_size):
            values = []
            body_parts = []

            for _ in range(self.action_size):
                amps   = np.random.uniform(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"], size=1) # amplitude
                period = np.random.uniform(self.arguments["period_range"]["MIN"],    self.arguments["period_range"]["MAX"], size=1) # amplitude
                shiftX = np.random.uniform(self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"], size=1)   # shift-x
                shiftY = np.random.uniform(self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"], size=1)   # shift-y
                values.append(np.concatenate([amps,period,shiftX,shiftY])) 
            values = np.array(values)

            if self.evolve_body:
                body_parts = np.array([])
                for value in self.body_part_mask:
                    if value:
                        assert isinstance(value, tuple) or isinstance(value, list)
                        _part = np.random.uniform(value[0], value[1],size=1)
                        body_parts = np.concatenate([body_parts,_part])
            body_parts = np.array([body_parts])

            individual = [values, body_parts]
            population.append(individual)

        return np.array(population, dtype=object)

    def selection(self, population, fitness_values):
        return Operators.tournament_selection(population, fitness_values, int(len(population)*0.1))
        # return Operators.roulette_selection(population, fitness_values)

    def crossover(self, population):
        return Operators.crossover_uniform(population, self.evolve_control, self.evolve_body)

    def mutation(self, population):
        if self.action_range == []:
            self.action_range  = [(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"], 1), 
                                  (self.arguments["period_range"]["MIN"],    self.arguments["period_range"]["MAX"],    1),
                                  (self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"],   1),
                                  (self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"],   1)]
            self.body_range    = [] 

            for value in self.body_part_mask:
                if value:
                    assert isinstance(value, tuple)
                    self.body_range.append((value[0], value[1]))

        return Operators.uniform_mutation(population, self)

class SineFuncHalfAgent(BaseAgent):
    # individual = amplitude, frequency, shift-x, shift-y for each leg
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, GUI=False):
        if not GUI:
            super(SineFuncHalfAgent, self).__init__(robot, body_parts, evo_type)
            self.action_size = self.action_size//2

        self.arguments = {}
        self.arguments = {"amplitude_range" : {"MIN":0.1,  "MAX":2},
                          "period_range"    : {"MIN":1,    "MAX":2},
                          "shift_x_range"   : {"MIN":0,    "MAX":2*math.pi}}
        self.arguments["shift_y_range"] = {"MIN": -self.arguments["amplitude_range"]["MAX"]/2, 
                                           "MAX":  self.arguments["amplitude_range"]["MAX"]/2}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

    @classmethod
    def for_GUI(cls):
        "Return default agent for GUI needs"
        return cls(None, None, GUI=True)

    @property
    def description(self):
        return "Sine Function Half agent\n    Similar to Sine Function Full agent, however only half of robot's motors are controlled by sine waves. The other half is symmetrical (point symmetry through center of the body)."

    def get_action(self, individual, step):
        values, _ = individual

        step = step/5

        values = np.array(values)
        amp    = values[:,0]
        period = values[:,1]
        shiftX = values[:,2]
        shiftY = values[:,3]
        actions = amp*np.sin((2*np.pi*step)/period + shiftX) + shiftY

        def __remap(x, in_min, in_max, out_min, out_max):
            return (((x - in_min)*(out_max-out_min))/(in_max-in_min)) + out_min

        for i in range(len(actions)):
            actions[i] = __remap(actions[i], self.min, self.max, -1, 1)

        full_action = np.concatenate([actions,-actions]) 

        return full_action

    def generate_population(self, population_size):
        population = []

        _amp   = self.arguments["amplitude_range"]["MAX"]
        _period = self.arguments["period_range"]["MAX"]
        _shiftX = 0
        _shiftY = self.arguments["shift_y_range"]["MAX"]

        _step = np.linspace(0, _period, 100000).reshape(-1,1)

        _min_max_search = _amp*np.sin((2*np.pi*_step)/_period + _shiftX) + _shiftY

        self.max = np.array([np.max(_min_max_search)])
        self.min = -self.max

        for _ in range(population_size):
            values = []
            body_parts = []

            for _ in range(self.action_size):
                amps   = np.random.uniform(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"], size=1) # amplitude
                period = np.random.uniform(self.arguments["period_range"]["MIN"],    self.arguments["period_range"]["MAX"], size=1) # amplitude
                shiftX = np.random.uniform(self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"], size=1)   # shift-x
                shiftY = np.random.uniform(self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"], size=1)   # shift-y
                values.append(np.concatenate([amps,period,shiftX,shiftY])) 
            values = np.array(values)

            if self.evolve_body:
                body_parts = np.array([])
                for value in self.body_part_mask:
                    if value:
                        assert isinstance(value, tuple) or isinstance(value, list)
                        _part = np.random.uniform(value[0], value[1],size=1)
                        body_parts = np.concatenate([body_parts,_part])
            body_parts = np.array([body_parts])

            individual = [values, body_parts]
            population.append(individual)

        return np.array(population, dtype=object)

    def selection(self, population, fitness_values):
        # return Operators.tournament_prob_selection(population, fitness_values, 0.8, int(len(population)*0.2))
        return Operators.roulette_selection(population, fitness_values)

    def crossover(self, population):
        return Operators.crossover_uniform(population, self.evolve_control, self.evolve_body)

    def mutation(self, population):
        if self.action_range == []:
            self.action_range  = [(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"], 1), 
                                  (self.arguments["period_range"]["MIN"],    self.arguments["period_range"]["MAX"],    1),
                                  (self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"],   1),
                                  (self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"],   1)]
            self.body_range    = [] 

            for value in self.body_part_mask:
                if value:
                    assert isinstance(value, tuple)
                    self.body_range.append((value[0], value[1]))

        # return Operators.uniform_shift_mutation(population, self)
        return Operators.uniform_mutation(population, self)

class FullRandomAgent(BaseAgent):
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, GUI=False):
        if not GUI:
            super(FullRandomAgent, self).__init__(robot, body_parts, evo_type)

        self.arguments = {"cycle_repeat": 25}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

    @classmethod
    def for_GUI(cls):
        "Return default agent for GUI needs"
        return cls(None,None,GUI=True)

    @property
    def description(self):
        return "Full Random agent\n    Starts off as a sequence of random actions for each motor for chosen amount of steps. Behavior of the agent is then made by repeating this sequence till end state is reached periodically."

    def get_action(self, individual, step):
        actions = []
        actions, _ = individual

        action = np.array(actions[step % self.arguments["cycle_repeat"]])

        return action

    def generate_population(self, population_size):
        population = []

        for _ in range(population_size):
            # gen actions
            actions = []
            body_parts = []

            actions = 2*np.random.random(size=(self.arguments["cycle_repeat"], self.action_size,)) - 1

            if self.evolve_body:
                body_parts = np.array([])
                for value in self.body_part_mask:
                    if value:
                        assert isinstance(value, tuple) or isinstance(value, list)
                        _part = np.random.uniform(value[0], value[1],size=1)
                        body_parts = np.concatenate([body_parts,_part])
            body_parts = np.array([body_parts])

            individual = [actions, body_parts]
            population.append(individual)

        return np.array(population, dtype=object)

    def selection(self, population, fitness_values):
        return Operators.tournament_prob_selection(population, fitness_values, 0.8, int(len(population)*0.2))

    def crossover(self, population):
        return Operators.crossover_uniform(population, self.evolve_control, self.evolve_body)

    def mutation(self, population):
        if self.action_range == []:
            self.action_range = [(-1.0,1.0,self.action_size)] 
            self.body_range    = [] 

            for value in self.body_part_mask:
                if value:
                    assert isinstance(value, tuple)
                    self.body_range.append((value[0], value[1]))

        return Operators.uniform_mutation(population, self)

# https://ic.unicamp.br/~reltech/PFG/2017/PFG-17-07.pdf
# https://web.fe.up.pt/~pro09025/papers/Shafii%20N.%20-%202009%20-%20A%20truncated%20fourier%20series%20with%20genetic%20algorithm%20for%20the%20control%20of%20biped%20locomotion.pdf
class TFSAgent(BaseAgent):
    def __init__(self, robot, body_part_mask, evo_type=EvoType.CONTROL, GUI=False):
        if not GUI:
            super(TFSAgent, self).__init__(robot, body_part_mask, evo_type)

        self.arguments = {}
        self.arguments = {"period":4,
                          "series_length":3,
                          "coeficient_range":1}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

    @classmethod
    def for_GUI(cls):
        "Return default agent for GUI needs"
        return cls(None,None,GUI=True)

    @property
    def description(self):
        return "TFSAgent\n    TFSAgent uses Truncated Fourier Series for each motor with potential of developing more complex periodical sequences with comparison to simpler sine wave agents."

    def get_action(self, individual, step):
        values, _ = individual

        step = step/5

        action = []

        def __remap(x, in_min, in_max, out_min, out_max):
            return (((x - in_min)*(out_max-out_min))/(in_max-in_min)) + out_min

        N = np.arange(self.arguments["series_length"]) + 1
        for i in range(self.action_size):
            amps   = values[i][:self.arguments["series_length"]]
            shifts = values[i][-self.arguments["series_length"]:]

            _out = np.sum(amps*np.sin((N*2*np.pi*step)/self.arguments["period"] + shifts))

            action.append(__remap(_out, self.min, self.max, -1, 1))

        return np.array(action)

    def generate_population(self, population_size):
        population = []

        N = np.arange(self.arguments["series_length"]) + 1

        _amps   = np.ones([self.arguments["series_length"]]) * self.arguments["coeficient_range"]
        _shifts = np.zeros([self.arguments["series_length"]])

        _step = np.linspace(0, 2*self.arguments["period"], 100000).reshape(-1,1)

        _min_max_search = np.sum(_amps*np.sin((N*2*np.pi*_step)/self.arguments["period"] + _shifts), axis=1)

        self.max = np.array([np.max(_min_max_search)])
        self.min = -self.max

        for _ in range(population_size):
            values = []
            body_parts = []

            # for each motor gen TFS coeficients
            for _ in range(self.action_size):

                amps = np.random.uniform(-self.arguments["coeficient_range"], self.arguments["coeficient_range"], size=self.arguments["series_length"])
                shifts = np.random.uniform(0, 2*np.pi, size=self.arguments["series_length"])

                values.append(np.concatenate([amps,shifts]))
            values = np.array(values)

            if self.evolve_body:
                body_parts = np.array([])
                for value in self.body_part_mask:
                    if value:
                        assert isinstance(value, tuple) or isinstance(value, list)
                        _part = np.random.uniform(value[0], value[1],size=1)
                        body_parts = np.concatenate([body_parts,_part])
            body_parts = np.array([body_parts])

            individual = [values, body_parts]
            population.append(individual)

        return np.array(population, dtype=object)

    def selection(self, population, fitness_values):
        # return GA.tournament_prob_selection(population, fitness_values, 0.8, int(len(population)*0.2))
        # return GA.tournament_selection(population, fitness_values, int(len(population)*0.2))
        return Operators.roulette_selection(population, fitness_values)

    def crossover(self, population):
        return Operators.crossover_uniform(population, self.evolve_control, self.evolve_body)

    def mutation(self, population):
        if self.action_range == []:
            self.action_range = [(-self.arguments["coeficient_range"], self.arguments["coeficient_range"], self.arguments["series_length"]), 
                                 (-2*np.pi, 2*np.pi, self.arguments["series_length"])]
            self.body_range    = [] 

            for value in self.body_part_mask:
                if value:
                    assert isinstance(value, tuple)
                    self.body_range.append((value[0], value[1]))

        return Operators.uniform_shift_mutation(population, self)
