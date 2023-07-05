#!/usr/bin/env python
"""Module for genetic agents.

Module that stores base agent abstract class :class:`BaseAgent` from which all
the other agent classes have to inherit.

Agent class is heavily used as an information source for GUI.

Usage example::

    import gaAgents

    robot = ... # Get some robot (type BaseRobot) 

    # regular TFSAgent instance 
    agent1 = gaAgents.TFSAgent(robot)

    # TFSAgent with unlocked robot body parts (first with range 1 to 5, last with range 2 to 3)
    agent2 = gaAgents.TFSAgent(robot, [(1,5), False, False, (2,3)])

Default implemented agents:
    * :class:`StepCycleFullAgent`
    * :class:`StepCycleHalfAgent`
    * :class:`SineFuncFullAgent`
    * :class:`SineFuncHalfAgent`
    * :class:`TFSAgent`
    * :class:`NEATAgent`
"""

from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import math
# import gym
import gymnasium as gym
import pickle
import lzma
from enum import Enum

import neat
import re
import tempfile
import copy
import os

import resources.agents.gaOperators as gaOperators
Operators = gaOperators.Operators

# DECORATORS - if gui was used, then operator selected by gui is used with selected arguments
#            - otherwise the orig function is used -> selected by code
def selection_deco( func ):
    def wrapper(self, population, fitness_values):
        if self.gui:
            method, arguments = self.genetic_operators["selection"] 
            return method(population, fitness_values, *arguments)
        else:
            return func( self, population, fitness_values)
    return wrapper

def crossover_deco( func ):
    def wrapper(self, population):
        if self.gui:
            method, arguments = self.genetic_operators["crossover"] 
            return method(population, *arguments)
        else:
            return func( self, population)
    return wrapper

def mutation_deco( func ):
    def wrapper(self, population):
        if self.gui:
            method, arguments = self.genetic_operators["mutation"] 
            return method(population, *arguments)
        else:
            return func( self, population)
    return wrapper

class EvoType(Enum):
    """EvoType enum
    Enum used for naming different types of evolution that agent can perform.
    """

    CONTROL = 0 
    BODY = 1   
    CONTROL_BODY_PARALEL = 2 
    CONTROL_BODY_SERIAL = 3 

class BaseAgent(ABC):
    """Base agent class

    This is the base abstract class from which every agent is inheriting and
    also calling in their init function. 

    In :func:`__init__` we get to initialise all default class fields which
    might be later used by the evolutionary algorithm, by the UI or by the
    agent itself.

    :ivar evo_type: Value of selected evolution type.
    :ivar continue_evo: Flag showing if the evolution of agent will continue
        after generations are finished (used for serial control-body evolution
        ``EvoType.CONTROL_BODY_SERIAL``).

    :ivar evolve_control: Flag showing if robot controls (`actions`) should be
        evolved.
    :ivar evolve_body: Flag showing if robot body should be evolved.

    :ivar body_part_mask: List of bools and tuples, showing which body parts
        should be unlocked for evolution and what ranges are allowed for them.
    :ivar orig_body_part_mask: Copy original :attr:`body_part_mask` parameter. 
        Used for evolution type switching.
    :ivar action_range: List of tuples (`ranges` of values) specifying allowed
        range of values for each action - creator of new agent class is
        ***obligated*** to specify this range, used in genetic operators. For
        examples look into implemented classes.
    :ivar body_range: Same as :attr:`action_range`, but for body parts. 

    :ivar action_size: Size of the action space that agents environment
        supports.
    :ivar observation_size: Size of the observation space that agents
        environment returns (`we often don't care about this one`).

    :ivar gui: Flag showing if this agent was created through GUI (possibly
        changing which genetic operators are used).

    :ivar arguments: Dictionary of agent specific arguments. Every agent
        populates this dictionary by itself depending on its needs. 
    :ivar arguments_tooltips: Dictionary of optional tooltips (hints) for
        :attr:`arguments` values. Used from GUI when describing agent options.
    :ivar genetic_operators: Dictionary of selected genetic operations for
        selection, crossover and mutation, which overrides default operators
        when the agent is created via GUI (:attr:`gui`).

    :ivar individual_mutation_prob: Mutation probability - allow mutation of
        selected individual.
    :ivar action_mutation_prob: Mutation probability - allow mutation of
        selected action.
    :ivar body_mutation_prob: Mutation probability - allow mutation of selected
        body part.

    :vartype evo_type: EvoType
    :vartype continue_evo: bool
    :vartype evolve_control: bool
    :vartype evolve_body: bool
    :vartype body_part_mask: List[False|Tuple[float, float]]
    :vartype action_range: List[Tuple[float, float]]
    :vartype body_range: List[Tuple[float]]
    :vartype gui: bool 
    :vartype arguments: Dict[str, float|{"MIN":float, "MAX":float}]
    :vartype arguments_tooltips: Dict[str, str]
    :vartype genetic_operators: Dict[str, List[Callable, List[Callable parameters ...]]]
    :vartype individual_mutation_prob: float
    :vartype action_mutation_prob: float
    :vartype body_mutation_prob: float
    """

    def  __init__(self, robot, body_part_mask, evo_type=EvoType.CONTROL, gui=False):

        # find out if evolution should continue after first evolution ended - (control evolution
        # changes to body evolution)
        self.evo_type = evo_type

        self.continue_evo   = True if evo_type == EvoType.CONTROL_BODY_SERIAL \
                              else False

        self.evolve_control = True if evo_type == EvoType.CONTROL or \
                                      evo_type == EvoType.CONTROL_BODY_PARALEL \
                              else False

        self.evolve_body    = True if evo_type == EvoType.BODY or \
                                      evo_type == EvoType.CONTROL_BODY_PARALEL \
                              else False

        # apply body_mask only if we actually want to evolve body
        # when not evolving body - robot stays with its default body values
        # (can be changed by GUI)
        self.orig_body_part_mask = body_part_mask
        if self.evolve_body:
            self.body_part_mask = body_part_mask
        else:
            self.body_part_mask = [False]*len(body_part_mask)

        self.action_range = [] 
        self.body_range   = [] 
       
        file = robot.create_default()
        file.close() # needs to be closed before gym opens it again (Windows)
        if file == None:
            default_env = gym.make(robot.environment_id)
        else:
            default_env = gym.make(robot.environment_id, xml_file=file.name)
            
        file.close() # ANOTHER ONE?
        os.unlink(file.name)

        self.action_size = default_env.action_space.shape[0]
        self.observation_size = default_env.observation_space.shape[0]
        default_env.close()

        self.gui = gui
        self.arguments = {}
        self.arguments_tooltips = {}
        # for gui selected operators 
        self.genetic_operators = {} 

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

class StepCycleFullAgent(BaseAgent):
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, gui=False):
        super(StepCycleFullAgent, self).__init__(robot, body_parts, evo_type, gui)

        self.arguments = {"cycle_repeat": 25}
        self.arguments_tooltips = {"cycle_repeat":"Length of the step sequence"}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

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

        # set ranges for actions and body parts
        self.action_range = [(-1.0,1.0,self.action_size)] 
        self.body_range    = [] 

        for value in self.body_part_mask:
            if value:
                assert isinstance(value, tuple)
                self.body_range.append((value[0], value[1]))

        return np.array(population, dtype=object)

    @selection_deco
    def selection(self, population, fitness_values):
        return Operators.tournament_prob_selection(population, fitness_values, 0.8, int(len(population)*0.2))

    @crossover_deco
    def crossover(self, population):
        return Operators.crossover_uniform(population, self)

    @mutation_deco
    def mutation(self, population):
        return Operators.uniform_mutation(population, self)

class StepCycleHalfAgent(BaseAgent):
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, gui=False):
        super(StepCycleHalfAgent, self).__init__(robot, body_parts, evo_type, gui)
        self.action_size = self.action_size//2

        self.arguments = {"cycle_repeat": 25}
        self.arguments_tooltips = {"cycle_repeat":"Length of the step sequence"}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

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

        # set ranges for actions and body parts
        self.action_range = [(-1.0,1.0,self.action_size)] 
        self.body_range    = [] 
        for value in self.body_part_mask:
            if value:
                assert isinstance(value, tuple)
                self.body_range.append((value[0], value[1]))

        return np.array(population, dtype=object)

    @selection_deco
    def selection(self, population, fitness_values):
        return Operators.tournament_selection(population, fitness_values, 5)

    @crossover_deco
    def crossover(self, population):
        return Operators.crossover_uniform(population, self)

    @mutation_deco
    def mutation(self, population):
        return Operators.uniform_mutation(population, self)

class SineFuncFullAgent(BaseAgent):
    # individual = amplitude, period, shift-x, shift-y for each leg
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, gui=False):
        super(SineFuncFullAgent, self).__init__(robot, body_parts, evo_type, gui)

        self.arguments = {"amplitude_range" : {"MIN":0.1,  "MAX":2},
                          "period_range"    : {"MIN":1,    "MAX":2},
                          "shift_x_range"   : {"MIN":0,    "MAX":2*math.pi}}
        self.arguments["shift_y_range"] = {"MIN": -self.arguments["amplitude_range"]["MAX"]/4, 
                                           "MAX":  self.arguments["amplitude_range"]["MAX"]/4}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

        _amp   = self.arguments["amplitude_range"]["MAX"]
        _period = self.arguments["period_range"]["MAX"]
        _shiftX = 0
        _shiftY = self.arguments["shift_y_range"]["MAX"]
        _step = np.linspace(0, _period, 100000).reshape(-1,1)
        _min_max_search = _amp*np.sin((2*np.pi*_step)/_period + _shiftX) + _shiftY
        self.max = np.array([np.max(_min_max_search)])
        self.min = -self.max

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
                        _part = np.random.uniform(value[0], value[1], size=1)
                        body_parts = np.concatenate([body_parts,_part])
            body_parts = np.array([body_parts])

            individual = [values, body_parts]
            population.append(individual)

        # set ranges for actions and body parts
        self.action_range  = [(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"], 1), 
                              (self.arguments["period_range"]["MIN"],    self.arguments["period_range"]["MAX"],    1),
                              (self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"],   1),
                              (self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"],   1)]
        self.body_range    = [] 

        for value in self.body_part_mask:
            if value:
                assert isinstance(value, tuple)
                self.body_range.append((value[0], value[1]))

        return np.array(population, dtype=object)

    @selection_deco
    def selection(self, population, fitness_values):
        return Operators.tournament_selection(population, fitness_values, int(len(population)*0.1))
        # return Operators.roulette_selection(population, fitness_values)

    @crossover_deco
    def crossover(self, population):
        return Operators.crossover_uniform(population, self)

    @mutation_deco
    def mutation(self, population):
        return Operators.uniform_mutation(population, self)

class SineFuncHalfAgent(BaseAgent):
    # individual = amplitude, frequency, shift-x, shift-y for each leg
    def __init__(self, robot, body_parts, evo_type=EvoType.CONTROL, gui=False):
        super(SineFuncHalfAgent, self).__init__(robot, body_parts, evo_type, gui)
        self.action_size = self.action_size//2

        self.arguments = {"amplitude_range" : {"MIN":0.1,  "MAX":2},
                          "period_range"    : {"MIN":1,    "MAX":2},
                          "shift_x_range"   : {"MIN":0,    "MAX":2*math.pi}}
        self.arguments["shift_y_range"] = {"MIN": -self.arguments["amplitude_range"]["MAX"]/2, 
                                           "MAX":  self.arguments["amplitude_range"]["MAX"]/2}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

        _amp   = self.arguments["amplitude_range"]["MAX"]
        _period = self.arguments["period_range"]["MAX"]
        _shiftX = 0
        _shiftY = self.arguments["shift_y_range"]["MAX"]
        _step = np.linspace(0, _period, 100000).reshape(-1,1)
        _min_max_search = _amp*np.sin((2*np.pi*_step)/_period + _shiftX) + _shiftY
        self.max = np.array([np.max(_min_max_search)])
        self.min = -self.max

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

        # set ranges for actions and body parts
        self.action_range  = [(self.arguments["amplitude_range"]["MIN"], self.arguments["amplitude_range"]["MAX"], 1), 
                              (self.arguments["period_range"]["MIN"],    self.arguments["period_range"]["MAX"],    1),
                              (self.arguments["shift_x_range"]["MIN"],   self.arguments["shift_x_range"]["MAX"],   1),
                              (self.arguments["shift_y_range"]["MIN"],   self.arguments["shift_y_range"]["MAX"],   1)]
        self.body_range    = [] 

        for value in self.body_part_mask:
            if value:
                assert isinstance(value, tuple)
                self.body_range.append((value[0], value[1]))

        return np.array(population, dtype=object)

    @selection_deco
    def selection(self, population, fitness_values):
        # return Operators.tournament_prob_selection(population, fitness_values, 0.8, int(len(population)*0.2))
        return Operators.roulette_selection(population, fitness_values)

    @crossover_deco
    def crossover(self, population):
        return Operators.crossover_uniform(population, self)

    @mutation_deco
    def mutation(self, population):
        return Operators.uniform_mutation(population, self)

# https://ic.unicamp.br/~reltech/PFG/2017/PFG-17-07.pdf
# https://web.fe.up.pt/~pro09025/papers/Shafii%20N.%20-%202009%20-%20A%20truncated%20fourier%20series%20with%20genetic%20algorithm%20for%20the%20control%20of%20biped%20locomotion.pdf
class TFSAgent(BaseAgent):
    def __init__(self, robot, body_part_mask, evo_type=EvoType.CONTROL, gui=False):
        super(TFSAgent, self).__init__(robot, body_part_mask, evo_type, gui)

        self.arguments = {"period":4,
                          "series_length":3,
                          "coeficient_range":1}

        self.individual_mutation_prob = 0.75
        self.action_mutation_prob     = 0.1
        self.body_mutation_prob       = 0.1

        # Get min and max of actions for remapping
        _N = np.arange(self.arguments["series_length"]) + 1
        _amps   = np.ones([self.arguments["series_length"]]) * self.arguments["coeficient_range"]
        _shifts = np.zeros([self.arguments["series_length"]])
        _step = np.linspace(0, 2*self.arguments["period"], 100000).reshape(-1,1)
        _min_max_search = np.sum(_amps*np.sin((_N*2*np.pi*_step)/self.arguments["period"] + _shifts), axis=1)
        self.max = np.array([np.max(_min_max_search)])
        self.min = -self.max

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

        # Get min and max of actions for remapping
        _N = np.arange(self.arguments["series_length"]) + 1
        _amps   = np.ones([self.arguments["series_length"]]) * self.arguments["coeficient_range"]
        _shifts = np.zeros([self.arguments["series_length"]])
        _step = np.linspace(0, 2*self.arguments["period"], 100000).reshape(-1,1)
        _min_max_search = np.sum(_amps*np.sin((_N*2*np.pi*_step)/self.arguments["period"] + _shifts), axis=1)
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

        # set ranges for actions and body parts
        self.action_range = [(-self.arguments["coeficient_range"], self.arguments["coeficient_range"], self.arguments["series_length"]), 
                             (-2*np.pi, 2*np.pi, self.arguments["series_length"])]
        self.body_range    = [] 

        for value in self.body_part_mask:
            if value:
                assert isinstance(value, tuple)
                self.body_range.append((value[0], value[1]))

        return np.array(population, dtype=object)

    @selection_deco
    def selection(self, population, fitness_values):
        return Operators.roulette_selection(population, fitness_values)

    @crossover_deco
    def crossover(self, population):
        return Operators.crossover_uniform(population, self)

    @mutation_deco
    def mutation(self, population):
        return Operators.uniform_shift_mutation(population, self)

class NEATAgent(BaseAgent):
    def __init__(self, robot, body_part_mask, evo_type=EvoType.CONTROL, gui=False):
        super(NEATAgent, self).__init__(robot, body_part_mask, evo_type, gui)

        self.config_source_file = ""
        with open("resources/agents/config_neat.txt") as file:
            lines = file.readlines()
            self.config_source_file= "".join(lines)

        raw_arguments = re.findall(r'\$[A-Za-z0-9_]+\([+-]?[0-9]*[.]?[0-9]+\)\$', self.config_source_file)
        for arg in raw_arguments:
            name = arg[1:arg.find("(")] #)
            self.arguments[name] = float(arg[arg.find("(")+1 : arg.find(")")])
            if name == "NET_NUM_OUTPUTS":
                self.arguments[name] = self.action_size 
            if name == "NET_NUM_INPUTS":
                self.arguments[name] = self.observation_size

    @property
    def description(self):
        return "NEATAgent\n    Agent using neuroevolution algorithm - NEAT"

    def evo_override(self, experiment_params):
        from dask.distributed import Client
        from gymnasium.wrappers.time_limit import TimeLimit
        TIME_LIMIT = 10000

        def custom_eval(env, net, render=False):
            """ Method for custom evaluation of neat-created nets in selected 
            environments.
            """

            steps = -1
            individual_reward = 0
            terminated = False
            truncated = False

            obs = env.reset()
            obs = obs[0]
            while True:
                steps += 1

                action = net.activate(obs)

                obs, reward, terminated, truncated, _ = env.step(action)

                individual_reward += reward

                if render:
                    env.render()

                if terminated or truncated:
                    break

            return individual_reward

        def eval_genomes(genomes, config): 
            """ Method for neat lib evaluator 

            Creating eval method for neat lib - running selected custom eval 
            envs.
            """

            client = Client(n_workers=1,threads_per_worker=1,scheduler_port=0)

            robot = experiment_params.robot
            agent = experiment_params.agent

            file = robot.create(agent.body_part_mask)
            env = None
            if file == None:
                env = gym.make(id=robot.environment_id,
                               max_episode_steps=TIME_LIMIT,
                               render_mode=None)
            else:
                env = gym.make(id=robot.environment_id,
                               xml_file=file.name,
                               reset_noise_scale=0.0,
                               disable_env_checker=True,
                               max_episode_steps=TIME_LIMIT,
                               render_mode=None)
                    
                file.close()

            futures = []
            for _, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, config) 
                futures.append(client.submit(custom_eval, env, net))

            fitness_values = client.gather(futures)

            for i, (_, genome) in enumerate(genomes):
                genome.fitness = fitness_values[i]

            client.close()

        population = self.generate_population(experiment_params.population_size)
        winner = population.run(eval_genomes, experiment_params.generation_count)

        file = experiment_params.robot.create(experiment_params.agent.body_part_mask)
        env = None
        if file == None:
            env = gym.make(id=experiment_params.robot.environment_id,
                           render_mode="human")
        else:
            env = gym.make(id=experiment_params.robot.environment_id,
                           xml_file=file.name,
                           reset_noise_scale=0.0,
                           disable_env_checker=True,
                           max_episode_steps=TIME_LIMIT,
                           render_mode="human")
            file.close()

        net = neat.nn.FeedForwardNetwork.create(winner, self.config) 
        print("READY FOR RENDER ...")
        input()
        print(custom_eval(env, net, True))

    def get_action(self, net, obs):
        return net.activate(obs)

    def generate_population(self, population_size):

        def key_to_regex(key):
            regex = key.replace("$", "\$")
            regex = regex.replace("(", "\(")
            regex = regex.replace(")", "\)")

            return regex

        tmp_file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")
        raw_text = copy.deepcopy(self.config_source_file)
        first_free_line_idx = raw_text.split("\n").index('')
        text = "".join([x+"\n" for x in raw_text.split("\n")[first_free_line_idx+1:]])

        raw_arguments = re.findall(r'\$[A-Za-z0-9_]+\([+-]?[0-9]*[.]?[0-9]+\)\$', self.config_source_file)
        argument_values = list(self.arguments.values())
        for i, key in enumerate(raw_arguments):
            if "POP_SIZE" in key:
                argument_values[i] = population_size

            regex = key_to_regex(key)
            # to check for int/float values
            orig_value = eval(key[key.find("(")+1 : key.find(")")])
            new_value = int(argument_values[i]) if isinstance(orig_value, int) else float(argument_values[i])
            text = re.sub(regex, str(new_value), text)

        tmp_file.seek(0)
        tmp_file.truncate()
        tmp_file.write(text)
        tmp_file.flush()

        self.config = neat.Config(neat.DefaultGenome, 
                                  neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation,
                                  tmp_file.name)
        tmp_file.close()

        pop = neat.Population(self.config)
        pop.add_reporter(neat.StdOutReporter(show_species_detail=True))

        return pop

    @selection_deco
    def selection(self, population, fitness_values):
        pass

    @crossover_deco
    def crossover(self, population):
        pass

    @mutation_deco
    def mutation(self, population):
        pass
