#!/usr/bin/env python

import typing
import time
import sys
import copy

import resources.gaAgents as gaAgents
import resources.robots.robots as robots
from resources.runParams import RunParams

class Experiments:
    """
    Class of user defined experiments. Experiment parameters for main
    `RunEvolution` method are accessed as via class methods or via
    `get_experiment` method which takes an experiment name as an argument. 

    When defining new experiments users need to create their own method for
    parameter creation and then add the experiment to `__experiments`
    dictionary with selected custom name.
    """

    __batch_dir = "./saves/batch_runs/@1_run_@2_@3_@4/"
    __experiments : typing.Dict[str, RunParams]

    def __init__(self):
        self.__experiments = {}
        self.__experiments["exp10_TFS"] = self.exp10_TFS_spotlike(False)
        self.__experiments["exp11_TFS"] = self.exp11_TFS_spotlike(False)
        self.__experiments["exp12_TFS"] = self.exp12_TFS_spotlike(False)
        self.__experiments["exp11_SineFull"] = self.exp11_SineFull_spotlike(False)
        self.__experiments["exp12_SineFull"] = self.exp12_SineFull_spotlike(False)

    def __create_batch_dir(self, robot, agent, note):
        batch_dir = self.__batch_dir.replace("@1", note).replace(
                                             "@2", type(robot).__name__).replace(
                                             "@3", type(agent).__name__).replace(
                                             "@4", str(time.time()))
        return batch_dir

    def __exp_start_note(self):
        print("Starting experiment:", sys._getframe(1).f_code.co_name)

    def get_experiment_names(self):
        return list(self.__experiments.keys());

    def get_experiment(self, name):
        assert name in self.get_experiment_names(), f"Unknown experiment name - list of created experiments {self.get_experiment_names()}"
        return copy.copy(self.__experiments[name])

    def exp10_TFS_spotlike(self, run=True):
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts))
        note = "exp1.0"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = RunParams(robot, 
                           agent,
                           ga_population_size=100,
                           ga_generation_count=500,
                           show_best=False,
                           save_best=True,
                           save_dir=batch_dir,
                           note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp11_TFS_spotlike(self, run=True):
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts))
        note = "exp1.1"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = RunParams(robot, 
                           agent,
                           ga_population_size=100,
                           ga_generation_count=200,
                           show_best=False,
                           save_best=True,
                           save_dir=batch_dir,
                           note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp12_TFS_spotlike(self, run=True):
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts))
        note = "exp1.2"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = RunParams(robot, 
                           agent,
                           ga_population_size=100,
                           ga_generation_count=100,
                           show_best=False,
                           save_best=True,
                           save_dir=batch_dir,
                           note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp11_SineFull_spotlike(self, run=True):
        robot = robots.SpotLike()
        agent = gaAgents.SineFuncFullAgent(robot, [False]*len(robot.body_parts))
        note = "exp1.1"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = RunParams(robot, 
                           agent,
                           ga_population_size=100,
                           ga_generation_count=200,
                           show_best=False,
                           save_best=True,
                           save_dir=batch_dir,
                           note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp12_SineFull_spotlike(self, run=True):
        robot = robots.SpotLike()
        agent = gaAgents.SineFuncFullAgent(robot, [False]*len(robot.body_parts))
        note = "exp1.2"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = RunParams(robot, 
                           agent,
                           ga_population_size=100,
                           ga_generation_count=100,
                           show_best=False,
                           save_best=True,
                           save_dir=batch_dir,
                           note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params
