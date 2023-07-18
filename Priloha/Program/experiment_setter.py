#!/usr/bin/env python
"""Experiment setter module

Module used for setting up, saving and loading experiments. Experiments are
stored in dictionary as ``(name, ExperimentParams)`` pairs.

See also:
    :mod:`experiment_params` : Module defining class of experiment parameters.

Usage example::

    from experiment_setter import Experiments
    experiments = Experiments()
    ...
    params = experiments.get_experiment("...")
"""

import typing
import time
import sys
import copy
import pickle
import lzma
import os

import resources.agents.gaAgents as gaAgents
import resources.robots.robots as robots
from resources.experiment_params import ExperimentParams 

class Experiments:
    """
    Class of user defined experiments. Experiment parameters for main
    :func:`roboEvo.run_experiment` method are accessed via class methods or via
    :func:`get_experiment` method which takes an experiment name as an argument. 

    **When defining new experiments users have to create their own method for
    parameter creation and then add the experiment to ``__experiments``
    dictionary with selected custom name.**

    :cvar __experiment: Dictionary of experiments initialised in :func:`__init__`
    :vartype __experiment: dict[str, ExperimentParams]
    """

    __batch_dir = "./saves/experiment_runs/@1run_@2_@3_@4/"

    __experiments : typing.Dict[str, ExperimentParams]

    def __init__(self):
        self.__experiments = {}
        self.__experiments["exp10_TFS"] = self.exp10_TFS_spotlike()
        self.__experiments["exp11_TFS_spot"] = self.exp11_TFS_spotlike()
        self.__experiments["exp12_TFS_ant"] = self.exp12_TFS_ant()
        self.__experiments["exp11_SineFull_spot"] = self.exp11_SineFull_spotlike()
        self.__experiments["exp12_SineFull_ant"] = self.exp12_SineFull_ant()

        self.__experiments["exp2_body_para"] = self.exp2_body_para()
        self.__experiments["exp2_body_serial"] = self.exp2_body_serial()

        self.__experiments["neat_test"] = self.neat_test()

        self.load_saved_experiments()

    def __create_batch_dir(self, robot, agent, note) -> str:
        """Method for automating save directory creation.

        Args:
            robot (BaseRobot) : Selected robot type.
            agent (BaseAgent) : Selected agent type.
            note (str) : Optional custom note.

        Returns:
            str : Save path.
        """

        note += "_" if len(note) != 0 else ""
        batch_dir = self.__batch_dir.replace("@1", note).replace(
                                             "@2", type(robot).__name__).replace(
                                             "@3", type(agent).__name__).replace(
                                             "@4", str(time.time()))
        return batch_dir

    def __exp_start_note(self):
        print(f"Starting experiment - {sys._getframe(1).f_code.co_name}")

    def save_experiment(self, name, params):
        """Method for saving experiments.

        Method used while experiments need to be saved from :mod:`GUI` .
        Saving to preset folder.

        Args:
            name (str) : Selected name under which the experiment preset will
                be saved.
            params (ExperimentParams) : Instance of experiment parameters to be
                saved.
        """

        path = "experiment_params"
        if not os.path.exists(path):
            os.makedirs(path)

        with lzma.open(f"{path}/{name}.expp", "wb") as save_file:
            pickle.dump(params, save_file)

    def load_saved_experiments(self):
        """Method for loading saved experiments.

        Method used for loading previously saved experiments from :mod:`GUI` .
        """

        path = "experiment_params"
        if not os.path.exists(path): return 

        saved_experiments = [x[:-5] for x in os.listdir(path) if x.endswith(".expp")]

        for name in saved_experiments:
            with lzma.open(f"{path}/{name}.expp", "rb") as save_file:
                params = pickle.load(save_file)
                self.__experiments[name] = params

    def get_experiment_names(self):
        """Method for listing all created experiments.
        """

        return list(self.__experiments.keys());

    def get_experiment(self, name) -> ExperimentParams:
        """Method for accessing stored experiments by their selected name.

        Method for accessing stored experiments in dictionary. Assert tests
        name validity.

        Args:
            name (str) : Selected experiment name.
        """
        assert name in self.get_experiment_names(), f"Unknown experiment name `{name}` - list of created experiments {self.get_experiment_names()}"
        return copy.copy(self.__experiments[name])

    def exp10_TFS_spotlike(self, run=False):
        """Example of experiment.

        This method is an example of how and experiment can be created through
        this class.

        Args: 
            run (bool) : Flag argument stating if function returns ExperimentParams that should be run immediately.

        Returns:
            ExperimentParams : Created experiment parameters.

        """

        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "exp1.0"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=100,
                                  generation_count=500,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp11_TFS_spotlike(self, run=False):
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.1"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=100,
                                  generation_count=200,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp12_TFS_ant(self, run=False):
        robot = robots.AntV3()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.2"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=100,
                                  generation_count=100,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp11_SineFull_spotlike(self, run=False):
        robot = robots.SpotLike()
        agent = gaAgents.SineFuncFullAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.1"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=100,
                                  generation_count=200,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp12_SineFull_ant(self, run=False):
        robot = robots.AntV3()
        agent = gaAgents.SineFuncFullAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.2"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=100,
                                  generation_count=100,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp2_body_para(self, run=False):
        robot = robots.AntV3()
        agent = gaAgents.TFSAgent(robot, [(0.1, 0.5), (0.15,0.5), (0.1, 0.5), (0.15,0.5), (0.1, 0.5), (0.15,0.5), (0.1, 0.5), (0.15,0.5)], evo_type=gaAgents.EvoType.CONTROL_BODY_PARALEL)
        note = "exp2_body_para"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=150,
                                  generation_count=200,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp2_body_serial(self, run=False):
        robot = robots.AntV3()
        agent = gaAgents.TFSAgent(robot, [(0.1, 0.5), (0.15,0.5), (0.1, 0.5), (0.15,0.5), (0.1, 0.5), (0.15,0.5), (0.1, 0.5), (0.15,0.5)], evo_type=gaAgents.EvoType.CONTROL_BODY_SERIAL)
        note = "exp2_body_serial"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=150,
                                  generation_count=100,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def neat_test(self, run=False):
        robot = robots.InvertedDoublePendulum()
        agent = gaAgents.NEATAgent(robot, [])
        note = "neat_robots_test"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  population_size=100,
                                  generation_count=100,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params
