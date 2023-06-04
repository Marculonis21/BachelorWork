#!/usr/bin/env python

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
    `RunEvolution` method are accessed as via class methods or via
    `get_experiment` method which takes an experiment name as an argument. 

    When defining new experiments users need to create their own method for
    parameter creation and then add the experiment to `__experiments`
    dictionary with selected custom name.
    """

    __batch_dir = "./saves/batch_runs/@1_run_@2_@3_@4/"
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

        self.__experiments["test_exp"] = self.test_exp_AntV3()

        self.__experiments["exp_BODYTEST"] = self.exp_BODYTEST()

        self.__experiments["a"] = self.exp_a()
        self.__experiments["b"] = self.exp_b()
        self.__experiments["c"] = self.exp_c()
        self.__experiments["d"] = self.exp_d()

        self.__experiments["neat_test"] = self.neat_test()

        self.load_saved_experiments()

    def __create_batch_dir(self, robot, agent, note):
        batch_dir = self.__batch_dir.replace("@1", note).replace(
                                             "@2", type(robot).__name__).replace(
                                             "@3", type(agent).__name__).replace(
                                             "@4", str(time.time()))
        return batch_dir

    def __exp_start_note(self):
        print(f"Starting experiment - {sys._getframe(1).f_code.co_name}")

    def save_experiment(self, name, params):
        path = "experiment_params"
        if not os.path.exists(path):
            os.makedirs(path)

        with lzma.open(f"{path}/{name}.expp", "wb") as save_file:
            pickle.dump(params, save_file)

    def load_saved_experiments(self):
        path = "experiment_params"
        if not os.path.exists(path): return 

        saved_experiments = [x[:-5] for x in os.listdir(path) if x.endswith(".expp")]
        print(saved_experiments)

        for name in saved_experiments:
            with lzma.open(f"{path}/{name}.expp", "rb") as save_file:
                params = pickle.load(save_file)
                self.__experiments[name] = params
                print(f"Loaded exp: {name} - {path}/{name}.expp")

    def get_experiment_names(self):
        return list(self.__experiments.keys());

    def get_experiment(self, name):
        assert name in self.get_experiment_names(), f"Unknown experiment name `{name}` - list of created experiments {self.get_experiment_names()}"
        return copy.copy(self.__experiments[name])

    def neat_test(self, run=False):
        robot = robots.Pendulum()
        agent = gaAgents.NEATAgent(robot, [])
        note = "neat_robots_test"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  ga_population_size=150,
                                  ga_generation_count=100,
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
                                  ga_population_size=150,
                                  ga_generation_count=200,
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
                                  ga_population_size=150,
                                  ga_generation_count=100,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp_a(self, run=False):
        robot = robots.StickAnt()
        agent = gaAgents.SineFuncFullAgent(robot, [(0.1, 0.5)]*len(robot.body_parts), evo_type=gaAgents.EvoType.CONTROL_BODY_SERIAL)
        note = "exp_a"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  ga_population_size=10,
                                  ga_generation_count=10,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp_b(self, run=False):
        robot = robots.StickAnt()
        agent = gaAgents.TFSAgent(robot, [(0.1, 0.5)]*len(robot.body_parts), evo_type=gaAgents.EvoType.CONTROL_BODY_SERIAL)
        note = "exp_b"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  ga_population_size=10,
                                  ga_generation_count=10,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp_c(self, run=False):
        robot = robots.StickAnt()
        agent = gaAgents.SineFuncHalfAgent(robot, [(0.1, 0.5)]*len(robot.body_parts), evo_type=gaAgents.EvoType.CONTROL_BODY_SERIAL)
        note = "exp_c"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  ga_population_size=10,
                                  ga_generation_count=10,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp_d(self, run=False):
        robot = robots.StickAnt()
        agent = gaAgents.StepCycleHalfAgent(robot, [(1.1, 0.5)]*len(robot.body_parts), evo_type=gaAgents.EvoType.CONTROL_BODY_SERIAL)
        note = "exp_d"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  ga_population_size=10,
                                  ga_generation_count=10,
                                  show_best=False,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def test_exp_AntV3(self, run=False):
        robot = robots.AntV3()
        agent = gaAgents.SineFuncFullAgent(robot, [(0.1, 0.5),(0.15,0,5), (0.1, 0.5),(0.15,0,5), (0.1, 0.5),(0.15,0,5), (0.1, 0.5),(0.15,0,5)], evo_type=gaAgents.EvoType.CONTROL)
        # agent = gaAgents.TFSAgent(robot, [(0.1, 0.5),(0.15,0,5), (0.1, 0.5),(0.15,0,5), (0.1, 0.5),(0.15,0,5), (0.1, 0.5),(0.15,0,5)], evo_type=gaAgents.EvoType.CONTROL)
        note = "test_exp"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  ga_population_size=100,
                                  ga_generation_count=100,
                                  show_best=True,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params


    def exp_BODYTEST(self, run=False):
        robot = robots.StickAnt()
        agent = gaAgents.TFSAgent(robot, [False, False, (1,2), False], gaAgents.EvoType.CONTROL_BODY_SERIAL)
        note = "serial_body_test"
        
        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
                                  agent,
                                  ga_population_size=8,
                                  ga_generation_count=50,
                                  show_best=True,
                                  save_best=True,
                                  save_dir=batch_dir,
                                  note="")

        if run: # print note before starting experiment
            self.__exp_start_note()

        return params

    def exp10_TFS_spotlike(self, run=False):
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "exp1.0"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
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

    def exp11_TFS_spotlike(self, run=False):
        robot = robots.SpotLike()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.1"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
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

    def exp12_TFS_ant(self, run=False):
        robot = robots.AntV3()
        agent = gaAgents.TFSAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.2"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
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

    def exp11_SineFull_spotlike(self, run=False):
        robot = robots.SpotLike()
        agent = gaAgents.SineFuncFullAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.1"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
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

    def exp12_SineFull_ant(self, run=False):
        robot = robots.AntV3()
        agent = gaAgents.SineFuncFullAgent(robot, [False]*len(robot.body_parts), gaAgents.EvoType.CONTROL)
        note = "Exp1.2"

        batch_dir = self.__create_batch_dir(robot, agent, note)

        params = ExperimentParams(robot, 
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
