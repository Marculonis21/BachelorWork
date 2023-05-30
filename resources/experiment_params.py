import resources.robots.robots as robots
import resources.agents.gaAgents as gaAgents
from enum import Enum

class ExperimentParams:
    def __init__(self, 
                 robot:robots.BaseRobot, 
                 agent:gaAgents.BaseAgent,
                 ga_population_size=100,
                 ga_generation_count=150,
                 show_best=False,
                 save_best=True,
                 save_dir='./saves/individuals',
                 pop_load_dir=None,
                 show_graph=True,
                 note=""):
        self.robot               = robot
        self.agent               = agent
        self.ga_population_size  = ga_population_size
        self.ga_generation_count = ga_generation_count
        self.show_best           = show_best
        self.save_best           = save_best
        self.save_dir            = save_dir
        self.show_graph          = show_graph 
        self.pop_load_dir        = pop_load_dir
        self.note                = note
