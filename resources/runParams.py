import resources.robots.robots as robots
import resources.gaAgents as gaAgents

class RunParams:
    def __init__(self, 
                 robot:robots.BaseRobot, 
                 agent:gaAgents.AgentType,
                 ga_population_size=100,
                 ga_generation_count=150,
                 show_best=False,
                 save_best=False,
                 save_dir='./saves/individuals',
                 note=""):
        self.robot               = robot
        self.agent               = agent
        self.ga_population_size  = ga_population_size
        self.ga_generation_count = ga_generation_count
        self.show_best           = show_best
        self.save_best           = save_best
        self.save_dir            = save_dir
        self.note                = note
