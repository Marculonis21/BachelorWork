import resources.robots.robots as robots
import resources.agents.gaAgents as gaAgents

class ExperimentParams:
    """
    Class for holding all experiment parameters that are used when setting up
    and running evolutionary algorithms.

    Args:
        robot (BaseRobot) : Selected robot instance.
        agent (BaseAgent) : Selected agent instance.
        population_size (int)  : Size of a population for evolutionary
            algorithm.
        generation_count (int) : Number of generations for evolutionary
            algorithm.
        show_best (bool) : Flag determining if best solution should be rendered
            after the experiment is done.
        save_best (bool) : Flag determining if best individual should be saved
            (now always True).
        save_dir (str) : Path of save dictionary (often auto-generated)
        show_graph (bool) : Flag determining if graph should be shown during
            experiment run.
        note (str) : Optional parameter used when saving experiment to
            customize folder name.
    """
    def __init__(self, 
                 robot:robots.BaseRobot, 
                 agent:gaAgents.BaseAgent,
                 population_size=100,
                 generation_count=150,
                 show_best=False,
                 save_best=True,
                 save_dir='./saves/experiment_runs/',
                 show_graph=True,
                 note=""):
        self.robot               = robot
        self.agent               = agent
        self.population_size     = population_size
        self.generation_count    = generation_count
        self.show_best           = show_best
        self.save_best           = save_best
        self.save_dir            = save_dir
        self.show_graph          = show_graph 
        self.note                = note
