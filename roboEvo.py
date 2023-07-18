#!/usr/bin/env python
"""Central module of the library used for running evolutionary algorithm.

This module is the central part of the library with which all the user interfaces 
communicate and it is then used for starting and running experiments with 
evolutionary algorithms.

Usage example::

    import roboEvo
    ...
    params = # GET EXPERIMENT PARAMETERS
    roboEvo.run_experiment(params)

Special global variables:
    This module works with a few global variables used for communication 
    between GUI and running experiments to enable drawing progress graphs.

    Variables:
        :GUI_GEN_NUMBER: (*int*) : Number of generations that EA has finished.
        :GUI_PREVIEW: (*bool*) : Flag raised when GUI wants to run a preview of best individual so far.
        :GUI_ABORT: (*bool*) : Flag raised when GUI wants to abort EA.
        :EPISODE_HISTORY: (*list[list[float]]*) : List of fitness values of all generations through the EA runtime. Saved at the end of the experiment and used for GUI graph drawing.
        :LAST_POP: (*list*) : Copy of all individuals from the last generation. Saved at the end of the experiment.
"""

# creating and registering custom environment
import resources.agents.gymnasiumCustomEnv as _

# import all modules to be reachable from central roboEvo
import resources.agents.gaAgents as gaAgents
import resources.robots.robots as robots
from resources.experiment_params import ExperimentParams

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit

import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import multiprocessing
import time
import os
import glfw

GUI_GEN_NUMBER = 0

GUI_PREVIEW = False
GUI_ABORT = False 

EPISODE_HISTORY = []
LAST_POP = None

def _simulation_run(env : gym.Env, agent : gaAgents.BaseAgent, individual, render=False) -> float:
    """Performs run in simulated environment.

    This function performs a single run in the simulated environment of a
    selected individual in specifie environment - used for evaluating and
    rendering individuals.

    Args:

        env (Env) : Selected Farama Gymnasium environment.
        agent (BaseAgent) : Selected agent type.
        individual : Genotype of an individual from EA.
        render (bool) : Optional flag, set to True if rendering of the run is desired.

    Returns:
        float : Reward of the individual from the environment.
    """

    steps = -1
    individual_reward = 0
    terminated = False
    truncated = False

    obs = env.reset()
    while True:
        steps += 1
        action = agent.get_action(individual, steps).squeeze()
        
        # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = env.step(action)

        """
        Calculating an intricate reward might need the user to redefine what
        information is returned from the custom environment.
        """
        if terminated or truncated: # calculate reward after finishing last step
            individual_reward = (info["x_position"]-0.5*abs(info["y_position"]))
            break

    if render:
        env.reset()
        env.close()

    # Optionally for future - return REWARD + some colected information ??? 
    return individual_reward

def render_run(agent : gaAgents.BaseAgent, robot : robots.BaseRobot, individual):
    """Function for rendering the simulation run.

    Specialized function used when rendering of the simulation run is desired.

    Args:
        agent (BaseAgent) : Selected agent type.
        robot (BaseRobot) : Selected robot type.
        individual : Genotype of an individual from EA.

    Returns:
        float : Reward of the individual from the simulation.

    See also:
        :func:`_simulation_run` : Function used for running the simulation.
    """

    run_reward = -1 
    file = robot.create(agent.body_part_mask, individual)
    file.close()
    try:
        env = gym.make(id=robot.environment_id,
                       xml_file=file.name,
                       reset_noise_scale=0.0,
                       disable_env_checker=True,
                       render_mode="human")
        env = TimeLimit(env, max_episode_steps=500)

        print("PREPARED FOR RENDERING... Press ENTER to start")
        input()
        run_reward = _simulation_run(env, agent, individual, render=True)
    finally:
        file.close()
        os.unlink(file.name)

    return run_reward

def _run_evolution(params : ExperimentParams, client : Client, load_population=[], gui=False, debug=False):
    """Function for setting up and running evolutionary algorithm.

    This function takes parameters from selected experiment and runs the
    evolution with parameters described by the experiment parameters.

    Args:
        params (ExperimentParams) : Class of multiple experiment parameters used in experiments.
        client (Client) : Client from Dask library, used for running evaluations in paralel. 
        load_population (list) : Optional population parameter to be loaded in instead of creating entirely new population. Used with morphology evolution.
        gui (bool) : Flag showing that module was started through GUI.
        debug (bool) : Debug flag. Used for testing features.

    Returns:
        Tuple[`best_individual`, Env] : A tuple of `best_individual` genome and environment used to run evaluation of the best individual.
    """

    global EPISODE_HISTORY, LAST_POP, GUI_GEN_NUMBER, GUI_PREVIEW

    population = params.agent.generate_population(params.population_size)

    # for serial conrol + body evolution ONLY
    if load_population != []:
        # reassign actions from the loaded population - keep only NEWLY GENERATED BODY VALUES
        for i in range(len(population)):
            population[i][0] = load_population[i][0] 
    # PROOF of concept: working population loading

    robot_source_files = []
    environments = []

    # create individual source files for different bodies (if needed - body parts evolutions)
    for i in range(len(population)):
        file = params.robot.create(params.agent.body_part_mask, population[i])
        file.close()

        robot_source_files.append(file)

        env = gym.make(id=params.robot.environment_id,
                       xml_file=file.name,
                       reset_noise_scale=0.0,
                       disable_env_checker=True,
                       render_mode=None)
        env = TimeLimit(env, max_episode_steps=500)

        environments.append(env)

        if not params.agent.evolve_body: # there exists only 1 environment if we don't need more!
            break

    best_individual = None 
    best_individual_env = None 

    for generations in range(params.generation_count+1):
        if GUI_ABORT:
            break

        GUI_GEN_NUMBER = generations

        if GUI_PREVIEW: # GUI FORCED PREVIEW
            run_reward = render_run(params.agent, params.robot, best_individual)
            print("Run reward:", run_reward)
            GUI_PREVIEW = False

        # Get fitness values
        fitness_values = []
        futures = []
        if debug:
            for index, individual in enumerate(population):
                fitness_values.append(_simulation_run(environments[index if params.agent.evolve_body else 0], params.agent, individual))
        else:
            for index, individual in enumerate(population):
                futures.append(client.submit(_simulation_run, environments[index if params.agent.evolve_body else 0], params.agent, individual))

            fitness_values = client.gather(futures)

        # for statistical reconstruction later
        EPISODE_HISTORY.append(fitness_values)
        if generations % 5 == 0: # frequency of info update

            if not gui:
                print("Best fitness: ", max(fitness_values))
                if params.show_graph:
                    plt.cla()
                    plt.title('Training')
                    plt.xlabel('5 Generations')
                    plt.ylabel('Fitness')
                    plt.plot(np.mean(EPISODE_HISTORY,axis=1), label='Mean')
                    plt.plot(np.min(EPISODE_HISTORY, axis=1), label='Min')
                    plt.plot(np.max(EPISODE_HISTORY, axis=1), label='Max')
                    plt.legend(loc='upper left', fontsize=9)
                    plt.tight_layout()
                    plt.pause(0.1)

        # get index of the best individuals for elitism
        elitism_count = int(params.population_size*0.10) # 10% of pop size

        fitness_values = np.array(fitness_values)
        top_indices = np.argpartition(fitness_values, -elitism_count)[-elitism_count:]
        sorted_top_indices = top_indices[np.argsort(fitness_values[top_indices])[::-1]]

        elite_individuals = []
        elite_individuals_envs = []
        for top_i in sorted_top_indices:
            elite_individuals.append(copy.deepcopy(population[top_i]))
            elite_individuals_envs.append(copy.deepcopy(environments[top_i if params.agent.evolve_body else 0]))

        best_individual = copy.deepcopy(elite_individuals[0])
        best_individual_env = copy.deepcopy(elite_individuals_envs[0])

        # Genetic operators - selection, crossover, mutation
        parents          = params.agent.selection(population,fitness_values)
        children         = params.agent.crossover(parents)
        mutated_children = params.agent.mutation(children)
        population = mutated_children

        # apply elitism - transfer top individuals
        for i in range(len(elite_individuals)):
            population[i] = elite_individuals[i]

        # change environments based on possible robot morphology changes
        for i in range(len(population)):
            environments[i].close()
            params.robot.create(params.agent.body_part_mask, population[i], tmp_file=robot_source_files[i])
            robot_source_files[i].close()

            if generations != params.generation_count:
                env = gym.make(id=params.robot.environment_id,
                               xml_file=robot_source_files[i].name,
                               reset_noise_scale=0.0,
                               disable_env_checker=True,
                               render_mode=None)
                env = TimeLimit(env, max_episode_steps=500)
                environments[i] = env

            if not params.agent.evolve_body:
                break

    # after evo tmp files cleanup
    for file in robot_source_files:
        file.close()
        os.unlink(file.name)

    if not gui:
        plt.close()

    # CONTINUE EVO = serial control + body evolution - switching vars to body evo
    if params.agent.continue_evo: 
        print("Evo type change")
        params.agent.switch_evo_phase()
        best_individual, best_individual_env = _run_evolution(params, client, population, gui, debug)
    else:
        LAST_POP = copy.deepcopy(population)

    return best_individual, best_individual_env

################################################################################
################################################################################

def run_experiment(params : ExperimentParams, gui=False, debug=False):
    """Main input function of the module.

    This function acts as a main input function from which both of the input
    interfaces start their experiments. It can also be used by the user who
    would possibly like to use this module connected to his own library. 

    This function takes experiment parameters and then takes care of starting
    the correct evolutionary algorithm and saving all of the data from the
    algorithm progress.

    Args:
        params (ExperimentParams) : Parameters of the selected experiment.
        gui (bool) : Flag showing if the method was called from the GUI.
        debug (bool) : Optional debug flag, used for debugging and testing.
    """

    global EPISODE_HISTORY
    EPISODE_HISTORY = []

    _params = copy.deepcopy(params)
    
    if isinstance(_params.agent, gaAgents.NEATAgent): # override EA with NEAT
        _params.agent.evolution_override(_params, EPISODE_HISTORY)
        return

    # Start threading client - use a bit less cpu threads than MAX! 
    # (GUI can have own thread)
    client = Client(n_workers=multiprocessing.cpu_count()-2,threads_per_worker=1,scheduler_port=0)

    # Run evolution
    try:
        print("RUNNING EVOLUTION")
        best_individual, best_individual_env = _run_evolution(_params,
                                                              client,
                                                              gui=gui,
                                                              debug=debug)
        print("EVOLUTION DONE")
    finally:
        client.close()

    # Set final reward (render best if set)
    best_reward = 0
    if _params.show_best: best_reward = render_run(_params.agent, _params.robot, best_individual)
    else:                 best_reward = _simulation_run(best_individual_env, _params.agent, best_individual)

    print("Saving data...")
    current_time = time.time()

    if _params.note != "":
        _params.note = _params.note + "_"

    if gui:
        folder_name = f"/{_params.note}run_{type(_params.robot).__name__}_{type(_params.agent).__name__}_{current_time}"
        _params.save_dir = _params.save_dir+folder_name

    # File saving
    if not os.path.exists(_params.save_dir):
        os.makedirs(_params.save_dir)

    if _params.save_best:
        gaAgents.BaseAgent.save(_params.agent, _params.robot, best_individual, _params.save_dir + f"/{_params.note}individual_run{current_time}_rew{best_reward}.save")

    episode_history = np.array(EPISODE_HISTORY)
    last_population = np.array(LAST_POP, dtype=object)
    np.save(_params.save_dir + f"/{_params.note}episode_history{current_time}", episode_history)
    np.save(_params.save_dir + f"/{_params.note}last_population{current_time}", last_population)

if __name__ == "__main__":
    pass
