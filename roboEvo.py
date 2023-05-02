#!/usr/bin/env python

# creating and registering custom environment
import resources.gymnasiumCustomEnv as _

import resources.gaAgents as gaAgents
import resources.robots.robots as robots
from resources.experiment_params import ExperimentParams

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
import matplotlib.pyplot as plt
import numpy as np
import copy
from dask.distributed import Client
import time
import tempfile
import os

GUI_FLAG = False
GUI_FITNESS = []
GUI_GEN_NUMBER = 0

GUI_PREVIEW = False
GUI_ABORT = False 

GRAPH_VALUES = [[],[],[]]
EPISODE_HISTORY = []
LAST_POP = None

def raise_preview():
    global GUI_PREVIEW
    GUI_PREVIEW = True

def raise_abort():
    global GUI_ABORT
    GUI_ABORT = True

def __simulation_run(env, agent, individual, render=False):
    steps = -1
    individual_reward = 0
    terminated = False
    truncated = False

    env.reset()
    while True:
        steps += 1
        action = agent.get_action(individual, steps).squeeze()
        
        # obs, reward, terminated, truncated, info
        _, _, terminated, truncated, info = env.step(action)

        if render:
            env.render()

        """
        Calculating an intricate reward might task user with redefining what
        information is returned from the custom environment returns - which
        shouldn't (is not) that hard.
        """
        if terminated or truncated: # calculate reward after finishing last step
            individual_reward = (info["x_position"]-0.5*abs(info["y_position"]))
            break

    # Optionally for future - return REWARD + some colected information ??? 
    return individual_reward

def render_run(agent, robot, individual):
    run_reward = -1 
    file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")
    try:
        robot.create(file, agent.body_part_mask, individual)

        env = gym.make(id=robot.environment_id,
                       xml_file=file.name,
                       reset_noise_scale=0.0,
                       disable_env_checker=True,
                       render_mode="human")
        env = TimeLimit(env, max_episode_steps=500)

        run_reward = __simulation_run(env, agent, individual, render=True)

        env.close()
    finally:
        file.close()

    return run_reward

def __run_evolution(robot, agent, client, generation_count, population_size, show_graph, debug=False):
    global GRAPH_VALUES, EPISODE_HISTORY, LAST_POP
    global GUI_GEN_NUMBER, GUI_FITNESS, GUI_PREVIEW

    GUI_FITNESS = []
    GRAPH_VALUES = [[],[],[]]
    EPISODE_HISTORY = []

    population = agent.generate_population(population_size)
    robot_source_files = []
        
    environments = []

    # create individual source files for different bodies (if needed - body parts evolutions)
    for i in range(len(population)):
        file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")

        robot.create(file, agent.body_part_mask, population[i])

        robot_source_files.append(file)

        env = gym.make(id=robot.environment_id,
                       xml_file=file.name,
                       reset_noise_scale=0.0,
                       disable_env_checker=True,
                       render_mode=None)
        env = TimeLimit(env, max_episode_steps=500)

        environments.append(env)

        if not agent.use_body_parts: # there exists only 1 environment if we don't need more!
            break

    best_individual = None 
    best_individual_env = None 

    for generations in range(generation_count+1):
        if GUI_ABORT:
            break

        GUI_GEN_NUMBER = generations

        if GUI_PREVIEW: # GUI FORCED PREVIEW
            render_run(agent, robot, best_individual)
            GUI_PREVIEW = False

        # Get fitness values
        fitness_values = []
        futures = []
        if debug:
            for index, individual in enumerate(population):
                fitness_values.append(__simulation_run(environments[index if agent.use_body_parts else 0], agent, individual))
        else:
            for index, individual in enumerate(population):
                futures.append(client.submit(__simulation_run, environments[index if agent.use_body_parts else 0], agent, individual))

            fitness_values = client.gather(futures)

        if GUI_FLAG:
            GUI_FITNESS = [np.mean(fitness_values), 
                           min(fitness_values),
                           max(fitness_values)]

        # for statistical reconstruction later
        EPISODE_HISTORY.append(fitness_values)
        if generations % 5 == 0: # frequency of info update
            GRAPH_VALUES[0].append(np.mean(fitness_values))
            GRAPH_VALUES[1].append(min(fitness_values))
            GRAPH_VALUES[2].append(max(fitness_values))

            if not GUI_FLAG:
                print("Best fitness: ", max(fitness_values))
                if show_graph:
                    plt.cla()
                    plt.title('Training')
                    plt.xlabel('5 Generations')
                    plt.ylabel('Fitness')
                    plt.plot(GRAPH_VALUES[0], label='Mean')
                    plt.plot(GRAPH_VALUES[1], label='Min')
                    plt.plot(GRAPH_VALUES[2], label='Max')
                    plt.legend(loc='upper left', fontsize=9)
                    plt.tight_layout()
                    plt.pause(0.1)

        # get index of the best individuals for elitism
        elitism_count = int(population_size*0.10) # 10% of pop size

        fitness_values = np.array(fitness_values)
        top_indices = np.argpartition(fitness_values, -elitism_count)[-elitism_count:]
        sorted_top_indices = top_indices[np.argsort(fitness_values[top_indices])[::-1]]

        elite_individuals = []
        elite_individuals_envs = []
        for top_i in sorted_top_indices:
            elite_individuals.append(copy.deepcopy(population[top_i]))
            elite_individuals_envs.append(copy.deepcopy(environments[top_i if agent.use_body_parts else 0]))

        best_individual = copy.deepcopy(elite_individuals[0])
        best_individual_env = copy.deepcopy(elite_individuals_envs[0])

        # Genetic operators - selection, crossover, mutation
        parents          = agent.selection(population,fitness_values)
        children         = agent.crossover(parents)
        mutated_children = agent.mutation(children)
        population = mutated_children

        # change environments based on possible robot morphology changes
        for i in range(len(population)):
            environments[i].close()
            robot.create(robot_source_files[i], agent.body_part_mask, population[i])

            if generations != generation_count:
                env = gym.make(id=robot.environment_id,
                               xml_file=robot_source_files[i].name,
                               reset_noise_scale=0.0,
                               disable_env_checker=True,
                               render_mode=None)
                env = TimeLimit(env, max_episode_steps=500)
                environments[i] = env

            if not agent.use_body_parts:
                break

        # apply elitism
        for i in range(len(elite_individuals)):
            population[i] = elite_individuals[i]
            if agent.use_body_parts:
                environments[i] = elite_individuals_envs[i]

    # after evo close tmp files
    for file in robot_source_files:
        file.close()

    if not GUI_FLAG:
        plt.close()

    LAST_POP = copy.deepcopy(population)

    return best_individual, best_individual_env

################################################################################
################################################################################

def run_experiment(params:ExperimentParams, gui=False, debug=False):
    global GUI_FLAG
    GUI_FLAG = gui

    # Start threading client
    client = Client(n_workers=11,threads_per_worker=1,scheduler_port=0)

    # Run evolution
    try:
        print("RUNNING EVOLUTION")
        best_individual, best_individual_env = __run_evolution(params.robot, 
                                                               params.agent,
                                                               client,
                                                               generation_count=params.ga_generation_count, 
                                                               population_size=params.ga_population_size, 
                                                               show_graph=params.show_graph,
                                                               debug=debug)
        print("EVOLUTION DONE")
    finally:
        client.close()

    # Set final reward (render best if set)
    best_reward = 0
    if params.show_best: best_reward = render_run(params.agent, params.robot, best_individual)
    else:                best_reward = __simulation_run(best_individual_env, params.agent, best_individual, render=False)

    # File saving
    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)

    current_time = time.time()
    if params.save_best:
        gaAgents.BaseAgent.save(params.agent, params.robot, best_individual, params.save_dir + f"/{params.note}_individual_run{current_time}_rew{best_reward}.save")

    episode_history = np.array(EPISODE_HISTORY)
    last_population = np.array(LAST_POP, dtype=object)
    np.save(params.save_dir + f"/{params.note}_episode_history{current_time}", episode_history)
    np.save(params.save_dir + f"/{params.note}_last_population{current_time}", last_population)

