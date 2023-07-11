#!/usr/bin/env python

import sys
import os
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)
import gymnasium as gym

import roboEvo

import numpy as np

import os
import tempfile
import re

def simulationRun(agent, individual):
    steps = -1
    individual_reward = 0
    env.reset()
    while True:
        if steps == 1:
            print("Press any key in console to start render")
            input()

        steps += 1
        action = agent.get_action(individual, steps).squeeze()

        # https://www.youtube.com/watch?v=7Wm6vy7yBNA&ab_channel=BostonDynamics
        action = np.zeros([len(action)])
        # lay down
        # action[0]  = -0.95
        # action[3]  = -0.95
        # action[6]  = -0.95
        # action[9]  = -0.95

        # action[1]  = 0.95
        # action[4]  = 0.95
        # action[7]  = 0.95
        # action[10] = 0.95

        # action[2]  = -1
        # action[5]  = -1
        # action[8]  = -1
        # action[11] = -1

        # stand up
        # action[1]  = -1
        # action[4]  = -1
        # action[7]  = -1
        # action[10] = -1

        # action[2]  = 1
        # action[5]  = 1
        # action[8]  = 1
        # action[11] = 1

        _, _, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            # sim end
            individual_reward = info["x_position"] # x-distance
            break

    return individual_reward

###################################################
###################################################

if __name__ == "__main__":
    robot = roboEvo.robots.Walker2D()
    agent = roboEvo.gaAgents.StepCycleFullAgent(robot, [])
    individual = agent.generate_population(1)[0]

    file = robot.create(agent.body_part_mask, individual)

    if file == None:
        env = gym.make(robot.environment_id, render_mode="human")
    else:
        env = gym.make(robot.environment_id, render_mode="human", xml_file=file.name)
        file.close()
    # env = roboEvo.TimeLimit(env, max_episode_steps=500)

    simulationRun(agent, individual)
    file.close()
    os.unlink(file.name)
    env.close()
    input()
    print("DONE")
