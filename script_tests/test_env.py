#!/usr/bin/env python

import gym
import gaAgent
import sys
import numpy as np

import os
import tempfile
import re

def simulationRun(agent, actions, render=False, render_start_paused=False):
    global step_cycle

    steps = -1
    individual_reward = 0
    done = False
    observation = env.reset()
    while not done:
        steps += 1

        action = agent.get_action(actions, steps)

        # https://www.youtube.com/watch?v=7Wm6vy7yBNA&ab_channel=BostonDynamics
        # action = np.zeros([len(action)])
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

        observation, reward, done, info = env.step(action)
        if render:
            env.render(start_paused=render_start_paused)

        if done:
            # sim end
            individual_reward = info["x_position"] # x-distance

    return individual_reward

###################################################
###################################################

from robots.robots import *

if __name__ == "__main__":
    robot = SpotLike()
    agent = gaAgent.TFSAgent(robot, [False for _ in range(len(robot.body_parts))])
    indiv = agent.generate_population(1)[0]

    file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")
    if agent.use_body_parts:
        robot.create(file, agent.body_part_mask, indiv)
    else:
        robot.create(file, agent.body_part_mask)

    env = gym.make('CustomAntLike-v1',
                    xml_file=file.name,
                    reset_noise_scale=0.0,
                    terminate_when_unhealthy=False)
    env._max_episode_steps = 500

    simulationRun(agent, indiv, render=True, render_start_paused=True)
    file.close()
    env.close()
    input()
    print("DONE")
