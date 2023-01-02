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
        # action = -np.ones([len(action)])

        observation, reward, done, info = env.step(action)
        if render:
            env.render(start_paused=render_start_paused)

        if done:
            # sim end
            individual_reward = info["x_position"] # x-distance

    return individual_reward

###################################################
###################################################

def printHelp():
    return
    # print("-h --help       ... Print help")
    # print("-o <individual> ... Select input file to play")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("not enough params")
        quit()

    from robots.robots import StickAnt

    sa = StickAnt()

    temp_file = tempfile.NamedTemporaryFile(mode="w",suffix=".xml",prefix="GArobot_")
    sa.create(temp_file, adjustment=[], adjustment_mask=[0,0,0,0])

    env = gym.make('CustomAntLike-v1', 
                   xml_file=temp_file.name,
                   reset_noise_scale=0.0)

    env._max_episode_steps = 10000

    if ("-h" in sys.argv or "--help" in sys.argv):
        printHelp()
        quit()

    # step_cycle = 25
    # agent = gaAgent.FullRandomAgent(100, 8) # Ant-v3
    agent = gaAgent.FullRandomAgent(100, 4, [1,1,0,0])
    indiv = agent.generate_population(1)[0]
    agent.mutation([indiv])
    simulationRun(agent, indiv, render=True, render_start_paused=True)
    temp_file.close()
    quit()
    print("DONE")
