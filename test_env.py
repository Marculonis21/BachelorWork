#!/usr/bin/env python

import gym
import gaAgent
import sys

def simulationRun(agent, actions, render=False, render_start_paused=False):
    global step_cycle

    steps = -1
    individual_reward = 0
    done = False
    observation = env.reset()
    while not done:
        steps += 1

        action = agent.get_action(actions, steps)

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

    chosen_env = sys.argv[1]
    env = gym.make(chosen_env, reset_noise_scale=0.0)
    env._max_episode_steps = 10000

    if ("-h" in sys.argv or "--help" in sys.argv):
        printHelp()
        quit()

    # step_cycle = 25
    agent = gaAgent.FullRandomAgent(100, 4)
    indiv = agent.generate_population(1)[0]
    simulationRun(agent, indiv, render=True, render_start_paused=True)
    print("DONE")
