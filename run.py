#!/usr/bin/env python

import sys
import gym
import numpy as np

max_steps = int(sys.argv[1])
individual_number = sys.argv[2]
individual = np.load(f"individuals/{individual_number}.npy")

env = gym.make('Ant-v3',
               reset_noise_scale=0.0)

env._max_episode_steps = max_steps

individual_reward = 0
steps = -1
done = False
observation = env.reset()

while not done:
    steps += 1
    observation, reward, done, info = env.step(individual[steps].reshape(8,))

    if done:
        # sim end
        individual_reward = info["x_position"] # x-distance
        if info["is_flipped"]:
            individual_reward = 0

print(f"{individual_number}-{individual_reward}")
