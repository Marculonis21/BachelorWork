#!/usr/bin/env python

import time
import random
import dask
import numpy as np
from dask.distributed import Client

import gym

def simCycle(id,actions):
    steps = -1
    individual_reward = 0
    done = False
    observation = env.reset()
    while not done:
        steps += 1

        action = actions[steps % 25]
        full_action = np.array([ action[0], action[1], action[2], action[3],
                                -action[0],-action[1],-action[2],-action[3]])

        observation, reward, done, info = env.step(full_action)

        if done:
            # sim end
            individual_reward = info["x_position"] # x-distance

    return (id,individual_reward)

if __name__ == "__main__":
    client = Client(n_workers=12,threads_per_worker=1,scheduler_port=0)
    # print(client)

    print("making env")
    env = gym.make('Ant-v3',
                terminate_when_unhealthy=False,
                reset_noise_scale=0.0)

    env._max_episode_steps = 1000
    print("done")

    np.random.seed(100)
    actionsList = [[(2*np.random.random(size=(25, 8//2,)) - 1) for _ in
                   range(25)] for _ in range(4)]

    start = time.time()
    results = []
    futures = []
    lazy = []

    print("sim")
    for _actionsList in actionsList:
        for id, actions in enumerate(_actionsList):
            # results.append(simCycle(id,actions))
            future = client.submit(simCycle, id,actions)
            futures.append(future)

    results = client.gather(futures)
    print(results)
    print("elapsed: ",time.time()-start)
    client.shutdown()
