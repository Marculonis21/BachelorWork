#!/usr/bin/env python

import matplotlib.pyplot as plt

import numpy as np
import argparse 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--open", default=False, const=True, nargs='?', type=str, help="Open saved history")
parser.add_argument("--tick_step", default=10, nargs='?', type=int, help="Step in x ticks")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if not args.open:
        print("Set --open to select file to for plotting")
        quit()

    episode_history_list = [x for x in os.listdir(args.open) if "episode_history" in x]

    history = []

    for i in range(len(episode_history_list)):
        _data = np.load(args.open+ "/" +episode_history_list[i])
        history.append(_data.T)

    history = np.array(history)
    per_gen_values = history.reshape(-1,history.shape[2])

    per_gen_max = np.max(history, axis=1)

    tick_step = args.tick_step
    n_ticks = int(np.ceil(per_gen_values.shape[1]/tick_step))

    index = np.arange(n_ticks)*tick_step

    for i in range(len(history)):
        plt.scatter(index[-1]/10, per_gen_max[:, -1][i], s=25, c='r', marker="*", zorder=2)

    plt.boxplot(per_gen_values[:,index], positions=np.arange(n_ticks), zorder=1)
    plt.plot(np.max(per_gen_values[:,index], axis=0), label="max fitness", zorder=1)
    plt.plot(np.min(per_gen_values[:,index], axis=0), c='r', label="min fitness", zorder=1)
    plt.xticks(ticks=np.arange(n_ticks),labels=np.arange(n_ticks)*tick_step, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Generace", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.legend()
    plt.show()

