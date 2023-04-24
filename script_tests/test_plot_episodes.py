#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import argparse 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--open", default=False, const=True, nargs='?', type=str, help="Open saved history")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if not args.open:
        print("Set --open to select file to for plotting")
        quit()


    episode_history_list = [x for x in os.listdir(args.open) if "npy" in x]

    fig, ax = plt.subplots(2,3)

    # axs[0, 0].plot(x, y)
    # axs[0, 0].set_title('Axis [0, 0]')
    # axs[0, 1].plot(x, y, 'tab:orange')
    # axs[0, 1].set_title('Axis [0, 1]')
    # axs[1, 0].plot(x, -y, 'tab:green')
    # axs[1, 0].set_title('Axis [1, 0]')
    # axs[1, 1].plot(x, -y, 'tab:red')
    # axs[1, 1].set_title('Axis [1, 1]')

    for i in range(5):
        x_plot = i%3
        y_plot = i//3

        history = np.load(args.open+ "/" +episode_history_list[i])
        max = np.max(history, axis=1)

        s_history = np.sort(history, axis=1)

        X = np.arange(len(history))
        min = np.min(history, axis=1)
        max = np.max(history, axis=1)
        # mean = np.mean(history, axis=1)

        for i in range(len(history[0])//5):
            alpha = 10/(len(history[0])*2) # Modify the alpha value for each iteration.

            ax[y_plot,x_plot].fill_between(X, s_history[:,(i*5)], s_history[:,-((i*5)+1)], color='blue', alpha=alpha)

        ax[y_plot,x_plot].plot(X, min)
        ax[y_plot,x_plot].plot(X, max)
    # ax.plot(X, mean)

    plt.show()
