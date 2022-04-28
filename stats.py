#!/usr/bin/env python

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

# 256x256 grid of values
afm_data = np.loadtxt('./afm_test/afm.txt')

# scale to nanometers
afm_data *= (10**9)

def heatmap():
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)

    # Remove x and y ticks
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    # colormaps colors
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    img = ax.imshow(afm_data,
                    origin='lower',
                    cmap='YlGnBu_r', # _r = revers
                    extent=(0, 2, 0, 2),
                    vmin=0,
                    vmax=200)

    # Create axis for colorbar
    cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)

    cbar = fig.colorbar(mappable=img, cax=cbar_ax)

    # Edit colorbar ticks and labels
    cbar.set_ticks([0, 50, 100, 150, 200])
    cbar.set_ticklabels(['0', '50', '100', '150', '200 nm'])

    plt.show()

def plot():
    # Create figure and add axis
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111, projection='3d')

    # Remove gray panes and axis grid
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)

    # Remove z-axis
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])

    # Create meshgrid
    X, Y = np.meshgrid(np.linspace(0, 2, len(afm_data)), np.linspace(0, 2, len(afm_data)))
    plot = ax.plot_surface(X=X, Y=Y, Z=afm_data, cmap='YlGnBu_r', vmin=0, vmax=200)

    ## Adjust plot view
    ax.view_init(elev=50, azim=225)
    ax.dist=11

    ## Add colorbar
    cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
    cbar.set_ticks([0, 50, 100, 150, 200])
    cbar.set_ticklabels(['0', '50', '100', '150', '200 nm'])

    ## Set tick marks
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    # ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))

    # Set axis labels
    # ax.set_xlabel(r'$\mathregular{\mu}$m', labelpad=20)
    # ax.set_ylabel(r'$\mathregular{\mu}$m', labelpad=20)

    ## Set z-limit
    ax.set_zlim(50, 200)

    ## Save and show figure
    ## plt.savefig('afm_3d_plot.png', dpi=100, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # heatmap()
    plot()
