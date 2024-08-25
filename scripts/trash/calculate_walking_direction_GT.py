"""
This script plot the GT data of a given experiment,
and then also plots the trajectory that will be estimated
by a perfect estimator.
Since that the estimated dl (of ~1sec movment) assums that the human 
had walked in a stright line, rather than a curved line,
we expect to see some differance as the window_size is getting bigger
"""

import utils.Classes as Classes
import matplotlib.pyplot as plt
import numpy as np
from utils.Functions import construct_traj


if __name__ == '__main__':
    #user inputs:
    exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Train/hang_leg1.csv'
    t_start = 0
    t_stop = 60
    window_size = 250

    # loading experiment:
    Exp = Classes.RidiExp(exp_path) 
    # clipping to [t_start,t_stop]:
    Exp.SegmentScenario([t_start, t_stop]) 
    # build the GT trajectory:
    pos_gt = Exp.Pos.arr()
    pos_gt = pos_gt - pos_gt[0]
    # build the trajectory that a perfect estimator will provide:
    dL = Exp.calc_dL(window_size)
    walking_angle = Exp.calc_walking_direction(window_size)[0]
    traj = construct_traj(dL, walking_angle)

    # Plot both walking angles:
    fig = plt.figure('walking angle')
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel=r"$idx$", ylabel=r"$rad$", title="walking angle"), ax.grid(True)
    ax.plot(walking_angle, color='blue', linestyle='-', linewidth=2, label='')
    
    # Plot both trajectories:
    fig = plt.figure('position plot')
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
    ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='')
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], color='black', linestyle='--', linewidth=2, label='')
    plt.show()