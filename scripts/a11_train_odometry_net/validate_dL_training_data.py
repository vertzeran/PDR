import os
import utils.Classes as Classes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.Functions as Functions
import random
from os import listdir
from os.path import join
from utils.Functions import construct_traj
from scripts.a11_train_odometry_net.process_data_for_training import get_segments_from_exp, get_exp_list, \
    rotate_traj_to_minimize_error, get_train_test_dir, get_XY_from_exp_list, get_segments_for_WDE, get_xy_pairs

import torch
import json
#from create_segments_for_WDE_SBG import get_dir_to_analyze, get_exp_list, rotate_traj_to_minimize_error
import scipy.io as sio


if __name__ == '__main__':
    WDE_opt_folder = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-03-16T14:49:12.177262optimization results on RIDI & SZ combined'
    exp_name = 'hao_leg2.csv'
    _, test_dir = get_train_test_dir(dataset='RIDI_ENU', mode='mixed')
    exp_path = join(test_dir, exp_name)
    main_dir = "/home/maint/git/walking_direction_estimation/"
    params_path = join(main_dir,
                       'scripts/a11_train_odometry_net/params.json')  # 'scripts/a7_training_on_SZ_dataset/params.json'
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    params = params["RIDI_params"]
    if params["dataset"] == 'AI_PDR':
        exp = Classes.AI_PDR_exp_w_SP_GT(exp_path)
    elif params["dataset"] == 'RIDI_ENU':
        exp = Classes.RidiExp_ENU(exp_path)
    else:
        raise 'invalid dataset'

    if params["traj_length_limit"] is not None:
        exp.limit_traj_length(params["traj_length_limit"])
    exp.define_walking_start_idx(th=params["walking_start_threshold"])
    exp_segments, exp_AHRS_results_segments, initial_WD_angles_segment = \
        get_segments_from_exp(exp, params, sample_id=0, use_GT_att=False)
    processed_exp_segments = get_segments_for_WDE(exp_segments, exp_AHRS_results_segments)
    X, Y1, Y2 = get_xy_pairs(params["window_size"], processed_exp_segments,
                                         add_quat=params["add_quat"], heading_angle_fix=False)
    walking_angle = []
    for y2 in Y2:
        walking_angle.append(np.arctan2(y2[1], y2[0]))
    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    traj = construct_traj(dl=np.array(Y1), walking_angle=np.array(walking_angle), plot_result=False, pos_gt=gt_pos)
    fig = plt.figure('construct_traj plot')
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
    ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='')
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='black', linestyle='--', linewidth=2, label='')
    plt.show()
