from utils import Classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
import json
from os import listdir, mkdir
import glob


def calc_traj_length_on_exp(exp_path):
    exp = Classes.AI_PDR_exp_w_SP_GT(exp_path)
    dL = exp.calc_dL(window_size=1)
    L = sum(dL)
    return L


def calc_traj_length_on_list(exp_path_list):
    exp_length_dic = {}
    N = len(exp_path_list)
    i = 1
    for exp_path in exp_path_list:
        exp_length_dic[exp_path] = calc_traj_length_on_exp(exp_path)
        print(str(round(i / N * 100, 2)) + '% completed')
        i += 1
    return exp_length_dic


def dic2arr(dic):
    """input is a dictoinary where all values are scalars"""
    keys = dic.keys()
    arr = np.array([])
    for key in keys:
        arr = np.hstack([arr, dic[key]])
    return arr


def calculate_traj_length_statistics(exp_length_dictionary, outputfolder):
    traj_lengths = dic2arr(exp_length_dictionary)
    plt.figure('traj_lengths.png')
    plt.hist(traj_lengths, 20)
    plt.grid(True)
    plt.savefig(join(outputfolder, 'traj_lengths.png'))
    plt.show()


if __name__ == '__main__':
    dir_to_analyze = r"C:\Eran\Onebox Sync Folder\Nav Projects\AI IMU\user walking direction\Datasets\dataset-ShenZhen"
    exp_path_list = glob.glob(dir_to_analyze + "/*.csv")
    # exp_names = listdir(dir_to_analyze)
    # for exp_name in exp_names:
    #     if '_AHRS_results.xlsx' in exp_name:
    #         exp_names.remove(exp_name)
    #     if 'json' in exp_name:
    #         exp_names.remove(exp_name)
    #     if '.png' in exp_name:
    #         exp_names.remove(exp_name)
    #     if '.txt' in exp_name:
    #         exp_names.remove(exp_name)
    # exp_path_list = [join(dir_to_analyze, exp_name) for exp_name in exp_names]
    N = len(exp_path_list)
    i = 1
    outputfolder = dir_to_analyze
    exp_length_dictionary = calc_traj_length_on_list(exp_path_list)
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    with open(join(outputfolder, 'trajectory_lengths.json'), 'w') as f:
        json.dump(exp_length_dictionary, f, indent=4)
        f.close()

    calculate_traj_length_statistics(exp_length_dictionary, outputfolder)