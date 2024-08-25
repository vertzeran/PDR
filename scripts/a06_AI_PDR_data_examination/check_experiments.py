import os
from os import listdir, mkdir
from os.path import join
import pandas as pd
import utils.Classes as Classes
import numpy as np
from datetime import datetime
import utils.Functions as Functions
import json

if __name__ == '__main__':
    dir_to_analyze = '/data/Datasets/Navigation/AI-PDR-dfx-record-frontAlign-printIMU-OriginalSixDof-ForTrain-IMUpose'
    person_list = listdir(dir_to_analyze)
    exp_path_list = listdir(dir_to_analyze)
    N = len(exp_path_list)
    i = 1
    res = {}
    for exp_name in exp_path_list:
        if '_AHRS_results.xlsx' not in exp_name:
            exp_path = join(dir_to_analyze, exp_name)
            exp = Classes.AI_PDR_exp_w_SP_GT(exp_path)
            mean_dt = np.mean(np.diff(exp.Time_IMU))
            std_dt = np.std(np.diff(exp.Time_IMU))
            max_dt = np.max(np.diff(exp.Time_IMU))
            res_temp = {"mean_dt": mean_dt, "std_dt": std_dt, "max_dt": max_dt}
            res[exp_path] = res_temp
        print(str(round(i / N * 100, 2)) + " % completed")
        i += 1
    with open(join(dir_to_analyze, 'dt_analysis.json'), "w") as outfile:
        json.dump(res, outfile, indent=4)