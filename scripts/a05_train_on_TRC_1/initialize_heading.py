import os

import matplotlib.pyplot as plt
import numpy as np

from test_on_full_exp import*
from create_segments_for_WDE_SBG import initialize_heading_on_exp
import random
from os import listdir
from os.path import join


def get_person_list(dir_to_analyze):
    person_list = listdir(dir_to_analyze)
    for i in range(len(person_list)):
        person_list[i] = join(dir_to_analyze, person_list[i])
    return person_list


def check_PCA_performnce_on_exp(exp=None, window_size=200, limit_exp_length=False, initialization_time=None):
    exp.define_walking_start_idx(th=1)
    if limit_exp_length:
        # varify round number of window sizes
        assert initialization_time is not None
        start_time = exp.Time_IMU[exp.index_of_walking_start]
        start_idx = abs(exp.Time_IMU - start_time).argmin()
        stop_time = exp.Time_IMU[exp.index_of_walking_start] + initialization_time
        stop_idx = abs(exp.Time_IMU - stop_time).argmin()
        stop_idx = start_idx + window_size * ((stop_idx - start_idx) / window_size).__floor__()
        stop_time = exp.Time_IMU[stop_idx]
        exp.SegmentScenario([exp.Time_IMU[0],
                             stop_time])
    # this code is taken from traj_est_using_PCA. it is duplicated because we need to keep local parameteres
    segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=exp,
                                                                                    window_size=window_size,
                                                                                    use_GT_att=False,
                                                                                    wind_size_for_heading_init=750)
    analyzied_segments = []
    errors = []
    for i in range(len(segment_list)):
        segment = segment_list[i]
        AHRS_results = AHRS_results_list[i]
        analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=False,
                                                                   lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
                                                                   Rot_est=AHRS_results.Rot,
                                                                   grv_est=AHRS_results.grv,
                                                                   Heading_est=AHRS_results.heading))
        analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        WD_angle_err, end_pos_err = analyzied_segments[-1].calc_error()
        errors.append(end_pos_err)
    return np.mean(np.array(errors))

if __name__ == '__main__':
    exp_path = '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_09_15_swing_zeev_R/outdoor_output_2022-09-15_09_34_28.csv'
    # '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_08_02_swing_ran_R/outdoor_output_2022-08-02_08_49_39.csv'
    # '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_08_01_swing_eran_L/outdoor_output_2022-08-01_18_18_15.csv'
    #'/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_08_02_swing_mani_R/outdoor_output_2022-08-02_07_52_01.csv'
    #'/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_nati_R/outdoor_output_2022-07-27_08_11_32.csv'
    window_size = 200
    limit_exp_length = False
    time_limit = 60
    # calc_on_directory = True # if false perform a single example
    # WD = os.getcwd()
    # outputfolder = join(WD, 'data', 'initial_heading_angles')
    # if not os.path.exists(outputfolder):
    #     os.mkdir(outputfolder)
    dir_to_analyze = get_dir_to_analyze(data_location='magneto', mode='swing')
    person_list = get_person_list(dir_to_analyze)
    exp_list = get_exp_list(list_of_persons=person_list, dir_to_analyze=dir_to_analyze)
    if exp_path is None:
        exp_idx = random.randint(0, len(exp_list) - 1)
        exp_path = exp_list[exp_idx]
    print(exp_path)
    exp = Classes.SbgExpRawData(exp_path)
    error_no_initialzation = check_PCA_performnce_on_exp(exp=exp.clone(), window_size=200,
                                                         limit_exp_length=limit_exp_length,
                                                         initialization_time=time_limit)
    exp.heading_fix, rot_traj = initialize_heading_on_exp(exp=exp.clone(), window_size=window_size, plot_results=True,
                                                          initialization_time=time_limit)
    error_w_initialzation = check_PCA_performnce_on_exp(exp=exp, window_size=200,
                                                         limit_exp_length=limit_exp_length,
                                                         initialization_time=time_limit)
    print('error_no_initialzation: ' + str(error_no_initialzation))
    print('error_w_initialzation: ' + str(error_w_initialzation))
    print('exp.heading_fix = ' + str(exp.heading_fix))
    plt.show()
