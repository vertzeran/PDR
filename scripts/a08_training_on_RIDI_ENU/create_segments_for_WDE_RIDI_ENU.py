import random
import os
from os import listdir, mkdir
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import utils.Classes
import utils.Classes as Classes
import utils.Functions as Functions
from utils.Functions import MyCDF
import scipy.io as sio
from datetime import datetime
import json
import pickle
import logging.handlers
from utils.Functions import rotate_trajectory, construct_traj
from scipy.optimize import minimize
import pandas as pd
import tracemalloc

def get_segments_from_exp(exp: Classes.AhrsExp, params, sample_id=0, use_GT_att=False):
    """
    Creating a list of segments from the experiment samples.
    Each segment contains window_size samples (~250)
    sample_id is useful if want to append such lists
    """
    seg_start_idx = 0
    # if isinstance(exp, Classes.SbgExpRawData):
    seg_start_idx = exp.index_of_walking_start
    exp.initial_heading = exp.Psi[seg_start_idx]
    seg_stop_idx = seg_start_idx + params["window_size"] - 1
    segment_list = []
    initial_WD_angles_list = []
    if not use_GT_att:
        AHRS_results_list = []
    # initialize WD angle only for SM heading method
    # analyzed_segment = Classes.WDE_performance_analysis(exp, use_GT_att=True)
    exp.initialize_WD_angle(wind_size_for_heading_init=params["wind_size_for_heading_init"], plot_results=False)
    initial_WD_angle = exp.initial_WD_angle_GT
    if not use_GT_att:
        t_est, lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(exp)
        exp.initial_heading = Heading_est[exp.index_of_walking_start]
    while seg_stop_idx <= exp.NumberOfSamples_IMU - 1:
        segment = exp.clone()
        segment.SegmentScenario([exp.Time_IMU[seg_start_idx], exp.Time_IMU[seg_stop_idx]])
        # print('[' + str(seg_start_idx) + ',' + str(seg_stop_idx) + ']')
        segment.id = sample_id
        if not use_GT_att:
            # AHRS results segmentation
            # t_start = segment.Time_IMU[0].round(11)
            # t_end = segment.Time_IMU[-1].round(11)
            # ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            ind_IMU = range(seg_start_idx, seg_stop_idx + 1)
            # ind_IMU = segment.IMU_valid_idx
            assert abs(t_est[ind_IMU[0]] - segment.Time_IMU[0]) < 1e-8
            assert abs(t_est[ind_IMU[-1]] - segment.Time_IMU[-1]) < 1e-8
            # if ind_IMU[0].shape[0] == 0:
            #     print('stop')
            # if seg_start_idx == 2292:
            #     print('stop')
            # print(seg_start_idx)
            # print(seg_stop_idx)
            AHRS_results_list.append(Classes.AHRS_results(t=t_est[ind_IMU],
                                                          lin_acc_b_frame=lin_acc_b_frame_est[ind_IMU],
                                                          grv=grv_est[ind_IMU],
                                                          Rot=Rot_est[ind_IMU],
                                                          heading=Heading_est[ind_IMU],
                                                          id=sample_id)
                                     )
        initial_WD_angles_list.append(segments_initial_WD_angles(initial_WD_angle=initial_WD_angle, id=sample_id))
        sample_id += 1
        segment_list.append(segment)
        seg_start_idx += params["window_size"]
        seg_stop_idx += params["window_size"]
    if use_GT_att:
        return segment_list, initial_WD_angles_list
    else:
        return segment_list, AHRS_results_list, initial_WD_angles_list


def get_segments_from_dir(exp_dir, window_size, shuffle=True, use_GT_att=False, wind_size_for_heading_init=750):
    """
    Same as get_segments_from_exp, but now we append all lists from all files
    after we had fixed their psi value
    exp_dir a path that includes experiment directories

    """
    segment_list = []
    initial_WD_angles_list = []
    if not use_GT_att:
        AHRS_results_segments_list = []
    exp_list = []
    sample_id = 0
    for item in listdir(exp_dir):
        if 'AHRS_results' not in item:
            # load exp file
            item_path = join(exp_dir, item)  # path to csv file
            exp = Classes.RidiExp(item_path)
            if use_GT_att:
                # Extract segments of the exp with the fixed psi flag
                exp_segments, initial_WD_angles_segment = \
                    get_segments_from_exp(exp, window_size, sample_id, use_GT_att=use_GT_att,
                                          wind_size_for_heading_init=wind_size_for_heading_init)
            else:
                exp_segments, exp_AHRS_results_segments, initial_WD_angles_segment = \
                    get_segments_from_exp(exp, window_size, sample_id, use_GT_att=use_GT_att,
                                          wind_size_for_heading_init=wind_size_for_heading_init)
                AHRS_results_segments_list.extend(exp_AHRS_results_segments)
            initial_WD_angles_list.extend(initial_WD_angles_segment)
            segment_list.extend(exp_segments)
            exp_list.append(exp)
            sample_id = segment_list[-1].id + 1

        if False:
            # exp.Pos.x = exp.Pos.x - exp.Pos.x[0]
            # exp.Pos.y = exp.Pos.y - exp.Pos.y[0]
            # exp.Pos.z = exp.Pos.z - exp.Pos.z[0]
            temp_obj = Classes.WDE_performance_analysis(exp)
            temp_obj.walking_direction_estimation_using_smartphone_heading(
                plot_results=True)
            plt.show()
    if shuffle:
        if use_GT_att:
            temp = list(zip(segment_list, initial_WD_angles_list))
            random.shuffle(temp)
            segment_list, initial_WD_angles_list = zip(*temp)
            segment_list, initial_WD_angles_list = list(segment_list), list(initial_WD_angles_list)
        else:
            temp = list(zip(segment_list, AHRS_results_segments_list, initial_WD_angles_list))
            random.shuffle(temp)
            segment_list, AHRS_results_segments_list, initial_WD_angles_list = zip(*temp)
            segment_list, AHRS_results_segments_list, initial_WD_angles_list = list(segment_list), \
                                                                               list(AHRS_results_segments_list), \
                                                                               list(initial_WD_angles_list)
    if use_GT_att:
        return segment_list, initial_WD_angles_list, exp_list
    else:
        return segment_list, AHRS_results_segments_list, initial_WD_angles_list, exp_list


def get_segments_from_person_list_SBG(person_list, dir_to_analyze, window_size, shuffle=True,
                                      wind_size_for_heading_init=750, logger=None, calc_heading_fix=False,
                                      traj_length_limit=None,
                                      heading_fix_initialization_time=30):
    """
    Same as get_segments_from_exp, but now we append all lists from all files
    exp_list is a list of experiment paths. we use person list to load GT file once
    """
    exp_path_list = get_exp_list(list_of_persons=person_list,
                                 dir_to_analyze=dir_to_analyze)
    N = len(exp_path_list)
    segment_list = []
    AHRS_results_segments_list = []
    sample_id = 0
    i = 1

    for person in person_list:
        person_path = join(dir_to_analyze, person)
        GT_path = join(person_path, 'ascii-output.txt')
        GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
        exp_list_in_person = get_exp_list(list_of_persons=[person],
                                          dir_to_analyze=dir_to_analyze)
        for exp_path in exp_list_in_person:
            info_message = 'working on ' + exp_path
            logger.info(info_message)
            print(info_message)
            exp = Classes.SbgExpRawData(exp_path, GT=GT)
            valid = exp.valid
            if not valid:
                info_message = join(exp.Path, exp.FileName) + ' is not valid!!!!!!!'
                if logger is not None:
                    logger.info(info_message)
                print('---------------------')
                print(join(exp.Path, exp.FileName) + ' is not valid!!!!!!!')
                continue
            if traj_length_limit is not None:
                exp.limit_traj_length(traj_length_limit)
            if exp.first_idx_of_time_gap_GPS.shape[0] != 0:
                info_message = 'time gap in GPS in '
                               # str(exp.Unixtime[exp.GPS_valid_idx][exp.first_idx_of_time_gap_GPS[0] - 1] -
                               #     exp.Unixtime[exp.GPS_valid_idx][0]) + ' [sec]'
                                # '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_08_02_swing_mani_R/outdoor_output_2022-08-02_07_50_28.csv'
                logger.info(info_message)
                print(info_message)
            if exp.first_idx_of_time_gap_IMU.shape[0] != 0:
                info_message = 'time gap in IMU in '
                               # + str(exp.Time_IMU[exp.first_idx_of_time_gap_IMU[0] - 1] -
                               #                             exp.Time_IMU[0]) + ' [sec]'
                               # exp_path = '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_nati_R/outdoor_output_2022-07-27_08_11_32.csv'
                logger.info(info_message)
                print(info_message)
            if calc_heading_fix:
                exp.heading_fix, _ = initialize_heading_on_exp(exp=exp.clone(), window_size=window_size,
                                                               plot_results=False,
                                                               initialization_time=heading_fix_initialization_time)
                info_message = 'heading angle fix is calculated on ' + join(exp.Path, exp.FileName)
                logger.info(info_message)
                print(info_message)
            exp_segments, exp_AHRS_results_segments, initial_WD_angles_segment = \
                get_segments_from_exp(exp, window_size, sample_id, use_GT_att=False,
                                      wind_size_for_heading_init=wind_size_for_heading_init)
            AHRS_results_segments_list.extend(exp_AHRS_results_segments)
            segment_list.extend(exp_segments)
            sample_id = segment_list[-1].id + 1
            info_message = exp_path + ' : ' + str(round(i / N * 100, 3)) + '% completed'
            logger.info(info_message)
            print(info_message)
            i += 1

    if shuffle:
        temp = list(zip(segment_list, AHRS_results_segments_list))
        random.shuffle(temp)
        segment_list, AHRS_results_segments_list = zip(*temp)
        segment_list, AHRS_results_segments_list = list(segment_list), list(AHRS_results_segments_list)

    return segment_list, AHRS_results_segments_list


def get_segments_from_exp_list_SZ(exp_list, params, shuffle=True, use_GT_att=False, logger=None):
    segment_list = []
    initial_WD_angles_list = []
    if not use_GT_att:
        AHRS_results_segments_list = []
    sample_id = 0
    i=1
    N = len(exp_list)
    tracemalloc.start()
    # displaying the memory
    print(tracemalloc.get_traced_memory())
    for item in exp_list:
        info_message = 'working on ' + item
        logger.info(info_message)
        print(info_message)
        # load exp file
        print(item + ' : ' + str(round(i/N * 100,3)) + '%')
        i += 1
        exp = Classes.AI_PDR_exp_w_SP_GT(item)
        if params["traj_length_limit"] is not None:
            exp.limit_traj_length(params["traj_length_limit"])
        exp.define_walking_start_idx(th=params["walking_start_threshold"])
        if use_GT_att:
            # Extract segments of the exp with the fixed psi flag
            exp_segments, initial_WD_angles_segment = \
                get_segments_from_exp(exp, params, sample_id, use_GT_att=use_GT_att)
        else:
            exp_segments, exp_AHRS_results_segments, initial_WD_angles_segment = \
                get_segments_from_exp(exp, params, sample_id, use_GT_att=use_GT_att)
            AHRS_results_segments_list.extend(exp_AHRS_results_segments)
        initial_WD_angles_list.extend(initial_WD_angles_segment)
        segment_list.extend(exp_segments)
        sample_id = segment_list[-1].id + 1
        # displaying the memory
        print(tracemalloc.get_traced_memory())
    if shuffle:
        if use_GT_att:
            temp = list(zip(segment_list, initial_WD_angles_list))
            random.shuffle(temp)
            segment_list, initial_WD_angles_list = zip(*temp)
            segment_list, initial_WD_angles_list = list(segment_list), list(initial_WD_angles_list)
        else:
            temp = list(zip(segment_list, AHRS_results_segments_list, initial_WD_angles_list))
            random.shuffle(temp)
            segment_list, AHRS_results_segments_list, initial_WD_angles_list = zip(*temp)
            segment_list, AHRS_results_segments_list, initial_WD_angles_list = list(segment_list), \
                                                                               list(AHRS_results_segments_list), \
                                                                               list(initial_WD_angles_list)
    if use_GT_att:
        return segment_list
    else:
        return segment_list, AHRS_results_segments_list


def get_segments_from_exp_list(exp_list, params, shuffle=True, use_GT_att=False, logger=None):
    segment_list = []
    initial_WD_angles_list = []
    if not use_GT_att:
        AHRS_results_segments_list = []
    sample_id = 0
    i=1
    N = len(exp_list)
    tracemalloc.start()
    # displaying the memory
    print(tracemalloc.get_traced_memory())
    for item in exp_list:
        info_message = 'working on ' + item
        logger.info(info_message)
        print(info_message)
        # load exp file
        print(item + ' : ' + str(round(i/N * 100,3)) + '%')
        i += 1
        if params["dataset"] == 'AI_PDR':
            exp = Classes.AI_PDR_exp_w_SP_GT(item)
        elif params["dataset"] == 'RIDI_ENU':
            exp = Classes.RidiExp_ENU(item)
        else:
            raise 'invalid dataset'
        if params["traj_length_limit"] is not None:
            exp.limit_traj_length(params["traj_length_limit"])
        exp.define_walking_start_idx(th=params["walking_start_threshold"])
        if use_GT_att:
            # Extract segments of the exp with the fixed psi flag
            exp_segments, initial_WD_angles_segment = \
                get_segments_from_exp(exp, params, sample_id, use_GT_att=use_GT_att)
        else:
            exp_segments, exp_AHRS_results_segments, initial_WD_angles_segment = \
                get_segments_from_exp(exp, params, sample_id, use_GT_att=use_GT_att)
            AHRS_results_segments_list.extend(exp_AHRS_results_segments)
        initial_WD_angles_list.extend(initial_WD_angles_segment)
        segment_list.extend(exp_segments)
        sample_id = segment_list[-1].id + 1
        # displaying the memory
        print(tracemalloc.get_traced_memory())
    if shuffle:
        if use_GT_att:
            temp = list(zip(segment_list, initial_WD_angles_list))
            random.shuffle(temp)
            segment_list, initial_WD_angles_list = zip(*temp)
            segment_list, initial_WD_angles_list = list(segment_list), list(initial_WD_angles_list)
        else:
            temp = list(zip(segment_list, AHRS_results_segments_list, initial_WD_angles_list))
            random.shuffle(temp)
            segment_list, AHRS_results_segments_list, initial_WD_angles_list = zip(*temp)
            segment_list, AHRS_results_segments_list, initial_WD_angles_list = list(segment_list), \
                                                                               list(AHRS_results_segments_list), \
                                                                               list(initial_WD_angles_list)
    if use_GT_att:
        return segment_list
    else:
        return segment_list, AHRS_results_segments_list


# def get_segments_from_exp_list(exp_list, dataset, window_size, shuffle=True, use_GT_att=False,
#                                wind_size_for_heading_init=750, logger=None, calc_heading_fix=False,
#                                heading_fix_initialization_time=30):
#     """
#     Same as get_segments_from_exp, but now we append all lists from all files
#     exp_list is a list of experiment paths
#     """
#     segment_list = []
#     initial_WD_angles_list = []
#     if not use_GT_att:
#         AHRS_results_segments_list = []
#     sample_id = 0
#     i=1
#     N = len(exp_list)
#     for item in exp_list:
#         # load exp file
#         print(item + ' : ' + str(round(i/N * 100,3)) + '%')
#         i += 1
#         if dataset == 'RIDI':
#             exp = Classes.RidiExp(item)
#             valid = True
#         elif dataset == 'TRC#1':
#             exp = Classes.SbgExpRawData(item)
#             valid = exp.valid
#         if not valid:
#             info_message = join(exp.Path ,exp.FileName) + ' is not valid!!!!!!!'
#             if logger is not None:
#                 logger.info(info_message)
#             print('---------------------')
#             print(join(exp.Path ,exp.FileName) + ' is not valid!!!!!!!')
#             continue
#         if exp.first_idx_of_time_gap_GPS.shape[0] != 0:
#             info_message = 'found a GPS time gap in ' + join(exp.Path, exp.FileName)
#             logger.info(info_message)
#             print(info_message)
#         if exp.first_idx_of_time_gap_IMU.shape[0] != 0:
#             info_message = 'found a IMU time gap in ' + join(exp.Path, exp.FileName)
#             logger.info(info_message)
#             print(info_message)
#         if calc_heading_fix:
#             exp.heading_fix, _ = initialize_heading_on_exp(exp=exp.clone(), window_size=window_size, plot_results=False,
#                                                            initialization_time=heading_fix_initialization_time)
#             info_message = 'heading angle fix is calculated on ' + join(exp.Path, exp.FileName)
#             logger.info(info_message)
#             print(info_message)
#         if use_GT_att:
#             # Extract segments of the exp with the fixed psi flag
#             exp_segments, initial_WD_angles_segment = \
#                 get_segments_from_exp(exp, window_size, sample_id, use_GT_att=use_GT_att,
#                                       wind_size_for_heading_init=wind_size_for_heading_init)
#         else:
#             exp_segments, exp_AHRS_results_segments, initial_WD_angles_segment = \
#                 get_segments_from_exp(exp, window_size, sample_id, use_GT_att=use_GT_att,
#                                       wind_size_for_heading_init=wind_size_for_heading_init)
#             AHRS_results_segments_list.extend(exp_AHRS_results_segments)
#         initial_WD_angles_list.extend(initial_WD_angles_segment)
#         segment_list.extend(exp_segments)
#         sample_id = segment_list[-1].id + 1
#
#         if False:
#             # exp.Pos.x = exp.Pos.x - exp.Pos.x[0]
#             # exp.Pos.y = exp.Pos.y - exp.Pos.y[0]
#             # exp.Pos.z = exp.Pos.z - exp.Pos.z[0]
#             temp_obj = Classes.WDE_performance_analysis(exp)
#             temp_obj.walking_direction_estimation_using_smartphone_heading(
#                 plot_results=True)
#             plt.show()
#     if shuffle:
#         if use_GT_att:
#             temp = list(zip(segment_list, initial_WD_angles_list))
#             random.shuffle(temp)
#             segment_list, initial_WD_angles_list = zip(*temp)
#             segment_list, initial_WD_angles_list = list(segment_list), list(initial_WD_angles_list)
#         else:
#             temp = list(zip(segment_list, AHRS_results_segments_list, initial_WD_angles_list))
#             random.shuffle(temp)
#             segment_list, AHRS_results_segments_list, initial_WD_angles_list = zip(*temp)
#             segment_list, AHRS_results_segments_list, initial_WD_angles_list = list(segment_list), \
#                                                                                list(AHRS_results_segments_list), \
#                                                                                list(initial_WD_angles_list)
#     if use_GT_att:
#         return segment_list, initial_WD_angles_list
#     else:
#         return segment_list, AHRS_results_segments_list, initial_WD_angles_list


def get_xy_pairs(window_size, segments_for_WDE, add_quat=False, heading_angle_fix=False):
    if add_quat:
        X = np.zeros((0, window_size, 7))
    else:
        X = np.zeros((0, window_size, 3))
    Y1 = np.zeros((0, 1))
    Y2 = np.zeros((0, 2))
    for i in range(len(segments_for_WDE)):
        segment_for_WDE = segments_for_WDE[i]
        lin_acc = segment_for_WDE.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment_for_WDE.Rot)  # window_sizeX3
        if heading_angle_fix:
            lin_acc_n_frame[:,0:2] = Functions.rotate_trajectory(traj=lin_acc_n_frame[:,0:2], alfa=segment_for_WDE.segment.heading_fix)
            ''' this function is used in initialize_heading.py to rotate a 2D traj. here same is used to rotate 2D
            acceleration vaectors '''
        if add_quat:
            quat_array = segment_for_WDE.Rot.as_quat()
            batch_of_quat = quat_array[:window_size, :]
            batch_of_a_nav_and_quat = np.hstack([lin_acc_n_frame, batch_of_quat])
            x = batch_of_a_nav_and_quat.reshape(1, window_size, 7)
        else:
            x = lin_acc_n_frame.reshape(1, window_size, 3)

        WD_vector = segment_for_WDE.WD_vector_GT.reshape(1, 2)
        dL = segment_for_WDE.dL.reshape(1, 1)
        X = np.concatenate((X, x), axis=0)
        # For odometry
        Y1 = np.concatenate((Y1, dL), axis=0)
        # For WDE
        Y2 = np.concatenate((Y2, WD_vector), axis=0)
    return X, Y1, Y2


def get_AHRS_results_for_exp(exp: utils.Classes.AhrsExp):
    suffix = '_AHRS_results.xlsx'
    AHRS_results_file_path = join(exp.Path, exp.FileName.split(sep='.')[0] + suffix)
    t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
        Functions.read_AHRS_results(AHRS_results_file_path)

    # lin_acc_b_frame = exp.Acc.arr() - grv_hat
    # Rot = Rot_hat
    # Heading = np.array(psi_hat) # + exp.Psi[0]

    ind_IMU = exp.IMU_valid_idx
    lin_acc_b_frame = exp.Acc.arr() - grv_hat[ind_IMU]
    Rot = Rot_hat[ind_IMU]
    Heading = np.array(psi_hat)[ind_IMU] + exp.Psi[0]
    grv_hat = grv_hat[ind_IMU]
    t_est = t_est[ind_IMU]
    return t_est, lin_acc_b_frame, grv_hat, Rot, Heading


class segments_initial_WD_angles():
    def __init__(self, initial_WD_angle=None, id=None):
        self.id = id
        self.initial_WD_angle = initial_WD_angle


def get_train_test_dir(dataset, mode):
    if dataset == 'AI_PDR':
        if mode == 'mixed':
            train_dir = r"/data/Datasets/Navigation/Shenzhen_datasets/dataset-ShenZhen/train"
            test_dir = r"/data/Datasets/Navigation/Shenzhen_datasets/dataset-ShenZhen/test"
        else:
            raise Exception('invalide mode')
    elif dataset == 'RIDI_ENU':
        if mode == 'mixed':
            train_dir = '/data/Datasets/Navigation/RIDI_dataset_train_test_ENU/RIDI_mixed_train'
            test_dir = '/data/Datasets/Navigation/RIDI_dataset_train_test_ENU/RIDI_mixed_test'
        else:
            raise Exception('invalide mode')
    else:
        raise Exception('invalide dataset')
    return train_dir, test_dir


def get_exp_list(list_of_persons, dir_to_analyze):
    exp_list = []
    for directory in list_of_persons:
        dir_path = join(dir_to_analyze, directory)
        if os.path.isdir(dir_path):
            exp_in_dir = [join(dir_path, item) for item in listdir(dir_path) if
                          'AHRS_results' not in item and 'ascii-output.txt' not in item]
            exp_list.extend(exp_in_dir)
    return exp_list


def get_random_person_list(dir_to_analyze, num_of_test_subjects):
    person_list = listdir(dir_to_analyze)
    num_of_train_subjects = len(person_list) - num_of_test_subjects
    person_idx_list = random.sample(range(len(person_list)), num_of_train_subjects)
    list_of_train_persons = []
    list_of_test_persons = []
    for i in range(len(person_list)):
        if i in person_idx_list:
            list_of_train_persons.append(person_list[i])
        else:
            list_of_test_persons.append(person_list[i])
    return list_of_train_persons, list_of_test_persons


def get_segments_for_WDE(segments: utils.Classes.AhrsExp, AHRS_results_segments_list: utils.Classes.AHRS_results):
    segments_for_WDE = []
    for i in range(len(segments)):
        segment = segments[i]
        AHRS_results_seg = AHRS_results_segments_list[i]
        # initial_WD_seg = train_initial_WD_angles_list[i]
        assert segment.id == AHRS_results_seg.id
        segments_for_WDE.append(Classes.WDE_performance_analysis(segment, use_GT_att=False,
                                                                 lin_acc_b_frame_est=AHRS_results_seg.lin_acc_b_frame,
                                                                 grv_est=AHRS_results_seg.grv,
                                                                 Heading_est=AHRS_results_seg.heading,
                                                                 Rot_est=AHRS_results_seg.Rot,
                                                                 arc_length_for_dl=True,
                                                                 pdr_net=None, use_GT_dl=True
                                                                 )
                                )
        segments_for_WDE[-1].initial_WD_angle_GT = segment.initial_WD_angle_GT
    return segments_for_WDE


def GetSwingTrainList():
    ListOfTrainDirs = ['22_07_12_swing_itzik',
    '22_07_27_swing_nati_L',
    '22_07_27_swing_zahi_R',
    '22_07_27_swing_zahi_L',
    '22_07_26_swing_ariel_L',
    '22_07_26_swing_ariel_R',
    '22_07_27_swing_nati_R',
    '22_08_01_swing_eran_L',
    '22_08_01_swing_eran_R',
    '22_08_02_swing_mani_R',
    '22_08_02_swing_mani_L',
    '22_08_02_swing_ofer_L',
    '22_08_02_swing_ofer_R',
    '22_08_02_swing_ran_L',
    '22_08_02_swing_ran_R',
    '22_09_15_swing_nadav_R',
    ]
    return ListOfTrainDirs


def GetSwingTestList():
    ListOfTestDirs = ['22_08_30_swing_sharon_R',
    '22_09_15_swing_yair_L',
    '22_09_15_swing_zeev_R']
    return ListOfTestDirs


def GetPocketTrainList():
    ListOfTrainDirs = ['21_11_10_omri',
    '21_11_28_eran',
    '21_11_28_itzik',
    '21_11_28_ofer',
    '21_11_28_omri',
    '22_09_15_pocket_nadav']
    return ListOfTrainDirs


def GetPocketTestList():
    ListOfTestDirs = ['22_08_30_pocket_sharon_R',
    '22_09_15_pocket_yair',
    '22_09_15_pocket_zeev']
    return ListOfTestDirs


def GetTextingTrainList():
    ListOfTrainDirs = ['21_10_31_eran',
    '21_11_07_mani',
    '21_11_07_ofer',
    '21_11_07_ran',
    '21_11_10_alex',
    '21_11_10_nati']
    return ListOfTrainDirs


def GetTextingTestList():
    ListOfTestDirs = ['21_11_10_demian',
    '21_11_07_firas',
    '21_11_10_omri']
    return ListOfTestDirs


def get_train_and_test_lists(mode):
    if mode == 'swing':
        list_of_train_persons = GetSwingTrainList()
        list_of_test_persons = GetSwingTestList()
    elif mode == 'pocket':
        list_of_train_persons = GetPocketTrainList()
        list_of_test_persons = GetPocketTestList()
    elif mode == 'text':
        list_of_train_persons = GetTextingTrainList()
        list_of_test_persons = GetTextingTestList()
    else:
        raise Exception('invalid mode')
    return list_of_train_persons, list_of_test_persons


def create_segments(params):
    # comment = params["comment"]
    # window_size = params["window_size"]
    # mode = params["mode"]
    # add_quat = params["add_quat"]
    # traj_length_limit = params["traj_length_limit"]
    # dataset_location = params["dataset_location"]
    # wind_size_for_heading_init = params["wind_size_for_heading_init"]
    # dataset = params["dataset"]
    # wind_size_for_heading_init = params["wind_size_for_heading_init"]

    # use_GT_att = False # use gournd truth linear acceleration and b->n rotations else use AHRS
    # for res net use the same window size as in training
    main_wd_path = os.getcwd() #'/'
    # error_type = 'end_pos' # could be: 'WD_angle', 'end_pos'
    # mode = 'swing'  # could be: 'text', 'pocket', 'body', 'bag', 'swing'
    # data_location = 'magneto'  # could be 'magneto' or 'local_machine'
    # dataset = 'AI_PDR'  # could be: 'RIDI', 'TRC#1',
    # sample_percentage = None
    # predefined_train_and_test_list_of_persons = True
    # num_of_test_subjects = None # to randomly choose
    create_validation_set = False
    now = datetime.isoformat(datetime.now())
    now = now.replace(":", "_")
    description = params["dataset"] + '_' + params["mode"] + '_' + 'wind_size_' + str(params["window_size"]) + params["comment"]
    print(description)
    outputfolder = join(main_wd_path, 'data', 'XY_pairs', now + description)
    mkdir(outputfolder)
    ### info file
    info_dict = params
    # info_dict['add_quat'] = add_quat
    # info_dict['window_size'] = window_size
    # info_dict['mode'] = mode
    info_dict['create_validation_set'] = create_validation_set
    info_dict['data_folder'] = outputfolder
    # info_dict['dataset'] = dataset
    with open(join(outputfolder, 'info_file.json'), "w") as outfile:
        json.dump(info_dict, outfile, indent=4)
    # logger
    logfile_location = repr(outputfolder)
    logfile_name = 'create_segments_log.log'
    handler = logging.handlers.WatchedFileHandler(
        os.environ.get("LOGFILE", join(outputfolder, logfile_name)))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    logger_name = 'create_segments_logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logger.addHandler(handler)
    train_dir, test_dir = get_train_test_dir(dataset=params["dataset"], mode=params["mode"])
    # get experiment list from directory
    print('get experiment list from directory')
    exp_list_train = [join(train_dir, item) for item in listdir(train_dir) if
                      'AHRS_results' not in item]
    exp_list_test = [join(test_dir, item) for item in listdir(test_dir) if
                      'AHRS_results' not in item]
    info_message = 'get segments train'
    logger.info(info_message)
    print(info_message)
    segments_train, AHRS_results_segments_list_train = \
        get_segments_from_exp_list(exp_list=exp_list_train,
                                      params=params,
                                      shuffle=False,
                                      use_GT_att=False,
                                      logger=logger)
    if create_validation_set:
        info_message = 'divide to test-validation segments'
        logger.info(info_message)
        print(info_message)
        num_of_train_samples = round(len(segments_train) * 0.8)
    # num_of_validation_samples = len(segments_train) - num_of_train_samples
    # num_of_test_samples = len(segments_test)
        train_idx = random.sample(range(len(segments_train)), num_of_train_samples)
        train_segments = list(map(segments_train.__getitem__, train_idx))
        train_AHRS_results_segments_list = list(map(AHRS_results_segments_list_train.__getitem__, train_idx))
    else:
        train_segments = segments_train
        train_AHRS_results_segments_list = AHRS_results_segments_list_train
    train_segments_for_WDE = get_segments_for_WDE(train_segments, train_AHRS_results_segments_list)
    del train_segments, train_AHRS_results_segments_list
    if create_validation_set:
        validation_idx = list(range(len(segments_train))) # start with 100% train and then remove train and leave a validation
        for idx in train_idx:
            validation_idx.remove(idx)
        validation_segments = list(map(segments_train.__getitem__, validation_idx))
        validation_AHRS_results_segments_list = list(map(AHRS_results_segments_list_train.__getitem__, validation_idx))
        validation_segments_for_WDE = get_segments_for_WDE(validation_segments, validation_AHRS_results_segments_list)
        del validation_segments, validation_AHRS_results_segments_list

    info_message = 'get segments test'
    logger.info(info_message)
    print(info_message)
    segments_test, AHRS_results_segments_list_test = \
        get_segments_from_exp_list(exp_list=exp_list_test,
                                      params=params,
                                      shuffle=False,
                                      use_GT_att=False,
                                      logger=logger
                                      )
    test_segments_for_WDE = get_segments_for_WDE(segments_test, AHRS_results_segments_list_test)
    del segments_test, AHRS_results_segments_list_test
    info_message = 'convert to XY pairs'
    logger.info(info_message)
    print(info_message)
    # convert train/validation_segments_for_WDE to X,Y. X->NX200X3 (lin acc in n frame), Y->N NX2 (WD vector)
    X_train, Y1_train, Y2_train = get_xy_pairs(params["window_size"], train_segments_for_WDE,
                                               add_quat=params["add_quat"], heading_angle_fix=False)
    # check_segmentation(segment_list=train_segments_for_WDE, X=X_train, Y1=Y1_train, Y2=Y2_train)
    if create_validation_set:
        X_validation, Y1_validation, Y2_validation = get_xy_pairs(params["window_size"], validation_segments_for_WDE,
                                                                  add_quat=params["add_quat"], heading_angle_fix=False)
    X_test, Y1_test, Y2_test = get_xy_pairs(params["window_size"], test_segments_for_WDE,
                                            add_quat=params["add_quat"], heading_angle_fix=False)
    info_message = 'saving results to ' + outputfolder
    logger.info(info_message)
    print(info_message)
    sio.savemat(join(outputfolder, 'train.mat'), {'X': X_train, 'Y1': Y1_train, 'Y2': Y2_train})
    info_message = 'saved train XY'
    logger.info(info_message)
    print(info_message)
    del X_train, Y1_train, Y2_train
    if create_validation_set:
        sio.savemat(join(outputfolder, 'validation.mat'), {'X': X_validation, 'Y1': Y1_validation, 'Y2': Y2_validation})
        info_message = 'saved validation XY'
        logger.info(info_message)
        print(info_message)
        del X_validation, Y1_validation, Y2_validation
    sio.savemat(join(outputfolder, 'test.mat'), {'X': X_test, 'Y1': Y1_test, 'Y2': Y2_test})
    info_message = 'saved test XY'
    logger.info(info_message)
    print(info_message)
    del X_test, Y1_test, Y2_test
    # this is commented out since it causes the process to be killed by the OS

    # with open(join(outputfolder, 'train_segments_for_WDE.pickle'), 'wb') as f:
    #     pickle.dump(train_segments_for_WDE, f)
    # info_message = 'saved train segments'
    info_message = 'skipped saving train segments'
    logger.info(info_message)
    print(info_message)
    del train_segments_for_WDE
    with open(join(outputfolder, 'test_segments_for_WDE.pickle'), 'wb') as f:
        pickle.dump(test_segments_for_WDE, f)
    info_message = 'saved test segments'
    logger.info(info_message)
    print(info_message)
    del test_segments_for_WDE
    if create_validation_set:
        with open(join(outputfolder, 'validation_segments_for_WDE.pickle'), 'wb') as f:
            pickle.dump(validation_segments_for_WDE, f)
        info_message = 'saved validation segments'
        logger.info(info_message)
        print(info_message)
        del validation_segments_for_WDE
    return outputfolder


def check_segmentation(segment_list, X, Y1, Y2):
    segment = segment_list[0]
    lin_acc = segment.lin_acc_b_frame
    lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment.Rot).reshape(1, 200, 3)
    X = X[0]
    e = lin_acc_n_frame - X
    print(np.linalg.norm(e))


def initialize_heading_on_exp(exp_path=None, exp=None, window_size=200, plot_results=False, initialization_time=10):
    """use PCA to calculate a 2D trajectory. then calculate the optimal heading angle fox to minimize the trajectory
    errors relative to GT."""
    if initialization_time is not None:
        limit_exp_length = True
    else:
        limit_exp_length = False
    if exp is None:
        assert exp_path is not None
        exp = Classes.SbgExpRawData(exp_path)
    else:
        assert exp_path is None
    exp.define_walking_start_idx(th=1)
    if limit_exp_length:
        # original_exp = exp.clone() ## todo: comment out!!
        # varify round number of window sizes
        start_time = exp.Time_IMU[exp.index_of_walking_start]
        start_idx = abs(exp.Time_IMU - start_time).argmin()
        stop_time = exp.Time_IMU[exp.index_of_walking_start] + initialization_time
        stop_idx = abs(exp.Time_IMU - stop_time).argmin()
        stop_idx = start_idx + window_size * ((stop_idx - start_idx) / window_size).__floor__()
        stop_time = exp.Time_IMU[stop_idx]
        exp.SegmentScenario([exp.Time_IMU[0],
                             stop_time])
        # we dont cut out the begining because it is performed in the next command: get_segments_from_exp

    # this code is taken from traj_est_using_PCA. it is duplicated because we need to keep local parameteres
    segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=exp,
                                                                                    window_size=window_size,
                                                                                    use_GT_att=False,
                                                                                    wind_size_for_heading_init=750)
    analyzied_segments = []
    walking_angle = []
    dL = []
    exp.define_walking_start_idx()
    est_time = [exp.Time_IMU[exp.index_of_walking_start - 1]]
    p = exp.Pos.arr()[exp.index_of_walking_start]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        est_time.append(segment.Time_GT[-1])
        AHRS_results = AHRS_results_list[i]
        analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=False,
                                                                   lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
                                                                   Rot_est=AHRS_results.Rot,
                                                                   grv_est=AHRS_results.grv,
                                                                   Heading_est=AHRS_results.heading))
        analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        # dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
        dp = segment.Pos.arr()[-1] - p
        p = segment.Pos.arr()[-1]
        dL.append(np.linalg.norm(dp[0:2]))
        walking_angle.append(analyzied_segments[-1].WD_angle_est)
    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    exp_temp = exp.clone()
    exp_temp.SegmentScenario([exp_temp.Time_IMU[exp_temp.index_of_walking_start],
                             exp_temp.Time_IMU[-1]])
    dL_ideal = exp_temp.calc_dL(window_size=window_size)
    walking_angle_ideal, WD_ideal = exp_temp.calc_walking_direction(window_size=window_size)
    traj_ideal = construct_traj(dL_ideal, walking_angle_ideal, plot_result=False, pos_gt=gt_pos)
    traj = construct_traj(np.array(dL), np.array(walking_angle), plot_result=False, pos_gt=gt_pos)
    ## this code is taken from traj_errors, traj_length, error_metrika = calculate_traj_error(Exp, np.array(est_time), traj, dL)
    gt_time = exp.Time_GT[exp.index_of_walking_start:]
    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    traj_length = np.array(dL).cumsum()
    gt_pos_interp = np.vstack([np.interp(est_time, gt_time, gt_pos[:, 0]),
                               np.interp(est_time, gt_time, gt_pos[:, 1]),
                               np.interp(est_time, gt_time, gt_pos[:, 2])]).T
    x0 = np.array(0.0)

    def minimization_function(alfa, traj_est, traj_gt):
        rot_traj = rotate_trajectory(traj_est, alfa[0])
        return Functions.traj_error(traj_est=rot_traj, traj_gt=traj_gt)
    res = minimize(minimization_function, x0, method='nelder-mead', args=(traj, gt_pos_interp[:, 0:2]),
                   options={'xatol': 1e-8, 'disp': False})
    # traj_errors = np.linalg.norm(traj - gt_pos_interp[:, 0:2], axis=1)
    rot_traj = rotate_trajectory(traj, res.x[0])
    if plot_results:
        # calculate the traj again with rotated acceleration and varify resulting traj equals the rotated traj
        # exp.heading_fix = res.x
        # segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=exp,
        #                                                                                 window_size=window_size,
        #                                                                                 use_GT_att=False,
        #                                                                                 wind_size_for_heading_init=750)
        # analyzied_segments = []
        # walking_angle = []
        # dL = []
        # exp.define_walking_start_idx()
        # est_time = [exp.Time_IMU[exp.index_of_walking_start - 1]]
        # p = exp.Pos.arr()[exp.index_of_walking_start]
        # for i in range(len(segment_list)):
        #     segment = segment_list[i]
        #     est_time.append(segment.Time_GT[-1])
        #     AHRS_results = AHRS_results_list[i]
        #     analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=False,
        #                                                                lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
        #                                                                Rot_est=AHRS_results.Rot,
        #                                                                grv_est=AHRS_results.grv,
        #                                                                Heading_est=AHRS_results.heading))
        #     analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        #     # dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
        #     dp = segment.Pos.arr()[-1] - p
        #     p = segment.Pos.arr()[-1]
        #     dL.append(np.linalg.norm(dp[0:2]))
        #     walking_angle.append(analyzied_segments[-1].WD_angle_est)
        #     # if True: # varify accuracy has improved
        #     #     segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=original_exp,
        #     #                                                                                     window_size=window_size,
        #     #                                                                                     use_GT_att=False,
        #     #                                                                                     wind_size_for_heading_init=750)
        #     #     analyzied_segments = []
        #     #     walking_angle = []
        #     #     dL = []
        #     #     original_exp.define_walking_start_idx()
        #     #     est_time = [original_exp.Time_IMU[exp.index_of_walking_start - 1]]
        #     #     p = original_exp.Pos.arr()[exp.index_of_walking_start]
        #     #     for i in range(len(segment_list)):
        #     #         segment = segment_list[i]
        #     #         est_time.append(segment.Time_GT[-1])
        #     #         AHRS_results = AHRS_results_list[i]
        #     #         analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=False,
        #     #                                                                    lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
        #     #                                                                    Rot_est=AHRS_results.Rot,
        #     #                                                                    grv_est=AHRS_results.grv,
        #     #                                                                    Heading_est=AHRS_results.heading))
        #     #         analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        #     #         # dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
        #     #         dp = segment.Pos.arr()[-1] - p
        #     #         p = segment.Pos.arr()[-1]
        #     #         dL.append(np.linalg.norm(dp[0:2]))
        #     #         walking_angle.append(analyzied_segments[-1].WD_angle_est)
        #
        #
        # traj_check = construct_traj(np.array(dL), np.array(walking_angle), plot_result=False, pos_gt=gt_pos)
        fig = plt.figure('initialize tarj heading')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='black', linestyle='--', linewidth=2, label='gt')
        ax.plot(traj_ideal[:, 0], traj_ideal[:, 1], color='green', linestyle='-', linewidth=2, label='ideal')
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='raw')
        ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
        # ax.plot(traj_check[:, 0], traj_check[:, 1], color='blue', linestyle='--', linewidth=2, label='rotated acc')
        ax.legend()
        ax.axis('equal')
    return res.x, rot_traj


def rotate_traj_to_minimize_error(est_traj, gt_traj):
    '''
    calculate the heading angle fix to align est_traj to gt_traj.
    gt_traj should be interpulated to the times of the estimation.
    '''
    x0 = np.array(0.0)

    def minimization_function(alfa, traj_est, traj_gt):
        rot_traj = rotate_trajectory(traj_est, alfa[0])
        return Functions.traj_error(traj_est=rot_traj, traj_gt=traj_gt)

    res = minimize(minimization_function, x0, method='nelder-mead', args=(est_traj, gt_traj),
                   options={'xatol': 1e-8, 'disp': False})
    # traj_errors = np.linalg.norm(traj - gt_pos_interp[:, 0:2], axis=1)
    rot_traj = rotate_trajectory(est_traj, res.x[0])
    return rot_traj, res.x[0]


if __name__ == '__main__':
    params = ''
    outputfolder = create_segments(params)
