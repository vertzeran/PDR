import random
import os
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import psutil
from tqdm import tqdm

import utils.Classes
import utils.Classes as Classes
import utils.Functions as Functions
import scipy.io as sio
from datetime import datetime
import json
import logging.handlers
from utils.Functions import rotate_trajectory, construct_traj
from scipy.optimize import minimize
import pandas as pd


def get_segments_from_exp(exp: Classes.AhrsExp, window_size, sample_id=0, use_gt_att=False,
                          win_size_for_heading_init=750):
    """
    Creating a list of segments from the experiment samples.
    Each segment contains window_size samples (~250)
    sample_id is useful if want to append such lists
    """
    exp.define_walking_start_idx()
    seg_start_idx = exp.index_of_walking_start
    exp.initial_heading = exp.Psi[seg_start_idx]
    seg_stop_idx = seg_start_idx + window_size - 1

    segment_list = []
    initial_wd_angles_list = []
    if not use_gt_att:
        ahrs_results_list = []

    # initialize WD angle only for SM heading method
    exp.initialize_WD_angle(wind_size_for_heading_init=win_size_for_heading_init, plot_results=False)
    initial_wd_angle = exp.initial_WD_angle_GT
    if not use_gt_att:
        t_est, lin_acc_b_frame_est, grv_est, rot_est, heading_est = get_ahrs_results_for_exp(exp)
        exp.initial_heading = heading_est[exp.index_of_walking_start]

    while seg_stop_idx <= exp.NumberOfSamples_IMU - 1:
        segment = exp.clone()  # TODO - is there a way to clone only the samples we need?
        segment.SegmentScenario([exp.Time_IMU[seg_start_idx], exp.Time_IMU[seg_stop_idx]])
        segment.id = sample_id
        if not use_gt_att:
            # AHRS results segmentation
            ind_imu = range(seg_start_idx, seg_stop_idx + 1)
            assert abs(t_est[ind_imu[0]] - segment.Time_IMU[0]) < 1e-8
            assert abs(t_est[ind_imu[-1]] - segment.Time_IMU[-1]) < 1e-8
            ahrs_results_list.append(Classes.AHRS_results(t=t_est[ind_imu],
                                                          lin_acc_b_frame=lin_acc_b_frame_est[ind_imu],
                                                          grv=grv_est[ind_imu],
                                                          Rot=rot_est[ind_imu],
                                                          heading=heading_est[ind_imu],
                                                          id=sample_id)
                                     )
        initial_wd_angles_list.append(SegmentsInitialWdAngles(initial_wd_angle=initial_wd_angle, id=sample_id))
        sample_id += 1
        segment_list.append(segment)
        seg_start_idx += window_size
        seg_stop_idx += window_size
    if use_gt_att:
        return segment_list, initial_wd_angles_list
    else:
        return segment_list, ahrs_results_list, initial_wd_angles_list


def get_segments_from_dir(exp_dir, window_size, shuffle=True, use_gt_att=False, win_size_for_heading_init=750):
    """
    Same as get_segments_from_exp, but now we append all lists from all files
    after we had fixed their psi value
    exp_dir a path that includes experiment directories

    """
    segment_list = []
    initial_wd_angles_list = []
    if not use_gt_att:
        ahrs_results_segments_list = []
    exp_list = []
    sample_id = 0
    for item in listdir(exp_dir):
        if 'AHRS_results' not in item:
            # load exp file
            item_path = join(exp_dir, item)  # path to csv file
            exp = Classes.RidiExp(item_path)
            if use_gt_att:
                # Extract segments of the exp with the fixed psi flag
                exp_segments, initial_wd_angles_segment = \
                    get_segments_from_exp(exp, window_size, sample_id, use_gt_att=use_gt_att,
                                          win_size_for_heading_init=win_size_for_heading_init)
            else:
                exp_segments, exp_ahrs_results_segments, initial_wd_angles_segment = \
                    get_segments_from_exp(exp, window_size, sample_id, use_gt_att=use_gt_att,
                                          win_size_for_heading_init=win_size_for_heading_init)
                ahrs_results_segments_list.extend(exp_ahrs_results_segments)
            initial_wd_angles_list.extend(initial_wd_angles_segment)
            segment_list.extend(exp_segments)
            exp_list.append(exp)
            sample_id = segment_list[-1].id + 1

    if shuffle:
        if use_gt_att:
            temp = list(zip(segment_list, initial_wd_angles_list))
            random.shuffle(temp)
            segment_list, initial_wd_angles_list = zip(*temp)
            segment_list, initial_wd_angles_list = list(segment_list), list(initial_wd_angles_list)
        else:
            temp = list(zip(segment_list, ahrs_results_segments_list, initial_wd_angles_list))
            random.shuffle(temp)
            segment_list, ahrs_results_segments_list, initial_wd_angles_list = zip(*temp)
            segment_list, ahrs_results_segments_list, initial_wd_angles_list = list(segment_list), \
                list(ahrs_results_segments_list), \
                list(initial_wd_angles_list)

    if use_gt_att:
        return segment_list, initial_wd_angles_list, exp_list
    else:
        return segment_list, ahrs_results_segments_list, initial_wd_angles_list, exp_list


def get_segments_from_person_list_sbg(person_list, dir_to_analyze, window_size, win_size_for_heading_init=750,
                                      logger=None, calc_heading_fix=False, traj_length_limit=None,
                                      heading_fix_initialization_time=30, add_quat=False, heading_angle_fix=False):
    """
    Same as get_segments_from_exp, but now we append all lists from all files
    exp_list is a list of experiment paths. we use person list to load GT file once
    """
    sample_id = 0
    x_list = []
    y1_list = []
    y2_list = []

    for i_person, person in enumerate(person_list):
        person_path = join(dir_to_analyze, person)
        gt_path = join(person_path, 'ascii-output.txt')
        gt = pd.read_csv(gt_path, sep='\t', skiprows=28)
        exp_list_in_person = get_exp_list(list_of_persons=[person], dir_to_analyze=dir_to_analyze)
        p_bar = tqdm(exp_list_in_person, desc=f'{person} ({i_person}/{len(person_list)})')
        for exp_path in p_bar:
            virt_mem = psutil.virtual_memory()
            p_bar.set_postfix({'Used RAM %': virt_mem[2], 'Used RAM (GB)': virt_mem[3] / 1000000000})

            info_message = 'working on ' + exp_path
            logger.info(info_message)
            print(info_message)
            exp = Classes.SbgExpRawData(exp_path, GT=gt)
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
                logger.info(info_message)
                print(info_message)
            if exp.first_idx_of_time_gap_IMU.shape[0] != 0:
                info_message = 'time gap in IMU in '
                logger.info(info_message)
                print(info_message)
            if calc_heading_fix:
                exp.heading_fix, _ = initialize_heading_on_exp(exp=exp.clone(), window_size=window_size,
                                                               plot_results=False,
                                                               initialization_time=heading_fix_initialization_time)
                info_message = 'heading angle fix is calculated on ' + join(exp.Path, exp.FileName)
                logger.info(info_message)
                print(info_message)
            exp_segments, exp_ahrs_results_segments, initial_wd_angles_segment = get_segments_from_exp(
                exp, window_size, sample_id, use_gt_att=False, win_size_for_heading_init=win_size_for_heading_init)

            sample_id = exp_segments[-1].id + 1

            # get segments for walking direction estimation
            for segment, ahrs_results_seg in zip(exp_segments, exp_ahrs_results_segments):
                assert segment.id == ahrs_results_seg.id
                segment_for_wde = Classes.WDE_performance_analysis(
                    segment, use_GT_att=False, lin_acc_b_frame_est=ahrs_results_seg.lin_acc_b_frame,
                    grv_est=ahrs_results_seg.grv, Heading_est=ahrs_results_seg.heading, Rot_est=ahrs_results_seg.Rot,
                    arc_length_for_dl=True, use_GT_dl=True)
                segment_for_wde.initial_WD_angle_GT = segment.initial_WD_angle_GT

                x, y1, y2 = get_seg_xy_pair(window_size, segment_for_wde,
                                            add_quat=add_quat, heading_angle_fix=heading_angle_fix)

                x_list.append(x)  # (1, window_size, 3/7)
                y1_list.append(y1)  # (1, 1)
                y2_list.append(y2)  # (1, 2)

    x_list = np.vstack(x_list)
    y1_list = np.vstack(y1_list)
    y2_list = np.vstack(y2_list)

    return x_list, y1_list, y2_list


def get_segments_from_exp_list(exp_list, dataset, window_size, shuffle=True, use_gt_att=False,
                               win_size_for_heading_init=750, logger=None, calc_heading_fix=False,
                               heading_fix_initialization_time=30):
    """
    Same as get_segments_from_exp, but now we append all lists from all files
    exp_list is a list of experiment paths
    """
    segment_list = []
    initial_wd_angles_list = []
    if not use_gt_att:
        ahrs_results_segments_list = []
    sample_id = 0
    i = 1
    n = len(exp_list)
    for item in exp_list:
        # load exp file
        print(item + ' : ' + str(round(i/n * 100,3)) + '%')
        i += 1
        if dataset == 'RIDI':
            exp = Classes.RidiExp(item)
            valid = True
        elif dataset == 'TRC#1':
            exp = Classes.SbgExpRawData(item)
            valid = exp.valid
        if not valid:
            info_message = join(exp.Path ,exp.FileName) + ' is not valid!!!!!!!'
            if logger is not None:
                logger.info(info_message)
            print('---------------------')
            print(join(exp.Path ,exp.FileName) + ' is not valid!!!!!!!')
            continue
        if exp.first_idx_of_time_gap_GPS.shape[0] != 0:
            info_message = 'found a GPS time gap in ' + join(exp.Path, exp.FileName)
            logger.info(info_message)
            print(info_message)
        if exp.first_idx_of_time_gap_IMU.shape[0] != 0:
            info_message = 'found a IMU time gap in ' + join(exp.Path, exp.FileName)
            logger.info(info_message)
            print(info_message)
        if calc_heading_fix:
            exp.heading_fix, _ = initialize_heading_on_exp(exp=exp.clone(), window_size=window_size, plot_results=False,
                                                           initialization_time=heading_fix_initialization_time)
            info_message = 'heading angle fix is calculated on ' + join(exp.Path, exp.FileName)
            logger.info(info_message)
            print(info_message)
        if use_gt_att:
            # Extract segments of the exp with the fixed psi flag
            exp_segments, initial_wd_angles_segment = \
                get_segments_from_exp(exp, window_size, sample_id, use_gt_att=use_gt_att,
                                      win_size_for_heading_init=win_size_for_heading_init)
        else:
            exp_segments, exp_ahrs_results_segments, initial_wd_angles_segment = \
                get_segments_from_exp(exp, window_size, sample_id, use_gt_att=use_gt_att,
                                      win_size_for_heading_init=win_size_for_heading_init)
            ahrs_results_segments_list.extend(exp_ahrs_results_segments)
        initial_wd_angles_list.extend(initial_wd_angles_segment)
        segment_list.extend(exp_segments)
        sample_id = segment_list[-1].id + 1

    if shuffle:
        if use_gt_att:
            temp = list(zip(segment_list, initial_wd_angles_list))
            random.shuffle(temp)
            segment_list, initial_wd_angles_list = zip(*temp)
            segment_list, initial_wd_angles_list = list(segment_list), list(initial_wd_angles_list)
        else:
            temp = list(zip(segment_list, ahrs_results_segments_list, initial_wd_angles_list))
            random.shuffle(temp)
            segment_list, ahrs_results_segments_list, initial_wd_angles_list = zip(*temp)
            segment_list, ahrs_results_segments_list, initial_wd_angles_list = list(segment_list), \
                list(ahrs_results_segments_list), \
                list(initial_wd_angles_list)
    if use_gt_att:
        return segment_list, initial_wd_angles_list
    else:
        return segment_list, ahrs_results_segments_list, initial_wd_angles_list


def get_seg_xy_pair(window_size, segment_for_wde, add_quat=False, heading_angle_fix=False):
    lin_acc = segment_for_wde.lin_acc_b_frame
    lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment_for_wde.Rot)  # window_sizeX3
    if heading_angle_fix:
        lin_acc_n_frame[:, 0:2] = Functions.rotate_trajectory(traj=lin_acc_n_frame[:, 0:2],
                                                              alfa=segment_for_wde.segment.heading_fix)
        ''' this function is used in initialize_heading.py to rotate a 2D traj. here same is used to rotate 2D
        acceleration vaectors '''
    if add_quat:
        quat_array = segment_for_wde.Rot.as_quat()
        batch_of_quat = quat_array[:window_size, :]
        batch_of_a_nav_and_quat = np.hstack([lin_acc_n_frame, batch_of_quat])
        x = batch_of_a_nav_and_quat.reshape((1, window_size, 7))
    else:
        x = lin_acc_n_frame.reshape((1, window_size, 3))

    walking_dir_vector = segment_for_wde.WD_vector_GT.reshape((1, 2))
    d_l = segment_for_wde.dL.reshape(1, 1)

    y1 = d_l
    y2 = walking_dir_vector

    return x, y1, y2


def get_xy_pairs(window_size, segments_for_wde, add_quat=False, heading_angle_fix=False):
    if add_quat:
        x = np.zeros((0, window_size, 7))
    else:
        x = np.zeros((0, window_size, 3))
    y1 = np.zeros((0, 1))
    y2 = np.zeros((0, 2))
    for i in range(len(segments_for_wde)):
        segment_for_wde = segments_for_wde[i]
        lin_acc = segment_for_wde.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment_for_wde.Rot)  # window_sizeX3
        if heading_angle_fix:
            lin_acc_n_frame[:, 0:2] = Functions.rotate_trajectory(traj=lin_acc_n_frame[:, 0:2],
                                                                  alfa=segment_for_wde.segment.heading_fix)
            ''' this function is used in initialize_heading.py to rotate a 2D traj. here same is used to rotate 2D
            acceleration vaectors '''
        if add_quat:
            quat_array = segment_for_wde.Rot.as_quat()
            batch_of_quat = quat_array[:window_size, :]
            batch_of_a_nav_and_quat = np.hstack([lin_acc_n_frame, batch_of_quat])
            x = batch_of_a_nav_and_quat.reshape((1, window_size, 7))
        else:
            x = lin_acc_n_frame.reshape((1, window_size, 3))

        walking_dir_vector = segment_for_wde.WD_vector_GT.reshape((1, 2))
        d_l = segment_for_wde.dL.reshape(1, 1)
        x = np.concatenate((x, x), axis=0)
        # For odometry
        y1 = np.concatenate((y1, d_l), axis=0)
        # For WDE
        y2 = np.concatenate((y2, walking_dir_vector), axis=0)
    return x, y1, y2


def get_ahrs_results_for_exp(exp: utils.Classes.AhrsExp):
    suffix = '_AHRS_results.xlsx'
    ahrs_results_file_path = join(exp.Path, exp.FileName.split(sep='.')[0] + suffix)
    t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, rot_hat = \
        Functions.read_AHRS_results(ahrs_results_file_path)
    # segmentation
    ind_imu = exp.IMU_valid_idx
    lin_acc_b_frame = exp.Acc.arr() - grv_hat[ind_imu]
    rot = rot_hat[ind_imu]
    heading = np.array(psi_hat)[ind_imu] + exp.Psi[0]
    grv_hat = grv_hat[ind_imu]
    t_est = t_est[ind_imu]
    return t_est, lin_acc_b_frame, grv_hat, rot, heading


class SegmentsInitialWdAngles:
    def __init__(self, initial_wd_angle=None, id=None):
        self.id = id
        self.initial_WD_angle = initial_wd_angle


def get_dir_to_analyze(data_location, mode):
    if data_location == 'magneto':
        root_to_analyze = '/data/Datasets/Navigation/SBG-PDR-DATA'
    elif data_location == 'wolverine':
        root_to_analyze = '/nfstemp/Datasets/Navigation/SBG-PDR-DATA'
    elif data_location == 'local_machine':
        root_to_analyze = '/home/maint/Eran/AHRS/SBG-PDR-DATA'
    else:
        raise ValueError(data_location)

    if mode == 'text':
        dir_to_analyze = os.path.join(root_to_analyze, 'texting')
    elif mode == 'pocket':
        dir_to_analyze = os.path.join(root_to_analyze, 'pocket')
    elif mode == 'swing':
        dir_to_analyze = os.path.join(root_to_analyze, 'swing')
    elif mode == 'mixed':
        dir_to_analyze = os.path.join(root_to_analyze, 'mixed')
    else:
        raise ValueError(f'invalid mode: {mode}')
    return dir_to_analyze


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


def get_segments_for_wde(segments: utils.Classes.AhrsExp, ahrs_results_segments_list: utils.Classes.AHRS_results):
    segments_for_wde = []
    for i in range(len(segments)):
        segment = segments[i]
        ahrs_results_seg = ahrs_results_segments_list[i]
        assert segment.id == ahrs_results_seg.id
        segments_for_wde.append(Classes.WDE_performance_analysis(segment, use_GT_att=False,
                                                                 lin_acc_b_frame_est=ahrs_results_seg.lin_acc_b_frame,
                                                                 grv_est=ahrs_results_seg.grv,
                                                                 Heading_est=ahrs_results_seg.heading,
                                                                 Rot_est=ahrs_results_seg.Rot,
                                                                 arc_length_for_dl=True,
                                                                 pdr_net=None, use_GT_dl=True
                                                                 )
                                )
        segments_for_wde[-1].initial_WD_angle_GT = segment.initial_WD_angle_GT
    return segments_for_wde


def get_swing_train_list():
    list_of_train_dirs = ['22_07_12_swing_itzik',
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
    return list_of_train_dirs


def get_swing_test_list():
    list_of_test_dirs = ['22_08_30_swing_sharon_R',
                         '22_09_15_swing_yair_L',
                         '22_09_15_swing_zeev_R']
    return list_of_test_dirs


def get_pocket_train_list():
    list_of_train_dirs = ['21_11_10_omri',
                          '21_11_28_eran',
                          '21_11_28_itzik',
                          '21_11_28_ofer',
                          '21_11_28_omri',
                          '22_09_15_pocket_nadav']
    return list_of_train_dirs


def get_pocket_test_list():
    list_of_test_dirs = ['22_08_30_pocket_sharon_R',
                         '22_09_15_pocket_yair',
                         '22_09_15_pocket_zeev']
    return list_of_test_dirs


def get_texting_train_list():
    list_of_train_dirs = ['21_10_31_eran',
                          '21_11_07_mani',
                          '21_11_07_ofer',
                          '21_11_07_ran',
                          '21_11_10_alex',
                          '21_11_10_nati']
    return list_of_train_dirs


def get_texting_test_list():
    list_of_test_dirs = ['21_11_10_demian',
                         '21_11_07_firas',
                         '21_11_10_omri']
    return list_of_test_dirs


def get_train_and_test_lists(mode):
    if mode == 'swing':
        list_of_train_persons = get_swing_train_list()
        list_of_test_persons = get_swing_test_list()
    elif mode == 'pocket':
        list_of_train_persons = get_pocket_train_list()
        list_of_test_persons = get_pocket_test_list()
    elif mode == 'text':
        list_of_train_persons = get_texting_train_list()
        list_of_test_persons = get_texting_test_list()
    elif mode == 'mixed':
        list_of_train_persons = get_swing_train_list() + get_pocket_train_list() + get_texting_train_list()
        list_of_test_persons = get_swing_test_list() + get_pocket_test_list() + get_texting_test_list()
    else:
        raise ValueError(f'invalid mode: {mode}')
    return list_of_train_persons, list_of_test_persons


def create_segments(comment, window_size=200, mode=None, add_quat=False, heading_fix=False, traj_length_limit=None,
                    heading_fix_initialization_time=30):
    win_size_for_heading_init = 1000

    # for res net use the same window size as in training
    main_wd_path = '/home/adam/git/walking_direction_estimation'

    data_location = 'wolverine'  # could be 'magneto' or 'local_machine'
    dataset = 'TRC#1'  # could be: 'RIDI', 'TRC#1',
    sample_percentage = None
    predefined_train_and_test_list_of_persons = True
    num_of_test_subjects = None  # to randomly choose
    create_validation_set = False
    now = datetime.isoformat(datetime.now())
    description = dataset + ' ' + mode + ' ' + 'wind_size ' + str(window_size) + comment
    print(description)

    outputfolder = join(main_wd_path, 'data', 'XY_pairs', now + description)
    os.makedirs(outputfolder)

    # info file
    info_dict = {'add_quat': add_quat, 'heading_fix': heading_fix, 'window_size': window_size, 'mode': mode,
                 'create_validation_set': create_validation_set, 'dataset': dataset}
    with open(join(outputfolder, 'info_file.json'), "w") as outfile:
        json.dump(info_dict, outfile, indent=4)
    # logger
    logfile_location = outputfolder
    logfile_name = 'create_segments_log.log'
    handler = logging.handlers.WatchedFileHandler(
        os.environ.get("LOGFILE", join(logfile_location, logfile_name)))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    logger_name = 'create_segments_logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    logger.addHandler(handler)
    dir_to_analyze = get_dir_to_analyze(data_location, mode)  # get main dir
    info_message = now + ": started working on folder " + dir_to_analyze
    logger.info(info_message)
    print(info_message)
    # get experiment list from directory
    print('get experiment list from directory')
    if predefined_train_and_test_list_of_persons:
        list_of_train_persons, list_of_test_persons = get_train_and_test_lists(mode)
    else:
        list_of_train_persons, list_of_test_persons = get_random_person_list(dir_to_analyze, num_of_test_subjects)

    # sample experiment list just for running fast and developing this script
    if sample_percentage is not None:
        num_of_idx_to_sample = 1
        idx = random.sample(range(len(list_of_train_persons)), num_of_idx_to_sample)
        list_of_train_persons = list(map(list_of_train_persons.__getitem__, idx))
        num_of_idx_to_sample = 1
        idx = random.sample(range(len(list_of_test_persons)), num_of_idx_to_sample)
        list_of_test_persons = list(map(list_of_test_persons.__getitem__, idx))
        del idx

    info_message = 'get segments train'
    logger.info(info_message)
    print(info_message)

    x_train, y1_train, y2_train = get_segments_from_person_list_sbg(
        person_list=list_of_train_persons, dir_to_analyze=dir_to_analyze, window_size=window_size,
        win_size_for_heading_init=win_size_for_heading_init, logger=logger, calc_heading_fix=heading_fix,
        add_quat=add_quat, traj_length_limit=traj_length_limit, heading_fix_initialization_time=heading_fix_initialization_time)
    info_message = 'get segments test'
    logger.info(info_message)
    print(info_message)

    if create_validation_set:
        info_message = 'divide to test-validation segments'
        logger.info(info_message)
        print(info_message)
        num_of_train_samples = round(len(x_train) * 0.8)
        train_idx = random.sample(range(len(x_train)), num_of_train_samples)
        val_idx = [i for i in list(range(len(x_train))) if i not in train_idx]

        x_val = list(map(x_train.__getitem__, val_idx))
        y1_val = list(map(y1_train.__getitem__, val_idx))
        y2_val = list(map(y2_train.__getitem__, val_idx))

        x_train = list(map(x_train.__getitem__, train_idx))
        y1_train = list(map(y1_train.__getitem__, train_idx))
        y2_train = list(map(y2_train.__getitem__, train_idx))
    else:
        x_val = y1_val = y2_val = None

    info_message = 'saving results to ' + outputfolder
    logger.info(info_message)
    print(info_message)
    train_test_division_list = {'train': list_of_train_persons, 'test': list_of_test_persons}
    with open(join(outputfolder, 'train_test_division_list.json'), 'w') as f:
        json.dump(train_test_division_list, f, indent=4)
        f.close()

    sio.savemat(join(outputfolder, 'train.mat'), {'X': x_train, 'Y1': y1_train, 'Y2': y2_train})
    info_message = 'saved train XY'
    logger.info(info_message)
    print(info_message)

    if create_validation_set:
        sio.savemat(join(outputfolder, 'validation.mat'), {'X': x_val, 'Y1': y1_val, 'Y2': y2_val})
        info_message = 'saved validation XY'
        logger.info(info_message)
        print(info_message)

    x_test, y1_test, y2_test = get_segments_from_person_list_sbg(
        person_list=list_of_test_persons, dir_to_analyze=dir_to_analyze, window_size=window_size,
        win_size_for_heading_init=win_size_for_heading_init, logger=logger, calc_heading_fix=heading_fix,
        add_quat=add_quat, heading_fix_initialization_time=heading_fix_initialization_time)

    sio.savemat(join(outputfolder, 'test.mat'), {'X': x_test, 'Y1': y1_test, 'Y2': y2_test})
    info_message = 'saved test XY'
    logger.info(info_message)
    print(info_message)

    return outputfolder


def check_segmentation(segment_list, x):
    segment = segment_list[0]
    lin_acc = segment.lin_acc_b_frame
    lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment.Rot).reshape(1, -1, 3)
    x = x[0]
    e = lin_acc_n_frame - x
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
    segment_list, ahrs_results_list, initial_wd_angles_list = get_segments_from_exp(exp=exp,
                                                                                    window_size=window_size,
                                                                                    use_gt_att=False,
                                                                                    win_size_for_heading_init=750)
    analyzied_segments = []
    walking_angle = []
    d_l = []
    exp.define_walking_start_idx()
    est_time = [exp.Time_IMU[exp.index_of_walking_start - 1]]
    p = exp.Pos.arr()[exp.index_of_walking_start]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        est_time.append(segment.Time_GT[-1])
        ahrs_results = ahrs_results_list[i]
        analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=False,
                                                                   lin_acc_b_frame_est=ahrs_results.lin_acc_b_frame,
                                                                   Rot_est=ahrs_results.Rot,
                                                                   grv_est=ahrs_results.grv,
                                                                   Heading_est=ahrs_results.heading))
        analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        # dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
        dp = segment.Pos.arr()[-1] - p
        p = segment.Pos.arr()[-1]
        d_l.append(np.linalg.norm(dp[0:2]))
        walking_angle.append(analyzied_segments[-1].WD_angle_est)
    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    exp_temp = exp.clone()
    exp_temp.SegmentScenario([exp_temp.Time_IMU[exp_temp.index_of_walking_start],
                              exp_temp.Time_IMU[-1]])
    d_l_ideal = exp_temp.calc_dL(window_size=window_size)
    walking_angle_ideal, wd_ideal = exp_temp.calc_walking_direction(window_size=window_size)
    traj_ideal = construct_traj(d_l_ideal, walking_angle_ideal, plot_result=False, pos_gt=gt_pos)
    traj = construct_traj(np.array(d_l), np.array(walking_angle), plot_result=False, pos_gt=gt_pos)

    # this code is taken from traj_errors, traj_length, error_metrika = calculate_traj_error(Exp, np.array(est_time), traj, dL)
    gt_time = exp.Time_GT[exp.index_of_walking_start:]
    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    traj_length = np.array(d_l).cumsum()
    gt_pos_interp = np.vstack([np.interp(est_time, gt_time, gt_pos[:, 0]),
                               np.interp(est_time, gt_time, gt_pos[:, 1]),
                               np.interp(est_time, gt_time, gt_pos[:, 2])]).T
    x0 = np.array(0.0)

    def minimization_function(alfa, traj_est, traj_gt):
        rot_traj = rotate_trajectory(traj_est, alfa[0])
        return Functions.traj_error(traj_est=rot_traj, traj_gt=traj_gt)
    res = minimize(minimization_function, x0, method='nelder-mead', args=(traj, gt_pos_interp[:, 0:2]),
                   options={'xatol': 1e-8, 'disp': False})

    rot_traj = rotate_trajectory(traj, res.x[0])
    if plot_results:
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
    outputfolder = create_segments(comment='test', window_size=200, mode='mixed', add_quat=True)
    print(outputfolder)
