import random
from os import listdir, mkdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

import utils.Classes as Classes
import utils.Functions as Functions
from utils.Functions import MyCDF
import scipy.io as sio
from datetime import datetime
import json
import pickle

def get_segments_from_exp(exp: Classes.AhrsExp, window_size, sample_id=0, use_GT_att=False, wind_size_for_heading_init=750):
    """
    Creating a list of segments from the experiment samples.
    Each segment contains window_size samples (~250)
    sample_id is useful if want to append such lists
    """
    seg_start_idx = 0
    if isinstance(exp, Classes.SbgExpRawData):
        exp.define_walking_start_idx()
        seg_start_idx = exp.index_of_walking_start
        exp.initial_heading = exp.Psi[seg_start_idx]
    seg_stop_idx = exp.index_of_walking_start + window_size - 1
    segment_list = []
    initial_WD_angles_list = []
    if not use_GT_att:
        AHRS_results_list = []
    # initialize WD angle only for SM heading method
    # analyzed_segment = Classes.WDE_performance_analysis(exp, use_GT_att=True)
    exp.initialize_WD_angle(wind_size_for_heading_init=wind_size_for_heading_init, plot_results=False)
    initial_WD_angle = exp.initial_WD_angle_GT
    if not use_GT_att:
        t_est, lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(exp)
        exp.initial_heading = Heading_est[exp.index_of_walking_start]
    while seg_stop_idx <= exp.NumberOfSamples_GT - 1:
        segment = exp.clone()  # TODO - is there a way to clone only the samples we need?
        segment.SegmentScenario([exp.Time_GT[seg_start_idx], exp.Time_GT[seg_stop_idx]])
        segment.id = sample_id
        if not use_GT_att:
            # AHRS results segmentation
            t_start = segment.Time_IMU[0].round(11)
            t_end = segment.Time_IMU[-1].round(11)
            ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
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
        seg_start_idx += window_size
        seg_stop_idx += window_size
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


def get_segments_from_exp_list(exp_list, dataset, window_size, shuffle=True, use_GT_att=False, wind_size_for_heading_init=750):
    """
    Same as get_segments_from_exp, but now we append all lists from all files
    exp_list is a list of experiment paths
    """
    segment_list = []
    initial_WD_angles_list = []
    if not use_GT_att:
        AHRS_results_segments_list = []
    sample_id = 0
    i=1
    N = len(exp_list)
    for item in exp_list:
        # load exp file
        print(str(round(i/N * 100,3)) + '%')
        i += 1
        if dataset == 'RIDI':
            exp = Classes.RidiExp(item)
        elif dataset == 'TRC#1':
            exp = Classes.SbgExpRawData(item)
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
        return segment_list, initial_WD_angles_list
    else:
        return segment_list, AHRS_results_segments_list, initial_WD_angles_list


# def EstimatePsiWithAHRS(exp, wind_size_for_estimating_heading, plot_heading_initialization=False):
#     """
#     Initiate AHRS module and calculate current psi based on the first {wind_size_for_learning} samples
#     Then, change (override) the exp.psi field
#     TODO:
#         3) add field for raw_psi and fix_psi
#     """
#     first_segment = exp.clone()
#     first_segment.SegmentScenario([exp.Time_GT[0], exp.Time_GT[wind_size_for_estimating_heading]])
#     class_instance_to_initialize_heading = Classes.WDE_performance_analysis(first_segment)
#     class_instance_to_initialize_heading.segment.Psi = class_instance_to_initialize_heading.segment.Psi - \
#                                                        class_instance_to_initialize_heading.segment.Psi[0] + \
#                                                        class_instance_to_initialize_heading.WD_angle_GT
#     estimated_psi = class_instance_to_initialize_heading.WD_angle_GT
#     if plot_heading_initialization:
#         class_instance_to_initialize_heading.walking_direction_estimation_using_smartphone_heading(plot_results=True)
#         plt.show()
#     return estimated_psi


def pos_acc_correlation(Exp: Classes.AhrsExp):
    """plot linear acceleration in navigation frame vs position 2nd derivative
    to verify correct coordinate frame handling"""
    # Position GT 2nd derivative:
    dt = np.diff(Exp.Time_GT).mean()
    pos_gt = Exp.Pos.arr()
    pos_der = np.diff(pos_gt, axis=0) / dt
    pos_2nd_der = np.diff(pos_der, axis=0) / dt
    # Accelaration GT;
    lin_acc_GT = Exp.LinAcc.arr()
    lin_acc_GT_n_frame = Functions.transform_vectors(lin_acc_GT, Exp.Rot)
    # Plots
    fig = plt.figure('Position Acc correlation')
    Ax_pos_x = fig.add_subplot(311)
    Ax_pos_x.plot(Exp.Time_GT[0: -2], pos_2nd_der[:, 0], color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
    Ax_pos_x.plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 0], color='red', linewidth=1, label = 'Accelerometer')
    Ax_pos_x.set_ylim(lin_acc_GT_n_frame[:, 0].min(), lin_acc_GT_n_frame[:, 0].max()),
    Ax_pos_x.set(title="Position 2nd derivative VS $a^n_L$", ylabel="$[m/sec^2]$")
    Ax_pos_x.grid(True);Ax_pos_x.legend()

    Ax_pos_y = fig.add_subplot(312, sharex=Ax_pos_x)
    Ax_pos_y.plot(Exp.Time_GT[0: -2], pos_2nd_der[:, 1], color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
    Ax_pos_y.plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 1], color='red', linewidth=1, label = 'Accelerometer')
    Ax_pos_y.set_ylim(lin_acc_GT_n_frame[:, 1].min(), lin_acc_GT_n_frame[:, 1].max()),
    Ax_pos_y.set(ylabel="$[m/sec^2]$")
    Ax_pos_y.grid(True);Ax_pos_y.legend()

    Ax_pos_z = fig.add_subplot(313, sharex=Ax_pos_x)
    Ax_pos_z.plot(Exp.Time_GT[0: -2], pos_2nd_der[:, 2], color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
    Ax_pos_z.plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 2], color='red', linewidth=1, label = 'Accelerometer')
    Ax_pos_z.set_ylim(lin_acc_GT_n_frame[:, 2].min(), lin_acc_GT_n_frame[:, 2].max()),
    Ax_pos_z.set(xlabel="Time [sec]", ylabel="$[m/sec^2]$")
    Ax_pos_z.grid(True);Ax_pos_z.legend()


def get_AHRS_results_for_exp(exp):
    suffix = '_AHRS_results.xlsx'
    AHRS_results_file_path = join(exp.Path, exp.FileName.split(sep='.')[0] + suffix)
    t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
        Functions.read_AHRS_results(AHRS_results_file_path)
    # segmentation
    t_start = exp.Time_IMU[0].round(11)
    t_end = exp.Time_IMU[-1].round(11)
    ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11)<= t_end))
    lin_acc_b_frame = exp.Acc.arr() - grv_hat[ind_IMU]
    Rot = Rot_hat[ind_IMU]
    Heading = np.array(psi_hat)[ind_IMU] + exp.Psi[0]
    return t_est, lin_acc_b_frame, grv_hat, Rot, Heading


class segments_initial_WD_angles():
    def __init__(self, initial_WD_angle=None, id=None):
        self.id = id
        self.initial_WD_angle = initial_WD_angle


if __name__ == '__main__':
    plot_WDE_example = False
    wind_size_for_heading_init = 1000
    segment_example_idx = None
    validate_segments_with_turn_identification = False # if True segments are either turns or straight lines
    analyze_staight_lines = False # if False, turns are analyzed
    use_GT_att = False # use gournd truth linear acceleration and b->n rotations else use AHRS
    experiment_example_idx = None # if work_on_directory==False, this specifies the experiment idx
    window_size = 200 # for res net use the same window size as in training
    work_on_directory = False
    analyze_WDE_performance = True
    plot_turn_identification = False
    view_valid_traj = False # these are used to visualize/debug the turn detection
    view_invalid_traj = False
    WD_est_method = 'PCA' # could be: 'SM_heading', 'PCA', 'inverted_pendulum', 'resnet18LinAcc', 'resnet18RawIMU' , 'compare_all_methods'
    main_wd_path = '/'
    error_type = 'end_pos' # could be: 'WD_angle', 'end_pos'
    mode = 'swing'  # could be: 'text', 'pocket', 'body', 'bag', 'swing'
    data_location = 'magneto'  # could be 'magneto' or 'local_machine'
    dataset = 'TRC#1'  # could be: 'RIDI', 'TRC#1',
    exp_list_train = None
    root_dir = None
    root_of_roots = None
    person_idx_list = None
    sample_percentage = None
    varify_with_PCA = False
    now = datetime.isoformat(datetime.now())
    description = dataset + ' ' + mode + ' ' + 'wind_size ' + str(window_size)
    print(description)
    outputfolder = join(main_wd_path, 'data', 'XY_pairs', now + description)
    num_of_test_subjects = 2
    ###########################################################################
    if exp_list_train is not None:
        assert (root_dir is None)
        assert (root_of_roots is None)
        with open(exp_list_train) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    ###########################################################################
    # get main dir
    if mode == 'text':
        if data_location == 'magneto':
            if dataset == 'RIDI':
                dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Text - Test'
            elif dataset == 'TRC#1':
                dir_to_analyze = '/data/Datasets/Navigation/SBG-PDR-DATA/texting'
        elif data_location == 'local_machine':
            if dataset == 'RIDI':
                dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Text_Test'
            elif dataset == 'TRC#1':
                dir_to_analyze = '/home/maint/Eran/AHRS/SBG-PDR-DATA/texting'
    elif mode == 'pocket':
        if data_location == 'magneto':
            if dataset == 'RIDI':
                dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Test - Test'
            elif dataset == 'TRC#1':
                dir_to_analyze = '/data/Datasets/Navigation/SBG-PDR-DATA/pocket'
        elif data_location == 'local_machine':
            if dataset == 'RIDI':
                dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Pocket_Test'
            elif dataset == 'TRC#1':
                dir_to_analyze = '/home/maint/Eran/AHRS/SBG-PDR-DATA/pocket'
    elif mode == 'bag':
        assert dataset == 'RIDI'
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Bag - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Bag_Test'
    elif mode == 'body':
        assert dataset == 'RIDI'
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Body - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Body_Test'
    elif mode == 'swing':
        assert dataset == 'TRC#1'
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/SBG-PDR-DATA/swing_R'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/SBG-PDR-DATA/swing_R'
    else:
        raise 'invalide mode'
    # get experiment list from directory
    print('get experiment list from directory')
    if dataset == 'RIDI':
        exp_list_train = [join(dir_to_analyze, item) for item in listdir(dir_to_analyze) if
                    'AHRS_results' not in item]
    elif dataset == 'TRC#1':
        person_list = listdir(dir_to_analyze)
        if person_idx_list is None:
            num_of_train_subjects = len(person_list) - num_of_test_subjects
            person_idx_list = random.sample(range(len(person_list)), num_of_train_subjects)
        list_of_train_persons = []
        list_of_test_persons = []
        for i in range(len(person_list)):
            if i in person_idx_list:
                list_of_train_persons.append(person_list[i])
            else:
                list_of_test_persons.append(person_list[i])
        person_list = list(map(person_list.__getitem__, person_idx_list))
        exp_list_train = []
        for directory in list_of_train_persons:
            dir_path = join(dir_to_analyze, directory)
            exp_in_dir = [join(dir_path, item) for item in listdir(dir_path) if
                          'AHRS_results' not in item and 'ascii-output.txt' not in item]
            exp_list_train.extend(exp_in_dir)
        exp_list_test = []
        for directory in list_of_test_persons:
            dir_path = join(dir_to_analyze, directory)
            exp_in_dir = [join(dir_path, item) for item in listdir(dir_path) if
                          'AHRS_results' not in item and 'ascii-output.txt' not in item]
            exp_list_test.extend(exp_in_dir)
    # sample experiment list just for running fast and developing this script
    if sample_percentage is not None:
        num_of_idx_to_sample = round(len(exp_list_train) * sample_percentage / 100)
        idx = random.sample(range(len(exp_list_train)), num_of_idx_to_sample)
        exp_list_train = list(map(exp_list_train.__getitem__, idx))
        num_of_idx_to_sample = round(len(exp_list_test) * sample_percentage / 100)
        idx = random.sample(range(len(exp_list_test)), num_of_idx_to_sample)
        exp_list_test = list(map(exp_list_test.__getitem__, idx))
        del idx
    # for debug of certain experiment  choose experiment_example_idx
    if experiment_example_idx is not None:
        print('idx = ' + str(experiment_example_idx))
        exp_list_train = [exp_list_train[experiment_example_idx]]
    # get segments
    if use_GT_att:
        assert dataset == 'RIDI'
        print('get segments train')
        segments_train, initial_WD_angles_list_train = \
            get_segments_from_exp_list(exp_list_train, dataset=dataset, window_size=window_size, shuffle=False, use_GT_att=use_GT_att,
                                       wind_size_for_heading_init=wind_size_for_heading_init)
        print('get segments test')
        segments_test, initial_WD_angles_list_test = \
            get_segments_from_exp_list(exp_list_test, dataset=dataset, window_size=window_size, shuffle=False,
                                       use_GT_att=use_GT_att,
                                       wind_size_for_heading_init=wind_size_for_heading_init)
    else:
        print('get segments train')
        segments_train, AHRS_results_segments_list_train, initial_WD_angles_list_train = \
            get_segments_from_exp_list(exp_list_train, dataset=dataset, window_size=window_size,
                                       shuffle=False,
                                       use_GT_att=use_GT_att,
                                       wind_size_for_heading_init=wind_size_for_heading_init)
        print('get segments test')
        segments_test, AHRS_results_segments_list_test, initial_WD_angles_list_test = \
            get_segments_from_exp_list(exp_list_test, dataset=dataset, window_size=window_size,
                                       shuffle=False,
                                       use_GT_att=use_GT_att,
                                       wind_size_for_heading_init=wind_size_for_heading_init)
    # divide to turn/straight line segments
    if validate_segments_with_turn_identification:
        print('divide to turn/straight line segments')
        staight_line_segments = []
        staight_line_segments_AHRS_results = []
        staight_line_initial_WD_angles = []
        turn_segments = []
        turn_segments_AHRS_results = []
        turn_initial_WD_angles = []
        train_PCA_errors = []

        for i in range(len(segments_train)):
            segment = segments_train[i]
            initial_WD_seg = initial_WD_angles_list_train[i]
            if not use_GT_att:
                AHRS_results_seg = AHRS_results_segments_list_train[i]
            candidate = Classes.WDE_performance_analysis(segment, use_GT_att=True)
            candidate.identify_turn(threshold=0.3, plot_results=False)
            if not candidate.turn_identified:
                staight_line_segments.append(segment)
                staight_line_initial_WD_angles.append(initial_WD_seg)
                if not use_GT_att:
                    staight_line_segments_AHRS_results.append(AHRS_results_seg)
            else:
                turn_segments.append(segment)
                turn_initial_WD_angles.append(initial_WD_seg)
                if not use_GT_att:
                    turn_segments_AHRS_results.append(AHRS_results_seg)
        if analyze_staight_lines:
            segments_train = staight_line_segments
            AHRS_results_segments_list_train = staight_line_segments_AHRS_results
            initial_WD_angles_list_train = staight_line_initial_WD_angles
        else:
            segments_train = turn_segments
            AHRS_results_segments_list_train = turn_segments_AHRS_results
            initial_WD_angles_list_train = turn_initial_WD_angles
        print('number of straight line segments = ' + str(len(staight_line_segments)))
        print('number of turn segments = ' + str(len(turn_segments)))
    # divide train, to train and validation.
    num_of_train_samples = round(len(segments_train) * 0.8)
    num_of_validation_samples = len(segments_train) - num_of_train_samples
    num_of_test_samples = len(segments_test)
    train_idx = random.sample(range(len(segments_train)), num_of_train_samples)
    validation_idx = list(range(len(segments_train)))
    for idx in train_idx:
        validation_idx.remove(idx)
    train_segments = list(map(segments_train.__getitem__, train_idx))
    validation_segments = list(map(segments_train.__getitem__, validation_idx))

    train_initial_WD_angles_list = list(map(initial_WD_angles_list_train.__getitem__, train_idx))
    validation_initial_WD_angles_list = list(map(initial_WD_angles_list_train.__getitem__, validation_idx))

    if not use_GT_att:
        train_AHRS_results_segments_list = list(map(AHRS_results_segments_list_train.__getitem__, train_idx))
        validation_AHRS_results_segments_list = list(map(AHRS_results_segments_list_train.__getitem__, validation_idx))

    train_segments_for_WDE = []
    train_PCA_errors = []
    print('initiate segments_for_WDE')
    for i in range(num_of_train_samples):
        segment = train_segments[i]
        if not use_GT_att:
            AHRS_results_seg = train_AHRS_results_segments_list[i]
        initial_WD_seg = train_initial_WD_angles_list[i]
        if not use_GT_att:
            assert segment.id == AHRS_results_seg.id
        if use_GT_att:
            train_segments_for_WDE.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        else:
            train_segments_for_WDE.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                           lin_acc_b_frame_est=AHRS_results_seg.lin_acc_b_frame,
                                                                           grv_est=AHRS_results_seg.grv,
                                                                           Heading_est=AHRS_results_seg.heading,
                                                                           Rot_est=AHRS_results_seg.Rot
                                                                           )
                                          )
        train_segments_for_WDE[-1].initial_WD_angle_GT = initial_WD_seg.initial_WD_angle
        if varify_with_PCA:
            train_segments_for_WDE[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
            WD_angle_err, end_pos_err = train_segments_for_WDE[-1].calc_error()
            if error_type == 'WD_angle':
                train_PCA_errors.append(WD_angle_err)
            elif error_type == 'end_pos':
                train_PCA_errors.append(end_pos_err)
    if varify_with_PCA:
        if error_type == 'WD_angle':
            print('mean error = ' + str(np.array(train_PCA_errors).mean() * 180 / np.pi))
            print('error std = ' + str(np.array(train_PCA_errors).std() * 180 / np.pi))
            print('mean abs error = ' + str(abs(np.array(train_PCA_errors)).mean() * 180 / np.pi))
            fig = plt.figure('WDE angle errors train')
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.plot(np.array(train_PCA_errors) * 180 / np.pi)
            ax1.set(ylabel='WD angle [deg]')
            ax2 = fig.add_subplot(212)
            ax2.hist(np.array(train_PCA_errors) * 180 / np.pi, 100)
            ax2.grid(True)
            ax2.set(ylabel='samples', xlabel='WD angle [deg]')

            plt.figure('Angle CDF')
            bins_count,cdf = MyCDF(np.array(train_PCA_errors))
            plt.plot(bins_count[1:]* 180 / np.pi,cdf,label='PCA')
            plt.grid();plt.legend()
            plt.xlabel('WD angle [deg]');plt.ylabel('CDF')
            plt.show()
        elif error_type == 'end_pos':
            print('mean error = ' + str(np.array(train_PCA_errors).mean()))
            print('error std = ' + str(np.array(train_PCA_errors).std()))
            print('mean abs error = ' + str(abs(np.array(train_PCA_errors)).mean()))
            fig = plt.figure('WDE pos errors')
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.plot(np.array(train_PCA_errors))
            ax1.set(ylabel='WD pos error [m]')
            ax2 = fig.add_subplot(212)
            ax2.hist(np.array(train_PCA_errors), 100)
            ax2.grid(True)
            ax2.set(ylabel='samples', xlabel='WD pos error [m]')

    validation_segments_for_WDE = []
    validation_PCA_errors = []

    for i in range(num_of_validation_samples):
        segment = validation_segments[i]
        if not use_GT_att:
            AHRS_results_seg = validation_AHRS_results_segments_list[i]
        initial_WD_seg = validation_initial_WD_angles_list[i]
        if not use_GT_att:
            assert segment.id == AHRS_results_seg.id
        if use_GT_att:
            validation_segments_for_WDE.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        else:
            validation_segments_for_WDE.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                           lin_acc_b_frame_est=AHRS_results_seg.lin_acc_b_frame,
                                                                           grv_est=AHRS_results_seg.grv,
                                                                           Heading_est=AHRS_results_seg.heading,
                                                                           Rot_est=AHRS_results_seg.Rot
                                                                           )
                                          )
        validation_segments_for_WDE[-1].initial_WD_angle_GT = initial_WD_seg.initial_WD_angle
        if varify_with_PCA:
            validation_segments_for_WDE[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
            WD_angle_err, end_pos_err = validation_segments_for_WDE[-1].calc_error()
            if error_type == 'WD_angle':
                validation_PCA_errors.append(WD_angle_err)
            elif error_type == 'end_pos':
                validation_PCA_errors.append(end_pos_err)
    if varify_with_PCA:
        if error_type == 'WD_angle':
            print('mean error = ' + str(np.array(validation_PCA_errors).mean() * 180 / np.pi))
            print('error std = ' + str(np.array(validation_PCA_errors).std() * 180 / np.pi))
            print('mean abs error = ' + str(abs(np.array(validation_PCA_errors)).mean() * 180 / np.pi))
            fig = plt.figure('WDE angle errors validation')
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.plot(np.array(validation_PCA_errors) * 180 / np.pi)
            ax1.set(ylabel='WD angle [deg]')
            ax2 = fig.add_subplot(212)
            ax2.hist(np.array(validation_PCA_errors) * 180 / np.pi, 100)
            ax2.grid(True)
            ax2.set(ylabel='samples', xlabel='WD angle [deg]')

            plt.figure('Angle CDF')
            bins_count, cdf = MyCDF(np.array(validation_PCA_errors))
            plt.plot(bins_count[1:] * 180 / np.pi, cdf, label='PCA')
            plt.grid();
            plt.legend()
            plt.xlabel('WD angle [deg]');
            plt.ylabel('CDF')
            plt.show()
        elif error_type == 'end_pos':
            print('mean error = ' + str(np.array(validation_PCA_errors).mean()))
            print('error std = ' + str(np.array(validation_PCA_errors).std()))
            print('mean abs error = ' + str(abs(np.array(validation_PCA_errors)).mean()))
            fig = plt.figure('WDE pos errors')
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.plot(np.array(validation_PCA_errors))
            ax1.set(ylabel='WD pos error [m]')
            ax2 = fig.add_subplot(212)
            ax2.hist(np.array(validation_PCA_errors), 100)
            ax2.grid(True)
            ax2.set(ylabel='samples', xlabel='WD pos error [m]')

    test_segments_for_WDE = []
    test_PCA_errors = []
    test_segments = segments_test
    test_AHRS_results_segments_list = AHRS_results_segments_list_test
    test_initial_WD_angles_list = initial_WD_angles_list_test
    for i in range(num_of_test_samples):
        segment = test_segments[i]
        if not use_GT_att:
            AHRS_results_seg = test_AHRS_results_segments_list[i]
        initial_WD_seg = test_initial_WD_angles_list[i]
        if not use_GT_att:
            assert segment.id == AHRS_results_seg.id
        if use_GT_att:
            test_segments_for_WDE.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        else:
            test_segments_for_WDE.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                                lin_acc_b_frame_est=AHRS_results_seg.lin_acc_b_frame,
                                                                                grv_est=AHRS_results_seg.grv,
                                                                                Heading_est=AHRS_results_seg.heading,
                                                                                Rot_est=AHRS_results_seg.Rot
                                                                                )
                                               )
        test_segments_for_WDE[-1].initial_WD_angle_GT = initial_WD_seg.initial_WD_angle
        if varify_with_PCA:
            test_segments_for_WDE[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
            WD_angle_err, end_pos_err = test_segments_for_WDE[-1].calc_error()
            if error_type == 'WD_angle':
                test_PCA_errors.append(WD_angle_err)
            elif error_type == 'end_pos':
                test_PCA_errors.append(end_pos_err)
    if varify_with_PCA:
        if error_type == 'WD_angle':
            print('mean error = ' + str(np.array(test_PCA_errors).mean() * 180 / np.pi))
            print('error std = ' + str(np.array(test_PCA_errors).std() * 180 / np.pi))
            print('mean abs error = ' + str(abs(np.array(test_PCA_errors)).mean() * 180 / np.pi))
            fig = plt.figure('WDE angle errors test')
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.plot(np.array(test_PCA_errors) * 180 / np.pi)
            ax1.set(ylabel='WD angle [deg]')
            ax2 = fig.add_subplot(212)
            ax2.hist(np.array(test_PCA_errors) * 180 / np.pi, 100)
            ax2.grid(True)
            ax2.set(ylabel='samples', xlabel='WD angle [deg]')

            plt.figure('Angle CDF')
            bins_count, cdf = MyCDF(np.array(test_PCA_errors))
            plt.plot(bins_count[1:] * 180 / np.pi, cdf, label='PCA')
            plt.grid()
            plt.legend()
            plt.xlabel('WD angle [deg]')
            plt.ylabel('CDF')
            plt.show()
        elif error_type == 'end_pos':
            print('mean error = ' + str(np.array(test_PCA_errors).mean()))
            print('error std = ' + str(np.array(test_PCA_errors).std()))
            print('mean abs error = ' + str(abs(np.array(test_PCA_errors)).mean()))
            fig = plt.figure('WDE pos errors')
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.plot(np.array(test_PCA_errors))
            ax1.set(ylabel='WD pos error [m]')
            ax2 = fig.add_subplot(212)
            ax2.hist(np.array(test_PCA_errors), 100)
            ax2.grid(True)
            ax2.set(ylabel='samples', xlabel='WD pos error [m]')
    print('convert to XY pairs')
    # convert train/validation_segments_for_WDE to X,Y. X->NX200X3 (lin acc in n frame), Y->N NX2 (WD vector)
    X_train = np.zeros((0, window_size, 3))
    Y1_train = np.zeros((0, 1))
    Y2_train = np.zeros((0, 2))
    for i in range(num_of_train_samples):
        segment_for_WDE = train_segments_for_WDE[i]
        lin_acc = segment_for_WDE.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment_for_WDE.Rot).reshape(1,window_size,3) # window_sizeX3
        WD_vector = segment_for_WDE.WD_vector_GT.reshape(1,2)
        dL = segment_for_WDE.dL.reshape(1, 1)
        X_train = np.concatenate((X_train, lin_acc_n_frame), axis=0)
        # For odometry
        Y1_train = np.concatenate((Y1_train, dL), axis=0)
        # For WDE
        Y2_train = np.concatenate((Y2_train, WD_vector), axis=0)
    X_validation = np.zeros((0, window_size, 3))
    Y1_validation = np.zeros((0, 1))
    Y2_validation = np.zeros((0, 2))
    for i in range(num_of_validation_samples):
        segment_for_WDE = validation_segments_for_WDE[i]
        lin_acc = segment_for_WDE.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment_for_WDE.Rot).reshape(1, window_size,
                                                                                            3)  # window_sizeX3
        WD_vector = segment_for_WDE.WD_vector_GT.reshape(1, 2)
        dL = segment_for_WDE.dL.reshape(1, 1)
        X_validation = np.concatenate((X_validation, lin_acc_n_frame), axis=0)
        # For odometry
        Y1_validation = np.concatenate((Y1_validation, dL), axis=0)
        # For WDE
        Y2_validation = np.concatenate((Y2_validation, WD_vector), axis=0)
    X_test = np.zeros((0, window_size, 3))
    Y1_test = np.zeros((0, 1))
    Y2_test = np.zeros((0, 2))
    for i in range(num_of_test_samples):
        segment_for_WDE = test_segments_for_WDE[i]
        lin_acc = segment_for_WDE.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment_for_WDE.Rot).reshape(1, window_size,
                                                                                            3)  # window_sizeX3
        WD_vector = segment_for_WDE.WD_vector_GT.reshape(1, 2)
        dL = segment_for_WDE.dL.reshape(1, 1)
        X_test = np.concatenate((X_test, lin_acc_n_frame), axis=0)
        # For odometry
        Y1_test = np.concatenate((Y1_test, dL), axis=0)
        # For WDE
        Y2_test = np.concatenate((Y2_test, WD_vector), axis=0)
    print('saving results to ' + outputfolder)
    mkdir(outputfolder)
    train_test_division_list = {'train': list_of_train_persons, 'test': list_of_test_persons}
    with open(join(outputfolder, 'train_test_division_list.json'), 'w') as f:
        json.dump(train_test_division_list, f, indent=4)
        f.close()
    sio.savemat(join(outputfolder, 'train.mat'), {'X': X_train, 'Y1': Y1_train, 'Y2': Y2_train})
    sio.savemat(join(outputfolder, 'validation.mat'), {'X': X_validation, 'Y1': Y1_validation, 'Y2': Y2_validation})
    sio.savemat(join(outputfolder, 'test.mat'), {'X': X_test, 'Y1': Y1_test, 'Y2': Y2_test})
    with open(join(outputfolder, 'train_segments_for_WDE.pickle'), 'wb') as f:
        pickle.dump(train_segments_for_WDE, f)
    with open(join(outputfolder, 'test_segments_for_WDE.pickle'), 'wb') as f:
        pickle.dump(test_segments_for_WDE, f)
    with open(join(outputfolder, 'validation_segments_for_WDE.pickle'), 'wb') as f:
        pickle.dump(validation_segments_for_WDE, f)
    plt.show()
