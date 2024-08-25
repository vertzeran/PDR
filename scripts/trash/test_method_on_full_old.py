import utils.Classes as Classes
import matplotlib.pyplot as plt
import numpy as np
import utils.Functions as Functions
import random
from os import listdir
from os.path import join
from utils.Functions import construct_traj


def get_segments_from_exp(exp: Classes.AhrsExp, window_size, sample_id=0):
    """
    Creating a list of segments from the experiment samples.
    Each segment contains window_size samples (~250)
    sample_id is useful if want to append such lists
    """
    seg_start_idx = exp.index_of_walking_start
    seg_stop_idx = exp.index_of_walking_start + window_size - 1
    segment_list = []
    wind_size_for_estimating_heading = 750

    # # Calculate the real psi with AHRS
    # if fix_psi:
    #     estimated_psi = estimiate_initial_traj_direction(exp, wind_size_for_estimating_heading)
    #     exp.Psi = exp.Psi - exp.Psi[0] + estimated_psi

    while seg_stop_idx <= exp.NumberOfSamples_GT - 1:
        segment = exp.clone()  # TODO - is there a way to clone only the samples we need?
        segment.SegmentScenario([exp.Time_GT[seg_start_idx], exp.Time_GT[seg_stop_idx]])
        segment.id = sample_id
        sample_id += 1
        segment_list.append(segment)
        seg_start_idx += window_size
        seg_stop_idx += window_size
    return segment_list


def get_sagments_from_dir(exp_dir, window_size, shuffle=True):
    """
    Same as get_segments_from_exp, but now we append all lists from all files
    after we had fixed their psi value
    exp_dir a path that includes experiment directories

    """
    segment_list = []
    exp_list = []
    sample_id = 0
    for item in listdir(exp_dir):
        if 'AHRS_results' not in item:
            # load exp file
            item_path = join(exp_dir, item)  # path to csv file
            exp = Classes.RidiExp(item_path)

            # Extract segments of the exp with the fixed psi flag
            segment_list.extend(get_segments_from_exp(exp, window_size, sample_id))
            exp_list.append(exp)
            sample_id = segment_list[-1].id + 1

        if False:
            # exp.Pos.x = exp.Pos.x - exp.Pos.x[0]
            # exp.Pos.y = exp.Pos.y - exp.Pos.y[0]
            # exp.Pos.z = exp.Pos.z - exp.Pos.z[0]
            temp_obj = WDE_performance_analysis(exp)
            temp_obj.walking_direction_estimation_using_smartphone_heading(
                plot_results=True)
            plt.show()

    if shuffle:
        random.shuffle(segment_list)

    return segment_list, exp_list


# def estimiate_initial_traj_direction(exp, wind_size_for_estimating_heading, plot_heading_initialization=False):
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

def traj_est_using_SP_heading(Exp, window_size, plot_result=False, use_GT_att=False):
    if not use_GT_att:
        t_est,lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(Exp)

    Exp.define_walking_start_idx(th=1)
    Exp.initialize_WD_angle(wind_size_for_heading_init=500)
    Heading_est = Heading_est - Heading_est[0] # we initialize heading using GT position
    segments = get_segments_from_exp(Exp, window_size)
    analyzied_segments = []
    traj = np.array([0, 0])
    gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:]
    gt_pos = gt_pos - gt_pos[0]
    for segment in segments:
        if not use_GT_att:
            # AHRS results segmentation
            t_start = segment.Time_IMU[0].round(11)
            t_end = segment.Time_IMU[-1].round(11)
            ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            Rot_seg = Rot_est[ind_IMU]
            Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=lin_acc_b_frame_seg,
                                                                       grv_est=grv_est,
                                                                       Rot_est=Rot_seg, Heading_est=Heading_seg))
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        analyzied_segments[-1].walking_direction_estimation_using_smartphone_heading(plot_results=False)
        traj = np.vstack([traj, traj[-1] + analyzied_segments[-1].est_traj])
    if plot_result:
        fig = plt.figure('construct_traj plot')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='')
        ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                color='black', linestyle='--', linewidth=2, label='')
        plt.show()

    return traj


def traj_est_using_inv_pend(Exp, window_size, plot_result=False, use_GT_att=False):
    if not use_GT_att:
        t_est,lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(Exp)

    Exp.define_walking_start_idx(th=1)
    Exp.initialize_WD_angle(wind_size_for_heading_init=1000)
    segments = get_segments_from_exp(Exp, window_size)
    analyzied_segments = []
    traj = np.array([0, 0])
    for segment in segments:
        if not use_GT_att:
            # AHRS results segmentation
            t_start = segment.Time_IMU[0].round(11)
            t_end = segment.Time_IMU[-1].round(11)
            ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            Rot_seg = Rot_est[ind_IMU]
            Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=lin_acc_b_frame_seg,
                                                                       grv_est=grv_est,
                                                                       Rot_est=Rot_seg, Heading_est=Heading_seg))
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        analyzied_segments[-1].inverted_pendulum_model(plot_results=False)
        traj = np.vstack([traj, traj[-1] + analyzied_segments[-1].est_traj])
        gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:]
        gt_pos = gt_pos - gt_pos[0]
    if plot_result:
        fig = plt.figure('construct_traj plot')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='')
        ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                color='black', linestyle='--', linewidth=2, label='')
        plt.show()

    return traj


def traj_est_using_PCA(Exp, window_size, plot_result=False, use_GT_att=False):
    if not use_GT_att:
        t_est,lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(Exp)
        # suffix = '_AHRS_results.xlsx'
        # AHRS_results_file_path = join(Exp.Path, Exp.FileName.split(sep='.')[0] + suffix)
        # exp_path = join(Exp.Path, Exp.FileName)
        # t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
        #     Functions.read_AHRS_results(AHRS_results_file_path)
        # lin_acc_b_frame_est = Exp.Acc.arr() - grv_hat
        # Heading_est = np.array(psi_hat) + Exp.Psi[0]
    segments = get_segments_from_exp(Exp, window_size)
    analyzied_segments = []
    walking_angle = []
    dL = []
    for segment in segments:
        if not use_GT_att:
            # AHRS results segmentation
            t_start = segment.Time_IMU[0].round(11)
            t_end = segment.Time_IMU[-1].round(11)
            ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            Rot_seg = Rot_est[ind_IMU]
            Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=lin_acc_b_frame_seg,
                                                                       grv_est=grv_est,
                                                                       Rot_est=Rot_seg, Heading_est=Heading_seg))
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
        walking_angle.append(analyzied_segments[-1].WD_angle_est)
    traj = construct_traj(np.array(dL), np.array(walking_angle), plot_result=plot_result, pos_gt=Exp.Pos.arr() - Exp.Pos.arr()[0, :])
    return traj, walking_angle


def traj_est_using_resnet18(Exp, window_size, dL, plot_result=False, use_GT_att=False):
    if not use_GT_att:
        t_est,lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(Exp)
    segments = get_segments_from_exp(Exp, window_size)
    analyzied_segments = []
    walking_angle = []
    dL = []
    for segment in segments:
        if not use_GT_att:
            # AHRS results segmentation
            t_start = segment.Time_IMU[0].round(11)
            t_end = segment.Time_IMU[-1].round(11)
            ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            Rot_seg = Rot_est[ind_IMU]
            Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=lin_acc_b_frame_seg,
                                                                       grv_est=grv_est,
                                                                       Rot_est=Rot_seg, Heading_est=Heading_seg))
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        analyzied_segments[-1].window_size = window_size
        analyzied_segments[-1].model_path = model_path
        analyzied_segments[-1].res18_direction_pred(plot_results=False, data_type = 'LinAcc')
        walking_angle.append(analyzied_segments[-1].WD_angle_est)
        dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
    traj = construct_traj(np.array(dL), np.array(walking_angle), plot_result=plot_result, pos_gt=Exp.Pos.arr() - Exp.Pos.arr()[0, :])
    return traj, walking_angle


def traj_est_using_resnet18_raw(Exp, window_size, dL, plot_result=False, use_GT_att=False):
    if not use_GT_att:
        t_est,lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(Exp)
    segments = get_segments_from_exp(Exp, window_size)
    analyzied_segments = []
    walking_angle = []
    for segment in segments:
        if not use_GT_att:
            # AHRS results segmentation
            t_start = segment.Time_IMU[0].round(11)
            t_end = segment.Time_IMU[-1].round(11)
            ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            Rot_seg = Rot_est[ind_IMU]
            Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=lin_acc_b_frame_seg,
                                                                       grv_est=grv_est,
                                                                       Rot_est=Rot_seg, Heading_est=Heading_seg))
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
        analyzied_segments[-1].window_size = window_size
        analyzied_segments[-1].model_raw_path = model_raw_path
        analyzied_segments[-1].res18_direction_pred(plot_results=False,data_type = 'RawIMU')
        walking_angle.append(analyzied_segments[-1].WD_angle_est)
    traj = construct_traj(dL, np.array(walking_angle), plot_result=plot_result, pos_gt=Exp.Pos.arr() - Exp.Pos.arr()[0, :])
    return traj, walking_angle


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


if __name__ == '__main__':
    window_size = 200
    wind_size_for_heading_init = 1000
    WD_est_method = 'inverted_pendulum'  # could be: 'SP_heading', 'PCA', 'inverted_pendulum', 'resnet18LinAcc', 'resnet18RawIMU', 'compare_all_methods'
    experiment_example_idx = 7
    person_idx = 4 # only for dataset==TRC#1 4 for text->ran_dev and 6 for pocket->omri_dev
    work_on_directory = True
    main_wd_path = '/'
    mode = 'text' # could be: 'text', 'pocket', 'body', 'bag'
    data_location = 'magneto' # could be 'magneto' or 'local_machine'
    limit_exp_length = False
    time_limit = 60
    use_GT_att = False
    dataset = 'TRC#1' #could be: 'RIDI', 'TRC#1',

    # define directory for analysis and resnet model paths
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
        if dataset == 'RIDI':
            model_path = join('data/models', 'WDE_regressor_Text_LinAcc_0.902.pth')
            model_raw_path = join('data/models', 'WDE_regressor_Text_RawIMU_0.902.pth')
        elif dataset == 'TRC#1':
            model_path = join('data/models', '')
            model_raw_path = join('data/models', '')
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
        if dataset == 'RIDI':
            model_path = join('data/models', 'WDE_regressor_Pocket_LinAcc_0.9.pth')
            model_raw_path = join('data/models', 'WDE_regressor_Pocket_RawIMU_0.902.pth')
        elif dataset == 'TRC#1':
            model_path = join('data/models', '')
            model_raw_path = join('data/models', '')
    elif mode == 'bag':
        assert dataset == 'RIDI'
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Bag - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Bag_Test'
        if dataset == 'RIDI':
            model_path = join('data/models', 'WDE_regressor_Bag_LinAcc_0.95.pth')
            model_raw_path = join('data/models', 'WDE_regressor_Bag_RawIMU_0.951.pth')
        elif dataset == 'TRC#1':
            model_path = join('data/models', '')
            model_raw_path = join('data/models', '')
    elif mode == 'body':
        assert dataset == 'RIDI'
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Body - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Body_Test'
        if dataset == 'RIDI':
            model_path = join('data/models', 'WDE_regressor_Body_LinAcc_0.901.pth')
            model_raw_path = join('data/models', 'WDE_regressor_Body_RawIMU_0.905.pth')
        elif dataset == 'TRC#1':
            model_path = join('data/models', '')
            model_raw_path = join('data/models', '')
    elif mode == 'swing':
        assert dataset == 'TRC#1'
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/SBG-PDR-DATA/swing'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/SBG-PDR-DATA/swing'
        model_path = join(main_wd_path, 'data/models', 'WDE_regressor_Swing_LinAcc_0.803.pth')
        model_raw_path = join('data/models', '')
    else:
        raise 'invalide mode'
    # get experiment list from directory
    if dataset == 'RIDI':
        exp_list = [join(dir_to_analyze, item) for item in listdir(dir_to_analyze) if
                    'AHRS_results' not in item]
    elif dataset == 'TRC#1':
        person_list = listdir(dir_to_analyze)
        if person_idx == None:
            person_idx = random.randint(0, len(person_list) - 1)
        dir_to_analyze = join(dir_to_analyze, person_list[person_idx])
        exp_list = [join(dir_to_analyze, item) for item in listdir(dir_to_analyze) if
                    'AHRS_results' not in item and 'ascii-output.txt' not in item]
    # choose exp
    if experiment_example_idx is None:
        experiment_example_idx = random.randint(0, len(exp_list) - 1)
    print('idx = ' + str(experiment_example_idx))
    if dataset == 'RIDI':
        Exp = Classes.RidiExp(exp_list[experiment_example_idx])
    elif dataset == 'TRC#1':
        Exp = Classes.SbgExpRawData(exp_list[experiment_example_idx])
    if limit_exp_length:
        Exp.SegmentScenario([0, time_limit])
    # example_sample = Exp
    # delete first few seconds before start to walk
    print(join(Exp.Path, Exp.FileName))
    if not use_GT_att:
        t_est, lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(Exp)
        analyzed_segment = Classes.WDE_performance_analysis(Exp, use_GT_att=use_GT_att,
                                                            lin_acc_b_frame_est=lin_acc_b_frame_est,
                                                            grv_est=grv_est,
                                                            Rot_est=Rot_est,
                                                            Heading_est=Heading_est)
    else:
        assert dataset == 'RIDI'
        analyzed_segment = Classes.WDE_performance_analysis(Exp, use_GT_att=use_GT_att)
    # analyzed_segment.initialize_WD_angle(wind_size_for_heading_init=wind_size_for_heading_init)
    if WD_est_method == 'PCA':
        # dL = Exp.calc_dL(window_size)
        traj, walking_angle = traj_est_using_PCA(Exp, window_size, plot_result=True, use_GT_att=use_GT_att)
    elif WD_est_method == 'SP_heading':
        traj_est_using_SP_heading(Exp, window_size, plot_result=True)
    elif WD_est_method == 'inverted_pendulum':
        traj_est_using_inv_pend(Exp, window_size, plot_result=True)
        # analyzed_segment.inverted_pendulum_model(plot_results=True)
    elif WD_est_method == 'resnet18LinAcc':
        dL = Exp.calc_dL(window_size)
        traj, walking_angle = traj_est_using_resnet18(Exp, window_size, dL, plot_result=True, use_GT_att=use_GT_att)
    elif WD_est_method == 'resnet18RawIMU':
        dL = Exp.calc_dL(window_size)
        traj, walking_angle = traj_est_using_resnet18_raw(Exp, window_size, dL, plot_result=True, use_GT_att=use_GT_att)
    elif WD_est_method == 'compare_all_methods':
        dL = Exp.calc_dL(window_size)
        est_traj_PCA, _ = traj_est_using_PCA(Exp, window_size, dL, plot_result=False, use_GT_att=use_GT_att)
        analyzed_segment.walking_direction_estimation_using_smartphone_heading(plot_results=False)
        est_traj_SP_heading = analyzed_segment.est_traj
        analyzed_segment.inverted_pendulum_model(plot_results=False)
        est_traj_inv_pend = analyzed_segment.est_traj
        dL = Exp.calc_dL(window_size)
        est_traj_resnet18, _ = traj_est_using_resnet18(Exp, window_size, dL, plot_result=False, use_GT_att=use_GT_att)
        est_traj_resnet18_raw, _ = traj_est_using_resnet18_raw(Exp, window_size, dL, plot_result=False, use_GT_att=use_GT_att)
        fig = plt.figure('walking_direction_estimation compare all methods')
        Ax = fig.add_subplot(111)
        Ax.plot(analyzed_segment.segment.Pos.arr()[:, 0] - analyzed_segment.segment.Pos.arr()[0, 0],
                analyzed_segment.segment.Pos.arr()[:, 1] - analyzed_segment.segment.Pos.arr()[0, 1],
                 color="black", linestyle='--', label='GT')
        Ax.plot(est_traj_PCA[:, 0], est_traj_PCA[:, 1], label='PCA')
        Ax.plot(est_traj_SP_heading[:, 0], est_traj_SP_heading[:, 1], label='SP_heading')
        Ax.plot(est_traj_inv_pend[:, 0], est_traj_inv_pend[:, 1], label='inv_pend')
        Ax.plot(est_traj_resnet18[:, 0], est_traj_resnet18[:, 1], label='resnet18')
        Ax.plot(est_traj_resnet18_raw[:, 0], est_traj_resnet18_raw[:, 1], label='resnet18-raw')
        Ax.grid(True)
        Ax.legend()
        Ax.axis('equal')

    plt.show()
