import utils.Classes as Classes
import matplotlib.pyplot as plt
import numpy as np
import utils.Functions as Functions
import random
from os import listdir, getcwd
from os.path import join
from utils.Functions import construct_traj,MyCDF


def get_segments_from_exp(exp: Classes.RidiExp, window_size, sample_id=0, use_GT_att=False, wind_size_for_heading_init=750):
    """
    Creating a list of segments from the experiment samples.
    Each segment contains window_size samples (~250)
    sample_id is useful if want to append such lists
    """
    seg_start_idx = 0
    seg_stop_idx = window_size - 1
    segment_list = []
    initial_WD_angles_list = []
    if not use_GT_att:
        AHRS_results_list = []
    # initialize WD angle only for SM heading method
    analyzed_segment = Classes.WDE_performance_analysis(exp, use_GT_att=True)
    analyzed_segment.initialize_WD_angle(wind_size_for_heading_init=wind_size_for_heading_init)
    initial_WD_angle = analyzed_segment.initial_WD_angle_GT
    if not use_GT_att:
        t_est, lin_acc_b_frame_est, grv_est, Rot_est, Heading_est = get_AHRS_results_for_exp(exp)
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

#
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
    validate_segments_with_turn_identification = True # if True segments are either turns or straight lines
    analyze_staight_lines = True # if False, turns are analyzed
    use_GT_att = False # use gournd truth linear acceleration and b->n rotations else use AHRS
    experiment_example_idx = None # if work_on_directory==False, this specifies the experiment idx
    window_size = 200 # for res net use the same window size as in training
    work_on_directory = True
    analyze_WDE_performance = True
    plot_turn_identification = False
    view_valid_traj = False # these are used to visualize/debug the turn detection
    view_invalid_traj = False
    WD_est_method = 'inverted_pendulum' # could be: 'SM_heading', 'PCA', 'inverted_pendulum', 'resnet18LinAcc', 'resnet18RawIMU' , 'compare_all_methods'
    main_wd_path = '/'
    error_type = 'end_pos' # could be: 'WD_angle', 'end_pos'
    mode = 'text'  # could be: 'text', 'pocket', 'body', 'bag'
    data_location = 'magneto'  # could be 'magneto' or 'local_machine'
    if mode == 'text':
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Text - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Text_Test'
        model_path = join('data/models', 'WDE_regressor_Text_LinAcc_0.902.pth')
        model_raw_path = join('data/models', 'WDE_regressor_Text_RawIMU_0.902.pth')
    elif mode == 'pocket':
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Pocket_Test'
        model_path = join('data/models', 'WDE_regressor_Pocket_LinAcc_0.9.pth')
        model_raw_path = join('data/models', 'WDE_regressor_Pocket_RawIMU_0.902.pth')
    elif mode == 'bag':
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Bag - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Bag_Test'
        model_path = join('data/models', 'WDE_regressor_Bag_LinAcc_0.95.pth')
        model_raw_path = join('data/models', 'WDE_regressor_Bag_RawIMU_0.951.pth')
    elif mode == 'body':
        if data_location == 'magneto':
            dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Body - Test'
        elif data_location == 'local_machine':
            dir_to_analyze = '/home/maint/Eran/AHRS/RIDI_dataset_train_test/Body_Test'
        model_path = join('data/models', 'WDE_regressor_Body_LinAcc_0.901.pth')
        model_raw_path = join('data/models', 'WDE_regressor_Body_RawIMU_0.905.pth')
    else:
        raise 'invalide mode'
    exp_list = [Classes.RidiExp(join(dir_to_analyze, item)) for item in listdir(dir_to_analyze) if
                'AHRS_results' not in item]
    if work_on_directory:
        if use_GT_att:
            segments, initial_WD_angles_list, exp_list = \
                get_segments_from_dir(dir_to_analyze, window_size, shuffle=False, use_GT_att=use_GT_att,
                                      wind_size_for_heading_init=wind_size_for_heading_init)
        else:
            segments, AHRS_results_segments_list, initial_WD_angles_list, exp_list = \
                get_segments_from_dir(dir_to_analyze, window_size,
                                      shuffle=False,
                                      use_GT_att=use_GT_att,
                                      wind_size_for_heading_init=wind_size_for_heading_init)
        if experiment_example_idx is None:
            experiment_example_idx = random.randint(0, len(exp_list) - 1)
            print('idx = ' + str(experiment_example_idx))
        Exp = exp_list[experiment_example_idx]
    else:
        # Exp = Classes.RidiExp('/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Train/hang_leg1.csv')
        if experiment_example_idx is None:
            experiment_example_idx = random.randint(0, len(exp_list) - 1)
        print('idx = ' + str(experiment_example_idx))
        Exp = exp_list[experiment_example_idx]
        Exp.SegmentScenario([0, 60])
        if use_GT_att:
            segments, initial_WD_angles_list = \
                get_segments_from_exp(Exp, window_size, use_GT_att=use_GT_att,
                                      wind_size_for_heading_init=wind_size_for_heading_init)
        else:
            segments, AHRS_results_segments_list, initial_WD_angles_list = \
                get_segments_from_exp(Exp, window_size, use_GT_att=use_GT_att,
                                      wind_size_for_heading_init=wind_size_for_heading_init)
    if validate_segments_with_turn_identification:
        staight_line_segments = []
        staight_line_segments_AHRS_results = []
        staight_line_initial_WD_angles = []
        turn_segments = []
        turn_segments_AHRS_results = []
        turn_initial_WD_angles = []
        errors = []

        for i in range(len(segments)):
            segment = segments[i]
            initial_WD_seg = initial_WD_angles_list[i]
            if not use_GT_att:
                AHRS_results_seg = AHRS_results_segments_list[i]
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
            segments = staight_line_segments
            AHRS_results_segments_list = staight_line_segments_AHRS_results
            initial_WD_angles_list = staight_line_initial_WD_angles
        else:
            segments = turn_segments
            AHRS_results_segments_list = turn_segments_AHRS_results
            initial_WD_angles_list = turn_initial_WD_angles
        print('number of straight line segments = ' + str(len(staight_line_segments)))
        print('number of turn segments = ' + str(len(turn_segments)))
    if analyze_WDE_performance:
        analyzied_segments = []
        if WD_est_method == 'compare_all_methods':
            errors_SM_heading = []
            errors_PCA = []
            errors_inv_pend = []
            errors_pca_raw = []
            errors_resnet18 = []
            errors_resnet18_raw = []
        else:
            errors = []
        for i in range(len(segments)):
            segment = segments[i]
            if not use_GT_att:
                AHRS_results_seg = AHRS_results_segments_list[i]
            initial_WD_seg = initial_WD_angles_list[i]
            segment.id == initial_WD_seg.id
            if not use_GT_att:
                assert segment.id == AHRS_results_seg.id
            if use_GT_att:
                analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att))
            else:
                analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                           lin_acc_b_frame_est=AHRS_results_seg.lin_acc_b_frame,
                                                                           grv_est=AHRS_results_seg.grv,
                                                                           Heading_est=AHRS_results_seg.heading,
                                                                           Rot_est=AHRS_results_seg.Rot
                                                                           )
                                          )
            analyzied_segments[-1].initial_WD_angle_GT = initial_WD_seg.initial_WD_angle
            if WD_est_method == 'PCA':
                analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
            elif WD_est_method == 'SM_heading':
                analyzied_segments[-1].walking_direction_estimation_using_smartphone_heading(plot_results=False)
            elif WD_est_method =='inverted_pendulum':
                analyzied_segments[-1].inverted_pendulum_model(plot_results=False)
            elif WD_est_method =='resnet18LinAcc':
                analyzied_segments[-1].window_size = window_size
                analyzied_segments[-1].model_path = model_path
                analyzied_segments[-1].res18_direction_pred(plot_results=False,data_type='LinAcc')
            elif WD_est_method =='resnet18RawIMU':
                analyzied_segments[-1].window_size = window_size
                analyzied_segments[-1].model_raw_path = model_raw_path
                analyzied_segments[-1].res18_direction_pred(plot_results=False,data_type='RawIMU')
            elif WD_est_method == 'compare_all_methods':
                analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
                WD_angle_err_PCA, end_pos_err_PCA = analyzied_segments[-1].calc_error()
                analyzied_segments[-1].walking_direction_estimation_using_smartphone_heading(plot_results=False)
                WD_angle_err_SM_heading, end_pos_err_SM_heading = analyzied_segments[-1].calc_error()
                analyzied_segments[-1].inverted_pendulum_model(plot_results=False)
                WD_angle_err_inv_pend, end_pos_err_inv_pend= analyzied_segments[-1].calc_error()
                analyzied_segments[-1].pca_on_raw_acc(plot_results=False)
                WD_angle_err_pca_raw, end_pos_err_pca_raw= analyzied_segments[-1].calc_error()
                analyzied_segments[-1].window_size = window_size
                analyzied_segments[-1].model_path = model_path
                analyzied_segments[-1].res18_direction_pred(plot_results=False,data_type='LinAcc')
                WD_angle_err_resnet18, end_pos_err_resnet18 = analyzied_segments[-1].calc_error()
                analyzied_segments[-1].model_raw_path = model_raw_path
                analyzied_segments[-1].res18_direction_pred(plot_results=False,data_type='RawIMU')
                WD_angle_err_resnet18_raw, end_pos_err_resnet18_raw = analyzied_segments[-1].calc_error()
            if WD_est_method == 'compare_all_methods':
                if error_type == 'WD_angle':
                    errors_PCA.append(WD_angle_err_PCA)
                    errors_SM_heading.append(WD_angle_err_SM_heading)
                    errors_inv_pend.append(WD_angle_err_inv_pend)
                    errors_pca_raw.append(WD_angle_err_pca_raw)
                    errors_resnet18.append(WD_angle_err_resnet18)
                    errors_resnet18_raw.append(WD_angle_err_resnet18_raw)
                elif error_type == 'end_pos':
                    errors_PCA.append(end_pos_err_PCA)
                    errors_SM_heading.append(end_pos_err_SM_heading)
                    errors_inv_pend.append(end_pos_err_inv_pend)
                    errors_pca_raw.append(end_pos_err_pca_raw)
                    errors_resnet18.append(end_pos_err_resnet18)
                    errors_resnet18_raw.append(end_pos_err_resnet18_raw)
            else:
                WD_angle_err, end_pos_err = analyzied_segments[-1].calc_error()
                if error_type == 'WD_angle':
                    errors.append(WD_angle_err)
                elif error_type == 'end_pos':
                    errors.append(end_pos_err)
        if error_type == 'WD_angle':
            if WD_est_method == 'compare_all_methods':
                print('mean error PCA= ' + str(np.array(errors_PCA).mean() * 180 / np.pi))
                print('mean error SM heading= ' + str(np.array(errors_SM_heading).mean() * 180 / np.pi))
                print('mean error inv pend= ' + str(np.array(errors_inv_pend).mean() * 180 / np.pi))
                print('mean error raw pca = ' + str(np.array(errors_pca_raw).mean() * 180 / np.pi))
                print('mean error resnet18 = ' + str(np.array(errors_resnet18).mean() * 180 / np.pi))
                print('mean error resnet18 raw = ' + str(np.array(errors_resnet18_raw).mean() * 180 / np.pi))
                fig = plt.figure('angle errors')
                ax1 = fig.add_subplot(111)
                ax1.grid(True)
                ax1.hist(np.array(errors_PCA) * 180 / np.pi, 100, label='PCA')
                ax1.hist(np.array(errors_SM_heading) * 180 / np.pi, 100, label='SM_heading')
                ax1.hist(np.array(errors_inv_pend) * 180 / np.pi, 100, label='inv pend')
                ax1.hist(np.array(errors_pca_raw) * 180 / np.pi, 100, label='raw pca')
                ax1.hist(np.array(errors_resnet18) * 180 / np.pi, 100, label='resnet18')
                ax1.hist(np.array(errors_resnet18_raw) * 180 / np.pi, 100, label='resnet18_raw')
                ax1.set(ylabel='samples', xlabel='WD angle [deg]')
                ax1.legend()               
            else:
                print('mean error = ' + str(np.array(errors).mean() * 180 / np.pi))
                print('error std = ' + str(np.array(errors).std() * 180 / np.pi))
                print('mean abs error = ' + str(abs(np.array(errors)).mean() * 180 / np.pi))
                fig = plt.figure('WDE angle errors')
                ax1 = fig.add_subplot(211)
                ax1.grid(True)
                ax1.plot(np.array(errors) * 180 / np.pi)
                ax1.set(ylabel='WD angle [deg]')
                ax2 = fig.add_subplot(212)
                ax2.hist(np.array(errors) * 180 / np.pi, 100)
                ax2.grid(True)
                ax2.set(ylabel='samples', xlabel='WD angle [deg]')
                
                plt.figure('Angle CDF')
                bins_count,cdf = MyCDF(np.array(errors))
                plt.plot(bins_count[1:]* 180 / np.pi,cdf,label='PCA')
                plt.grid();plt.legend()
                plt.xlabel('WD angle [deg]');plt.ylabel('CDF')    
                plt.show()
        if error_type == 'end_pos':
            if WD_est_method == 'compare_all_methods':
                print('mean error PCA= ' + str(np.array(errors_PCA).mean().round(3)))
                print('mean error SM heading= ' + str(np.array(errors_SM_heading).mean().round(3)))
                print('mean error inv pend= ' + str(np.array(errors_inv_pend).mean().round(3)))
                print('mean error raw pca = ' + str(np.array(errors_pca_raw).mean().round(3)))
                print('mean error resnet18 = ' + str(np.array(errors_resnet18).mean().round(3)))
                print('mean error resnet18_raw = ' + str(np.array(errors_resnet18_raw).mean().round(3)))
                fig = plt.figure('WDE pos errors')
                ax1 = fig.add_subplot(111)
                ax1.hist(np.array(errors_PCA), 100, label='PCA')
                ax1.hist(np.array(errors_SM_heading), 100, label='SM_heading')
                ax1.hist(np.array(errors_inv_pend), 100, label='inv pend')
                ax1.hist(np.array(errors_pca_raw), 100, label='raw pca')
                ax1.hist(np.array(errors_resnet18), 100, label='resnet18')
                ax1.hist(np.array(errors_resnet18_raw), 100, label='resnet18_raw')
                ax1.legend()
                ax1.grid(True)
                ax1.set(ylabel='samples', xlabel='WD pos error [m]')
        
                plt.figure('Pos CDF')
                bins_count,cdf = MyCDF(np.array(errors_PCA))
                plt.plot(bins_count[1:],cdf,label='PCA')
                bins_count,cdf = MyCDF(np.array(errors_SM_heading))
                plt.plot(bins_count[1:],cdf,label='SM_heading')
                bins_count,cdf = MyCDF(np.array(errors_inv_pend))
                plt.plot(bins_count[1:],cdf,label='inv pend')
                bins_count,cdf = MyCDF(np.array(errors_pca_raw))
                plt.plot(bins_count[1:],cdf,label='raw pca')
                bins_count, cdf = MyCDF(np.array(errors_resnet18))
                plt.plot(bins_count[1:], cdf, label='resnet18')
                bins_count, cdf = MyCDF(np.array(errors_resnet18_raw))
                plt.plot(bins_count[1:], cdf, label='resnet18_raw')
                plt.grid();plt.legend()

                plt.xlabel('WD pos error [m]');plt.ylabel('CDF')    
                plt.show()
            else:
                print('mean error = ' + str(np.array(errors).mean()))
                print('error std = ' + str(np.array(errors).std() ))
                print('mean abs error = ' + str(abs(np.array(errors)).mean()))
                fig = plt.figure('WDE pos errors')
                ax1 = fig.add_subplot(211)
                ax1.grid(True)
                ax1.plot(np.array(errors))
                ax1.set(ylabel='WD pos error [m]')
                ax2 = fig.add_subplot(212)
                ax2.hist(np.array(errors), 100)
                ax2.grid(True)
                ax2.set(ylabel='samples', xlabel='WD pos error [m]')
        if plot_turn_identification:
            dP_angles_variance_list = [segment.dP_angles_variance for segment in analyzied_segments]
            fig = plt.figure('dP angle variances for turn identification')
            ax = fig.add_subplot(111)
            ax.plot(dP_angles_variance_list)
            ax.set(title='dP_angles_variances')
    if view_valid_traj:
        number_of_test_segments = 10
        test_segment_idx = random.sample(range(0, len(analyzied_segments) - 1), number_of_test_segments)
        for idx in test_segment_idx:
            print(idx)
            analyzied_segments[idx].PlotPosition()
            analyzied_segments[idx].identify_turn(plot_results=True)
            plt.show()
    if view_invalid_traj:
        number_of_test_segments = 10
        test_segment_idx = random.sample(range(0, len(turn_segments) - 1), number_of_test_segments)
        for idx in test_segment_idx:
            print(idx)
            turn_segments[idx].PlotPosition()
            turn_segments[idx].identify_turn(plot_results=True)
            plt.show()
    if plot_WDE_example:
        if segment_example_idx is None:
            segment_example_idx = random.randint(0, len(segments) - 1)
            print('idx = ' + str(segment_example_idx))
        example_sample = segments[segment_example_idx]
        AHRS_results_seg = AHRS_results_segments_list[segment_example_idx]
        initial_WD_seg = initial_WD_angles_list[segment_example_idx]
        assert example_sample.id == AHRS_results_seg.id and example_sample.id == initial_WD_seg.id
        # example_sample.PlotPosition()
        if use_GT_att:
            analyzed_segment = Classes.WDE_performance_analysis(example_sample, use_GT_att=use_GT_att)
        else:
            analyzed_segment = Classes.WDE_performance_analysis(example_sample, use_GT_att=use_GT_att,
                                                                lin_acc_b_frame_est=AHRS_results_seg.lin_acc_b_frame,
                                                                grv_est=AHRS_results_seg.grv,
                                                                Heading_est=AHRS_results_seg.heading,
                                                                Rot_est=AHRS_results_seg.Rot
                                                                )
        analyzed_segment.initial_WD_angle_GT = initial_WD_seg.initial_WD_angle
        if WD_est_method == 'PCA':
            analyzed_segment.PCA_direction_analysis(plot_results=True)
            WD_angle_err, end_pos_err = analyzed_segment.calc_error()
            if error_type == 'WD_angle':
                error = WD_angle_err
                print('error = ' + str(error * 180 / np.pi) + ' deg')
            elif error_type == 'end_pos':
                error = end_pos_err
        elif WD_est_method == 'SM_heading':
            analyzed_segment.walking_direction_estimation_using_smartphone_heading(plot_results=True)
            WD_angle_err, end_pos_err = analyzed_segment.calc_error()
            if error_type == 'WD_angle':
                error = WD_angle_err
                print('error = ' + str(error * 180 / np.pi) + ' deg')
            elif error_type == 'end_pos':
                error = end_pos_err
        elif WD_est_method == 'inverted_pendulum':
            analyzed_segment.inverted_pendulum_model(plot_results=True)
            WD_angle_err, end_pos_err = analyzed_segment.calc_error()
            if error_type == 'WD_angle':
                error = WD_angle_err
                print('error = ' + str(error * 180 / np.pi) + ' deg')
            elif error_type == 'end_pos':
                error = end_pos_err
        elif WD_est_method == 'raw pca':
            analyzed_segment.pca_on_raw_acc(plot_results=True)
            WD_angle_err, end_pos_err = analyzed_segment.calc_error()
            if error_type == 'WD_angle':
                error = WD_angle_err
                print('error = ' + str(error * 180 / np.pi) + ' deg')
            elif error_type == 'end_pos':
                error = end_pos_err
        elif WD_est_method == 'resnet18':
            analyzed_segment.window_size = window_size
            analyzed_segment.model_path = model_path
            analyzed_segment.res18_direction_pred(plot_results=True)
        elif WD_est_method == 'compare_all_methods':
            analyzed_segment.PCA_direction_analysis(plot_results=False)
            est_traj_PCA = analyzed_segment.est_traj
            WD_angle_err_PCA, end_pos_err_PCA = analyzed_segment.calc_error()
            analyzed_segment.walking_direction_estimation_using_smartphone_heading(plot_results=False)
            est_traj_SM_heading = analyzed_segment.est_traj
            WD_angle_err_SM_heading, end_pos_err_SM_heading = analyzed_segment.calc_error()
            analyzed_segment.inverted_pendulum_model(plot_results=False)
            est_traj_inv_pend = analyzed_segment.est_traj
            WD_angle_err_inv_pend, end_pos_err_inv_pend = analyzed_segment.calc_error()
            analyzed_segment.pca_on_raw_acc(plot_results=False)
            est_traj_raw_pca = analyzed_segment.est_traj
            WD_angle_err_raw_pca, end_pos_err_raw_pca = analyzed_segment.calc_error()
            print('SM_heading error = ' + str(end_pos_err_SM_heading) + ' [m]')
            print('PCA error = ' + str(end_pos_err_PCA) + ' [m]')
            print('inv_pend error = ' + str(end_pos_err_inv_pend) + ' [m]')
            print('raw pca error = ' + str(end_pos_err_raw_pca) + ' [m]')
            fig = plt.figure('walking_direction_estimation compare all methods')
            Ax = fig.add_subplot(111)
            Ax.plot(analyzed_segment.segment.Pos.arr()[:, 0] - analyzed_segment.segment.Pos.arr()[0, 0],
                    analyzed_segment.segment.Pos.arr()[:, 1] - analyzed_segment.segment.Pos.arr()[0, 1], label='GT')
            Ax.plot(est_traj_PCA[:, 0], est_traj_PCA[:, 1], label='PCA')
            Ax.plot(est_traj_SM_heading[:, 0], est_traj_SM_heading[:, 1], label='SM_heading')
            Ax.plot(est_traj_inv_pend[:, 0], est_traj_inv_pend[:, 1], label='inv_pend')
            Ax.plot(est_traj_raw_pca[:, 0], est_traj_raw_pca[:, 1], label='raw pca')
            Ax.grid(True)
            Ax.legend()
            Ax.axis('equal')
        if plot_turn_identification:
            analyzed_segment.identify_turn(plot_results=True)
    if plot_WDE_example or \
            analyze_WDE_performance:
        plt.show()
