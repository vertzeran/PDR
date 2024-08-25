import utils.Classes as Classes
import matplotlib.pyplot as plt
import numpy as np
import utils.Functions as Functions
import random
from os import listdir
from os.path import join
from utils.Functions import construct_traj,MyCDF


def get_segments_from_exp(exp: Classes.RidiExp, window_size, sample_id=0, fix_psi=False):
    """
    Creating a list of segments from the experiment samples.
    Each segment contains window_size samples (~250)
    sample_id is useful if want to append such lists
    """
    seg_start_idx = 0
    seg_stop_idx = window_size - 1
    segment_list = []
    wind_size_for_estimating_heading = 750

    # Calculate the real psi with AHRS
    if fix_psi:
        estimated_psi = EstimatePsiWithAHRS(exp, wind_size_for_estimating_heading)
        exp.Psi = exp.Psi - exp.Psi[0] + estimated_psi

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

        # load exp file
        item_path = join(exp_dir, item)  # path to csv file
        exp = Classes.RidiExp(item_path)

        # Extract segments of the exp with the fixed psi flag
        segment_list.extend(get_segments_from_exp(exp, window_size, sample_id, fix_psi=True))
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
        random.shuffle(segment_list)

    return segment_list, exp_list


def EstimatePsiWithAHRS(exp, wind_size_for_estimating_heading, plot_heading_initialization=False):
    """   
    Initiate AHRS module and calculate current psi based on the first {wind_size_for_learning} samples
    Then, change (override) the exp.psi field
    TODO: 
        3) add field for raw_psi and fix_psi
    """
    first_segment = exp.clone()
    first_segment.SegmentScenario([exp.Time_GT[0], exp.Time_GT[wind_size_for_estimating_heading]]) 
    class_instance_to_initialize_heading = Classes.WDE_performance_analysis(first_segment)
    class_instance_to_initialize_heading.segment.Psi = class_instance_to_initialize_heading.segment.Psi - \
                                                       class_instance_to_initialize_heading.segment.Psi[0] + \
                                                       class_instance_to_initialize_heading.WD_angle_GT
    estimated_psi = class_instance_to_initialize_heading.WD_angle_GT       
    if plot_heading_initialization:
        class_instance_to_initialize_heading.walking_direction_estimation_using_smartphone_heading(plot_results=True)
        plt.show()               
    return estimated_psi


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



###############################################################################
###############################################################################


if __name__ == '__main__':
    plot_walking_angle = False
    plot_traj = False
    plot_WDE_example = True
    segment_example_idx = None
    validate_segments_with_turn_identification = True
    analyze_staight_lines = True
    experiment_example_idx = None
    plot_acc_vs_pos_der = False
    window_size = 250
    work_on_directory = True
    dir_to_analyze = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Text - Train'
    analyze_WDE_performance = False
    plot_turn_identification = False
    view_valid_traj = False
    view_invalid_traj = False
    WD_est_method = 'PCA' # could be: 'SM_heading', 'PCA', 'inverted_pendulum', 'compare_all_methods'
    error_type = 'end_pos' # could be: 'WD_angle', 'end_pos'

    if work_on_directory:
        segments, exp_list = get_sagments_from_dir(dir_to_analyze, window_size, shuffle=False)
        if experiment_example_idx is None:
            experiment_example_idx = random.randint(0, len(exp_list) - 1)
            print('idx = ' + str(experiment_example_idx))
            Exp = exp_list[experiment_example_idx]
        else:
            Exp = exp_list[experiment_example_idx]
    else:
        Exp = Classes.RidiExp('/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Train/hang_leg1.csv')
        Exp.SegmentScenario([0, 60])
        segments = get_segments_from_exp(Exp, window_size)
    if validate_segments_with_turn_identification:
        staight_line_segments = []
        turn_segments = []
        errors = []
        for segment in segments:
            candidate = Classes.WDE_performance_analysis(segment)
            candidate.identify_turn(threshold=0.3, plot_results=False)
            if not candidate.turn_identified:
                staight_line_segments.append(segment)
            else:
                turn_segments.append(segment)
        if analyze_staight_lines:
            segments = staight_line_segments
        else:
            segments = turn_segments
        print('number of straight line segments = ' + str(len(staight_line_segments)))
        print('number of turn segments = ' + str(len(turn_segments)))
    if analyze_WDE_performance:
        analyzied_segments = []
        if WD_est_method == 'compare_all_methods':
            errors_SM_heading = []
            errors_PCA = []
            errors_inv_pend = []
            errors_pca_raw = []
        else:
            errors = []
        for segment in segments:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment))
            if WD_est_method == 'PCA':
                analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
            elif WD_est_method == 'SM_heading':
                analyzied_segments[-1].walking_direction_estimation_using_smartphone_heading(plot_results=False)
            elif WD_est_method =='inverted_pendulum':
                analyzied_segments[-1].inverted_pendulum_model(plot_results=False)
            elif WD_est_method == 'compare_all_methods':
                analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
                WD_angle_err_PCA, end_pos_err_PCA = analyzied_segments[-1].calc_error()
                analyzied_segments[-1].walking_direction_estimation_using_smartphone_heading(plot_results=False)
                WD_angle_err_SM_heading, end_pos_err_SM_heading = analyzied_segments[-1].calc_error()
                analyzied_segments[-1].inverted_pendulum_model(plot_results=False)
                WD_angle_err_inv_pend, end_pos_err_inv_pend= analyzied_segments[-1].calc_error()
                analyzied_segments[-1].pca_on_raw_acc(plot_results=False)
                WD_angle_err_pca_raw, end_pos_err_pca_raw= analyzied_segments[-1].calc_error()


            if WD_est_method == 'compare_all_methods':
                if error_type == 'WD_angle':
                    errors_PCA.append(WD_angle_err_PCA)
                    errors_SM_heading.append(WD_angle_err_SM_heading)
                    errors_inv_pend.append(WD_angle_err_inv_pend)
                    errors_pca_raw.append(WD_angle_err_pca_raw)
                elif error_type == 'end_pos':
                    errors_PCA.append(end_pos_err_PCA)
                    errors_SM_heading.append(end_pos_err_SM_heading)
                    errors_inv_pend.append(end_pos_err_inv_pend)
                    errors_pca_raw.append(end_pos_err_pca_raw)
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
                fig = plt.figure('angle errors')
                ax1 = fig.add_subplot(111)
                ax1.grid(True)
                ax1.hist(np.array(errors_PCA) * 180 / np.pi, 100, label='PCA')
                ax1.hist(np.array(errors_SM_heading) * 180 / np.pi, 100, label='SM_heading')
                ax1.hist(np.array(errors_inv_pend) * 180 / np.pi, 100, label='inv pend')
                ax1.hist(np.array(errors_pca_raw) * 180 / np.pi, 100, label='raw pca')
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
                bins_count,cdf = MyCDF(np.array(errors_PCA))
                plt.plot(bins_count[1:]* 180 / np.pi,cdf,label='PCA')
                bins_count,cdf = MyCDF(np.array(errors_SM_heading))
                plt.plot(bins_count[1:]* 180 / np.pi,cdf,label='SM_heading')
                bins_count,cdf = MyCDF(np.array(errors_inv_pend))
                plt.plot(bins_count[1:]* 180 / np.pi,cdf,label='inv pend')
                bins_count,cdf = MyCDF(np.array(errors_pca_raw))
                plt.plot(bins_count[1:]* 180 / np.pi,cdf,label='raw pca')
                plt.grid();plt.legend()
                plt.xlabel('WD angle [deg]');plt.ylabel('CDF')    
                plt.show()
                
        if error_type == 'end_pos':
            if WD_est_method == 'compare_all_methods':
                print('mean error PCA= ' + str(np.array(errors_PCA).mean()))
                print('mean error SM heading= ' + str(np.array(errors_SM_heading).mean()))
                print('mean error inv pend= ' + str(np.array(errors_inv_pend).mean()))
                print('mean error raw pca = ' + str(np.array(errors_pca_raw).mean()))
                fig = plt.figure('WDE pos errors')
                ax1 = fig.add_subplot(111)
                ax1.hist(np.array(errors_PCA), 100, label='PCA')
                ax1.hist(np.array(errors_SM_heading), 100, label='SM_heading')
                ax1.hist(np.array(errors_inv_pend), 100, label='inv pend')
                ax1.hist(np.array(errors_pca_raw), 100, label='raw pca')
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
    if plot_acc_vs_pos_der:
        pos_acc_correlation(Exp)
    if plot_WDE_example:
        if analyze_staight_lines:
            if segment_example_idx is None:
                segment_example_idx = random.randint(0, len(segments) - 1)
                print('idx = ' + str(segment_example_idx))
                example_sample = segments[segment_example_idx]
            else:
                example_sample = segments[segment_example_idx]
        else:
            # segment_example_idx = random.randint(0, len(turn_segments) - 1)
            # invalid_segment = turn_segments[segment_example_idx]
            # example_sample = invalid_segment.segment
            if segment_example_idx is None:
                segment_example_idx = random.randint(0, len(segments) - 1)
                print('idx = ' + str(segment_example_idx))
                example_sample = segments[segment_example_idx]
            else:
                example_sample = segments[segment_example_idx]
        # example_sample.PlotPosition()
        analyzed_segment = Classes.WDE_performance_analysis(example_sample)
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
    if plot_walking_angle:
        walking_angle = Exp.calc_walking_direction(window_size)
        fig = plt.figure('walking angle')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$idx$", ylabel=r"$deg$", title="walking angle"), ax.grid(True)
        ax.plot(walking_angle * 180 / np.pi, color='blue', linestyle='-', linewidth=2, label='')
    if plot_traj:
        dL = Exp.calc_dL(window_size)
        pos_gt = Exp.Pos.arr()
        pos_gt = pos_gt - pos_gt[0]
        traj = construct_traj(dL, walking_angle)
        fig = plt.figure('position plot')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(traj[:, 0], traj[:, 1], color='r', linestyle='-', linewidth=2, label='')
        ax.plot(pos_gt[:, 0], pos_gt[:, 1], color='black', linestyle='--', linewidth=2, label='')
    if plot_traj or \
            plot_walking_angle or \
            plot_WDE_example or \
            plot_acc_vs_pos_der or \
            analyze_WDE_performance:
        plt.show()
