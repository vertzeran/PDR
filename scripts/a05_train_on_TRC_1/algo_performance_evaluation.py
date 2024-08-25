import matplotlib.pyplot as plt
import numpy as np
import utils.Functions as Functions
from os.path import join
from utils.Functions import MyCDF
from scripts.train_WDE_on_TRC1 import MySBGPytorchDS
import pickle
import torch
import math
import json
# from create_segments_for_WDE_SBG import get_AHRS_results_for_exp

# def pos_acc_correlation(Exp: Classes.AhrsExp):
#     """plot linear acceleration in navigation frame vs position 2nd derivative
#     to verify correct coordinate frame handling"""
#     # Position GT 2nd derivative:
#     dt = np.diff(Exp.Time_GT).mean()
#     pos_gt = Exp.Pos.arr()
#     pos_der = np.diff(pos_gt, axis=0) / dt
#     pos_2nd_der = np.diff(pos_der, axis=0) / dt
#     # Accelaration GT;
#     lin_acc_GT = Exp.LinAcc.arr()
#     lin_acc_GT_n_frame = Functions.transform_vectors(lin_acc_GT, Exp.Rot)
#     # Plots
#     fig = plt.figure('Position Acc correlation')
#     Ax_pos_x = fig.add_subplot(311)
#     Ax_pos_x.plot(Exp.Time_GT[0: -2], pos_2nd_der[:, 0], color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
#     Ax_pos_x.plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 0], color='red', linewidth=1, label = 'Accelerometer')
#     Ax_pos_x.set_ylim(lin_acc_GT_n_frame[:, 0].min(), lin_acc_GT_n_frame[:, 0].max()),
#     Ax_pos_x.set(title="Position 2nd derivative VS $a^n_L$", ylabel="$[m/sec^2]$")
#     Ax_pos_x.grid(True);Ax_pos_x.legend()
#
#     Ax_pos_y = fig.add_subplot(312, sharex=Ax_pos_x)
#     Ax_pos_y.plot(Exp.Time_GT[0: -2], pos_2nd_der[:, 1], color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
#     Ax_pos_y.plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 1], color='red', linewidth=1, label = 'Accelerometer')
#     Ax_pos_y.set_ylim(lin_acc_GT_n_frame[:, 1].min(), lin_acc_GT_n_frame[:, 1].max()),
#     Ax_pos_y.set(ylabel="$[m/sec^2]$")
#     Ax_pos_y.grid(True);Ax_pos_y.legend()
#
#     Ax_pos_z = fig.add_subplot(313, sharex=Ax_pos_x)
#     Ax_pos_z.plot(Exp.Time_GT[0: -2], pos_2nd_der[:, 2], color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
#     Ax_pos_z.plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 2], color='red', linewidth=1, label = 'Accelerometer')
#     Ax_pos_z.set_ylim(lin_acc_GT_n_frame[:, 2].min(), lin_acc_GT_n_frame[:, 2].max()),
#     Ax_pos_z.set(xlabel="Time [sec]", ylabel="$[m/sec^2]$")
#     Ax_pos_z.grid(True);Ax_pos_z.legend()


# def get_AHRS_results_for_exp(exp):
#     suffix = '_AHRS_results.xlsx'
#     AHRS_results_file_path = join(exp.Path, exp.FileName.split(sep='.')[0] + suffix)
#     t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
#         Functions.read_AHRS_results(AHRS_results_file_path)
#     # segmentation
#     t_start = exp.Time_IMU[0].round(11)
#     t_end = exp.Time_IMU[-1].round(11)
#     ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11)<= t_end))
#     lin_acc_b_frame = exp.Acc.arr() - grv_hat[ind_IMU]
#     Rot = Rot_hat[ind_IMU]
#     Heading = np.array(psi_hat)[ind_IMU] + exp.Psi[0]
#     return t_est, lin_acc_b_frame, grv_hat, Rot, Heading


# class segments_initial_WD_angles():
#     def __init__(self, initial_WD_angle=None, id=None):
#         self.id = id
#         self.initial_WD_angle = initial_WD_angle


def calculate_performance(data_location, res18_optimization_results_location, model_name,
                          WD_est_method='resnet18LinAcc', add_quat=False):
    # WD_est_method = 'compare_all_methods'  # could be: 'SM_heading', 'PCA', 'inverted_pendulum', 'resnet18LinAcc', 'resnet18RawIMU' , 'compare_all_methods'
    main_wd_path = '/'
    error_type = 'end_pos' # could be: 'WD_angle', 'end_pos'
    # data_location = \
    # '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-01-22T19:51:40.411866TRC#1 swing wind_size 200swing left right mix'
    print('data location: ' + data_location)
    test_ds = MySBGPytorchDS(join(data_location, 'test.mat'))
    window_size = test_ds.window_size
    print('loading test segments')
    with open(join(data_location, 'test_segments_for_WDE.pickle'), 'rb') as file:
        analyzied_segments = pickle.load(file)
    print('loading model')
    if WD_est_method == 'resnet18LinAcc' or WD_est_method == 'compare_all_methods':
        # res18_optimization_results_location = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-01-22T21:36:17.392113optimization results on Swing, window size: 200 swing left right mix'
        # model_path = join(res18_optimization_results_location, 'WDE_regressor_Swing_LinAcc_0.82.pth')
        model_path = join(res18_optimization_results_location, model_name)
        print('model path: ' + model_path)
        res18model = torch.load(model_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        res18model.to(device)
    if WD_est_method == 'compare_all_methods':
        errors_SM_heading = []
        errors_PCA = []
        errors_inv_pend = []
        # errors_pca_raw = []
        errors_resnet18 = []
        # errors_resnet18_raw = []
    else:
        errors = []
    i = 1
    N = len(analyzied_segments)
    print(N)
    for analyzied_segment in analyzied_segments:
        if np.mod(i, 500) == 0:
            print(str(round(i/N*100,3)) + '% completed')
        i += 1
        if WD_est_method == 'PCA':
            analyzied_segment.PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        elif WD_est_method == 'SM_heading':
            analyzied_segment.walking_direction_estimation_using_smartphone_heading(plot_results=False)
        elif WD_est_method =='inverted_pendulum':
            analyzied_segment.inverted_pendulum_model(plot_results=False)
        elif WD_est_method =='resnet18LinAcc':
            analyzied_segment.res18model = res18model
            analyzied_segment.window_size = window_size
            analyzied_segment.model_path = model_path
            analyzied_segment.res18_direction_pred(plot_results=False,data_type='LinAcc', device=device, add_quat=add_quat)
        # elif WD_est_method =='resnet18RawIMU':
        #     analyzied_segment.window_size = window_size
        #     analyzied_segment.model_raw_path = model_raw_path
        #     analyzied_segment.res18_direction_pred(plot_results=False,data_type='RawIMU')
        elif WD_est_method == 'compare_all_methods':
            analyzied_segment.PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
            WD_angle_err_PCA, end_pos_err_PCA = analyzied_segment.calc_error()
            analyzied_segment.walking_direction_estimation_using_smartphone_heading(plot_results=False)
            WD_angle_err_SM_heading, end_pos_err_SM_heading = analyzied_segment.calc_error()
            analyzied_segment.inverted_pendulum_model(plot_results=False)
            WD_angle_err_inv_pend, end_pos_err_inv_pend= analyzied_segment.calc_error()
            # analyzied_segment.pca_on_raw_acc(plot_results=False)
            # WD_angle_err_pca_raw, end_pos_err_pca_raw= analyzied_segment.calc_error()
            analyzied_segment.res18model = res18model
            analyzied_segment.window_size = window_size
            analyzied_segment.model_path = model_path
            analyzied_segment.res18_direction_pred(plot_results=False, data_type='LinAcc', device=device, add_quat=add_quat)
            WD_angle_err_resnet18, end_pos_err_resnet18 = analyzied_segment.calc_error()
            # analyzied_segment.model_raw_path = model_raw_path
            # analyzied_segment.res18_direction_pred(plot_results=False,data_type='RawIMU')
            # WD_angle_err_resnet18_raw, end_pos_err_resnet18_raw = analyzied_segment.calc_error()
        if WD_est_method == 'compare_all_methods':
            if error_type == 'WD_angle':
                errors_PCA.append(WD_angle_err_PCA)
                errors_SM_heading.append(WD_angle_err_SM_heading)
                errors_inv_pend.append(WD_angle_err_inv_pend)
                # errors_pca_raw.append(WD_angle_err_pca_raw)
                errors_resnet18.append(WD_angle_err_resnet18)
                # errors_resnet18_raw.append(WD_angle_err_resnet18_raw)
            elif error_type == 'end_pos':
                errors_PCA.append(end_pos_err_PCA)
                errors_SM_heading.append(end_pos_err_SM_heading)
                if not math.isnan(end_pos_err_inv_pend):
                    errors_inv_pend.append(end_pos_err_inv_pend)
                else:
                    print('nan in inverted pend')
                # errors_pca_raw.append(end_pos_err_pca_raw)
                errors_resnet18.append(end_pos_err_resnet18)
                # errors_resnet18_raw.append(end_pos_err_resnet18_raw)
        else:
            WD_angle_err, end_pos_err = analyzied_segment.calc_error()
            if error_type == 'WD_angle':
                errors.append(WD_angle_err)
            elif error_type == 'end_pos':
                errors.append(end_pos_err)
    if error_type == 'WD_angle':
        if WD_est_method == 'compare_all_methods':
            print('mean error PCA= ' + str(np.array(errors_PCA).mean() * 180 / np.pi))
            print('mean error SM heading= ' + str(np.array(errors_SM_heading).mean() * 180 / np.pi))
            print('mean error inv pend= ' + str(np.array(errors_inv_pend).mean() * 180 / np.pi))
            # print('mean error raw pca = ' + str(np.array(errors_pca_raw).mean() * 180 / np.pi))
            print('mean error resnet18 = ' + str(np.array(errors_resnet18).mean() * 180 / np.pi))
            # print('mean error resnet18 raw = ' + str(np.array(errors_resnet18_raw).mean() * 180 / np.pi))
            results_dic = {
                'e_PCA': np.array(errors_PCA).mean() * 180 / np.pi,
                'e_SP_heading': np.array(errors_SM_heading).mean() * 180 / np.pi,
                'e_inv_pend': np.array(errors_inv_pend).mean() * 180 / np.pi,
                'e_resnet18': np.array(errors_resnet18).mean() * 180 / np.pi
            }
            fig = plt.figure('angle errors')
            ax1 = fig.add_subplot(111)
            ax1.grid(True)
            ax1.hist(np.array(errors_PCA) * 180 / np.pi, 100, label='PCA')
            ax1.hist(np.array(errors_SM_heading) * 180 / np.pi, 100, label='SM_heading')
            ax1.hist(np.array(errors_inv_pend) * 180 / np.pi, 100, label='inv pend')
            # ax1.hist(np.array(errors_pca_raw) * 180 / np.pi, 100, label='raw pca')
            ax1.hist(np.array(errors_resnet18) * 180 / np.pi, 100, label='resnet18')
            # ax1.hist(np.array(errors_resnet18_raw) * 180 / np.pi, 100, label='resnet18_raw')
            ax1.set(ylabel='samples', xlabel='WD angle [deg]')
            ax1.legend()
        else:
            print('mean error = ' + str(np.array(errors).mean() * 180 / np.pi))
            print('error std = ' + str(np.array(errors).std() * 180 / np.pi))
            print('mean abs error = ' + str(abs(np.array(errors)).mean() * 180 / np.pi))
            results_dic = {
                'mean error': np.array(errors).mean() * 180 / np.pi,
                'error std': np.array(errors).std() * 180 / np.pi,
                'mean abs error': abs(np.array(errors)).mean() * 180 / np.pi
            }
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
    if error_type == 'end_pos':
        if WD_est_method == 'compare_all_methods':
            print('mean error PCA= ' + str(np.array(errors_PCA).mean().round(3)))
            print('mean error SM heading= ' + str(np.array(errors_SM_heading).mean().round(3)))
            print('mean error inv pend= ' + str(np.array(errors_inv_pend).mean().round(3)))
            # print('mean error raw pca = ' + str(np.array(errors_pca_raw).mean().round(3)))
            print('mean error resnet18 = ' + str(np.array(errors_resnet18).mean().round(3)))
            # print('mean error resnet18_raw = ' + str(np.array(errors_resnet18_raw).mean().round(3)))
            results_dic = {
                'e_PCA': np.array(errors_PCA).mean().round(3),
                'e_SP_heading': np.array(errors_SM_heading).mean().round(3),
                'e_inv_pend': np.array(errors_inv_pend).mean().round(3),
                'e_resnet18': np.array(errors_resnet18).mean().round(3)
            }
            fig = plt.figure('WDE pos errors')
            ax1 = fig.add_subplot(111)
            ax1.hist(np.array(errors_PCA), 100, label='PCA')
            ax1.hist(np.array(errors_SM_heading), 100, label='SM_heading')
            ax1.hist(np.array(errors_inv_pend), 100, label='inv pend')
            # ax1.hist(np.array(errors_pca_raw), 100, label='raw pca')
            ax1.hist(np.array(errors_resnet18), 100, label='resnet18')
            # ax1.hist(np.array(errors_resnet18_raw), 100, label='resnet18_raw')
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
            # bins_count,cdf = MyCDF(np.array(errors_pca_raw))
            # plt.plot(bins_count[1:],cdf,label='raw pca')
            bins_count, cdf = MyCDF(np.array(errors_resnet18))
            plt.plot(bins_count[1:], cdf, label='resnet18')
            # bins_count, cdf = MyCDF(np.array(errors_resnet18_raw))
            # plt.plot(bins_count[1:], cdf, label='resnet18_raw')
            plt.grid(), plt.legend()
            plt.xlabel('WD pos error [m]'), plt.ylabel('CDF')
        else:
            print('mean error = ' + str(np.array(errors).mean()))
            print('error std = ' + str(np.array(errors).std() ))
            print('mean abs error = ' + str(abs(np.array(errors)).mean()))
            results_dic = {
                'mean error': np.array(errors).mean().round(3),
                'error std': np.array(errors).std().round(3),
                'mean abs error': abs(np.array(errors)).mean().round(3)
            }
            fig = plt.figure('WDE pos errors')
            ax1 = fig.add_subplot(211)
            ax1.grid(True)
            ax1.plot(np.array(errors))
            ax1.set(ylabel='WD pos error [m]')
            ax2 = fig.add_subplot(212)
            ax2.hist(np.array(errors), 100)
            ax2.grid(True)
            ax2.set(ylabel='samples', xlabel='WD pos error [m]')
    plt.savefig(join(res18_optimization_results_location, 'prformance.png'))
    with open(join(res18_optimization_results_location, 'prformance.json'), 'w') as f:
        json.dump(results_dic, f, indent=4)
        f.close()


def debug_segmentation(data_location, res18_optimization_results_location, model_name):
    # WD_est_method = 'compare_all_methods'  # could be: 'SM_heading', 'PCA', 'inverted_pendulum', 'resnet18LinAcc', 'resnet18RawIMU' , 'compare_all_methods'
    main_wd_path = '/'
    print('data location: ' + data_location)
    test_ds = MySBGPytorchDS(join(data_location, 'test.mat'))
    window_size = test_ds.window_size
    print('loading test segments')
    with open(join(data_location, 'test_segments_for_WDE.pickle'), 'rb') as file:
        analyzied_segments = pickle.load(file)
    print('loading model')
    ### 1st test compare values :
    # test first segment
    if False:
        e = []
        for (segment, i) in zip(analyzied_segments, range(len(analyzied_segments))):
            lin_acc = segment.lin_acc_b_frame
            lin_acc_n_frame = Functions.transform_vectors(lin_acc, segment.Rot).reshape(1, 200, 3)
            X = test_ds.X
            X = X.permute(0, 2, 1, 3)
            X = X[i].squeeze().numpy()
            e.append(np.mean(lin_acc_n_frame - X))
        print(np.mean(np.array(e)))
    ### 2nd test compare resnet inference
    if True:
        # prepare network
        model_path = join(res18_optimization_results_location, model_name)
        print('model path: ' + model_path)
        res18model = torch.load(model_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        res18model.to(device)
        # calculate using the class routine
        analyzied_segment = analyzied_segments[0]
        analyzied_segment.res18model = res18model
        analyzied_segment.window_size = window_size
        analyzied_segment.model_path = model_path
        analyzied_segment.res18_direction_pred(plot_results=False, data_type='LinAcc', device=device)
        WD_angle_err_resnet18, end_pos_err_resnet18 = analyzied_segment.calc_error()
        #### in the class methos
        LinAcc = analyzied_segment.lin_acc_b_frame # 200 X 3
        DCM_vec = analyzied_segment.Rot.as_matrix()  # Nx3x3
        a_meas_nav = np.einsum('ijk,ik->ij', DCM_vec, LinAcc)  # Lin accelaration at Nav frame
        batch_of_a_nav = a_meas_nav[:window_size, :]  # x,y,z
        batch_of_a_nav = batch_of_a_nav[None, :, :]  # adding batch dim

        net = res18model
        net.to(device)
        net.eval()
        from torch.utils.data import DataLoader

        test_dl = DataLoader(test_ds, len(test_ds), shuffle=False)
        LinAccNav_test, dL_GT_test, WDE_GT_test = next(iter(test_dl))
        LinAccNav_test, dL_GT_test, WDE_GT_test = LinAccNav_test[0], dL_GT_test[0], WDE_GT_test[0]
        ## check input
        e_input = batch_of_a_nav.squeeze() - LinAccNav_test.squeeze().permute(1, 0).to('cpu').numpy()
        e_input = np.linalg.norm(e_input, axis=1).mean()
        ### passed
        ### check output
        #### in the class method
        resent_input = Functions.PrepareInputForResnet18(batch_of_a_nav)
        resent_input.to(device)  # this cannot be done in operational function
        resent_input = resent_input.cuda()
        class_output = net(resent_input)
        ## in training
        LinAccNav_test, dL_GT_test, WDE_GT_test = next(iter(test_dl))
        LinAccNav_test = LinAccNav_test.to(device)
        training_outputs = net(LinAccNav_test)
        training_output = training_outputs[0]
        ## passed check error calculation
        ### check dL
        analyzied_segment.dL - dL_GT_test[0].numpy()[0]
        ### passed
        #### in training
        WDE_GT_test = WDE_GT_test / torch.linalg.norm(WDE_GT_test, dim=1).unsqueeze(1)
        WDE_GT_test = WDE_GT_test[0]
        dL_GT_test = dL_GT_test[0]
        training_output = training_output / torch.linalg.norm(training_output)
        import torch.nn as nn
        criterion = nn.L1Loss(reduction='mean')
        training_loss = criterion(training_output * dL_GT_test, WDE_GT_test * dL_GT_test)
        torch.linalg.norm(training_output * dL_GT_test - WDE_GT_test * dL_GT_test)
        f = nn.MSELoss()
        proposed_loss = torch.sqrt(f(training_output * dL_GT_test, WDE_GT_test * dL_GT_test))

        ### in the class
        WD_vector_est = class_output.cpu().detach().numpy().squeeze()
        end_pos_est = WD_vector_est / np.linalg.norm(WD_vector_est) * analyzied_segment.dL
        end_pos_error = np.linalg.norm(end_pos_est - analyzied_segment.WD_vector_GT)

        # errors_PCA = []
        # errors_resnet18 = []
        #
        # i = 1
        # N = len(analyzied_segments)
        # print(N)
        # for analyzied_segment in analyzied_segments:
        #     if np.mod(i, 500) == 0:
        #         print(str(round(i/N*100,3)) + '% completed')
        #     i += 1
        #
        #     analyzied_segment.PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        #     WD_angle_err_PCA, end_pos_err_PCA = analyzied_segment.calc_error()
        #     analyzied_segment.res18model = res18model
        #     analyzied_segment.window_size = window_size
        #     analyzied_segment.model_path = model_path
        #     analyzied_segment.res18_direction_pred(plot_results=False, data_type='LinAcc', device=device)
        #     WD_angle_err_resnet18, end_pos_err_resnet18 = analyzied_segment.calc_error()
        #
        #     errors_PCA.append(end_pos_err_PCA)
        #     errors_resnet18.append(end_pos_err_resnet18)
        #
        #     print('mean error PCA= ' + str(np.array(errors_PCA).mean().round(3)))
        #     print('mean error resnet18 = ' + str(np.array(errors_resnet18).mean().round(3)))
        #     results_dic = {
        #         'e_PCA': np.array(errors_PCA).mean().round(3),
        #         'e_resnet18': np.array(errors_resnet18).mean().round(3)
        #     }
        #     fig = plt.figure('WDE pos errors')
        #     ax1 = fig.add_subplot(111)
        #     ax1.hist(np.array(errors_PCA), 100, label='PCA')
        #     ax1.hist(np.array(errors_SM_heading), 100, label='SM_heading')
        #     ax1.hist(np.array(errors_inv_pend), 100, label='inv pend')
        #     ax1.hist(np.array(errors_resnet18), 100, label='resnet18')
        #     ax1.legend()
        #     ax1.grid(True)
        #     ax1.set(ylabel='samples', xlabel='WD pos error [m]')
        #
        #     plt.figure('Pos CDF')
        #     bins_count,cdf = MyCDF(np.array(errors_PCA))
        #     plt.plot(bins_count[1:],cdf,label='PCA')
        #     bins_count, cdf = MyCDF(np.array(errors_resnet18))
        #     plt.plot(bins_count[1:], cdf, label='resnet18')
        #     plt.grid(), plt.legend()
        #     plt.xlabel('WD pos error [m]'), plt.ylabel('CDF')


if __name__ == '__main__':
    opt_folder ='/home/maint/git/walking_direction_estimation/data/optimization_results/2023-02-05T13:19:45.474374optimization results on Swing, window size: 200_heading_angle_fix'
    info_file_path = join(opt_folder, 'info_file.json')
    with open(info_file_path, "r") as f:
        info = json.loads(f.read())
    data_folder = info["root_dir"]
    model_name = info["best_saved_model"]
    add_quat = True
    WD_est_method = 'compare_all_methods'#'compare_all_methods'
    calculate_performance(data_location=data_folder,
                          res18_optimization_results_location=opt_folder,
                          model_name=model_name,
                          WD_est_method=WD_est_method,
                          add_quat=add_quat)
    # debug_segmentation(
    #     data_location= '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-01-22T19:51:40.411866TRC#1 swing wind_size 200swing left right mix',
    #     res18_optimization_results_location= '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-01-22T21:36:17.392113optimization results on Swing, window size: 200 swing left right mix',
    #     model_name='WDE_regressor_Swing_LinAcc_0.82.pth'
    # )