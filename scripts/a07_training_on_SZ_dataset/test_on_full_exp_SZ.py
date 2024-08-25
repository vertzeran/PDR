import os
import utils.Classes as Classes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils.Functions as Functions
import random
from os import listdir
from os.path import join
from utils.Functions import construct_traj
from scripts.a07_training_on_SZ_dataset.create_segments_for_WDE_SZ import get_segments_from_exp, get_exp_list, \
    rotate_traj_to_minimize_error, get_tain_test_dir
import torch
import json
#from create_segments_for_WDE_SBG import get_dir_to_analyze, get_exp_list, rotate_traj_to_minimize_error
import scipy.io as sio


def traj_est_using_SP_heading(Exp, params, plot_result=False, use_GT_att=False,
                              predict_dL_with_PDRnet=False,
                              pdr_net=None, allign_traj=False):
    if use_GT_att:
        segment_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                     params=params,
                                                                     use_GT_att=use_GT_att,
                                                                     )
    else:
        segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                     params=params,
                                                                     use_GT_att=use_GT_att,
                                                                     )
        # psi0 = AHRS_results_list[0].heading[0]
    analyzied_segments = []
    traj = np.array([0, 0])
    dL = np.array([])
    gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:] - Exp.Pos.arr()[Exp.index_of_walking_start,:]
    # Exp.define_walking_start_idx()
    est_time = [Exp.Time_IMU[Exp.index_of_walking_start - 1]]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        est_time.extend(segment.Time_GT)
        if not use_GT_att:
            # AHRS results segmentation
            AHRS_results = AHRS_results_list[i]
            # t_start = segment.Time_IMU[0].round(11)
            # t_end = segment.Time_IMU[-1].round(11)
            # ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            # lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            # Rot_seg = Rot_est[ind_IMU]
            # Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
                                                                       Rot_est=AHRS_results.Rot,
                                                                       grv_est=AHRS_results.grv,
                                                                       Heading_est=AHRS_results.heading,
                                                                       use_GT_dl=not predict_dL_with_PDRnet,
                                                                       pdr_net=pdr_net,
                                                                       arc_length_for_dl=True),
                                      )
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       use_GT_dl=not predict_dL_with_PDRnet,
                                                                       pdr_net=pdr_net,
                                                                       arc_length_for_dl=True)
                                      )
        # analyzied_segments[-1].Heading -= segment.initial_heading
        analyzied_segments[-1].walking_direction_estimation_using_smartphone_heading(plot_results=False)
        traj = np.vstack([traj, traj[-1] + analyzied_segments[-1].est_traj])

    dL = Exp.calc_dL(window_size=1)
    dL = np.interp(est_time[1:], Exp.Time_IMU[1:], dL)
    traj_errors, traj_length, error_metrika, rot_traj = calculate_traj_error(Exp, np.array(est_time), traj, dL,
                                                                             fix_heading=allign_traj)
    if plot_result:
        fig = plt.figure('traj_est_using_SP_heading')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='black', linestyle='--', linewidth=2, label='gt')
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='raw')
        ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
        ax.legend()
        ax.axis('equal')
    return rot_traj, traj_errors, traj_length, error_metrika


def traj_est_using_inv_pend(Exp, window_size, plot_result=False, use_GT_att=False, wind_size_for_heading_init=1000,pdr_net=None):
    if use_GT_att:
        segment_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                     window_size=window_size,
                                                                     use_GT_att=use_GT_att,
                                                                     wind_size_for_heading_init=wind_size_for_heading_init)
    else:
        segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                                        window_size=window_size,
                                                                                        use_GT_att=use_GT_att,
                                                                                        wind_size_for_heading_init=wind_size_for_heading_init)
    analyzied_segments = []
    traj = np.array([0, 0])
    gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:] - Exp.Pos.arr()[Exp.index_of_walking_start,:]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        if not use_GT_att:
            # AHRS results segmentation
            AHRS_results = AHRS_results_list[i]
            # t_start = segment.Time_IMU[0].round(11)
            # t_end = segment.Time_IMU[-1].round(11)
            # ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            # lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            # Rot_seg = Rot_est[ind_IMU]
            # Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
                                                                       Rot_est=AHRS_results.Rot,
                                                                       grv_est=AHRS_results.grv,
                                                                       Heading_est=AHRS_results.heading,
                                                                       pdr_net=pdr_net))
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,pdr_net=pdr_net))
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


def traj_est_using_PCA(Exp, params, plot_result=False, use_GT_att=False,
                              predict_dL_with_PDRnet=False, pdr_net=None, allign_traj=False):
    if use_GT_att:
        segment_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                     params=params,
                                                                     use_GT_att=use_GT_att,
                                                                     )
    else:
        segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                                        params=params,
                                                                                        use_GT_att=False)
    analyzied_segments = []
    walking_angle = []
    dL = []
    est_time = [Exp.Time_IMU[Exp.index_of_walking_start - 1]]
    p = Exp.Pos.arr()[Exp.index_of_walking_start]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        est_time.append(segment.Time_GT[-1])
        if not use_GT_att:
            # AHRS results segmentation
            AHRS_results = AHRS_results_list[i]
            # t_start = segment.Time_IMU[0].round(11)
            # t_end = segment.Time_IMU[-1].round(11)
            # ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            # lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            # Rot_seg = Rot_est[ind_IMU]
            # Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
                                                                       Rot_est=AHRS_results.Rot,
                                                                       grv_est=AHRS_results.grv,
                                                                       Heading_est=AHRS_results.heading,
                                                                       use_GT_dl=not predict_dL_with_PDRnet,
                                                                       pdr_net=pdr_net,
                                                                       arc_length_for_dl=False)
                                      )
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       use_GT_dl=not predict_dL_with_PDRnet,
                                                                       pdr_net=pdr_net,
                                                                       arc_length_for_dl=False)
                                      )
        analyzied_segments[-1].PCA_direction_analysis(plot_results=False, use_GT_to_solve_amguity=True)
        # dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
        # dp = segment.Pos.arr()[-1] - p
        # p = segment.Pos.arr()[-1]
        # dL.append(np.linalg.norm(dp[0:2]))
        dL.append(analyzied_segments[-1].dL)
        walking_angle.append(analyzied_segments[-1].WD_angle_est)
    gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:] - Exp.Pos.arr()[Exp.index_of_walking_start, :]
    traj = construct_traj(np.array(dL), np.array(walking_angle), plot_result=False, pos_gt=gt_pos)
    traj_errors, traj_length, error_metrika, rot_traj = calculate_traj_error(Exp, np.array(est_time), traj, dL,
                                                                             fix_heading=allign_traj)
    if plot_result:
        fig = plt.figure('traj_est_using_PCA')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='black', linestyle='--', linewidth=2, label='gt')
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='raw')
        ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
        ax.legend()
        ax.axis('equal')
    return rot_traj, walking_angle, traj_errors, traj_length, error_metrika


def traj_est_using_resnet18(Exp, params, plot_result=False, use_GT_att=False, res18model=None, device='cpu',
                            model_path=None, predict_dL_with_PDRnet=False, pdr_net=None, allign_traj=False):
    if use_GT_att:
        segment_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                     params=params,
                                                                     use_GT_att=use_GT_att,
                                                                     )
    else:
        segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
                                                                     params=params,
                                                                     use_GT_att=use_GT_att,
                                                                     )
    analyzied_segments = []
    walking_angle = []
    dL = []
    est_time = [Exp.Time_IMU[Exp.index_of_walking_start - 1]]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        est_time.append(segment.Time_GT[-1])
        if not use_GT_att:
            # AHRS results segmentation
            AHRS_results = AHRS_results_list[i]
            # t_start = segment.Time_IMU[0].round(11)
            # t_end = segment.Time_IMU[-1].round(11)
            # ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
            # lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
            # Rot_seg = Rot_est[ind_IMU]
            # Heading_seg = Heading_est[ind_IMU]
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
                                                                       Rot_est=AHRS_results.Rot,
                                                                       grv_est=AHRS_results.grv,
                                                                       Heading_est=AHRS_results.heading,
                                                                       use_GT_dl=not predict_dL_with_PDRnet,
                                                                       pdr_net=pdr_net,
                                                                       arc_length_for_dl=False)
                                      )
        else:
            analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
                                                                       use_GT_dl=not predict_dL_with_PDRnet,
                                                                       pdr_net=pdr_net,
                                                                       arc_length_for_dl=False)
                                      )
        analyzied_segments[-1].res18model = res18model
        analyzied_segments[-1].window_size = params["window_size"]
        analyzied_segments[-1].model_path = model_path
        analyzied_segments[-1].res18_direction_pred(plot_results=False,
                                                    data_type='LinAcc',
                                                    device=device, add_quat=params["add_quat"])
        walking_angle.append(analyzied_segments[-1].WD_angle_est)
        # dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
        # dp = segment.Pos.arr()[-1] - p
        # p = segment.Pos.arr()[-1]
        # dL.append(np.linalg.norm(dp[0:2]))
        dL.append(analyzied_segments[-1].dL)
    gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:] - Exp.Pos.arr()[Exp.index_of_walking_start, :]
    traj = construct_traj(np.array(dL), np.array(walking_angle), plot_result=False, pos_gt=gt_pos)
    traj_errors, traj_length, error_metrika, rot_traj = calculate_traj_error(Exp, np.array(est_time), traj, dL,
                                                                             fix_heading=allign_traj)
    if plot_result:
        fig = plt.figure('traj_est_using_PCA')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='black', linestyle='--', linewidth=2, label='gt')
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='raw')
        ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
        ax.legend()
        ax.axis('equal')
    return rot_traj, walking_angle, traj_errors, traj_length, error_metrika


# def traj_est_using_resnet18RV(Exp, window_size, plot_result=False, use_GT_att=False, res18RVmodel=None, device='cpu', model_path = None):
#     if use_GT_att:
#         segment_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
#                                                                      window_size=window_size,
#                                                                      use_GT_att=use_GT_att,
#                                                                      wind_size_for_heading_init=750)
#     else:
#         segment_list, AHRS_results_list, initial_WD_angles_list = get_segments_from_exp(exp=Exp,
#                                                                                         window_size=window_size,
#                                                                                         use_GT_att=use_GT_att,
#                                                                                         wind_size_for_heading_init=750)
#     analyzied_segments = []
#     walking_angle = []
#     dL = []
#     Exp.define_walking_start_idx()
#     est_time = [Exp.Time_IMU[Exp.index_of_walking_start - 1]]
#     for i in range(len(segment_list)):
#         segment = segment_list[i]
#         if not use_GT_att:
#             # AHRS results segmentation
#             AHRS_results = AHRS_results_list[i]
#             # t_start = segment.Time_IMU[0].round(11)
#             # t_end = segment.Time_IMU[-1].round(11)
#             # ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
#             # lin_acc_b_frame_seg = lin_acc_b_frame_est[ind_IMU]
#             # Rot_seg = Rot_est[ind_IMU]
#             # Heading_seg = Heading_est[ind_IMU]
#             analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att,
#                                                                        lin_acc_b_frame_est=AHRS_results.lin_acc_b_frame,
#                                                                        Rot_est=AHRS_results.Rot,
#                                                                        grv_est=AHRS_results.grv,
#                                                                        Heading_est=AHRS_results.heading,pdr_net=pdr_net))
#         else:
#             analyzied_segments.append(Classes.WDE_performance_analysis(segment, use_GT_att=use_GT_att, pdr_net=pdr_net))
#         analyzied_segments[-1].res18RVmodel = res18RVmodel
#         analyzied_segments[-1].window_size = window_size
#         analyzied_segments[-1].model_path = model_path
#         analyzied_segments[-1].res18RV_direction_pred(plot_results=False, device=device)
#         walking_angle.append(analyzied_segments[-1].WD_angle_est)
#         dL.append(analyzied_segments[-1].segment.calc_dL(window_size=1).sum())
#     gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:] - Exp.Pos.arr()[Exp.index_of_walking_start, :]
#     traj = construct_traj(np.array(dL), np.array(walking_angle), plot_result=plot_result, pos_gt=gt_pos)
#     traj_errors, traj_length, error_metrika = calculate_traj_error(Exp, np.array(est_time), traj, dL)
#     return traj, walking_angle, traj_errors, traj_length, error_metrika


def test_on_exp(params, WD_est_method = 'compare_all_methods',
                experiment_path=None, GT=None, outputfolder=None, res18model=None, predict_dL_with_PDRnet=False,
                pdr_net=None, allign_traj=False, save_n_close_fig=True):
    use_GT_att = False
    if params["dataset"] == 'RIDI':
        Exp = Classes.RidiExp(path=experiment_path)
    elif params["dataset"] == 'TRC#1':
        Exp = Classes.SbgExpRawData(path=experiment_path, GT=GT)
        Exp.define_walking_start_idx(th=1)
    elif params["dataset"] == 'AI_PDR':
        Exp = Classes.AI_PDR_exp_w_SP_GT(path=experiment_path)
        Exp.define_walking_start_idx(th=3)
    elif params["dataset"] == 'RIDI_ENU':
        Exp = Classes.RidiExp_ENU(path=experiment_path)
        Exp.define_walking_start_idx(th=1)
    else:
        raise 'invalid dataset'
    if params["traj_length_limit"] is not None:
        Exp.limit_traj_length(limit=params["traj_length_limit"])
    Exp.define_walking_start_idx(th=params["walking_start_threshold"])
    print(join(Exp.Path, Exp.FileName))
    if WD_est_method == 'PCA':
        traj, walking_angle, traj_errors, traj_length, error_metrika = traj_est_using_PCA(Exp,
                                                                                          params["window_size"],
                                                                                          plot_result=True,
                                                                                          use_GT_att=use_GT_att,
                                                                                          predict_dL_with_PDRnet=predict_dL_with_PDRnet,
                                                                                          pdr_net=pdr_net)
        errors_dic = None
    elif WD_est_method == 'SP_heading':
        traj_est_using_SP_heading(Exp, params["window_size"], wind_size_for_heading_init=params["wind_size_for_heading_init"],
                                  pdr_net=pdr_net, predict_dL_with_PDRnet=predict_dL_with_PDRnet,
                                  plot_result=True)
        errors_dic = None
    elif WD_est_method == 'inverted_pendulum':
        traj_est_using_inv_pend(Exp, params["window_size"], plot_result=True,
                                wind_size_for_heading_init=params["wind_size_for_heading_init"],
                                pdr_net=pdr_net, predict_dL_with_PDRnet=predict_dL_with_PDRnet)
    elif WD_est_method == 'resnet18LinAcc':
        dL = Exp.calc_dL(params["window_size"])
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        traj, walking_angle, traj_errors, traj_length, error_metrika = traj_est_using_resnet18(Exp, params["window_size"],
                                                                                               plot_result=True,
                                                                                               use_GT_att=use_GT_att,
                                                                                               device=device,
                                                                                               res18model=res18model,
                                                                                               pdr_net = pdr_net,
                                                                                               predict_dL_with_PDRnet=predict_dL_with_PDRnet,
                                                                                               model_path=None,
                                                                                               add_quat=params["add_quat"])
        errors_dic = None
    elif WD_est_method == 'compare_all_methods':
        est_traj_PCA, _, traj_errors_PCA, traj_length_PCA, error_metrika_PCA = \
            traj_est_using_PCA(Exp, params, plot_result=False, use_GT_att=use_GT_att,
                              predict_dL_with_PDRnet=predict_dL_with_PDRnet,
                               pdr_net=pdr_net, allign_traj=allign_traj)
        est_traj_SP_heading, traj_errors_SP_heading, traj_length_SP_heading, error_metrika_SP_heading = \
            traj_est_using_SP_heading(Exp=Exp, params=params, plot_result=False, use_GT_att=use_GT_att,
                              predict_dL_with_PDRnet=predict_dL_with_PDRnet,
                              pdr_net=pdr_net, allign_traj=allign_traj)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        est_traj_resnet18, _, traj_errors_resnet18, traj_length_resnet18, error_metrika_resnet18 = \
            traj_est_using_resnet18(Exp, params=params, plot_result=False, use_GT_att=use_GT_att,
                                    device=device, res18model=res18model, pdr_net=pdr_net, model_path=None,
                                    predict_dL_with_PDRnet=predict_dL_with_PDRnet, allign_traj=allign_traj)
        print('PCA error: ' + str(error_metrika_PCA) + ' %')
        print('SP heading error: ' + str(error_metrika_SP_heading) + ' %')
        print('resnet 18 error: ' + str(error_metrika_resnet18) + ' %')
        PCA_errors_dic = {'traj_errors': traj_errors_PCA,
                          'traj_length': traj_length_PCA,
                          'error_metrika': error_metrika_PCA}
        SP_heading_errors_dic = {'traj_errors': traj_errors_SP_heading,
                                 'traj_length': traj_length_SP_heading,
                                 'error_metrika': error_metrika_SP_heading}
        resnet18_errors_dic = {'traj_errors': traj_errors_resnet18,
                               'traj_length': traj_length_resnet18,
                               'error_metrika': error_metrika_resnet18}
        errors_dic = {"PCA": PCA_errors_dic, "SP heading": SP_heading_errors_dic, "resnet18": resnet18_errors_dic}

        fig = plt.figure('walking_direction_estimation compare all methods')
        Ax = fig.subplots()
        Ax.set_title(Exp.FileName)

        gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:] - Exp.Pos.arr()[Exp.index_of_walking_start, :]
        Ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                 color="black", linestyle='--', label='GT')
        Ax.plot(est_traj_PCA[:, 0], est_traj_PCA[:, 1], label='PCA')
        Ax.plot(est_traj_SP_heading[:, 0], est_traj_SP_heading[:, 1], label='SP_heading')
        Ax.plot(est_traj_resnet18[:, 0], est_traj_resnet18[:, 1], label='resnet18')
        Ax.grid(True)
        Ax.legend()
        Ax.axis('equal')
        if save_n_close_fig:
            plt.savefig(join(outputfolder, Exp.FileName + '_traj.png'))
            plt.close(fig)
    return errors_dic

def calculate_traj_error(Exp, est_time, est_pos, dL, fix_heading=False):
    gt_time = Exp.Time_GT[Exp.index_of_walking_start:]
    gt_pos = Exp.Pos.arr()[Exp.index_of_walking_start:] - Exp.Pos.arr()[Exp.index_of_walking_start, :]
    traj_length = np.array(dL).cumsum()
    gt_pos_interp = np.vstack([np.interp(est_time, gt_time, gt_pos[:, 0]),
                               np.interp(est_time, gt_time, gt_pos[:, 1]),
                               np.interp(est_time, gt_time, gt_pos[:, 2])]).T
    if fix_heading:
        est_pos, rot_angle = rotate_traj_to_minimize_error(est_pos, gt_pos_interp[:, 0:2])
    traj_errors = np.linalg.norm(est_pos - gt_pos_interp[:, 0:2], axis=1)
    normalized_errors = traj_errors[1:] / traj_length
    error_metrika = normalized_errors.mean() * 100
    return traj_errors, traj_length, error_metrika, est_pos


def calc_error_statistics(X, Y, dx, n_round, plot_results=False):
    """plot statistics of many graphs. X and Y's are lists since they are different sizes"""
    assert len(X) == len(Y)
    num_of_graphs = len(X)
    starts = np.array([])
    ends = np.array([])
    for (x, y) in zip(X, Y):
        starts = np.hstack([starts, x[0]])
        ends = np.hstack([ends, x[-1]])
    start = starts.min()
    end = ends.max()
    x_common = np.linspace(start.round(n_round), end.round(n_round), round((end - start)/dx + 1))
    y_mean = np.array([])
    y_std = np.array([])
    y_max = np.array([])
    y_min = np.array([])
    i = 0
    for x_sample in x_common:
        y_sample = np.array([])
        for (x, y) in zip(X, Y):
            if x[0] <= x_sample and x_sample <= x[-1]:
                y_interp = np.interp(x_sample, x, y)
                y_sample = np.hstack([y_sample, y_interp])
        if y_sample.size == 0:
            x_common = np.delete(x_common, i, 0)
        else:
            y_mean = np.hstack([y_mean, y_sample.mean()])
            y_std = np.hstack([y_std, y_sample.std()])
            y_max = np.hstack([y_max, y_sample.max()])
            y_min = np.hstack([y_min, y_sample.min()])
            i += 1
    if plot_results:
        plt.figure('error_statistics_plot')
        plt.plot(x_common, y_mean,'-', color='gray')
        plt.fill_between(x_common, y_mean - y_std, y_mean + y_std, color='gray', alpha=0.2)
        plt.show()
    return x_common, y_mean, y_std, y_max, y_min


def test_on_list(exp_list, params, traj_folder, res18model):
    file_names = []
    PCA_tarj_errors = []
    PCA_tarj_lengths = []
    PCA_errors = []
    PCA_errors_normalized = []
    SP_heading_tarj_errors = []
    SP_heading_tarj_lengths = []
    SP_heading_errors = []
    SP_heading_errors_normalized = []
    resnet18_tarj_errors = []
    resnet18_tarj_lengths = []
    resnet18_errors = []
    resnet18_errors_normalized = []
    data_to_save = {}
    i = 1
    N = len(exp_list)
    for exp in exp_list:
        errors = test_on_exp(params, WD_est_method = 'compare_all_methods',
                             experiment_path=exp, GT=None, outputfolder=traj_folder, res18model=res18model,
                             predict_dL_with_PDRnet=False, pdr_net=None, allign_traj=False, save_n_close_fig=True)
        PCA_tarj_errors.append(errors["PCA"]["traj_errors"][1:])  # removing first component = 0
        PCA_tarj_lengths.append(errors["PCA"]["traj_length"])
        PCA_errors.append(errors["PCA"]["traj_errors"].mean())
        PCA_errors_normalized.append(errors["PCA"]["error_metrika"])
        SP_heading_tarj_errors.append(errors["SP heading"]["traj_errors"][1:])
        SP_heading_tarj_lengths.append(errors["SP heading"]["traj_length"])
        SP_heading_errors.append(errors["SP heading"]["traj_errors"].mean())
        SP_heading_errors_normalized.append(errors["SP heading"]["error_metrika"])
        resnet18_tarj_errors.append(errors["resnet18"]["traj_errors"][1:])
        resnet18_tarj_lengths.append(errors["resnet18"]["traj_length"])
        resnet18_errors.append(errors["resnet18"]["traj_errors"].mean())
        resnet18_errors_normalized.append(errors["resnet18"]["error_metrika"])
        file_names.append(exp)
        # head, tail = ntpath.split(exp)
        # print(tail)
        print(str((i / N * 100).__round__(3)) + ' % done')
        i += 1
    x_common_PCA, y_mean_PCA, y_std_PCA, y_max_PCA, y_min_PCA = calc_error_statistics(X=PCA_tarj_lengths, Y=PCA_tarj_errors, dx=0.1,
                                                                n_round=1, plot_results=False)
    sio.savemat(join(traj_folder, 'PCA_L_vs_e.mat'),
                {'x_common': x_common_PCA, 'y_mean': y_mean_PCA, 'y_std': y_std_PCA})
    x_common_SP_heading, y_mean_SP_heading, y_std_SP_heading, y_max_SP_heading, y_min_SP_heading = calc_error_statistics(X=SP_heading_tarj_lengths,
                                                                                     Y=SP_heading_tarj_errors,
                                                                                     dx=0.1, n_round=1,
                                                                                     plot_results=False)
    sio.savemat(join(traj_folder, 'SP_heading_L_vs_e.mat'),
                {'x_common': x_common_SP_heading, 'y_mean': y_mean_SP_heading, 'y_std': y_std_SP_heading})
    x_common_resnet18, y_mean_resnet18, y_std_resnet18, y_max_resnet18, y_min_resnet18 = calc_error_statistics(X=resnet18_tarj_lengths,
                                                                               Y=resnet18_tarj_errors, dx=0.1,
                                                                               n_round=1, plot_results=False)
    sio.savemat(join(traj_folder, 'resnet18_L_vs_e.mat'),
                {'x_common': x_common_resnet18, 'y_mean': y_mean_resnet18, 'y_std': y_std_resnet18})
    plt.figure('error_statistics_plot')
    plt.plot(x_common_PCA, y_mean_PCA, '-', color='gray', label='PCA')
    plt.fill_between(x_common_PCA, y_min_PCA, y_max_PCA, color='gray', alpha=0.2)
    plt.plot(x_common_SP_heading, y_mean_SP_heading, '-', color='green', label='SP heading')
    plt.fill_between(x_common_SP_heading, y_min_SP_heading, y_max_SP_heading,
                     color='green', alpha=0.2)
    plt.plot(x_common_resnet18, y_mean_resnet18, '-', color='blue', label='resnet18')
    plt.fill_between(x_common_resnet18, y_min_resnet18, y_max_resnet18,
                     color='blue', alpha=0.2)
    plt.legend()
    plt.grid(True)
    plt.savefig(join(traj_folder, 'error_statistics.png'))
    plt.show()
    data_to_save["file_names"] = file_names
    data_to_save["PCA_errors"] = PCA_errors
    data_to_save["PCA_errors_normalized"] = PCA_errors_normalized
    data_to_save["SP_heading_errors"] = SP_heading_errors
    data_to_save["SP_heading_errors_normalized"] = SP_heading_errors_normalized
    data_to_save["resnet18_errors"] = resnet18_errors
    data_to_save["resnet18_errors_normalized"] = resnet18_errors_normalized
    file_path = join(traj_folder, 'performance summary.xlsx')
    Functions.save_to_excel_file(path=file_path, dic=data_to_save, print_message=True)
    print('PCA mean normalized errors: ' + str(np.array(PCA_errors_normalized).mean()))
    print('SP_heading mean normalized errors: ' + str(np.array(SP_heading_errors_normalized).mean()))
    print('resnet18 mean normalized errors: ' + str(np.array(resnet18_errors_normalized).mean()))


def test_on_person_list(person_list, dir_to_analyze, info, traj_folder, res18model, predict_dL_with_PDRnet, pdr_net, add_quat):
    file_names = []
    PCA_tarj_errors = []
    PCA_tarj_lengths = []
    PCA_errors = []
    PCA_errors_normalized = []
    SP_heading_tarj_errors = []
    SP_heading_tarj_lengths = []
    SP_heading_errors = []
    SP_heading_errors_normalized = []
    resnet18_tarj_errors = []
    resnet18_tarj_lengths = []
    resnet18_errors = []
    resnet18_errors_normalized = []
    data_to_save = {}
    exp_path_list = get_exp_list(list_of_persons=person_list,
                                 dir_to_analyze=dir_to_analyze)
    N = len(exp_path_list)
    i = 1
    for person in person_list:
        person_path = join(dir_to_analyze, person)
        GT_path = join(person_path, 'ascii-output.txt')
        GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
        exp_list_in_person = get_exp_list(list_of_persons=[person],
                                          dir_to_analyze=dir_to_analyze)
        for exp_path in exp_list_in_person:
            errors = test_on_exp(window_size=info["training_window_size"], wind_size_for_heading_init=1000,
                                 WD_est_method='compare_all_methods', experiment_path=exp_path, GT=GT,
                                 outputfolder=traj_folder, res18model=res18model,
                                 predict_dL_with_PDRnet=predict_dL_with_PDRnet, pdr_net=pdr_net, add_quat=add_quat,
                                 dataset='TRC#1')
            PCA_tarj_errors.append(errors["PCA"]["traj_errors"][1:])  # removing first component = 0
            PCA_tarj_lengths.append(errors["PCA"]["traj_length"])
            PCA_errors.append(errors["PCA"]["traj_errors"].mean())
            PCA_errors_normalized.append(errors["PCA"]["error_metrika"])
            SP_heading_tarj_errors.append(errors["SP heading"]["traj_errors"][1:])
            SP_heading_tarj_lengths.append(errors["SP heading"]["traj_length"])
            SP_heading_errors.append(errors["SP heading"]["traj_errors"].mean())
            SP_heading_errors_normalized.append(errors["SP heading"]["error_metrika"])
            resnet18_tarj_errors.append(errors["resnet18"]["traj_errors"][1:])
            resnet18_tarj_lengths.append(errors["resnet18"]["traj_length"])
            resnet18_errors.append(errors["resnet18"]["traj_errors"].mean())
            resnet18_errors_normalized.append(errors["resnet18"]["error_metrika"])
            file_names.append(exp_path)
            # head, tail = ntpath.split(exp)
            # print(tail)
            print(str((i / N * 100).__round__(3)) + ' % done')
            i += 1

    x_common_PCA, y_mean_PCA, y_std_PCA, y_max_PCA, y_min_PCA = calc_error_statistics(X=PCA_tarj_lengths,
                                                                                      Y=PCA_tarj_errors, dx=0.1,
                                                                                      n_round=1, plot_results=False)

    sio.savemat(join(traj_folder, 'PCA_L_vs_e.mat'),
                {'x_common': x_common_PCA, 'y_mean': y_mean_PCA, 'y_std': y_std_PCA, 'y_max': y_max_PCA, 'y_min': y_min_PCA})
    x_common_SP_heading, y_mean_SP_heading, y_std_SP_heading, y_max_SP_heading, y_min_SP_heading = calc_error_statistics(X=SP_heading_tarj_lengths,
                                                                                     Y=SP_heading_tarj_errors,
                                                                                     dx=0.1, n_round=1,
                                                                                     plot_results=False)
    sio.savemat(join(traj_folder, 'SP_heading_L_vs_e.mat'),
                {'x_common': x_common_SP_heading, 'y_mean': y_mean_SP_heading, 'y_std': y_std_SP_heading, 'y_max': y_max_SP_heading, 'y_min': y_min_SP_heading})
    x_common_resnet18, y_mean_resnet18, y_std_resnet18, y_max_resnet18, y_min_resnet18 = calc_error_statistics(X=resnet18_tarj_lengths,
                                                                               Y=resnet18_tarj_errors, dx=0.1,
                                                                               n_round=1, plot_results=False)
    sio.savemat(join(traj_folder, 'resnet18_L_vs_e.mat'),
                {'x_common': x_common_resnet18, 'y_mean': y_mean_resnet18, 'y_std': y_std_resnet18, 'y_max': y_max_resnet18, 'y_min': y_min_resnet18})

    # y_mean_PCA = y_mean_PCA / x_common_PCA * 100
    # y_std_PCA = y_std_PCA / x_common_PCA * 100
    # y_min_PCA = y_min_PCA / x_common_PCA * 100
    # y_max_PCA = y_max_PCA / x_common_PCA * 100
    #
    # y_mean_SP_heading = y_mean_SP_heading / x_common_SP_heading * 100
    # y_std_SP_heading = y_std_SP_heading / x_common_SP_heading * 100
    # y_min_SP_heading = y_min_SP_heading / x_common_SP_heading * 100
    # y_max_SP_heading = y_max_SP_heading / x_common_SP_heading * 100
    #
    # y_mean_resnet18 = y_mean_resnet18 / x_common_resnet18 * 100
    # y_std_resnet18 = y_std_resnet18 / x_common_resnet18 * 100
    # y_min_resnet18 = y_min_resnet18 / x_common_resnet18 * 100
    # y_max_resnet18 = y_max_resnet18 / x_common_resnet18 * 100

    plt.figure('error_statistics_plot')
    plt.plot(x_common_PCA, y_mean_PCA, '-', color='gray', label='PCA')
    plt.fill_between(x_common_PCA, y_min_PCA ,y_max_PCA, color='gray', alpha=0.2)
    plt.plot(x_common_SP_heading, y_mean_SP_heading, '-', color='green', label='SP heading')
    plt.fill_between(x_common_SP_heading, y_min_SP_heading, y_max_SP_heading,
                     color='green', alpha=0.2)
    plt.plot(x_common_resnet18, y_mean_resnet18, '-', color='blue', label='resnet18')
    plt.fill_between(x_common_resnet18, y_min_resnet18, y_max_resnet18,
                     color='blue', alpha=0.2)
    plt.legend()
    plt.grid(True)
    plt.savefig(join(traj_folder, 'error_statistics.png'))
    plt.show()
    data_to_save["file_names"] = file_names
    data_to_save["PCA_errors"] = PCA_errors
    data_to_save["PCA_errors_normalized"] = PCA_errors_normalized
    data_to_save["SP_heading_errors"] = SP_heading_errors
    data_to_save["SP_heading_errors_normalized"] = SP_heading_errors_normalized
    data_to_save["resnet18_errors"] = resnet18_errors
    data_to_save["resnet18_errors_normalized"] = resnet18_errors_normalized
    file_path = join(traj_folder, 'performance summary.xlsx')
    Functions.save_to_excel_file(path=file_path, dic=data_to_save, print_message=True)
    print('PCA mean normalized errors: ' + str(np.array(PCA_errors_normalized).mean()))
    print('SP_heading mean normalized errors: ' + str(np.array(SP_heading_errors_normalized).mean()))
    print('resnet18 mean normalized errors: ' + str(np.array(resnet18_errors_normalized).mean()))


if __name__ == '__main__':
    calculate_on_test_list = False
    sample_percentage = None
    opt_folder = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-03-13T09:30:56.999354optimization results on mixed, window size: 250_SZ_mixed_modes'
    #'/data/Work/Navigation/MyCopy'
    dataset_folder = '/data/Datasets/Navigation/Shenzhen_datasets/dataset-ShenZhen'
    train_test = 'test'
    exp_name = 'texting-0005.csv'
    exp_path = join(dataset_folder, train_test, exp_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    predict_dL_with_PDRnet = False
    WD_est_method = 'compare_all_methods'#'resnet18LinAcc'#'SP_heading'#'PCA'#'compare_all_methods'
    info_file_path = join(opt_folder, 'info_file.json')
    with open(info_file_path, "r") as f:
        params = json.loads(f.read())
    if predict_dL_with_PDRnet:
        if params["mode"] == 'Swing': # todo: this needs to be fixed ->  add info file to create segments and inherid mode in training
            pdr_path = '/data/Work/Navigation/SBGResults/PDR_net/IMU/swing/checkpoints/checkpoint_31.pt'
        if params["mode"] == 'text':
            pdr_path = '/data/Work/Navigation/SBGResults/PDR_net/IMU/text/checkpoints/checkpoint_30.pt'
        if params["mode"] == 'pocket':
            pdr_path = '/data/Work/Navigation/SBGResults/PDR_net/IMU/pocket/checkpoints/checkpoint_30.pt'
        pdr_net = torch.load(pdr_path, map_location=lambda storage, location: storage)
        pdr_net.to(device)
        pdr_net.eval()
    else:
        pdr_net = None
    if WD_est_method == 'compare_all_methods' or WD_est_method == 'resnet18LinAcc':
        data_folder = params["data_folder"]
        model_name = params["best_saved_model"]
        model_path = join(opt_folder, model_name)
        res18model = torch.load(model_path)
        res18model.to(device)
    else:
        data_folder = ''
        res18model = None

    _, test_dir = get_tain_test_dir(mode=params["mode"])
    exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                'AHRS_results' not in item]
    traj_folder = join(opt_folder, 'traj')
    if not os.path.exists(traj_folder):
        os.mkdir(traj_folder)
    # for running on a single example chosen randomly from list
    if exp_path is None and not calculate_on_test_list:
        exp_idx = random.randint(0, len(exp_list) - 1)
        exp_path = exp_list[exp_idx]
    traj_folder = join(opt_folder, 'traj')
    if not os.path.exists(traj_folder):
        os.mkdir(traj_folder)
    if calculate_on_test_list:
        load_results = False # if already performed and only want to plot graph
        if not load_results:
            # if sample_percentage is not None:
                # # only for debug running script on one person
                # num_of_idx_to_sample = round(len(test_persons) * sample_percentage / 100)
                # num_of_idx_to_sample = 1
                # idx = random.sample(range(len(test_persons)), num_of_idx_to_sample)
                # test_persons = list(map(test_persons.__getitem__, idx))
                # del idx
            test_on_list(exp_list=exp_list, params=params, traj_folder=traj_folder, res18model=res18model)
        else:
            # with open(join(traj_folder, 'PCA_tarj_lengths.pickle'), 'rb') as file:
            #     PCA_tarj_lengths = pickle.load(file)
            # with open(join(traj_folder, 'PCA_tarj_errors.pickle'), 'rb') as file:
            #     PCA_tarj_errors = pickle.load(file)
            # x_common_PCA, y_mean_PCA, y_std_PCA = calc_error_statistics(X=PCA_tarj_lengths, Y=PCA_tarj_errors, dx=0.1,
            #                                                             n_round=1, plot_results=False)
            x_common_PCA = sio.loadmat(join(traj_folder, 'PCA_L_vs_e.mat'))['x_common'].squeeze()
            y_mean_PCA = sio.loadmat(join(traj_folder, 'PCA_L_vs_e.mat'))['y_mean'].squeeze()
            y_std_PCA = sio.loadmat(join(traj_folder, 'PCA_L_vs_e.mat'))['y_std'].squeeze()
            y_min_PCA = sio.loadmat(join(traj_folder, 'PCA_L_vs_e.mat'))['y_min'].squeeze()
            y_max_PCA = sio.loadmat(join(traj_folder, 'PCA_L_vs_e.mat'))['y_max'].squeeze()
            x_common_SP_heading = sio.loadmat(join(traj_folder, 'SP_heading_L_vs_e.mat'))['x_common'].squeeze()
            y_mean_SP_heading = sio.loadmat(join(traj_folder, 'SP_heading_L_vs_e.mat'))['y_mean'].squeeze()
            y_std_SP_heading = sio.loadmat(join(traj_folder, 'SP_heading_L_vs_e.mat'))['y_std'].squeeze()
            y_min_SP_heading = sio.loadmat(join(traj_folder, 'SP_heading_L_vs_e.mat'))['y_min'].squeeze()
            y_max_SP_heading = sio.loadmat(join(traj_folder, 'SP_heading_L_vs_e.mat'))['y_max'].squeeze()
            x_common_resnet18 = sio.loadmat(join(traj_folder, 'resnet18_L_vs_e.mat'))['x_common'].squeeze()
            y_mean_resnet18 = sio.loadmat(join(traj_folder, 'resnet18_L_vs_e.mat'))['y_mean'].squeeze()
            y_std_resnet18 = sio.loadmat(join(traj_folder, 'resnet18_L_vs_e.mat'))['y_std'].squeeze()
            y_min_resnet18 = sio.loadmat(join(traj_folder, 'resnet18_L_vs_e.mat'))['y_min'].squeeze()
            y_max_resnet18 = sio.loadmat(join(traj_folder, 'resnet18_L_vs_e.mat'))['y_max'].squeeze()

            # y_mean_PCA = y_mean_PCA / x_common_PCA * 100
            # y_std_PCA = y_std_PCA / x_common_PCA * 100
            # y_min_PCA = y_min_PCA / x_common_PCA * 100
            # y_max_PCA = y_max_PCA / x_common_PCA * 100
            #
            # y_mean_SP_heading = y_mean_SP_heading / x_common_SP_heading * 100
            # y_std_SP_heading = y_std_SP_heading / x_common_SP_heading * 100
            # y_min_SP_heading = y_min_SP_heading / x_common_SP_heading * 100
            # y_max_SP_heading = y_max_SP_heading / x_common_SP_heading * 100
            #
            # y_mean_resnet18 = y_mean_resnet18 / x_common_resnet18 * 100
            # y_std_resnet18 = y_std_resnet18 / x_common_resnet18 * 100
            # y_min_resnet18 = y_min_resnet18 / x_common_resnet18 * 100
            # y_max_resnet18 = y_max_resnet18 / x_common_resnet18 * 100

            plt.figure('error_statistics_plot')
            plt.plot(x_common_PCA, y_mean_PCA, '-', color='gray', label='PCA')
            plt.fill_between(x_common_PCA, y_min_PCA, y_max_PCA, color='gray', alpha=0.2)
            plt.plot(x_common_SP_heading, y_mean_SP_heading, '-', color='green', label='SP heading')
            plt.fill_between(x_common_SP_heading, y_min_SP_heading, y_max_SP_heading ,
                             color='green', alpha=0.2)
            plt.plot(x_common_resnet18, y_mean_resnet18, '-', color='blue', label='resnet18')
            plt.fill_between(x_common_resnet18, y_min_resnet18, y_max_resnet18,
                             color='blue', alpha=0.2)
            plt.legend()
            plt.grid(True)
            plt.savefig(join(traj_folder, 'error_statistics.png'))
            plt.show()
    else:
        test_on_exp(params=params, WD_est_method = 'compare_all_methods',
                experiment_path=exp_path, GT=None, outputfolder=traj_folder, res18model=res18model, predict_dL_with_PDRnet=False,
                pdr_net=None, allign_traj=False, save_n_close_fig=False)
        plt.show()