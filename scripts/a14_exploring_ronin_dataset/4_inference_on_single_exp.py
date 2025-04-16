import os
import utils.Classes
from os.path import join
from utils.Classes import RidiExp_ENU
import numpy as np
import matplotlib.pyplot as plt
import utils.Functions as Functions
from utils import Classes, AAE
import ntpath
from scripts.a01_AHRS.calculate_AHRS_results_on_list import CalcResultsOnFile
from scipy import signal
from scripts.a08_training_on_RIDI_ENU.create_segments_for_WDE_RIDI_ENU import get_AHRS_results_for_exp
from scipy.spatial.transform import Rotation as Rotation
from create_segments import get_segments_from_exp_no_gt, get_segments_from_exp
import torch
from torch import nn
from params import EXP_DIR, EXP_TIME_INTERVAL
import json
from tqdm import tqdm
from scipy.optimize import minimize


def calculate_traj_error(exp, est_time, est_pos, d_l, fix_heading=False):
    gt_time = exp.Time_GT[exp.index_of_walking_start:]
    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    if hasattr(exp, 'heading_fix') and exp.heading_fix != 0:
        gt_pos[:, :2] = Functions.rotate_trajectory(traj=gt_pos[:, :2], alfa=-exp.heading_fix)
    traj_length = np.array(d_l).cumsum()
    gt_pos_interp = np.vstack([np.interp(est_time, gt_time, gt_pos[:, 0]),
                               np.interp(est_time, gt_time, gt_pos[:, 1]),
                               np.interp(est_time, gt_time, gt_pos[:, 2])]).T
    gt_pos_interp_xy = gt_pos_interp[:, 0:2]  # only xy
    if fix_heading:
        est_pos, rot_angle = rotate_traj_to_minimize_error(est_pos, gt_pos_interp[:, 0:2])
    traj_errors = np.linalg.norm(est_pos - gt_pos_interp_xy, axis=1)
    normalized_errors = traj_errors[1:] / traj_length
    error_metric = normalized_errors.mean() * 100
    return traj_errors, traj_length, error_metric, est_pos, gt_pos_interp_xy


def rotate_traj_to_minimize_error(est_traj, gt_traj):
    '''
    calculate the heading angle fix to align est_traj to gt_traj.
    gt_traj should be interpulated to the times of the estimation.
    '''
    x0 = np.array(0.0)

    def minimization_function(alfa, traj_est, traj_gt):
        rot_traj = Functions.rotate_trajectory(traj_est, alfa[0])
        return Functions.traj_error(traj_est=rot_traj, traj_gt=traj_gt)

    res = minimize(minimization_function, x0, method='nelder-mead', args=(est_traj, gt_traj),
                   options={'xatol': 1e-8, 'disp': False})
    # traj_errors = np.linalg.norm(traj - gt_pos_interp[:, 0:2], axis=1)
    rot_traj = Functions.rotate_trajectory(est_traj, res.x[0])
    return rot_traj, res.x[0]


def check_for_valid_ahrs_results_file(exp: Classes.AhrsExp):
    suffix = '_AHRS_results.xlsx'
    ahrs_results_file_path = join(exp.Path, exp.FileName.split(sep='.')[0] + suffix)
    if not os.path.exists(ahrs_results_file_path):
        print("no AHRS results file found in data path ")
        return False
    print('found AHRS results file in data path: ' + ahrs_results_file_path)
    t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, rot_hat = \
        Functions.read_AHRS_results(ahrs_results_file_path)
    if len(list(exp.Time_IMU)) != len(list(t_est)):
        print("AHRS results file time vector invalid")
        return False
    if np.max(np.array(exp.Time_IMU) - np.array(t_est)) > 1e-3:
        print("AHRS results file time vector invalid")
        return False
    print('AHRS results file is valid')
    return True


class DlWdModel(nn.Module):
    def __init__(self, dl_model, wd_model):
        super().__init__()
        self.dl_model = dl_model
        self.wd_model = wd_model

    def forward(self, x):
        dl = self.dl_model(x)
        wd = self.wd_model(x)
        wd = nn.functional.normalize(wd, dim=1)
        y = dl * wd
        return y


if __name__ == '__main__':
    save_traj_csv = True
    run_on_a_single_exp = True
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')

    dl_model_path = join('PDR_models', 'V3__dl_regressor_epoch459_dl_loss_mse_loss_0.015_traced_model.pt')
    dl_model = torch.jit.load(dl_model_path, map_location='cpu')
    wd_model_path = join('PDR_models', 'V3__wd_regressor_epoch150_wd_loss_norm_loss_0.205_traced_model.pt')
    wd_model = torch.jit.load(wd_model_path, map_location='cpu')

    info_file_path = join('PDR_models', 'params_wd_model.json')
    with open(info_file_path, "r") as f:
        params = json.loads(f.read())
    model = DlWdModel(dl_model=dl_model, wd_model=wd_model)
    model.to(device)

    data_path = 'C:\\Users\\EranVertzberger\\PHD\\ronin_dataset\\dataset\\train_dataset_1\\a000_3'
    exp = Classes.RoninExp(data_path)
    exp.SegmentScenario([0, 150])
    print('resampling...')
    exp.resample(new_SF=250)
    exp.define_walking_start_idx(th=1, plot_res=True)
    if not check_for_valid_ahrs_results_file(exp):
        AAE_AHRS = AAE.AtitudeEstimator(Ka=0.005, coor_sys_convention=exp.Frame)
        print('performing AHRS analysis on: \n' + exp.Path)
        AAE_AHRS.run_exp(exp=exp, return_grv=True, return_euler=True,
                                             save_results_to_file=True, visualize=True)
    segment_list, ahrs_results_list = get_segments_from_exp_no_gt(
        exp=exp, window_size=200, use_gt_att=False,
        win_size_for_heading_init=1000, AHRS_results_path=data_path)
    print("num segments = " + str(len(segment_list)))
    print("num AHRS segments = " + str(len(ahrs_results_list)))
    convert_quat_to_rot6d = params["convert_quat_to_rot6d"]
    window_size = params["window_size"]
    analyzied_segments = []
    dx_dy = []
    d_l = []
    walking_angle = []
    est_time = [exp.Time_IMU[exp.index_of_walking_start - 1]]
    for i in tqdm(range(len(segment_list)), desc='PDRNet infer segments'):
        segment = segment_list[i]
        est_time.append(segment.Time_IMU[-1])
        dp = segment.Pos.arr()[-1] - segment.Pos.arr()[0]
        d_l.append(np.linalg.norm(dp[0:2]))
        # AHRS results segmentation
        ahrs_results = ahrs_results_list[i]
        lin_acc_b_frame_est = ahrs_results.lin_acc_b_frame
        grv_est = ahrs_results.grv
        Heading_est = ahrs_results.heading
        Rot = ahrs_results.Rot

        t = segment.Time_IMU.reshape(-1, 1)
        lin_acc_n_frame = Functions.transform_vectors(lin_acc_b_frame_est, Rot)  # window_sizeX3

        if convert_quat_to_rot6d:
            m = Rot.as_matrix()
            r6d = Functions.rotation_matrix_to_r6d(m)[:window_size, :]
            x = np.hstack([lin_acc_n_frame, r6d]).reshape(1, window_size, 9)
        else:
            quat_array = Rot.as_quat()
            batch_of_quat = quat_array[:window_size, :]
            x = np.hstack([lin_acc_n_frame, batch_of_quat]).reshape(1, window_size, 7)
        x_tensor = Functions.PrepareInputForResnet18(x, add_dim=False)
        dx_dy.append(model(x_tensor).detach().numpy())

    est_traj = Functions.construct_traj(dx_dy=np.vstack(dx_dy), method='dx_dy', plot_result=False, pos_gt=None)
    traj_errors, traj_length, error_metric, est_traj, gt_pos_interp_xy = calculate_traj_error(
        exp, np.array(est_time), est_traj, d_l, fix_heading=True)

    data_to_save = {"x": est_traj[:, 0], "y": est_traj[:, 1]}

    # Functions.save_csv(data_to_save, path=join(EXP_DIR, 'PDR_traj.csv'), print_message=True)//
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
    if exp.Pos.x is not None:
        Pos_GT_XY = exp.Pos.arr()[exp.index_of_walking_start:, 0:2] - exp.Pos.arr()[exp.index_of_walking_start, 0:2]
        Pos_GT_X = Pos_GT_XY[:, 0]
        Pos_GT_Y = Pos_GT_XY[:, 1]
        ax.plot(Pos_GT_X, Pos_GT_Y, color='black', linestyle='-', linewidth=1, label='GT')
        ax.plot(Pos_GT_X[0], Pos_GT_Y[0], 'ok', label='GTStart')
    ax.plot(est_traj[:, 0], est_traj[:, 1], color='red', linestyle='-', linewidth=1, label='predicted')
    ax.plot(est_traj[0, 0], est_traj[0, 1], 'or', label='predictedStart')
    # ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
    ax.legend()
    ax.axis('equal')
    # plt.savefig(join(EXP_DIR, 'PDR.png'))
    plt.show()
    plt.close()
