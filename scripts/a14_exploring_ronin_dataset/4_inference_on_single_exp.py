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
        raise NotImplementedError()
        est_pos, rot_angle = rotate_traj_to_minimize_error(est_pos, gt_pos_interp[:, 0:2])
    traj_errors = np.linalg.norm(est_pos - gt_pos_interp_xy, axis=1)
    normalized_errors = traj_errors[1:] / traj_length
    error_metric = normalized_errors.mean() * 100
    return traj_errors, traj_length, error_metric, est_pos, gt_pos_interp_xy


def traj_est_using_pdrnet(exp, params, plot_result=False, use_gt_att=False, pdrnet_model=None, device='cpu',
                          model_path=None, align_traj=False, outputfolder=None):
    if use_gt_att:
        segment_list, initial_wd_angles_list = get_segments_from_exp(
            exp=exp, window_size=params['window_size'], use_gt_att=use_gt_att,
            win_size_for_heading_init=params['TRC_params']['win_size_for_heading_init'])
    else:
        segment_list, ahrs_results_list, initial_wd_angles_list = get_segments_from_exp(
            exp=exp, window_size=params['window_size'], use_gt_att=use_gt_att,
            win_size_for_heading_init=params['TRC_params']['win_size_for_heading_init'])

    analyzied_segments = []
    dx_dy = []
    d_l = []
    walking_angle = []
    est_time = [exp.Time_IMU[exp.index_of_walking_start - 1]]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        est_time.append(segment.Time_GT[-1])
        if not use_gt_att:
            # AHRS results segmentation
            ahrs_results = ahrs_results_list[i]
            analyzied_segment = Classes.WDE_performance_analysis(
                segment, use_GT_att=use_gt_att, lin_acc_b_frame_est=ahrs_results.lin_acc_b_frame,
                Rot_est=ahrs_results.Rot, grv_est=ahrs_results.grv, Heading_est=ahrs_results.heading,
                use_GT_dl=True, dl_net=None, arc_length_for_dl=False
            )
        else:
            analyzied_segment = Classes.WDE_performance_analysis(
                segment, use_GT_att=use_gt_att, use_GT_dl=False, dl_net=None, arc_length_for_dl=False)

        analyzied_segment.PDR_net_res18model = pdrnet_model
        analyzied_segment.window_size = params["window_size"]
        analyzied_segment.WDE_model_path = model_path
        analyzied_segment.PDR_net_V2_pred(
            plot_results=False, data_type='LinAcc', device=device, add_quat=params["add_quat"],
            add_dim=not params["resnet1d"], convert_quat_to_rot6d=params["convert_quat_to_rot6d"])

        dx_dy.append(analyzied_segment.end_pos_est)
        d_l.append(analyzied_segment.dL)
        walking_angle.append(analyzied_segment.WD_angle_est)
        analyzied_segments.append(analyzied_segment)

    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    if hasattr(exp, 'heading_fix') and exp.heading_fix != 0:
        gt_pos[:, :2] = Functions.rotate_trajectory(traj=gt_pos[:, :2], alfa=-exp.heading_fix)

    traj = Functions.construct_traj(dx_dy=np.array(dx_dy), method='dx_dy', plot_result=False, pos_gt=gt_pos)
    traj_errors, traj_length, error_metric, rot_traj, gt_pos_interp_xy = calculate_traj_error(
        exp, np.array(est_time), traj, d_l, fix_heading=align_traj)

    if plot_result:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='black', linestyle='--', linewidth=2, label='gt')
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='predicted')
        # ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
        ax.legend()
        ax.axis('equal')
        if outputfolder is not None:
            fig_folder = join(outputfolder, os.path.basename(exp.Path))
            os.makedirs(fig_folder, exist_ok=True)
            plt.savefig(join(fig_folder, f'{os.path.splitext(exp.FileName)[0]}.png'))
            plt.close()

    return rot_traj, dx_dy, traj_errors, traj_length, error_metric, gt_pos_interp_xy


def traj_est_using_pdrnet_no_gt(exp, params, plot_result=False, use_gt_att=False, pdrnet_model=None, device='cpu',
                                model_path=None, align_traj=False, outputfolder=None):

    segment_list, ahrs_results_list, initial_wd_angles_list = get_segments_from_exp(
        exp=exp, window_size=params['window_size'], use_gt_att=use_gt_att,
        win_size_for_heading_init=params['TRC_params']['win_size_for_heading_init'])

    analyzied_segments = []
    dx_dy = []
    d_l = []
    walking_angle = []
    est_time = [exp.Time_IMU[exp.index_of_walking_start - 1]]
    for i in range(len(segment_list)):
        segment = segment_list[i]
        est_time.append(segment.Time_GT[-1])
        # AHRS results segmentation
        ahrs_results = ahrs_results_list[i]
        lin_acc_b_frame_est = ahrs_results.lin_acc_b_frame
        grv_est = ahrs_results.grv
        Heading_est = ahrs_results.heading
        PDR_net_res18model = pdrnet_model
        window_size = params["window_size"]
        WDE_model_path = model_path


        analyzied_segment = Classes.WDE_performance_analysis(
            segment, use_GT_att=use_gt_att, lin_acc_b_frame_est=ahrs_results.lin_acc_b_frame,
            Rot_est=ahrs_results.Rot, grv_est=ahrs_results.grv, Heading_est=ahrs_results.heading,
            use_GT_dl=True, dl_net=None, arc_length_for_dl=False
        )

        analyzied_segment.PDR_net_res18model = pdrnet_model
        analyzied_segment.window_size = params["window_size"]
        analyzied_segment.WDE_model_path = model_path
        analyzied_segment.PDR_net_V2_pred(
            plot_results=False, data_type='LinAcc', device=device, add_quat=params["add_quat"],
            add_dim=not params["resnet1d"], convert_quat_to_rot6d=params["convert_quat_to_rot6d"])

        dx_dy.append(analyzied_segment.end_pos_est)
        d_l.append(analyzied_segment.dL)
        walking_angle.append(analyzied_segment.WD_angle_est)
        analyzied_segments.append(analyzied_segment)

    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    if hasattr(exp, 'heading_fix') and exp.heading_fix != 0:
        gt_pos[:, :2] = Functions.rotate_trajectory(traj=gt_pos[:, :2], alfa=-exp.heading_fix)

    traj = Functions.construct_traj(dx_dy=np.array(dx_dy), method='dx_dy', plot_result=False, pos_gt=gt_pos)
    traj_errors, traj_length, error_metric, rot_traj, gt_pos_interp_xy = calculate_traj_error(
        exp, np.array(est_time), traj, d_l, fix_heading=align_traj)

    if plot_result:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], color='black', linestyle='--', linewidth=2, label='gt')
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='predicted')
        # ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
        ax.legend()
        ax.axis('equal')
        if outputfolder is not None:
            fig_folder = join(outputfolder, os.path.basename(exp.Path))
            os.makedirs(fig_folder, exist_ok=True)
            plt.savefig(join(fig_folder, f'{os.path.splitext(exp.FileName)[0]}.png'))
            plt.close()

    return rot_traj, dx_dy, traj_errors, traj_length, error_metric, gt_pos_interp_xy


def test_on_exp(params, experiment_path=None, gt=None, pdrnet_model=None, device=None, outputfolder=None):
    use_gt_att = False

    exp = Classes.SbgExpRawData(path=experiment_path, GT=gt)
    if params["TRC_params"]["traj_length_limit"] is not None:
        exp.limit_traj_length(limit=params["TRC_params"]["traj_length_limit"])
    exp.define_walking_start_idx(th=params["TRC_params"]["walking_start_threshold"])
    if params["TRC_params"].get("heading_fix", False):
        file_path = os.sep.join(join(exp.Path, exp.FileName).split(os.sep)[-2:])
        row = trc_heading_fix_table[trc_heading_fix_table['file_path'] == file_path]
        # r=rotation, rs=rotation&scaling
        # using (-) since angle computed for GT rotation
        exp.heading_fix = -float(row['angle_rs'])

    device = torch.device(device if torch.cuda.is_available() and (device is not None) else 'cpu')
    traj, walking_angle, traj_errors, traj_length, error_metric, gt_pos_interp_xy = traj_est_using_pdrnet(
        exp, params, plot_result=True, use_gt_att=use_gt_att, device=device, pdrnet_model=pdrnet_model,
        outputfolder=outputfolder)

    errors_dict = {'traj_errors': traj_errors, 'traj_length': traj_length, 'error_metric': error_metric}
    traj_dict = {'gt': gt_pos_interp_xy, 'est': traj}

    return errors_dict, traj_dict


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

    data_path = 'C:\\Users\\EranVertzberger\\PHD\\ronin_dataset\\dataset\\train_dataset_1\\a000_10'
    exp = Classes.RoninExp(data_path)
    exp.SegmentScenario([0, 500])
    exp_resampled = exp.clone()
    print('resampling...')
    exp_resampled.resample(new_SF=250)

    segment_list, ahrs_results_list = get_segments_from_exp_no_gt(
        exp=exp_resampled, window_size=200, use_gt_att=False,
        win_size_for_heading_init=1000, AHRS_results_path=data_path)
    print("num segments = " + str(len(segment_list)))
    print("num AHRS segments = " + str(len(ahrs_results_list)))
    convert_quat_to_rot6d = params["convert_quat_to_rot6d"]
    window_size = params["window_size"]
    analyzied_segments = []
    dx_dy = []
    d_l = []
    walking_angle = []
    est_time = []
    for i in tqdm(range(len(segment_list)), desc='PDRNet infer segments'):
        segment = segment_list[i]
        est_time.append(segment.Time_IMU[-1])
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

    traj = Functions.construct_traj(dx_dy=np.vstack(dx_dy), method='dx_dy', plot_result=False, pos_gt=None)
    data_to_save = {"x": traj[:, 0], "y": traj[:, 1]}

    # Functions.save_csv(data_to_save, path=join(EXP_DIR, 'PDR_traj.csv'), print_message=True)//
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
    if exp.Pos.x is not None:
        ax.plot(exp.Pos.x, exp.Pos.y, color='black', linestyle='-', linewidth=1, label='GT')
        ax.plot(exp.Pos.x[0], exp.Pos.y[0], 'ok', label='GTStart')
    ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=1, label='predicted')
    ax.plot(traj[0, 0], traj[0, 1], 'or', label='predictedStart')
    # ax.plot(rot_traj[:, 0], rot_traj[:, 1], color='gray', linestyle='-', linewidth=2, label='rotated traj')
    ax.legend()
    ax.axis('equal')
    # plt.savefig(join(EXP_DIR, 'PDR.png'))
    plt.show()
    plt.close()
