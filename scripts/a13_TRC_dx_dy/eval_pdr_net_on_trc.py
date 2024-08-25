from os.path import join
import json
import torch
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from scripts.a13_TRC_dx_dy.create_segments import get_train_and_test_lists, get_dir_to_analyze, get_exp_list, \
    get_segments_from_exp
from utils import Functions, Classes
from utils.Functions import construct_traj


def calculate_traj_error(exp, est_time, est_pos, d_l, fix_heading=False):
    gt_time = exp.Time_GT[exp.index_of_walking_start:]
    gt_pos = exp.Pos.arr()[exp.index_of_walking_start:] - exp.Pos.arr()[exp.index_of_walking_start, :]
    traj_length = np.array(d_l).cumsum()
    gt_pos_interp = np.vstack([np.interp(est_time, gt_time, gt_pos[:, 0]),
                               np.interp(est_time, gt_time, gt_pos[:, 1]),
                               np.interp(est_time, gt_time, gt_pos[:, 2])]).T
    if fix_heading:
        raise NotImplementedError()
        est_pos, rot_angle = rotate_traj_to_minimize_error(est_pos, gt_pos_interp[:, 0:2])
    traj_errors = np.linalg.norm(est_pos - gt_pos_interp[:, 0:2], axis=1)
    normalized_errors = traj_errors[1:] / traj_length
    error_metric = normalized_errors.mean() * 100
    return traj_errors, traj_length, error_metric, est_pos


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
    traj = construct_traj(dx_dy=np.array(dx_dy), method='dx_dy', plot_result=False, pos_gt=gt_pos)
    traj_errors, traj_length, error_metric, rot_traj = calculate_traj_error(
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

    return rot_traj, dx_dy, traj_errors, traj_length, error_metric


def test_on_exp(params, experiment_path=None, gt=None, pdrnet_model=None, device=None, outputfolder=None):
    use_gt_att = False

    exp = Classes.SbgExpRawData(path=experiment_path, GT=gt)
    if params["TRC_params"]["traj_length_limit"] is not None:
        exp.limit_traj_length(limit=params["TRC_params"]["traj_length_limit"])
    exp.define_walking_start_idx(th=params["TRC_params"]["walking_start_threshold"])

    device = torch.device(device if torch.cuda.is_available() and (device is not None) else 'cpu')
    traj, walking_angle, traj_errors, traj_length, error_metric = traj_est_using_pdrnet(
        exp, params, plot_result=True, use_gt_att=use_gt_att, device=device, pdrnet_model=pdrnet_model,
        outputfolder=outputfolder)

    errors_dict = {'traj_errors': traj_errors, 'traj_length': traj_length, 'error_metric': error_metric}

    return errors_dict


def main():
    data_location = 'wolverine'
    device = torch.device(1 if torch.cuda.is_available() else 'cpu')
    model_path = '/home/adam/git/walking_direction_estimation/data/optimization_results/2023-06-01T13:39:48.056090_resnet18-rot6d-norm_loss_with_length_mse/dx_dy_regressor_epoch239_loss_norm_loss_with_length_mse_0.345.pth'
    optimization_folder = os.path.split(model_path)[0]

    info_file_path = join(optimization_folder, 'info_file.json')
    with open(info_file_path, "r") as f:
        params = json.loads(f.read())

    model = torch.load(model_path, map_location='cpu')
    model.to(device)

    traj_folder = join(optimization_folder, 'traj_trc')
    os.makedirs(traj_folder, exist_ok=True)

    pdrnet_traj_lengths = []
    pdrnet_traj_errors = []
    pdrnet_errors = []

    file_names = []
    # TRC
    mode = params["TRC_params"]["mode"]
    dir_to_analyze = get_dir_to_analyze(data_location, mode)  # get main dir
    _, test_persons_list = get_train_and_test_lists(mode)
    for i_person, person in enumerate(test_persons_list):
        person_path = join(dir_to_analyze, person)
        gt_path = join(person_path, 'ascii-output.txt')
        gt = pd.read_csv(gt_path, sep='\t', skiprows=28)
        exp_list_in_person = get_exp_list(list_of_persons=[person], dir_to_analyze=dir_to_analyze)

        for exp_path in tqdm(exp_list_in_person, desc=f'{person} ({i_person}/{len(test_persons_list)})'):
            errors_dict = test_on_exp(params, experiment_path=exp_path, gt=gt, pdrnet_model=model, device=device,
                                      outputfolder=traj_folder)

            pdrnet_traj_lengths.extend(errors_dict['traj_length'])
            pdrnet_traj_errors.extend(errors_dict['traj_errors'])
            pdrnet_errors.append(errors_dict['error_metric'])
            file_names.append(exp_path)

    data_to_save = {
        "file_names": file_names,
        "pdrnet_errors": pdrnet_errors,
    }

    file_path = join(traj_folder, 'performance_on_test_scenarios.csv')
    Functions.save_csv(path=file_path, dic=data_to_save, print_message=True)

    data_to_save = {'pdrnet_mean_errors': np.array(pdrnet_errors).mean()}
    with open(join(traj_folder, 'performance_summery.json'), "w") as outfile:
        json.dump(data_to_save, outfile, indent=4)


if __name__ == '__main__':
    main()
