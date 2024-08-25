import argparse
from os.path import join
from utils.Classes import SbgExpRawData
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import utils.Functions as Functions
from scipy.spatial.transform import Rotation
from scipy import signal


def transform_b2u(acc_b, Rnb):
    """
    transform linear accelerations to the u-frame
    :param acc_b: linear accelerations in the b frame  n_samples X 3
    :param Rnb: rotation object with n_samples instances
    :return: acc_u: linear accelerations in the u frame  n_samples X 3
    :return: Rub: rotation object with n_samples instances
    """
    assert len(acc_b.shape) == 2
    assert acc_b.shape[1] == 3
    assert acc_b.shape[0] == len(Rnb)
    n_samples = len(Rnb)
    Rub = []
    acc_u = np.zeros(acc_b.shape)
    for i in range(n_samples):
        Rnb_i = Rnb[i].as_matrix()
        # u-frame z axis projected to the b-frame
        zu_b = Rnb_i.T.dot(np.array([0, 0, 1]))
        # b-frame x and y axes projected to the n-frame
        xb_n = Rnb_i.dot(np.array([1, 0, 0]).T)
        # yb_n = Rnb_i.dot(np.array([0, 1, 0]).T)
        # u-frame x and y axes projected to the n-frame (zeroing z component and normalizing)
        xu_n_tag = xb_n.reshape([1,3]).dot(np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 0]]))
        xu_n = xu_n_tag / np.linalg.norm(xu_n_tag)
        # yu_n_tag = yb_n.reshape([1, 3]).dot(np.array([[1, 0, 0],
        #                                               [0, 1, 0],
        #                                               [0, 0, 0]]))
        # yu_n = yu_n_tag / np.linalg.norm(yu_n_tag)
        # yu_n_tag = np.cross(xu_n, np.array([0, 0, 1]))
        # yu_n = yu_n_tag/np.linalg.norm(yu_n_tag)
        # Transform to b-frame
        xu_b = Rnb_i.T.dot(xu_n.reshape([3,1])).squeeze()
        # yu_b = Rnb_i.T.dot(yu_n.reshape([3,1]))
        yu_b_tag = np.cross(xu_b, zu_b)
        yu_b = yu_b_tag / np.linalg.norm(yu_b_tag)
        xu_b = xu_b.reshape([3, 1])
        yu_b = yu_b.reshape([3, 1])
        zu_b = zu_b.reshape([3, 1])
        Rub_i = np.hstack([xu_b, yu_b, zu_b])
        Rub_i = Functions.OrthonormalizeRotationMatrix(Rub_i) # doto: check if needed
        acc_u[i, :] = Rub_i.dot(acc_b[i, :])
        Rub.append(Rotation.from_matrix(Rub_i))
    return acc_u, Rub


def pos_acc_correlation(t, pos, lin_acc_n_frame, filter_derivatives=False, filter_freq = 1, filter_order = 4):
    """plot linear acceleration in navigation frame vs position 2nd derivative
    to verify correct coordinate frame handling"""
    # Position GT 2nd derivative:
    dt = np.diff(t).mean()
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    pos_z = pos[:, 2]
    if filter_derivatives:
        fs = 1 / dt
        filter = signal.butter(N=filter_order, Wn=[filter_freq], btype='low', analog=False, fs=fs, output='sos')
        pos_x = signal.sosfiltfilt(filter, pos_x)
        pos_y = signal.sosfiltfilt(filter, pos_y)
        pos_z = signal.sosfiltfilt(filter, pos_z)
    pos_x_der = np.diff(pos_x, axis=0) / dt
    pos_y_der = np.diff(pos_y, axis=0) / dt
    pos_z_der = np.diff(pos_z, axis=0) / dt
    pos_x_2nd_der = np.diff(pos_x_der, axis=0) / dt
    pos_y_2nd_der = np.diff(pos_y_der, axis=0) / dt
    pos_z_2nd_der = np.diff(pos_z_der, axis=0) / dt

    # fig = plt.figure('der')
    # ax1 = fig.add_subplot(311)
    # ax1.plot(pos_x), ax1.grid(True)
    # ax2 = fig.add_subplot(312, sharex=ax1)
    # ax2.plot(pos_x_der), ax2.grid(True)
    # ax3 = fig.add_subplot(313, sharex=ax1)
    # ax3.plot(pos_x_2nd_der), ax3.grid(True)
    # Plots
    fig = plt.figure('Position Acc correlation')
    Ax_pos_x = fig.add_subplot(311)
    Ax_pos_x.plot(t[0: -2], pos_x_2nd_der, color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
    Ax_pos_x.plot(t, lin_acc_n_frame[:, 0], color='red', linewidth=1, label = 'Accelerometer')
    # Ax_pos_x.set_ylim(lin_acc_n_frame[:, 0].min(), lin_acc_n_frame[:, 0].max()),
    Ax_pos_x.set(title="Position 2nd derivative VS $a^n_L$", ylabel="$[m/sec^2]$")
    Ax_pos_x.grid(True),Ax_pos_x.legend()

    Ax_pos_y = fig.add_subplot(312, sharex=Ax_pos_x)
    Ax_pos_y.plot(t[0: -2], pos_y_2nd_der, color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
    Ax_pos_y.plot(t, lin_acc_n_frame[:, 1], color='red', linewidth=1, label = 'Accelerometer')
    # Ax_pos_y.set_ylim(lin_acc_n_frame[:, 1].min(), lin_acc_n_frame[:, 1].max()),
    Ax_pos_y.set(ylabel="$[m/sec^2]$")
    Ax_pos_y.grid(True),Ax_pos_y.legend()

    Ax_pos_z = fig.add_subplot(313, sharex=Ax_pos_x)
    Ax_pos_z.plot(t[0: -2], pos_z_2nd_der, color='blue', linewidth=1, label = 'Pos GT 2nd derivative')
    Ax_pos_z.plot(t, lin_acc_n_frame[:, 2], color='red', linewidth=1, label = 'Accelerometer')
    # Ax_pos_z.set_ylim(lin_acc_n_frame[:, 2].min(), lin_acc_n_frame[:, 2].max()),
    Ax_pos_z.set(xlabel="Time [sec]", ylabel="$[m/sec^2]$")
    Ax_pos_z.grid(True), Ax_pos_z.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_list', type=str, default=None, help='if not given work on the whole directory')
    parser.add_argument('--root_dir', type=str, default='/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_firas',
                        help='Path to data directory')
    args = parser.parse_args()
    plot_acc_vs_pos_der = True

    if args.exp_list is not None:
        with open(args.exp_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
        suffix = '.csv'
    else:
        data_list = listdir(args.root_dir)
        suffix = ''
    i = 1
    for file_name in data_list:
        file_path = join(args.root_dir, file_name + suffix)
        if '_AHRS_results.xlsx' not in file_path and 'ascii-output.txt' not in file_path:
            exp = SbgExpRawData(file_path)
            exp.define_walking_start_idx(th=1)
            exp.SegmentScenario([exp.Time_IMU[exp.index_of_walking_start], exp.Time_IMU[-1]])
            exp.SBG_yaw = exp.SBG_yaw[exp.index_of_walking_start:]
            AHRS_results_files = exp.check_for_AHRS_results()
            if len(AHRS_results_files) == 0:
                print('no AHRS analysis on ' + exp.FileName)
            else:
                print('found AHRS results for ' + file_name)
                t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
                    Functions.read_AHRS_results(join(args.root_dir, AHRS_results_files[0]))
                fig = plt.figure('Euler angles')
                axes = []
                n_rows = 3
                n_cols = 1
                for i in range(n_rows * n_cols):
                    if i == 0:
                        axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
                    else:
                        axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
                ax_idx = 0
                axes[ax_idx].set(ylabel=r"$\phi [deg]$", title=""), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Phi * 180 / np.pi, color='black', linewidth=2, label='android')
                axes[ax_idx].plot(t_est, phi_hat * 180 / np.pi, color='red', linestyle='dashed',
                                  label='AE')
                axes[ax_idx].legend()
                ax_idx = 1
                axes[ax_idx].set(ylabel=r"$\theta [deg]$"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Theta * 180 / np.pi, color='black', linewidth=2, label='android')
                axes[ax_idx].plot(t_est, theta_hat * 180 / np.pi, color='red', linestyle='dashed',
                                  label='AE')
                ax_idx = 2
                exp.Psi = Functions.ContinuousAngle(Functions.FoldAngles(exp.Psi))* 180 / np.pi
                psi_hat = Functions.ContinuousAngle(Functions.FoldAngles(psi_hat))* 180 / np.pi
                walking_angle, _ = exp.calc_walking_direction()
                walking_angle = Functions.ContinuousAngle(Functions.FoldAngles(walking_angle))
                axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$\psi [deg]$"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Psi, color='black', linewidth=2, label='android')
                axes[ax_idx].plot(t_est, psi_hat, color='red', linestyle='dashed',
                                  label='AE')
                exp.SBG_yaw = Functions.ContinuousAngle(Functions.FoldAngles(exp.SBG_yaw)) * 180 / np.pi
                axes[ax_idx].plot(exp.Time_GT, -exp.SBG_yaw, color='green', linestyle='dashed',
                                  label='SBG')
                axes[ax_idx].plot(exp.Time_GT[1:], walking_angle * 180 / np.pi, color='blue', linestyle='dotted',
                                  label=r'atan(dP)')
                axes[ax_idx].legend()
                if plot_acc_vs_pos_der:
                    t_start = exp.Time_IMU[0].round(11)
                    t_end = exp.Time_IMU[-1].round(11)
                    ind_IMU = np.where((t_est.round(11) >= t_start) & (t_est.round(11) <= t_end))
                    lin_acc_est_b_frame = exp.Acc.arr() - grv_hat[ind_IMU]

                    lin_acc_est_n_frame = Functions.transform_vectors(lin_acc_est_b_frame, Rot_hat[ind_IMU])
                    pos_acc_correlation(t=exp.Time_IMU, pos=exp.Pos.arr(), lin_acc_n_frame=lin_acc_est_n_frame,
                                        filter_derivatives=True, filter_freq=10, filter_order=1)
                plt.show()
