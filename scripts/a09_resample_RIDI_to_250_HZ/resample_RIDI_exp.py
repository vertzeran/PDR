import os
import utils.Classes
from os.path import join
from utils.Classes import RidiExp_ENU
import numpy as np
import matplotlib.pyplot as plt
import utils.Functions as Functions
import ntpath
from scripts.a01_AHRS.calculate_AHRS_results_on_list import CalcResultsOnFile
from scipy import signal
from scripts.a08_training_on_RIDI_ENU.create_segments_for_WDE_RIDI_ENU import get_AHRS_results_for_exp
from scipy.spatial.transform import Rotation as Rotation


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


def resample_RIDI_exp(exp: utils.Classes.RidiExp_ENU, new_SF):
    t = exp.Time_IMU
    num_of_samples = int((t[-1] - t[0]) * new_SF)
    exp.NumberOfSamples_IMU = num_of_samples
    exp.NumberOfSamples_GT = num_of_samples
    new_t = np.linspace(t[0], t[-1], num=num_of_samples, endpoint=True)

    # Euler
    exp.Psi = np.interp(new_t, exp.Time_IMU, exp.Psi)
    exp.Theta = np.interp(new_t, exp.Time_IMU, exp.Theta)
    exp.Phi = np.interp(new_t, exp.Time_IMU, exp.Phi)
    euler_array = np.vstack([exp.Psi, exp.Theta, exp.Phi]).T
    exp.Rot = Rotation.from_euler('ZYX', euler_array, degrees=False)

    # IMU
    exp.Gyro.x = np.interp(new_t, exp.Time_IMU, exp.Gyro.x)
    exp.Gyro.y = np.interp(new_t, exp.Time_IMU, exp.Gyro.y)
    exp.Gyro.z = np.interp(new_t, exp.Time_IMU, exp.Gyro.z)

    exp.Acc.x = np.interp(new_t, exp.Time_IMU, exp.Acc.x)
    exp.Acc.y = np.interp(new_t, exp.Time_IMU, exp.Acc.y)
    exp.Acc.z = np.interp(new_t, exp.Time_IMU, exp.Acc.z)

    exp.Mag.x = np.interp(new_t, exp.Time_IMU, exp.Mag.x)
    exp.Mag.y = np.interp(new_t, exp.Time_IMU, exp.Mag.y)
    exp.Mag.z = np.interp(new_t, exp.Time_IMU, exp.Mag.z)

    # lin acc and grv
    exp.Grv.x = np.interp(new_t, exp.Time_IMU, exp.Grv.x)
    exp.Grv.y = np.interp(new_t, exp.Time_IMU, exp.Grv.y)
    exp.Grv.z = np.interp(new_t, exp.Time_IMU, exp.Grv.z)

    exp.LinAcc.x = np.interp(new_t, exp.Time_IMU, exp.LinAcc.x)
    exp.LinAcc.y = np.interp(new_t, exp.Time_IMU, exp.LinAcc.y)
    exp.LinAcc.z = np.interp(new_t, exp.Time_IMU, exp.LinAcc.z)

    # pos
    exp.Pos.x = np.interp(new_t, exp.Time_IMU, exp.Pos.x)
    exp.Pos.y = np.interp(new_t, exp.Time_IMU, exp.Pos.y)
    exp.Pos.z = np.interp(new_t, exp.Time_IMU, exp.Pos.z)

    # time vectors
    exp.Time_IMU = new_t
    exp.Time_GT = new_t
    exp.IMU_valid_idx = list(range(exp.NumberOfSamples_IMU))
    exp.dt = np.mean(np.diff(exp.Time_IMU))
    return exp


def compare_positions_of_two_experiments(exp1: utils.Classes.RidiExp_ENU, exp2: utils.Classes.RidiExp_ENU):
    fig = plt.figure('compare_positions_of_two_experiments')
    axes = []
    n_rows = 3
    n_cols = 1
    for i in range(n_rows * n_cols):
        if i == 0:
            axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
        else:
            axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
    ax_idx = 0
    axes[ax_idx].set(ylabel=r"$x [m]$", title="pso compare"), axes[ax_idx].grid(True)
    axes[ax_idx].scatter(exp1.Time_IMU, exp1.Pos.x, color='black', label='1')
    axes[ax_idx].scatter(exp2.Time_IMU, exp2.Pos.x, color='red', label='2')
    axes[ax_idx].legend()
    ax_idx = 1
    axes[ax_idx].set(ylabel=r"$y [m]$"), axes[ax_idx].grid(True)
    axes[ax_idx].scatter(exp1.Time_IMU, exp1.Pos.y, color='black', label='1')
    axes[ax_idx].scatter(exp2.Time_IMU, exp2.Pos.y, color='red', label='2')
    ax_idx = 2
    axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [m]$"), axes[ax_idx].grid(True)
    axes[ax_idx].scatter(exp1.Time_IMU, exp1.Pos.z, color='black', label='1')
    axes[ax_idx].scatter(exp2.Time_IMU, exp2.Pos.z, color='red', label='2')


if __name__ == '__main__':
    exp_path = r"/data/Datasets/Navigation/RIDI_dataset_train_test_ENU/RIDI - Text - Test/huayi2.csv"
    head, tail = ntpath.split(exp_path)
    root_dir = head
    file_name = tail
    exp = RidiExp_ENU(exp_path)
    exp_new = resample_RIDI_exp(exp=exp.clone(), new_SF=250)
    if False:
        compare_positions_of_two_experiments(exp, exp_new)
        plt.show()
    # save to csv
    new_path = join(os.getcwd(), 'test_resampling.csv')
    exp_new.save_csv(new_path)
    # exp.PlotSensors()
    calculate_AHRS = False
    if calculate_AHRS:
        CalcResultsOnFile(new_path, 'RIDI_ENU', override=True, GT=None)
    AHRS_results_files = exp_new.check_for_AHRS_results()
    if len(AHRS_results_files) == 0:
        print('no AHRS analysis on ' + exp.FileName)
    else:
        print('found AHRS results for ' + file_name)
        t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
            Functions.read_AHRS_results(join(exp_new.Path, AHRS_results_files[0]))
        fig = plt.figure('euler angles Plot')
        axes = []
        n_rows = 3
        n_cols = 2
        for i in range(n_rows * n_cols):
            if i == 0:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
            else:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
        ax_idx = 0
        axes[ax_idx].set(ylabel=r"$x [rad]$", title="euler angles"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(exp.Time_IMU, exp.Phi, color='black', linewidth=2, label='phi')
        axes[ax_idx].plot(t_est, phi_hat,  color='red', linestyle='dashed', label='phi hat')
        axes[ax_idx].legend()
        ax_idx = 1
        axes[ax_idx].grid(True), axes[ax_idx].set(title="euler angles errors")
        axes[ax_idx].plot(t_est, phi_e, color='black', linewidth=2, label='grv error ')

        ax_idx = 2
        axes[ax_idx].set(ylabel=r"$y [rad]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(exp.Time_IMU, exp.Theta, color='black', linewidth=2, label='theta')
        axes[ax_idx].plot(t_est, theta_hat, color='red', linestyle='dashed',
                          label='theta hat')
        ax_idx = 3
        axes[ax_idx].grid(True)
        axes[ax_idx].plot(t_est, theta_e, color='black', linewidth=2,
                          label='theta error')
        ax_idx = 4
        axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [rad]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(exp.Time_GT, exp.Psi, color='black', linewidth=2, label='psi')
        axes[ax_idx].plot(t_est, psi_hat, color='red', linestyle='dashed',
                          label='psi hat')
        ax_idx = 5
        axes[ax_idx].set(xlabel=r"$[sec]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(t_est, psi_e, color='black', linewidth=2,
                          label='phi error ')
        t_est, lin_acc_b_frame, grv_hat, Rot, Heading = get_AHRS_results_for_exp(exp_new)
        lin_acc_n_frame = Functions.transform_vectors(lin_acc_b_frame, Rot)
        pos_acc_correlation(exp_new.Time_IMU, exp_new.Pos.arr(), lin_acc_n_frame, filter_derivatives=True,
                            filter_freq=2 * 2 * np.pi, filter_order=4)
        plt.show()