from os.path import join
from utils.Classes import RidiExp_ENU
import numpy as np
import matplotlib.pyplot as plt
import utils.Functions as Functions
import ntpath
from scripts.a01_AHRS.calculate_AHRS_results_on_list import CalcResultsOnFile
from scipy import signal
from scripts.a08_training_on_RIDI_ENU.create_segments_for_WDE_RIDI_ENU import get_AHRS_results_for_exp

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
    exp_path = r"/data/Datasets/Navigation/RIDI_dataset_train_test_ENU/RIDI - Text - Test/huayi2.csv"
    head, tail = ntpath.split(exp_path)
    root_dir = head
    file_name = tail
    exp = RidiExp_ENU(exp_path)
    # exp.PlotSensors()
    calculate_AHRS = False
    if calculate_AHRS:
        CalcResultsOnFile(exp_path, 'RIDI_ENU', override=True, GT=None)
    AHRS_results_files = exp.check_for_AHRS_results()
    if len(AHRS_results_files) == 0:
        print('no AHRS analysis on ' + exp.FileName)
    else:
        print('found AHRS results for ' + file_name)
        t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
            Functions.read_AHRS_results(join(root_dir, AHRS_results_files[0]))
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
        axes[ax_idx].plot(exp.Time_GT, psi_e, color='black', linewidth=2,
                          label='phi error ')
        t_est, lin_acc_b_frame, grv_hat, Rot, Heading = get_AHRS_results_for_exp(exp)
        lin_acc_n_frame = Functions.transform_vectors(lin_acc_b_frame, Rot)
        pos_acc_correlation(exp.Time_IMU, exp.Pos.arr(), lin_acc_n_frame, filter_derivatives=True,
                            filter_freq=2 * 2 * np.pi, filter_order=4)

        plt.show()