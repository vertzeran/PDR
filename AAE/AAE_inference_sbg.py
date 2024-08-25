import utils.Classes as Classes
from utils.AAE import AtitudeEstimator
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import utils.Functions as Functions
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


def pos_acc_correlation(t, pos, lin_acc_n_frame, filter_derivatives=False, filter_freq = 1):
    """plot linear acceleration in navigation frame vs position 2nd derivative
    to verify correct coordinate frame handling"""
    # Position GT 2nd derivative:
    dt = np.diff(t).mean()
    pos_x = pos[:, 0]
    pos_y = pos[:, 1]
    pos_z = pos[:, 2]
    if filter_derivatives:
        fs = 1 / dt
        filter = signal.butter(N=3, Wn=[filter_freq], btype='low', analog=False, fs=fs, output='sos')
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
    Ax_pos_z.grid(True),Ax_pos_z.legend()


if __name__ == '__main__':
    # main_dir = '/home/maint/git/walking_direction_estimation/data/android rec' # '/data/Datasets/Navigation/SBG-PDR-DATA/swing'
    # main_dir_child = None #'22_07_26_swing_ariel_R'
    # if main_dir_child is None:
    #     main_dir_child = listdir(main_dir)[random.randint(0, len(listdir(main_dir)) - 1)]
    # exp_imu_file = None #'outdoor_output_2022-07-26_08_39_40.csv'
    # if exp_imu_file is None:
    #     list_of_files = glob.glob(join(main_dir, main_dir_child) + '/*.csv') # without the "ascii-output.txt"
    #     exp_path = list_of_files[random.randint(0, len(list_of_files) - 1)]
    # else:
    #     exp_path = join(main_dir, main_dir_child, exp_imu_file)
    exp_path = '/data/Datasets/Navigation/SBG-PDR-DATA/pocket/21_11_28_omri_dev/outdoor_output_2021-11-28_12_30_25.csv'
    # '/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_ran_dev/outdoor_output_2021-11-07_09_28_34.csv'
    # '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_nati_R/outdoor_output_2022-07-27_08_26_33.csv'
    print(exp_path)
    Exp = Classes.SbgExpRawData(exp_path)
    # Exp.PlotSensors()
    # Exp.SegmentScenario([0, 5])
    # Exp.PlotAngles()
    # plt.show()
    plot_heading = True
    plot_grv = False
    plot_acc_vs_pos_der = False
    plot_position = False
    calc_AHRS = plot_heading or plot_grv or plot_acc_vs_pos_der
    if plot_position:
        Exp.PlotPosition()
        fig = plt.figure('')
        Ax_pos_x = fig.add_subplot(311)
        Ax_pos_x.plot(Exp.Time_IMU, Exp.Pos.x)
        Ax_pos_x.grid(True)

        Ax_pos_y = fig.add_subplot(312, sharex=Ax_pos_x)
        Ax_pos_y.plot(Exp.Time_IMU, Exp.Pos.y)
        Ax_pos_y.grid(True)

        Ax_pos_z = fig.add_subplot(313, sharex=Ax_pos_x)
        Ax_pos_z.plot(Exp.Time_IMU, Exp.Pos.z)
        Ax_pos_z.grid(True)
    # Exp.SegmentScenario([0, 5])
    if calc_AHRS:
        AAE_AHRS = AtitudeEstimator(Ka=0.005, coor_sys_convention=Exp.Frame)
        grv_est, RotHat, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e = AAE_AHRS.run_exp(exp=Exp, visualize=True,
                                                                                           return_grv=True, return_euler=True,
                                                                                           save_results_to_file=False)
        acc = np.vstack([np.array(Exp.Acc.x), np.array(Exp.Acc.y), np.array(Exp.Acc.z)]).T
        Pn = np.vstack([np.array(Exp.Pos.x), np.array(Exp.Pos.y), np.array(Exp.Pos.z)]).T
        Pn = Pn - Pn[0, :]
        lin_acc_est_b_frame = acc - grv_est
        lin_acc_est_n_frame = Functions.transform_vectors(lin_acc_est_b_frame, RotHat)
    if plot_grv:
        fig = plt.figure('')
        Ax_pos_x = fig.add_subplot(311)
        Ax_pos_x.plot(grv_est[:, 0])
        Ax_pos_x.grid(True)

        Ax_pos_y = fig.add_subplot(312, sharex=Ax_pos_x)
        Ax_pos_y.plot(grv_est[:, 1])
        Ax_pos_y.grid(True)

        Ax_pos_z = fig.add_subplot(313, sharex=Ax_pos_x)
        Ax_pos_z.plot(grv_est[:, 2])
        Ax_pos_z.grid(True)
    if plot_heading:
        fig = plt.figure('heading')
        ax = fig.add_subplot(111)
        ang = psi_hat
        ax.plot(Exp.Time_IMU, ang, label='AE')
        ax.plot(Exp.Time_IMU, Exp.Psi, label='android')
        ax.plot(Exp.Time_IMU, -(Exp.SBG_yaw), label='SBG')
        ax.grid(True)
        ax.legend()
    # fig = plt.figure('lin acc n frame Plot')
    # axes = []
    # n_rows = 3
    # n_cols = 1
    # for i in range(n_rows * n_cols):
    #     if i == 0:
    #         axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
    #     else:
    #         axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
    # ax_idx = 0
    # axes[ax_idx].set(ylabel=r"$x [m/sec^2]$", title="linear acceleration"), axes[ax_idx].grid(True)
    # axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 0], color='red', linestyle='dashed', label='lin acc est')
    # axes[ax_idx].legend()
    # ax_idx = 1
    # axes[ax_idx].set(ylabel=r"$y [m/sec^2]$"), axes[ax_idx].grid(True)
    # axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 1], color='red', linestyle='dashed', label='lin acc est')
    # ax_idx = 2
    # axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [m/sec^2]$"), axes[ax_idx].grid(True)
    # axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 2], color='red', linestyle='dashed', label='lin acc est')
        # axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 2], color='red', linestyle='dashed', label='lin acc est')
    if plot_acc_vs_pos_der:
        pos_acc_correlation(t=Exp.Time_IMU, pos=Exp.Pos.arr(), lin_acc_n_frame=lin_acc_est_n_frame,
                            filter_derivatives=True, filter_freq=5)
    plt.show()