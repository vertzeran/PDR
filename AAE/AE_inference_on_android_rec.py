import utils.Classes as Classes
from utils.AAE import AtitudeEstimator
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import utils.Functions as Functions
from os.path import join


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


if __name__ == '__main__':
    main_dir = '/home/maint/git/walking_direction_estimation/data/android rec' # '/data/Datasets/Navigation/SBG-PDR-DATA/swing'
    # main_dir_child = None #'22_07_26_swing_ariel_R'
    # if main_dir_child is None:
    #     main_dir_child = listdir(main_dir)[random.randint(0, len(listdir(main_dir)) - 1)]
    # exp_imu_file = None #'outdoor_output_2022-07-26_08_39_40.csv'
    # if exp_imu_file is None:
    #     list_of_files = glob.glob(join(main_dir, main_dir_child) + '/*.csv') # without the "ascii-output.txt"
    #     exp_path = list_of_files[random.randint(0, len(list_of_files) - 1)]
    # else:
    #     exp_path = join(main_dir, main_dir_child, exp_imu_file)
    exp_path = join(main_dir, 'outdoor_output_2022-08-01_18_09_33_Eran_swing.csv')
    # 'outdoor_output_2021-10-31_10_12_22_Eran_texting.csv'
    # 'indoor_output_2022-01-04_18_32_14_yaw_movement.csv'
    # 'outdoor_output_2021-11-28_10_13_18_Eran_pocket.csv'
    # 'outdoor_output_2022-08-01_18_09_33_Eran_swing.csv'

    print(exp_path)
    Exp = Classes.AndroidExp(exp_path)
    Exp.PlotSensors()
    # Exp.SegmentScenario([0, 50])
    android_psi = Exp.Psi
    android_psi = Functions.ContinuousAngle(android_psi)
    android_theta = Exp.Theta
    android_phi = Exp.Phi

    android_GRV_quat = Exp.QuatArray
    EulerArray = Rotation.from_quat(android_GRV_quat).as_euler("ZYX", degrees=False)
    android_grv_psi = EulerArray[:, 0]
    android_grv_psi = android_grv_psi - android_grv_psi[0]
    android_grv_psi = Functions.ContinuousAngle(android_grv_psi)
    android_grv_theta = EulerArray[:, 1]
    android_grv_phi = EulerArray[:, 2]
    # Psi = Functions.ContinuousAngle(Psi)

    # apply attitude estimator:
    AAE_AHRS = AtitudeEstimator(Ka=0.005, coor_sys_convention=Exp.Frame)
    phi_hat, _, theta_hat, _, psi_hat, _ = AAE_AHRS.run_exp(exp=Exp, visualize=False,
                                                                          return_grv=False, return_euler=True,
                                                                          save_results_to_file=False)
    fig = plt.figure('Euler')
    axes = []
    n_rows = 3
    n_cols = 1
    for i in range(n_rows * n_cols):
        if i == 0:
            axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
        else:
            axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
    ax_idx = 0
    axes[ax_idx].set(ylabel=r"$x [rad]$", title=""), axes[ax_idx].grid(True)
    axes[ax_idx].plot(Exp.Time_IMU, android_phi,label='android_orientation_angles')
    axes[ax_idx].plot(Exp.Time_IMU, phi_hat, label='AE')
    axes[ax_idx].plot(Exp.grv_time - Exp.grv_time[0], android_grv_phi, label='androide_grv_q_as_Euler')
    axes[ax_idx].legend()

    ax_idx = 1
    axes[ax_idx].grid(True), axes[ax_idx].set(title="")
    axes[ax_idx].plot(Exp.Time_IMU, android_theta, label='android_theta')
    axes[ax_idx].plot(Exp.Time_IMU, theta_hat, label='AE')
    axes[ax_idx].plot(Exp.grv_time - Exp.grv_time[0], android_grv_theta, label='android_grv_theta')

    ax_idx = 2
    axes[ax_idx].grid(True)
    axes[ax_idx].plot(Exp.Time_IMU, android_psi, label='android_psi')
    axes[ax_idx].plot(Exp.Time_IMU, psi_hat, label='AE')
    axes[ax_idx].plot(Exp.grv_time - Exp.grv_time[0], android_grv_psi, label='android_grv_psi')
    plt.show()
