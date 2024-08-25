import utils.Classes as Classes
from utils.AAE import AtitudeEstimator
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import utils.Functions as Functions


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
    Exp = Classes.RidiExp('/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Text - Test/tang2.csv')
    plot_grv = False
    plot_lin_acc_b_frame = False
    plot_lin_acc_n_frame = False
    plot_lin_acc_u_frame = False
    plot_acc_trace = False
    plot_euler = True
    plot_position = False

    # Exp.SegmentScenario([0, 60])
    AAE_AHRS = AtitudeEstimator(Ka=0.005)
    t0 = time.time()
    # if
    grv_est, RotHat, _, _, _, _, _, _ = AAE_AHRS.run_exp(exp=Exp, visualize=plot_grv or plot_euler, return_grv=True,
                                                         return_euler=True, save_results_to_file=False)
    acc = np.vstack([np.array(Exp.Acc.x), np.array(Exp.Acc.y), np.array(Exp.Acc.z)]).T
    Pn = np.vstack([np.array(Exp.Pos.x), np.array(Exp.Pos.y), np.array(Exp.Pos.z)]).T
    Pn = Pn - Pn[0, :]
    lin_acc_GT = np.vstack([np.array(Exp.LinAcc.x), np.array(Exp.LinAcc.y), np.array(Exp.LinAcc.z)]).T
    lin_acc_est_b_frame = acc - grv_est
    lin_acc_est_n_frame = Functions.transform_vectors(lin_acc_est_b_frame, RotHat)
    lin_acc_GT_n_frame = Functions.transform_vectors(lin_acc_GT, Exp.Rot)
    lin_acc_GT_u_frame, Rub = transform_b2u(lin_acc_GT, Exp.Rot)
    lin_acc_est_u_frame, Rub_hat = transform_b2u(lin_acc_est_b_frame, RotHat)

    if plot_lin_acc_b_frame:
        fig = plt.figure('lin acc b frame Plot')
        axes = []
        n_rows = 3
        n_cols = 2
        for i in range(n_rows * n_cols):
            if i == 0:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
            else:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
        ax_idx = 0
        axes[ax_idx].set(ylabel=r"$x [m/sec^2]$", title="linear acceleration"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, Exp.LinAcc.x, color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_b_frame[:, 0], color='red', linestyle='dashed', label='lin acc est')
        axes[ax_idx].legend()
        ax_idx = 1
        axes[ax_idx].grid(True), axes[ax_idx].set(title="lin acc errors")
        axes[ax_idx].plot(Exp.Time_GT, Exp.LinAcc.x - lin_acc_est_b_frame[:, 0], color='black', linewidth=2, label='lin acc error ')
        ax_idx = 2
        axes[ax_idx].set(ylabel=r"$y [m/sec^2]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, Exp.LinAcc.y, color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_b_frame[:, 1], color='red', linestyle='dashed', label='lin acc est')
        ax_idx = 3
        axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, Exp.LinAcc.y - lin_acc_est_b_frame[:, 1], color='black', linewidth=2, label='lin acc error ')
        ax_idx = 4
        axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [m/sec^2]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, Exp.LinAcc.z, color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_b_frame[:, 2], color='red', linestyle='dashed', label='lin acc est')
        ax_idx = 5
        axes[ax_idx].set(xlabel=r"$[sec]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, Exp.LinAcc.z - lin_acc_est_b_frame[:, 2], color='black', linewidth=2, label='lin acc error ')
        if plot_acc_trace:
            fig = plt.figure('acc trace b frame')
            ax = fig.add_subplot(projection='3d')
            ax.set(xlabel=r"$x [m/sec^2]$", ylabel=r"$y [m/sec^2]$", zlabel=r"$z [m/sec^2]$", title="acceleration trace b frame"), axes[ax_idx].grid(True)
            ax.plot(Exp.LinAcc.x, Exp.LinAcc.y, Exp.LinAcc.z, color='black', linewidth=2, label='')
    if plot_lin_acc_n_frame:
        fig = plt.figure('lin acc n frame Plot')
        axes = []
        n_rows = 3
        n_cols = 2
        for i in range(n_rows * n_cols):
            if i == 0:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
            else:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
        ax_idx = 0
        axes[ax_idx].set(ylabel=r"$x [m/sec^2]$", title="linear acceleration"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 0], color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 0], color='red', linestyle='dashed', label='lin acc est')
        axes[ax_idx].legend()
        ax_idx = 1
        axes[ax_idx].grid(True), axes[ax_idx].set(title="lin acc errors")
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 0] - lin_acc_est_n_frame[:, 0], color='black', linewidth=2, label='lin acc error')
        ax_idx = 2
        axes[ax_idx].set(ylabel=r"$y [m/sec^2]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 1], color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 1], color='red', linestyle='dashed', label='lin acc est')
        ax_idx = 3
        axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 1] - lin_acc_est_n_frame[:, 1], color='black', linewidth=2, label='lin acc error')
        ax_idx = 4
        axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [m/sec^2]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 2], color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 2], color='red', linestyle='dashed', label='lin acc est')
        ax_idx = 5
        axes[ax_idx].set(xlabel=r"$[sec]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 2] - lin_acc_est_n_frame[:, 2], color='black', linewidth=2, label='lin acc error')
        if plot_acc_trace:
            fig = plt.figure('acc trace n frame')
            ax = fig.add_subplot(projection='3d')
            ax.set(xlabel=r"$x [m/sec^2]$", ylabel=r"$y [m/sec^2]$", zlabel=r"$z [m/sec^2]$", title="acceleration trace n frame"), axes[ax_idx].grid(True)
            ax.plot(lin_acc_GT_n_frame[:, 0], lin_acc_GT_n_frame[:, 1], lin_acc_GT_n_frame[:, 2], color='black', linewidth=2, label='')
    if plot_lin_acc_u_frame:
        fig = plt.figure('lin acc u frame Plot')
        axes = []
        n_rows = 3
        n_cols = 2
        for i in range(n_rows * n_cols):
            if i == 0:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
            else:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
        ax_idx = 0
        axes[ax_idx].set(ylabel=r"$x [m/sec^2]$", title="linear acceleration"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_u_frame[:, 0], color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_u_frame[:, 0], color='red', linestyle='dashed', label='lin acc est')
        axes[ax_idx].legend()
        ax_idx = 1
        axes[ax_idx].grid(True), axes[ax_idx].set(title="lin acc errors")
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_u_frame[:, 0] - lin_acc_est_u_frame[:, 0], color='black', linewidth=2, label='lin acc error')
        ax_idx = 2
        axes[ax_idx].set(ylabel=r"$y [m/sec^2]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_u_frame[:, 1], color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_u_frame[:, 1], color='red', linestyle='dashed', label='lin acc est')
        ax_idx = 3
        axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_u_frame[:, 1] - lin_acc_est_u_frame[:, 1], color='black', linewidth=2, label='lin acc error')
        ax_idx = 4
        axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [m/sec^2]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_u_frame[:, 2], color='black', linewidth=2, label='lin acc GT')
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_est_n_frame[:, 2], color='red', linestyle='dashed', label='lin acc est')
        ax_idx = 5
        axes[ax_idx].set(xlabel=r"$[sec]$"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(Exp.Time_GT, lin_acc_GT_n_frame[:, 2] - lin_acc_est_u_frame[:, 2], color='black', linewidth=2, label='lin acc error')
        if plot_acc_trace:
            fig = plt.figure('acc trace u frame')
            ax = fig.add_subplot(projection='3d')
            ax.set(xlabel=r"$x [m/sec^2]$", ylabel=r"$y [m/sec^2]$", zlabel=r"$z [m/sec^2]$", title="acceleration trace u frame"), axes[ax_idx].grid(True)
            ax.plot(lin_acc_GT_u_frame[:, 0], lin_acc_GT_u_frame[:, 1], lin_acc_GT_u_frame[:, 2], color='black', linewidth=2, label='')
    if plot_position:
        fig = plt.figure('plosition plot')
        axes = []
        n_rows = 1
        n_cols = 2
        for i in range(n_rows * n_cols):
            if i == 0:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
            else:
                axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
        px = Pn[:, 0] - Pn[0, 0]
        py = Pn[:, 1] - Pn[0, 1]
        pz = Pn[:, 2] - Pn[0, 2]
        ax_idx = 0
        axes[ax_idx].set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position n frame"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(px, py, color='black', linewidth=2, label='')
        # Rot_bn_hat = []
        # for R in RotHat:
        #     Rbn_i = R.as_matrix().T
        #     Rot_bn_hat.append(Rotation.from_matrix(Rbn_i))
        Pu = Functions.transform_vectors(Pn, Rub)
        px = Pu[:, 0]
        py = Pu[:, 1]
        pz = Pu[:, 2]
        ax_idx = 1
        axes[ax_idx].set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position u frame"), axes[ax_idx].grid(True)
        axes[ax_idx].plot(px, py, color='black', linewidth=2, label='')

    if plot_lin_acc_n_frame or \
            plot_euler or \
            plot_lin_acc_b_frame or \
            plot_lin_acc_u_frame or \
            plot_position:
        plt.show()