from os.path import join
from utils.Classes import AI_PDR_exp_w_SP_GT
import matplotlib.pyplot as plt
import utils.Functions as Functions
import ntpath
from scripts.a01_AHRS.calculate_AHRS_results_on_list import CalcResultsOnFile

if __name__ == '__main__':
    exp_path = r'/data/Datasets/Navigation/Shenzhen_datasets/dataset-ShenZhen/train/texting-0008-rectangular.csv'
    head, tail = ntpath.split(exp_path)
    root_dir = head
    file_name = tail
    exp = AI_PDR_exp_w_SP_GT(exp_path)
    # exp.PlotSensors()
    calculate_AHRS = False
    if calculate_AHRS:
        CalcResultsOnFile(exp_path, 'AI_PDR', override=True, GT=None)
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
        plt.show()