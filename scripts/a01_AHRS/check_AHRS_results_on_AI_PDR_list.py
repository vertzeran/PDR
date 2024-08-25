import argparse
from os.path import join
from utils.Classes import AI_PDR_exp_w_SP_GT
from os import listdir
import matplotlib.pyplot as plt
import utils.Functions as Functions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_list', type=str, default=None, help='if not given work on the whole directory')
    parser.add_argument('--root_dir', type=str, default='/data/Datasets/Navigation/AI_PDR_train_test/test',
                        help='Path to data directory')

    """
    to run (example):
    python calculate_AHRS_results_on_list.py 
    --exp_list /home/maint/git/pdrnet/lists/sbg_test_swing.txt
    --root_dir /home/maint/Eran/AHRS/RIDI_dataset_train_test/Pocket_Test
    """

    args = parser.parse_args()
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
        if '_AHRS_results.xlsx' not in file_path:
            exp = AI_PDR_exp_w_SP_GT(file_path)
            AHRS_results_files = exp.check_for_AHRS_results()
            if len(AHRS_results_files) == 0:
                print('no AHRS analysis on ' + exp.FileName)
            else:
                print('found AHRS results for ' + file_name)
                t_est, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv_hat, Rot_hat = \
                    Functions.read_AHRS_results(join(args.root_dir, AHRS_results_files[0]))
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
                axes[ax_idx].plot(t_est, exp.Grv.y - grv_hat[:, 1], color='black', linewidth=2,
                                  label='theta error')
                ax_idx = 4
                axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [rad]$"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.z, color='black', linewidth=2, label='psi')
                axes[ax_idx].plot(t_est, grv_hat[:, 2], color='red', linestyle='dashed',
                                  label='psi hat')
                ax_idx = 5
                axes[ax_idx].set(xlabel=r"$[sec]$"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.z - grv_hat[:, 2], color='black', linewidth=2,
                                  label='phi error ')
                plt.show()