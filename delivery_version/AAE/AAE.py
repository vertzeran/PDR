import numpy as np
import delivery_version.utils.Functions
import delivery_version.utils.Functions as Functions
from scipy.spatial.transform import Rotation as Rotation
from delivery_version.utils.Classes import Euler_angles_plot
from os.path import join
import matplotlib.pyplot as plt
import random


class AtitudeEstimator:
    def __init__(self, dt=1, Rnb=np.identity(3), Ka=0.0, coor_sys_convention='NED'):
        super(AtitudeEstimator, self).__init__()
        self.dt = dt
        self.Rnb = Rnb  # np array
        Euler = Functions.RotMat2euler(self.Rnb)
        self.Phi = Euler[0]
        self.Theta = Euler[1]
        self.Psi = Euler[2]

        if coor_sys_convention == 'NED':
            self.gn = np.array([0, 0, -1])
            self.gb = np.array([0, 0, -1])
            self.Mn = np.array([1, 0, 0])
            self.Mb = np.array([1, 0, 0])
        elif coor_sys_convention == 'ENU':
            self.gb = np.array([0, 0, 1])
            self.gn = np.array([0, 0, 1])
            self.Mn = np.array([0, 1, 0])
            self.Mb = np.array([0, 1, 0])
        else:
            raise Exception('unfamiliar convention')
        if isinstance(Ka, np.ndarray):
            Ka = Ka[0]  # optimization function passes as array
        self.Kgx = Ka
        self.Kgy = Ka
        self.Kgz = Ka
        self.Kmx = 0
        self.Kmy = 0
        self.Kmz = 0

        self.g = 9.81

    def GyroPromotion(self, Gyro):
        omega_nb_b = Gyro
        Omega_nb_b = Functions.Vec2SkewSimetric(omega_nb_b)
        Rnb_dot_m = self.Rnb.dot(Omega_nb_b)
        Rnb_m = self.Rnb + Rnb_dot_m * self.dt
        Rnb_m = Functions.OrthonormalizeRotationMatrix(Rnb_m)
        self.Rnb = Rnb_m
        return

    def UpdateGravity(self, Acc):
        Rnb_m = self.Rnb
        # gb_m = Rnb_m[2, :].T
        gb_m = Rnb_m.T.dot(self.gn)
        # e = -Acc / self.g - gb_m
        e = Acc / self.g - gb_m
        gb_p = gb_m + np.diag([self.Kgx, self.Kgy, self.Kgz]).dot(e)
        gb_p = gb_p / np.linalg.norm(gb_p)
        # fb = -gb_p
        MagneticField = self.Mn
        mb = Rnb_m.T.dot(MagneticField)
        # fn = self.gn
        mn = MagneticField
        # Cnb = Functions.TRIAD(fb, mb / np.linalg.norm(mb), fn, mn / np.linalg.norm(mn))
        Cnb = Functions.TRIAD(gb_p, mb / np.linalg.norm(mb), self.gn, mn / np.linalg.norm(mn))
        Rnb_p = Cnb
        self.Rnb = Rnb_p
        # self.gb = self.Rnb[2, :].T
        self.gb = self.Rnb.T.dot(self.gn)
        return

    def run_exp(self, exp: delivery_version.utils.Classes.AhrsExp, add_errors=False, errors_amp = 0.5, visualize=False, return_grv=True, return_euler=False,
                save_results_to_file=False, path=None):
        """
        given a sequence of IMU readings and reference attitude calculate the estimation state and errors
        :param exp:  instance of utils.Classes.Ahrsexp
        :return: phi_hat, theta_hat, psi_hat, phi_e, theta_e, psi_e
        """
        Gyro = np.vstack([np.array(exp.Gyro.x).reshape(1, exp.NumberOfSamples_IMU),
                          np.array(exp.Gyro.y).reshape(1, exp.NumberOfSamples_IMU),
                          np.array(exp.Gyro.z).reshape(1, exp.NumberOfSamples_IMU)])
        Acc = np.vstack([np.array(exp.Acc.x).reshape(1, exp.NumberOfSamples_IMU),
                         np.array(exp.Acc.y).reshape(1, exp.NumberOfSamples_IMU),
                         np.array(exp.Acc.z).reshape(1, exp.NumberOfSamples_IMU)])
        Rnb_0 = exp.Rot[0].as_matrix()


        if add_errors:
            ePhi0 = random.uniform(-errors_amp, errors_amp)
            eTheta0 = random.uniform(-errors_amp, errors_amp)
            R_e = Rotation.from_euler('ZYX', [0, eTheta0, ePhi0], degrees=True).as_matrix()
            Rnb_0 = R_e.dot(Rnb_0)

        # todo : take a closer measurement with interpulation
        self.dt = exp.dt
        self.Rnb = Rnb_0
        RnbHat = np.zeros([exp.NumberOfSamples_IMU, 3, 3])
        grv = np.zeros([exp.NumberOfSamples_IMU, 3])
        for i in range(exp.NumberOfSamples_IMU):
            if i > 0:
                self.dt = exp.Time_IMU[i] - exp.Time_IMU[i - 1]
            self.GyroPromotion(Gyro[:, i])
            if isinstance(exp, delivery_version.utils.Classes.AI_IMU_DS_Exp):
                acc_b = exp.Acc_bias
            else:
                acc_b = np.array([0.0, 0.0, 0.0])
            self.UpdateGravity(Acc[:, i] + acc_b)
            RnbHat[i, :, :] = self.Rnb
            grv[i,:] = self.gb
        grv *= self.g
        RotHat = Rotation.from_matrix(RnbHat)
        if isinstance(exp, delivery_version.utils.Classes.AI_IMU_DS_Exp):
            IMU_time_offset = exp.SLAM_time_offset
        else:
            IMU_time_offset = 0.0

        if return_euler:
            EulerArray = RotHat.as_euler('ZYX', degrees=False)
            psi_hat = EulerArray[:, 0]
            theta_hat = EulerArray[:, 1]
            phi_hat = EulerArray[:, 2]
            psi_hat = Functions.ContinuousAngle(psi_hat)
            theta_hat = Functions.ContinuousAngle(theta_hat)
            phi_hat = Functions.ContinuousAngle(phi_hat)
            res_visualization = Euler_angles_plot()
            res_visualization.add_ref(time=exp.Time_GT, roll=exp.Phi, pitch=exp.Theta, heading=exp.Psi)
            res_visualization.add_est("AE", time=exp.Time_IMU + IMU_time_offset, roll=phi_hat, pitch=theta_hat,
                                  heading=psi_hat)
            res_visualization.calc_est_err("AE")
            phi_e = res_visualization.estimates["AE"]["roll_error"]
            theta_e = res_visualization.estimates["AE"]["pitch_error"]
            psi_e = res_visualization.estimates["AE"]["heading_error"]
        if visualize:
            if return_euler:
                res_visualization.plot_values = True
                res_visualization.plot_errors = True
                res_visualization.plot_heading = True
                e_att = Functions.att_error(phi_e, theta_e)
                print(exp.Path + '/' + exp.FileName + "\n" + "att. error = " + str(e_att))
                res_visualization.plot_fig()
            if return_grv:
                fig = plt.figure('Gravity Plot')
                axes = []
                n_rows = 3
                n_cols = 2
                for i in range(n_rows * n_cols):
                    if i == 0:
                        axes.append(fig.add_subplot(n_rows, n_cols, i + 1))
                    else:
                        axes.append(fig.add_subplot(n_rows, n_cols, i + 1, sharex=axes[0]))
                ax_idx = 0
                axes[ax_idx].set(ylabel=r"$x [m/sec^2]$", title="gravity"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.x, color='black', linewidth=2, label='grv GT')
                axes[ax_idx].plot(exp.Time_GT, grv[:, 0], color='red', linestyle='dashed', label='grv est')
                axes[ax_idx].legend()
                ax_idx = 1
                axes[ax_idx].grid(True), axes[ax_idx].set(title="grv errors")
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.x - grv[:, 0] , color='black', linewidth=2, label='grv GT error ')
                ax_idx = 2
                axes[ax_idx].set(ylabel=r"$y [m/sec^2]$"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.y, color='black', linewidth=2, label='grv GT')
                axes[ax_idx].plot(exp.Time_GT, grv[:, 1], color='red', linestyle='dashed', linewidth=2, label='grv est')
                ax_idx = 3
                axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.y - grv[:, 1], color='black', linewidth=2,
                                  label='grv GT error ')
                ax_idx = 4
                axes[ax_idx].set(xlabel=r"$[sec]$", ylabel=r"$z [m/sec^2]$"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.z, color='black', linewidth=2, label='grv GT')
                axes[ax_idx].plot(exp.Time_GT, grv[:, 2], color='red', linestyle='dashed', linewidth=2, label='grv est')
                ax_idx = 5
                axes[ax_idx].set(xlabel=r"$[sec]$"), axes[ax_idx].grid(True)
                axes[ax_idx].plot(exp.Time_GT, exp.Grv.z - grv[:, 2], color='black', linewidth=2,
                                  label='grv GT error ')

        if save_results_to_file:
            data_to_saved = {}
            if return_euler and not return_grv:
                data_to_saved["time_IMU"] = exp.Time_IMU
                data_to_saved["phi_hat"] = phi_hat
                data_to_saved["phi_e"] = phi_e
                data_to_saved["theta_hat"] = theta_hat
                data_to_saved["theta_e"] = theta_e
                data_to_saved["psi_hat"] = psi_hat
                data_to_saved["psi_e"] = psi_e
            if return_grv and not return_euler:
                data_to_saved["time_IMU"] = exp.Time_IMU
                data_to_saved["grv_x"] = grv[:, 0].squeeze()
                data_to_saved["grv_y"] = grv[:, 1].squeeze()
                data_to_saved["grv_z"] = grv[:, 2].squeeze()
                data_to_saved["Rnb_11"] = RnbHat[:, 0, 0]
                data_to_saved["Rnb_12"] = RnbHat[:, 0, 1]
                data_to_saved["Rnb_13"] = RnbHat[:, 0, 2]
                data_to_saved["Rnb_21"] = RnbHat[:, 1, 0]
                data_to_saved["Rnb_22"] = RnbHat[:, 1, 1]
                data_to_saved["Rnb_23"] = RnbHat[:, 1, 2]
                data_to_saved["Rnb_31"] = RnbHat[:, 2, 0]
                data_to_saved["Rnb_32"] = RnbHat[:, 2, 1]
                data_to_saved["Rnb_33"] = RnbHat[:, 2, 2]
            if return_grv and return_euler:
                data_to_saved["time_IMU"] = exp.Time_IMU
                data_to_saved["phi_hat"] = phi_hat
                data_to_saved["phi_e"] = phi_e
                data_to_saved["theta_hat"] = theta_hat
                data_to_saved["theta_e"] = theta_e
                data_to_saved["psi_hat"] = psi_hat
                data_to_saved["psi_e"] = psi_e
                data_to_saved["grv_x"] = grv[:, 0].squeeze()
                data_to_saved["grv_y"] = grv[:, 1].squeeze()
                data_to_saved["grv_z"] = grv[:, 2].squeeze()
                data_to_saved["Rnb_11"] = RnbHat[:, 0, 0]
                data_to_saved["Rnb_12"] = RnbHat[:, 0, 1]
                data_to_saved["Rnb_13"] = RnbHat[:, 0, 2]
                data_to_saved["Rnb_21"] = RnbHat[:, 1, 0]
                data_to_saved["Rnb_22"] = RnbHat[:, 1, 1]
                data_to_saved["Rnb_23"] = RnbHat[:, 1, 2]
                data_to_saved["Rnb_31"] = RnbHat[:, 2, 0]
                data_to_saved["Rnb_32"] = RnbHat[:, 2, 1]
                data_to_saved["Rnb_33"] = RnbHat[:, 2, 2]
            if path == None:
                file_name = exp.FileName.split(sep='.')[0] + '_AHRS_results.xlsx'
                file_path = join(exp.Path, file_name)
            else:
                file_path = path
            Functions.save_to_excel_file(path=file_path, dic=data_to_saved, print_message=True)
        if return_grv and not return_euler:
            return grv, RotHat
        if return_euler and not return_grv:
            return phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e
        if return_grv and return_euler:
            return grv, RotHat, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e
        if not return_euler and not return_grv:
            return