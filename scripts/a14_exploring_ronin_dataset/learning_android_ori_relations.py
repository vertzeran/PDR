import utils.Classes as Classes
import utils.Functions as Functions
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from utils.AAE import AtitudeEstimator

file_path = Path(r'C:\Users\EranVertzberger\PHD\indoor recordings toga\indoor_output_2023-07-03_10_35_58.csv')
android_exp = Classes.AndroidExp(file_path)
android_exp.SegmentScenario([200, 250])
t = android_exp.Time_IMU
Euler_ENU_psi = android_exp.Psi
Euler_ENU_theta = android_exp.Theta
Euler_ENU_phi = android_exp.Phi


Rot = Rotation.from_quat(android_exp.QuatArray)
EulerArray = Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
Psi_grv = Functions.ContinuousAngle(EulerArray[:, 0])
initial_heading = Psi_grv[0]
Theta_grv = Functions.ContinuousAngle(EulerArray[:, 1])
Phi_grv = Functions.ContinuousAngle(EulerArray[:, 2])

Ka = 0.00026096
AAE_obj = AtitudeEstimator(Ka=Ka, coor_sys_convention=android_exp.Frame)
phi_hat, _, theta_hat, _, psi_hat, _ = AAE_obj.run_exp(exp=android_exp, return_grv=False, return_euler=True,
                                           save_results_to_file=False, visualize=False)
plot_handle = Classes.Euler_angles_plot()
plot_handle.add_ref(time=t, roll=Euler_ENU_phi, pitch=Euler_ENU_theta, heading=Euler_ENU_psi)
plot_handle.add_est(time=android_exp.grv_time - android_exp.grv_time[0], roll=Phi_grv, pitch=Theta_grv, heading=Psi_grv, name='grv')
plot_handle.add_est(time=t, roll=phi_hat, pitch=theta_hat, heading=psi_hat, name='AHRS')
plot_handle.plot_errors=False
plot_handle.plot_heading = True
plot_handle.plot_fig()
plt.show()
