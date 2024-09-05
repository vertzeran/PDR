# import h5py
# import json
# from os import path as osp
# import numpy as np
# import quaternion
from utils import Classes, AAE
import matplotlib.pyplot as plt

data_path = 'C:\\Users\\EranVertzberger\\PHD\\ronin_dataset\\dataset\\train_dataset_1\\a001_1'
# with open(osp.join(data_path, 'info.json')) as f:
#     info = json.load(f)
# info['path'] = osp.split(data_path)[-1]
# with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
#     ori = np.copy(f['synced/game_rv'])
# with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
#     gyro_uncalib = f['synced/gyro_uncalib']
#     acce_uncalib = f['synced/acce']
#     gyro = gyro_uncalib - np.array(info['imu_init_gyro_bias'])
#     acce = np.array(info['imu_acce_scale']) * (acce_uncalib - np.array(info['imu_acce_bias']))
#     ts = np.copy(f['synced/time'])
#     tango_pos = np.copy(f['pose/tango_pos'])
#     init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
exp = Classes.RoninExp(data_path)
exp.SegmentScenario([0,60])
exp.PlotSensors()
Ka = 0.005 # 0.00026096
AAE_AHRS = AAE.AtitudeEstimator(Ka=Ka, coor_sys_convention=exp.Frame)

print('performing AHRS analysis on: \n' + exp.Path)
_, _, _, _, _, _, = AAE_AHRS.run_exp(exp=exp, return_grv=False, return_euler=True,
                                           save_results_to_file=False, visualize=True)
plt.show()
