import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.transform import Slerp
import os
import tkinter.filedialog
import ntpath
import colorsys
import h5py
import utils.Functions as Functions
from os.path import join
import json
from os import listdir
from datetime import datetime
import copy
from scipy import signal
import torch


class Vec3d:
    x = None
    y = None
    z = None

    def arr(self):
        """return a n x 3 numpy array"""
        return np.vstack([np.array(self.x), np.array(self.y), np.array(self.z)]).T


class AhrsExp():
    def __init__(self):
        self.Time_IMU = None
        self.IMU_valid_idx = None
        self.Time_GT = None
        self.NumberOfSamples_IMU = None
        self.NumberOfSamples_GT = None
        self.dt = None  # IMU
        self.Pos = Vec3d()  # GT

        self.Rot = None  # GT

        self.Psi = None  # GT
        self.Theta = None  # GT
        self.Phi = None  # GT

        self.ePhi0 = None
        self.eTheta0 = None

        self.Gyro = Vec3d()

        self.Acc = Vec3d()

        self.Mag = Vec3d()

        self.LinAcc = Vec3d()

        self.Grv = Vec3d()

        self.QuietPeriods = None
        self.Frame = None  # 'ENU' or 'NED'
        self.id = None
        self.index_of_walking_start = 0
        self.initial_WD_angle_GT = None
        self.initial_heading = None
        self.Path = None
        self.FileName = None
        self.valid = None

    def PlotAngles(self):
        fig = plt.figure('Euler Angles Plot')
        Ax_x = fig.add_subplot(311)
        Ax_y = fig.add_subplot(312, sharex=Ax_x)
        Ax_z = fig.add_subplot(313, sharex=Ax_x)

        Ax_x.plot(self.Time_GT, self.Phi, color='blue', linewidth=1)
        Ax_x.set_xlim(self.Time_GT[0], self.Time_GT[-1]), Ax_x.set(title=r"$\phi$", xlabel="Time [sec]",
                                                                   ylabel="[rad]"), Ax_x.grid(True)
        Ax_y.plot(self.Time_GT, self.Theta, color='blue', linewidth=1)
        Ax_y.set(title=r"$\theta$", xlabel="Time [sec]", ylabel="[rad]"), Ax_y.grid(True)
        Ax_z.plot(self.Time_GT, self.Psi, color='blue', linewidth=1)
        Ax_z.set(title=r"$\psi$", xlabel="Time [sec]", ylabel="[rad]"), Ax_z.grid(True)
        plt.tight_layout()

    def PlotSensors(self):
        fig = plt.figure('Sensors Plot')
        # acc
        Ax_acc_x = fig.add_subplot(321)
        Ax_acc_x.plot(self.Time_IMU, self.Acc.x, color='blue', linewidth=1)
        Ax_acc_x.set_xlim(self.Time_IMU[0], self.Time_IMU[-1])
        Ax_acc_x.set(title=r"Accelerometers", ylabel="$[m/sec^2]$")
        Ax_acc_x.grid(True)

        Ax_acc_y = fig.add_subplot(323, sharex=Ax_acc_x)
        Ax_acc_y.plot(self.Time_IMU, self.Acc.y, color='blue', linewidth=1)
        Ax_acc_y.set_xlim(self.Time_IMU[0], self.Time_IMU[-1])
        Ax_acc_y.set(ylabel="$[m/sec^2]$")
        Ax_acc_y.grid(True)

        Ax_acc_z = fig.add_subplot(325, sharex=Ax_acc_x)
        Ax_acc_z.plot(self.Time_IMU, self.Acc.z, color='blue', linewidth=1)
        Ax_acc_z.set_xlim(self.Time_IMU[0], self.Time_IMU[-1])
        Ax_acc_z.set(xlabel="Time [sec]", ylabel="$[m/sec^2]$")
        Ax_acc_z.grid(True)
        # gyro
        Ax_gyro_x = fig.add_subplot(322, sharex=Ax_acc_x)
        Ax_gyro_x.plot(self.Time_IMU, self.Gyro.x, color='blue', linewidth=1)
        Ax_gyro_x.set_xlim(self.Time_IMU[0], self.Time_IMU[-1])
        Ax_gyro_x.set(title=r"Gyro", ylabel="$[rad/sec^2]$")
        Ax_gyro_x.grid(True)

        Ax_gyro_y = fig.add_subplot(324, sharex=Ax_acc_x)
        Ax_gyro_y.plot(self.Time_IMU, self.Gyro.y, color='blue', linewidth=1)
        Ax_gyro_y.set_xlim(self.Time_IMU[0], self.Time_IMU[-1])
        Ax_gyro_y.set(ylabel="$[rad/sec^2]$")
        Ax_gyro_y.grid(True)

        Ax_gyro_z = fig.add_subplot(326, sharex=Ax_acc_x)
        Ax_gyro_z.plot(self.Time_IMU, self.Gyro.z, color='blue', linewidth=1)
        Ax_gyro_z.set_xlim(self.Time_IMU[0], self.Time_IMU[-1])
        Ax_gyro_z.set(xlabel="Time [sec]", ylabel="$[rad/sec^2]$")
        Ax_gyro_z.grid(True)

    def PlotPosition(self, temporal=False, XY=True):
        if temporal:
            fig = plt.figure('Position Temporal Plot')
            Ax_pos_x = fig.add_subplot(311)
            Ax_pos_x.plot(self.Time_GT, self.Pos.x, color='blue', linewidth=1)
            Ax_pos_x.set_xlim(self.Time_GT[0], self.Time_GT[-1])
            Ax_pos_x.set(title=r"Position", ylabel="$[m]$")
            Ax_pos_x.grid(True)

            Ax_pos_y = fig.add_subplot(312)
            Ax_pos_y.plot(self.Time_GT, self.Pos.y, color='blue', linewidth=1)
            Ax_pos_y.set_xlim(self.Time_GT[0], self.Time_GT[-1])
            Ax_pos_y.set(ylabel="$[m]$")
            Ax_pos_y.grid(True)

            Ax_pos_z = fig.add_subplot(313)
            Ax_pos_z.plot(self.Time_GT, self.Pos.z, color='blue', linewidth=1)
            Ax_pos_z.set_xlim(self.Time_GT[0], self.Time_GT[-1])
            Ax_pos_z.set(xlabel="Time [sec]", ylabel="$[m]$")
            Ax_pos_z.grid(True)
        if XY:
            fig = plt.figure('Position XY Plot')
            Ax = fig.add_subplot(111)
            Ax.plot(self.Pos.x, self.Pos.y, color='blue', linewidth=1)
            Ax.set(xlabel="$[m]$", ylabel="$[m]$")
            Ax.grid(True)

    def SegmentScenario(self, StartStopTime):
        t_start = max(StartStopTime[0], self.Time_IMU[0])  # prevent someone to ask for times that are not exist
        t_stop = min(StartStopTime[1], self.Time_IMU[-1])  # prevent someone to ask for times that are not exist
        start_idx = abs(self.Time_IMU - t_start).argmin()
        stop_idx = abs(self.Time_IMU - t_stop).argmin()
        ind_IMU = list(range(start_idx, stop_idx + 1))  # to include stop_idx
        # ind_IMU = np.where((self.Time_IMU >= t_start) & (self.Time_IMU <= t_stop))
        start_idx = abs(self.Time_GT - t_start).argmin()
        stop_idx = abs(self.Time_GT - t_stop).argmin()
        # ind_GT = np.where((self.Time_GT >= t_start) & (self.Time_GT <= t_stop))
        ind_GT = list(range(start_idx, stop_idx + 1))

        self.Time_IMU = self.Time_IMU[ind_IMU]
        if self.IMU_valid_idx is not None:
            self.IMU_valid_idx = list(map(self.IMU_valid_idx.__getitem__, ind_IMU))
        self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
        self.Time_GT = self.Time_GT[ind_GT]
        self.NumberOfSamples_GT = self.Time_GT.shape[0]
        self.Pos.x = np.array(self.Pos.x)[ind_GT]
        self.Pos.y = np.array(self.Pos.y)[ind_GT]
        self.Pos.z = np.array(self.Pos.z)[ind_GT]

        self.Rot = self.Rot[ind_GT]
        self.Psi = self.Psi[ind_GT]
        self.Theta = self.Theta[ind_GT]
        self.Phi = self.Phi[ind_GT]

        self.Gyro.x = np.array(self.Gyro.x)[ind_IMU]
        self.Gyro.y = np.array(self.Gyro.y)[ind_IMU]
        self.Gyro.z = np.array(self.Gyro.z)[ind_IMU]

        self.Acc.x = np.array(self.Acc.x)[ind_IMU]
        self.Acc.y = np.array(self.Acc.y)[ind_IMU]
        self.Acc.z = np.array(self.Acc.z)[ind_IMU]

        if isinstance(self, RidiExp):
            self.Mag.x = np.array(self.Mag.x)[ind_IMU]
            self.Mag.y = np.array(self.Mag.y)[ind_IMU]
            self.Mag.z = np.array(self.Mag.z)[ind_IMU]
        if isinstance(self, RidiExp) or isinstance(self, SbgExpRawData):
            self.LinAcc.x = np.array(self.LinAcc.x)[ind_GT]
            self.LinAcc.y = np.array(self.LinAcc.y)[ind_GT]
            self.LinAcc.z = np.array(self.LinAcc.z)[ind_GT]
        if isinstance(self, RidiExp) or isinstance(self, SbgExpRawData):
            self.Grv.x = np.array(self.Grv.x)[ind_GT]
            self.Grv.y = np.array(self.Grv.y)[ind_GT]
            self.Grv.z = np.array(self.Grv.z)[ind_GT]

    def check_if_quite(self, idx, STD_Threshold=10e-3, Max_Threshold=10e-3):
        gx = self.Gyro.x[idx]
        gy = self.Gyro.y[idx]
        gz = self.Gyro.z[idx]
        nx = np.std(gx)
        ny = np.std(gy)
        nz = np.std(gz)
        mx = max(abs(gx))
        my = max(abs(gy))
        mz = max(abs(gz))

        Quiet = sum(np.array([nx, ny, nz]) < STD_Threshold) + sum(np.array([mx, my, mz]) < Max_Threshold) == 6
        return Quiet

    def identify_static_periods(self, MinimalWindowLength=250, plot_results=False):
        QuietPeriods = []
        StartIndex = 0
        StopIndex = MinimalWindowLength
        NumOfQuitePeriods = 0
        QuietPeriodUndentified = False
        while StopIndex <= self.NumberOfSamples_IMU:
            Quiet = self.check_if_quite(range(StartIndex, StopIndex))
            if Quiet:
                if QuietPeriodUndentified:  # extend current quiet period
                    StopIndex = StopIndex + 10
                    QuietPeriods[NumOfQuitePeriods - 1] = [StartIndex, StopIndex]
                else:  # new quite period identified
                    QuietPeriodUndentified = True
                    QuietPeriods.append([StartIndex, StopIndex])
                    NumOfQuitePeriods = NumOfQuitePeriods + 1
            else:
                if QuietPeriodUndentified:  # end of a quiet period
                    QuietPeriodUndentified = False
                    StartIndex = StopIndex
                    StopIndex = StartIndex + MinimalWindowLength
                else:  # move window
                    StartIndex = StartIndex + 10
                    StopIndex = StartIndex + MinimalWindowLength
        if len(QuietPeriods) > 0:
            QuietPeriods[-1][1] = min(QuietPeriods[-1][1], self.NumberOfSamples_IMU)
        self.QuietPeriods = QuietPeriods

        if plot_results:
            # convert to indexes vector
            NumOfPeriods = len(QuietPeriods)
            QuietIndexes = []
            for i in range(NumOfPeriods):
                QuietIndexes.extend(range(QuietPeriods[i][0], QuietPeriods[i][1]))
            fig = plt.figure('static periods')
            ax = fig.add_subplot(111)
            ax.plot(self.Gyro.x)
            ax.plot(self.Gyro.y)
            ax.plot(self.Gyro.z)
            for period in QuietPeriods:
                ax.axvline(x=period[0], color='r', ls='--')
                ax.axvline(x=period[1], color='r')
            ax.set(title="static periods identification", xlabel="samples", ylabel="[rad/sec]"), ax.grid(True)

    def calc_dL(self, window_size):
        gt_pos = self.Pos.arr()[:, 0:2]  # us only x,y
        L = np.zeros(self.NumberOfSamples_GT)
        for j in range(1, self.NumberOfSamples_GT):
            dp = gt_pos[j] - gt_pos[j - 1]
            L[j] = np.linalg.norm(dp) + L[j - 1]
        dL_dense = L[window_size:] - L[:-window_size]
        idx = range(0, dL_dense.shape[0], window_size)
        dL = dL_dense[idx]
        return dL

    def calc_walking_direction(self, window_size=1):
        """return the walking direction angles and vectors based on the sampled position"""
        gt_pos = self.Pos.arr()[:, 0:2]
        idx = range(0, self.NumberOfSamples_GT, window_size)
        pos_samples = gt_pos[idx]
        walking_angle = []
        v_wd = []
        for i in range(1, pos_samples.shape[0]):
            v_wd.append(pos_samples[i] - pos_samples[i - 1])
            walking_angle.append(np.arctan2(v_wd[-1][1], v_wd[-1][0]))
        return np.array(walking_angle), np.array(v_wd)

    def check_for_AHRS_results(self):
        AHRS_Results = []
        suffix_to_identify_results = '_AHRS_results.xlsx'
        suffix_length = len(suffix_to_identify_results)
        for item in listdir(self.Path):
            if self.FileName.split(sep='.')[0] in item:
                if item[-suffix_length:] == suffix_to_identify_results and item[:-18] == self.FileName.split(sep='.')[
                    0]:
                    AHRS_Results.append(item)
        return AHRS_Results

    def define_walking_start_idx(self, th=1):
        # dP_angles, dP_vectors = self.calc_walking_direction(window_size=1)
        # dL = np.linalg.norm(dP_vectors, axis=1)
        method = 'distance_for_origin'  # could be: 'accumulated_trajectory_length'
        if method == 'accumulated_trajectory_length':
            dL = self.calc_dL(window_size=1)
            dL_cumsum = dL.cumsum()
            self.index_of_walking_start = next(i for i in range(dL.shape[0]) if dL_cumsum[i] > th)
        elif method == 'distance_for_origin':
            pos = self.Pos.arr() - self.Pos.arr()[0]
            pos_norm = np.linalg.norm(pos, axis=1)
            self.index_of_walking_start = next(i for i in range(pos_norm.shape[0]) if pos_norm[i] > th)

    def initialize_WD_angle(self, wind_size_for_heading_init=750, plot_results=False):
        # index_of_walking_start is initialized is pre-calculated or zero consider calculating here
        if self.index_of_walking_start + wind_size_for_heading_init >= self.NumberOfSamples_IMU:
            print('check this file:  ' + join(self.Path, self.FileName))
        overall_d_pos = self.Pos.arr()[self.index_of_walking_start + wind_size_for_heading_init] - \
                        self.Pos.arr()[self.index_of_walking_start]
        initial_WD_vector_GT = overall_d_pos[0:2]
        self.initial_WD_angle_GT = Functions.FoldAngles(np.arctan2(initial_WD_vector_GT[1], initial_WD_vector_GT[0]))
        if plot_results:
            plt.figure()
            plt.plot(self.Pos.x[self.index_of_walking_start:self.index_of_walking_start + wind_size_for_heading_init],
                     self.Pos.y[self.index_of_walking_start:self.index_of_walking_start + wind_size_for_heading_init])
            plt.show()

    def validate_data_using_time_vectors(self, time_IMU, time_GPS, IMU_time_gap_th, GPS_time_gap_th, min_length_th):
        """
        1. look for time gaps in IMU
        2. segment GPS in IMu frame
        3. look for time gaps in GPS
        4. segment IMU in GPS frame
        5. varify the residual that is left (after time gaps segmentation) is long enough
        """
        """
        if False: # simulation
            print('utils.Classes.AhrsExp.validate_data_using_time_vectors: simulation is on !!!')
            # import numpy as np
            time_IMU = np.linspace(2, 8, 10000)
            time_of_IMU_time_gap = 6.8
            IMU_time_gap_length = 1
            IMU_time_gap_idx = np.where((time_IMU > time_of_IMU_time_gap) &
                                        (time_IMU < time_of_IMU_time_gap + IMU_time_gap_length)
                                        )
            time_IMU = np.delete(time_IMU, IMU_time_gap_idx)
            # 1. look for time gaps in IMU:  2->6.8 should be left in time_IMU[IMU_idx]
            time_GPS = np.linspace(0, 10, 1000)
            # 2. segment GPS in IMu frame: 2->6.8 should be left in time_GPS[GPS_idx]]
            time_of_GPS_time_gap = 6
            GPS_time_gap_length = 1
            GPS_time_gap_idx = np.where(
                (time_GPS > time_of_GPS_time_gap) &
                (time_GPS < time_of_GPS_time_gap + GPS_time_gap_length)
            )
            time_GPS = np.delete(time_GPS, GPS_time_gap_idx)
            # 3. look for time gaps in GPS:  2 -> 6 should be left in time_GPS[GPS[idx]]
            # 4. segment IMU in GPS frame:   2 -> 6 should be left in time_IMU[IMU_idx]
            # 5. varify the residual that is left (after time gaps segmentation) is long enough : 4>3 -> valid
            IMU_time_gap_th = 0.5
            GPS_time_gap_th = 0.1
            min_length_th = 3
            """
        # 1. look for time gaps in IMU

        # iterator = (idx for idx in range(time_IMU.shape[0] - 1) if
        #             time_IMU[idx + 1] - time_IMU[idx] > IMU_time_gap_th)
        # self.first_idx_of_time_gap_IMU = next(iterator, None)
        # if self.first_idx_of_time_gap_IMU is not None:
        #     IMU_idx = list(range(time_IMU.shape[0])[:self.first_idx_of_time_gap_IMU])
        #     print('time gap in IMU')
        # else:
        #     IMU_idx = list(range(time_IMU.shape[0]))

        self.first_idx_of_time_gap_IMU = np.where(np.diff(time_IMU) > IMU_time_gap_th)[0]
        if self.first_idx_of_time_gap_IMU.shape[0] != 0:
            IMU_idx = list(range(time_IMU.shape[0])[:self.first_idx_of_time_gap_IMU[0]])
            print(
                'time gap in IMU in ')  # + str(time_IMU[self.first_idx_of_time_gap_IMU[0] + 1] - time_IMU[0]) + ' [sec]')
        else:
            IMU_idx = list(range(time_IMU.shape[0]))
        # 2. segment GPS in IMU time frame
        t_start_IMU = time_IMU[IMU_idx][0]
        t_stop_IMU = time_IMU[IMU_idx][-1]
        # GPS_idx = np.where((time_GPS >= t_start_IMU) & (time_GPS <= t_stop_IMU))

        start_idx = abs(time_GPS - t_start_IMU).argmin()
        start_idx = max(0, start_idx - 1)
        stop_idx = abs(time_GPS - t_stop_IMU).argmin()
        stop_idx = min(stop_idx + 1, time_GPS.shape[0])
        GPS_idx = list(range(start_idx, stop_idx + 1))  # to include stop_idx

        # 3. look for time gaps in GPS

        # iterator = (idx for idx in range(time_GPS[GPS_idx].shape[0] - 1) if
        #             time_GPS[GPS_idx][idx + 1] - time_GPS[GPS_idx][idx] > GPS_time_gap_th)
        # self.first_idx_of_time_gap_GPS = next(iterator, None) # notice this is an idx in an idx vector
        # if self.first_idx_of_time_gap_GPS is not None:                                           ##    |
        #     GPS_idx = GPS_idx[:self.first_idx_of_time_gap_GPS] ## <------------------------------------|
        #     print('time GPS in IMU')

        self.first_idx_of_time_gap_GPS = np.where(np.diff(time_GPS[GPS_idx]) > GPS_time_gap_th)[0]
        # notice this is an idx in an idx vector
        if self.first_idx_of_time_gap_GPS.shape[0] != 0:  ##    |
            GPS_idx = GPS_idx[:self.first_idx_of_time_gap_GPS[0]]  ## <--------------------------|
            print(
                'time gap in GPS in ')  # + str(time_GPS[GPS_idx][self.first_idx_of_time_gap_GPS[0] + 1] - time_GPS[GPS_idx][0]) + ' [sec]')
        # 4. varify that IMU data is processed only in times where GPS data available

        t_start_GPS = time_GPS[GPS_idx][0]
        t_stop_GPS = time_GPS[GPS_idx][-1]
        IMU_idx_of_idx = np.where((time_IMU[IMU_idx] >= t_start_GPS) &
                                  (time_IMU[IMU_idx] <= t_stop_GPS))  # notice this is an idx in an idx vector
        IMU_idx_of_idx = IMU_idx_of_idx[0].astype(int)
        IMU_idx = list(map(IMU_idx.__getitem__, IMU_idx_of_idx))  ## <--------------------------------------|
        # varify the residual that is left (after time gaps segmentation) is long enough
        if time_IMU[IMU_idx][-1] - time_IMU[IMU_idx][0] < min_length_th:
            self.valid = False
        else:
            self.valid = True

        return IMU_idx, GPS_idx

    def limit_traj_length(self, limit):
        traj_length_vector = np.cumsum(self.calc_dL(window_size=1))
        if np.max(traj_length_vector) > limit:
            idx_of_traj_length_limit = np.argmin(abs(traj_length_vector - limit))
            self.SegmentScenario([0, self.Time_GT[idx_of_traj_length_limit]])

    def clone(self):
        return copy.deepcopy(self)


class RidiExp(AhrsExp):
    def __init__(self, path=None):
        super(RidiExp, self).__init__()
        if path is None:
            curr_directory = os.getcwd()  # will get current working directory
            path = tkinter.filedialog.askopenfilename(initialdir=curr_directory, title="Select A File",
                                                      filetype=(("csv files", "*.csv"), ("all files", "*.*")))
        Exp = pd.read_csv(path)
        head, tail = ntpath.split(path)
        self.Path = head
        self.FileName = tail
        time = np.array(Exp.time)
        self.Time_IMU = (time - time[0]) * 1e-9
        self.Time_GT = (time - time[0]) * 1e-9
        self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
        self.NumberOfSamples_GT = self.Time_GT.shape[0]
        self.dt = np.mean(np.diff(self.Time_IMU))

        self.Pos.x = Exp.pos_x
        self.Pos.y = -Exp.pos_y
        self.Pos.z = -Exp.pos_z

        QuatArray = np.array([Exp.ori_x, -Exp.ori_y, -Exp.ori_z, Exp.ori_w]).T  # size is nX4 Transfered to ned
        self.Rot = Rotation.from_quat(QuatArray)
        EulerArray = self.Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        self.Psi = Functions.ContinuousAngle(EulerArray[:, 0])
        self.initial_heading = self.Psi[0]
        self.Theta = Functions.ContinuousAngle(EulerArray[:, 1])
        self.Phi = Functions.ContinuousAngle(EulerArray[:, 2])

        self.Gyro.x = Exp.gyro_x
        self.Gyro.y = -Exp.gyro_y  # conversion from ENU to NED
        self.Gyro.z = -Exp.gyro_z  # conversion from ENU to NED

        self.Acc.x = Exp.acce_x
        self.Acc.y = -Exp.acce_y  # conversion from ENU to NED
        self.Acc.z = -Exp.acce_z  # conversion from ENU to NED

        self.Mag.x = Exp.magnet_x
        self.Mag.y = -Exp.magnet_y  # conversion from ENU to NED
        self.Mag.z = -Exp.magnet_z  # conversion from ENU to NED

        self.LinAcc.x = Exp.linacce_x
        self.LinAcc.y = -Exp.linacce_y  # conversion from ENU to NED
        self.LinAcc.z = -Exp.linacce_z  # conversion from ENU to NED

        self.Grv.x = Exp.grav_x
        self.Grv.y = -Exp.grav_y  # conversion from ENU to NED
        self.Grv.z = -Exp.grav_z  # conversion from ENU to NED

        self.Frame = 'NED'


class RidiExp_ENU(AhrsExp):
    def __init__(self, path=None):
        super(RidiExp_ENU, self).__init__()
        if path is None:
            curr_directory = os.getcwd()  # will get current working directory
            path = tkinter.filedialog.askopenfilename(initialdir=curr_directory, title="Select A File",
                                                      filetype=(("csv files", "*.csv"), ("all files", "*.*")))
        Exp = pd.read_csv(path)
        head, tail = ntpath.split(path)
        self.Path = head
        self.FileName = tail
        time = np.array(Exp.time)
        self.Time_IMU = (time - time[0]) * 1e-9
        self.Time_GT = (time - time[0]) * 1e-9
        self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
        self.NumberOfSamples_GT = self.Time_GT.shape[0]
        self.IMU_valid_idx = list(range(self.NumberOfSamples_IMU))
        self.dt = np.mean(np.diff(self.Time_IMU))

        self.Pos.x = Exp.pos_x
        self.Pos.y = Exp.pos_y
        self.Pos.z = Exp.pos_z

        QuatArray = np.array([Exp.ori_x, Exp.ori_y, Exp.ori_z, Exp.ori_w]).T  # size is nX4 Transfered to ned
        self.Rot = Rotation.from_quat(QuatArray)
        EulerArray = self.Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        self.Psi = Functions.ContinuousAngle(EulerArray[:, 0])
        self.initial_heading = self.Psi[0]
        self.Theta = Functions.ContinuousAngle(EulerArray[:, 1])
        self.Phi = Functions.ContinuousAngle(EulerArray[:, 2])

        self.Gyro.x = Exp.gyro_x
        self.Gyro.y = Exp.gyro_y  # conversion from ENU to NED
        self.Gyro.z = Exp.gyro_z  # conversion from ENU to NED

        self.Acc.x = Exp.acce_x
        self.Acc.y = Exp.acce_y  # conversion from ENU to NED
        self.Acc.z = Exp.acce_z  # conversion from ENU to NED

        self.Mag.x = Exp.magnet_x
        self.Mag.y = Exp.magnet_y  # conversion from ENU to NED
        self.Mag.z = Exp.magnet_z  # conversion from ENU to NED

        self.LinAcc.x = Exp.linacce_x
        self.LinAcc.y = Exp.linacce_y  # conversion from ENU to NED
        self.LinAcc.z = Exp.linacce_z  # conversion from ENU to NED

        self.Grv.x = Exp.grav_x
        self.Grv.y = Exp.grav_y  # conversion from ENU to NED
        self.Grv.z = Exp.grav_z  # conversion from ENU to NED

        self.Frame = 'ENU'
    def save_csv(self, new_path):
        quat = self.Rot.as_quat()
        # create dictionary
        dic = {'time': self.Time_IMU * 1e9,
               'gyro_x': self.Gyro.x,
               'gyro_y': self.Gyro.y,
               'gyro_z': self.Gyro.z,
               'acce_x': self.Acc.x,
               'acce_y': self.Acc.y,
               'acce_z': self.Acc.z,
               'linacce_x': self.LinAcc.x,
               'linacce_y': self.LinAcc.y,
               'linacce_z': self.LinAcc.z,
               'grav_x': self.Grv.x,
               'grav_y': self.Grv.y,
               'grav_z': self.Grv.z,
               'magnet_x': self.Mag.x,
               'magnet_y': self.Mag.y,
               'magnet_z': self.Mag.z,
               'pos_x': self.Pos.x,
               'pos_y': self.Pos.y,
               'pos_z': self.Pos.z,
               'ori_w': quat[:, 3],
               'ori_x': quat[:, 0],
               'ori_y': quat[:, 1],
               'ori_z': quat[:, 2],
               }
        Functions.save_csv(dic=dic, path=new_path, print_message=True)
        head, tail = ntpath.split(new_path)
        self.Path = head
        self.FileName = tail


class AI_PDR_exp_w_SP_GT(AhrsExp):
    """
    this class is for recordings made by china team with SLAM pose for IMU phone
    """

    def __init__(self, path=None):
        super(AI_PDR_exp_w_SP_GT, self).__init__()
        if path is None:
            curr_directory = os.getcwd()  # will get current working directory
            path = tkinter.filedialog.askopenfilename(initialdir=curr_directory, title="Select A File",
                                                      filetype=(("csv files", "*.csv"), ("all files", "*.*")))
        Exp = pd.read_csv(path)
        head, tail = ntpath.split(path)
        self.Path = head
        self.FileName = tail
        time = np.array(Exp.time)
        self.Time_IMU = (time - time[0]) * 1e-9
        self.Time_GT = (time - time[0]) * 1e-9
        self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
        self.NumberOfSamples_GT = self.Time_GT.shape[0]
        self.IMU_valid_idx = list(range(self.NumberOfSamples_IMU))
        self.dt = np.mean(np.diff(self.Time_IMU))

        self.Pos.x = Exp.pos_x
        self.Pos.y = Exp.pos_y
        self.Pos.z = Exp.pos_z

        self.Pos_IMU = Vec3d()
        self.Pos_IMU.x = Exp.imu_pos_x
        self.Pos_IMU.y = Exp.imu_pos_y
        self.Pos_IMU.z = Exp.imu_pos_z

        QuatArray = np.array(
            [Exp.imu_ori_x, Exp.imu_ori_y, Exp.imu_ori_z, Exp.imu_ori_w]).T  # size is nX4 Transfered to ned
        self.Rot = Rotation.from_quat(QuatArray)
        EulerArray = self.Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        self.Psi = Functions.ContinuousAngle(EulerArray[:, 0])
        self.initial_heading = self.Psi[0]
        self.Theta = Functions.ContinuousAngle(EulerArray[:, 1])
        self.Phi = Functions.ContinuousAngle(EulerArray[:, 2])

        self.Gyro.x = Exp.gyro_x
        self.Gyro.y = Exp.gyro_y
        self.Gyro.z = Exp.gyro_z

        self.Acc.x = Exp.acce_x
        self.Acc.y = Exp.acce_y
        self.Acc.z = Exp.acce_z

        self.Frame = 'ENU'

    def PlotPosition(self, temporal=False, XY=True):
        if temporal:
            fig = plt.figure('Position Temporal Plot')
            Ax_pos_x = fig.add_subplot(311)
            Ax_pos_x.plot(self.Time_GT, self.Pos.x, color='blue', linewidth=1)
            Ax_pos_x.plot(self.Time_GT, self.Pos_IMU.x, color='red', linewidth=1)
            Ax_pos_x.set_xlim(self.Time_GT[0], self.Time_GT[-1])
            Ax_pos_x.set(title=r"Position", ylabel="$[m]$")
            Ax_pos_x.grid(True)

            Ax_pos_y = fig.add_subplot(312, sharex=Ax_pos_x)
            Ax_pos_y.plot(self.Time_GT, self.Pos.y, color='blue', linewidth=1)
            Ax_pos_y.plot(self.Time_GT, self.Pos_IMU.y, color='red', linewidth=1)
            Ax_pos_y.set_xlim(self.Time_GT[0], self.Time_GT[-1])
            Ax_pos_y.set(ylabel="$[m]$")
            Ax_pos_y.grid(True)

            Ax_pos_z = fig.add_subplot(313, sharex=Ax_pos_x)
            Ax_pos_z.plot(self.Time_GT, self.Pos.z, color='blue', linewidth=1)
            Ax_pos_z.plot(self.Time_GT, self.Pos_IMU.z, color='red', linewidth=1)
            Ax_pos_z.set_xlim(self.Time_GT[0], self.Time_GT[-1])
            Ax_pos_z.set(xlabel="Time [sec]", ylabel="$[m]$")
            Ax_pos_z.grid(True)
        if XY:
            fig = plt.figure('Position XY Plot')
            Ax = fig.add_subplot(111)
            Ax.plot(self.Pos.x, self.Pos.y, color='blue', linewidth=1)
            Ax.plot(self.Pos_IMU.x, self.Pos_IMU.y, color='red', linewidth=1)
            Ax.set(xlabel="$[m]$", ylabel="$[m]$")
            Ax.grid(True)
            Ax.axis('equal')


class SbgExpRawData(AhrsExp):
    def __init__(self, path=None, GT=None):
        super(SbgExpRawData, self).__init__()
        if path is None:
            curr_directory = os.getcwd()  # will get current working directory
            path = tkinter.filedialog.askopenfilename(initialdir=curr_directory, title="Select A File",
                                                      filetype=(("csv files", "*.csv"), ("all files", "*.*")))
        head, tail = ntpath.split(path)
        self.Path = head
        self.FileName = tail
        Exp = pd.read_csv(path)

        if GT is None:
            GT_path = os.path.split(path)[:-1][0] + "/ascii-output.txt"
            GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
        ind = range(0, len(GT))
        sec = GT['GPS Time'].values[ind]
        Timestamp = Exp['gyroTimestamp'].values / 1000  # traeted as IMU time vector
        IMU_idx = np.array(range(Timestamp.shape[0]))
        # Timestamp_date = datetime.fromtimestamp(Timestamp)
        date = datetime.strptime(GT['UTC Date'].values[0], "%Y-%m-%d")
        (gpsweek, gpsseconds) = Functions.utctoweekseconds(date, 18)
        self.Unixtime = Functions.gpsTimeToUnixTime(gpsweek, sec)
        self.first_idx_of_time_gap_IMU = None
        self.first_idx_of_time_gap_GPS = None
        IMU_valid_idx, GPS_valid_idx = self.validate_data_using_time_vectors(time_IMU=Timestamp,
                                                                             time_GPS=self.Unixtime,
                                                                             IMU_time_gap_th=0.02,
                                                                             GPS_time_gap_th=0.1,
                                                                             min_length_th=10)

        if self.valid:
            t0 = Timestamp[
                0]  # this value is used for initializing the time vectors so it would be consistent with the AHRS results files
            Timestamp = Timestamp[IMU_valid_idx]
            self.IMU_valid_idx = IMU_valid_idx
            self.Gyro.x = Exp.gyroX[IMU_valid_idx]
            self.Gyro.y = Exp.gyroY[IMU_valid_idx]
            self.Gyro.z = Exp.gyroZ[IMU_valid_idx]

            self.Acc.x = Exp.accX[IMU_valid_idx]
            self.Acc.y = Exp.accY[IMU_valid_idx]
            self.Acc.z = Exp.accZ[IMU_valid_idx]
            self.Unixtime = self.Unixtime[GPS_valid_idx]
            self.GPS_valid_idx = GPS_valid_idx
            x_ECEF, y_ECEF, z_ECEF = Functions.LLA2ECEF(lat=GT['Latitude'].values[GPS_valid_idx],
                                                        lon=GT['Longitude'].values[GPS_valid_idx],
                                                        alt=GT['Altitude MSL'].values[GPS_valid_idx])
            ECEF_arr = np.vstack([x_ECEF, y_ECEF, z_ECEF]).T
            DCM = Functions.DCM_ECEF2NED(Long=GT['Longitude'].values[ind][0],
                                         Lat=GT['Latitude'].values[ind][0])
            n_e_d = np.dot(DCM, ECEF_arr.T)
            Pn = n_e_d[0, :].squeeze()
            Pe = n_e_d[1, :].squeeze()
            Pd = n_e_d[2, :].squeeze()
            # Pn, Pe, Pd = Functions.LLLN2NED(GT['Latitude'].values[ind], GT['Longitude'].values[ind], GT['Altitude MSL'].values[ind],
            #                       RN, RM)
            Pn = np.interp(Timestamp, self.Unixtime, Pn)
            Pe = np.interp(Timestamp, self.Unixtime, Pe)
            Pd = np.interp(Timestamp, self.Unixtime, Pd)

            pos = np.array([Pe, Pn, -Pd]).T  # NED to ENU conversion
            self.Pos.x = pos[:, 0] - pos[1, 0]
            self.Pos.y = pos[:, 1] - pos[1, 1]
            self.Pos.z = pos[:, 2] - pos[1, 2]

            SBG_Yaw = np.array(GT['Yaw'] * np.pi / 180)
            SBG_Yaw = Functions.ContinuousAngle(SBG_Yaw)
            self.SBG_yaw = np.interp(Timestamp, self.Unixtime, SBG_Yaw[GPS_valid_idx])

            head, tail = ntpath.split(path)
            self.Path = head
            self.FileName = tail
            self.Time_IMU = Timestamp - t0
            self.Time_GT = Timestamp - t0
            self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
            self.NumberOfSamples_GT = self.Time_GT.shape[0]
            self.dt = np.mean(np.diff(self.Time_IMU))

            ori_nonzero = Exp.orientationTimestamp.to_numpy().nonzero()
            self.ori_time = np.array(Exp.orientationTimestamp.iloc[ori_nonzero]).T / 1000  # ori_time != Time_IMU

            Psi = - np.array(Exp.azimut.iloc[ori_nonzero]).T * np.pi / 180
            Phi = - np.array(Exp.pitch.iloc[ori_nonzero]).T * np.pi / 180
            Theta = - np.array(Exp.roll.iloc[ori_nonzero]).T * np.pi / 180
            """notice pitch and roll are switched"""
            self.Psi = Functions.ContinuousAngle(Psi)
            self.initial_heading = self.Psi[0]
            self.Theta = Functions.ContinuousAngle(Theta)
            self.Phi = Functions.ContinuousAngle(Phi)
            """these are the Euler angles estimations of the android OS. this estimation is called orientation angles.
            this sensor uses magnetometers for heading estimation."""
            interpulate_to_IMU_time = True
            if interpulate_to_IMU_time:
                self.Psi = np.interp(Timestamp, self.ori_time, self.Psi)
                self.Theta = np.interp(Timestamp, self.ori_time, self.Theta)
                self.Phi = np.interp(Timestamp, self.ori_time, self.Phi)

            euler_array = np.vstack([self.Psi, self.Theta, self.Phi]).T
            self.Rot = Rotation.from_euler('ZYX', euler_array, degrees=False)
            '''notice we use android orientation angles for the self.Rot property. this is later used in AAE to 
            initialize the rotation matrix. since this data was colected outdoors  the magnetometers data is valid and 
            gives a rough estimate of the heading angle.'''
            '''capital letters in the rotation sequence reffers to Intrinsic rotaions.'''
            self.heading_fix = 0.0
            '''this property is an angle in radians to fix the heading and rotate the hole estimated trajectory'''
            use_RV = False
            if use_RV:
                rv_nonzero = Exp.RV_timestamp.to_numpy().nonzero()
                self.rv_time = np.array(Exp.RV_timestamp.iloc[rv_nonzero]).T / 1000  # ori_time != Time_IMU
                rv_qx = np.array(Exp.RV_qx.iloc[rv_nonzero]).T
                rv_qy = np.array(Exp.RV_qy.iloc[rv_nonzero]).T
                rv_qz = np.array(Exp.RV_qz.iloc[rv_nonzero]).T
                rv_qw = np.array(Exp.RV_qw.iloc[rv_nonzero]).T
                self.QuatArray = np.array([rv_qx, rv_qy, rv_qz, rv_qw]).T  # size is nX4

            else:
                grv_nonzero = Exp.GRV_timestamp.to_numpy().nonzero()
                self.grv_time = np.array(Exp.GRV_timestamp.iloc[grv_nonzero]).T / 1000  # ori_time != Time_IMU
                grv_qx = np.array(Exp.GRV_qx.iloc[grv_nonzero]).T
                grv_qy = np.array(Exp.GRV_qy.iloc[grv_nonzero]).T
                grv_qz = np.array(Exp.GRV_qz.iloc[grv_nonzero]).T
                grv_qw = np.array(Exp.GRV_qw.iloc[grv_nonzero]).T
                self.QuatArray = np.array([grv_qx, grv_qy, grv_qz, grv_qw]).T  # size is nX4

            # Rot = Rotation.from_quat(QuatArray)
            # EulerArray = Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
            # Psi = EulerArray[:, 0]
            # Theta = EulerArray[:, 1]
            # Phi = EulerArray[:, 2]
            # Psi = Functions.ContinuousAngle(Psi)
            # Theta = Functions.ContinuousAngle(Theta)
            # Phi = Functions.ContinuousAngle(Phi)
            # self.Psi = np.interp(Timestamp, grv_time, Psi)
            # self.Theta = np.interp(Timestamp, grv_time, Theta)
            # self.Phi = np.interp(Timestamp, grv_time, Phi)
            # euler_array = np.vstack([self.Psi, self.Theta, self.Phi]).T
            # self.Rot = Rotation.from_euler('zyx', euler_array, degrees=False)
            """this is another estimation by the android OS called game rotation vector (GRV). 
            In order to use as transformation, an interpulation to the IMU time is needed. bare in mind that interpulating 
             quaternions/martices doesn't work. Therefore, convert to euler angles->interpulate->convert to rotations."""
            gn = np.array([0, 0, 1])
            grv_arr = Functions.transform_vectors(gn, self.Rot)
            self.Grv.x = grv_arr[:, 0]
            self.Grv.y = grv_arr[:, 1]
            self.Grv.z = grv_arr[:, 2]

            self.LinAcc.x = self.Acc.x - self.Grv.x
            self.LinAcc.y = self.Acc.y - self.Grv.y
            self.LinAcc.z = self.Acc.z - self.Grv.z

            self.Frame = 'ENU'
        else:
            print(join(self.Path, self.FileName) + ' is not valid')


class AndroidExp(AhrsExp):
    def __init__(self, path=None, initialdir=None):
        super(AndroidExp, self).__init__()
        if path is None:
            if initialdir is None:
                curr_directory = os.getcwd()  # will get current working directory
                path = tkinter.filedialog.askopenfilename(initialdir=curr_directory, title="Select A File")
            else:
                path = tkinter.filedialog.askopenfilename(initialdir=initialdir, title="Select A File")
        Exp = pd.read_csv(path)

        Timestamp = Exp['gyroTimestamp'].values / 1000
        # Timestamp_date = datetime.fromtimestamp(Timestamp)

        head, tail = ntpath.split(path)
        self.Path = head
        self.FileName = tail
        self.Time_IMU = (Timestamp - Timestamp[0])
        self.Time_GT = (Timestamp - Timestamp[0])
        self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
        self.NumberOfSamples_GT = self.Time_GT.shape[0]
        self.dt = np.mean(np.diff(self.Time_IMU))

        ori_nonzero = Exp.orientationTimestamp.to_numpy().nonzero()
        self.ori_time = np.array(Exp.orientationTimestamp.iloc[ori_nonzero]).T / 1000  # ori_time != Time_IMU

        Psi = - np.array(Exp.azimut.iloc[ori_nonzero]).T * np.pi / 180
        Psi = Psi - Psi[0]
        Phi = - np.array(Exp.pitch.iloc[ori_nonzero]).T * np.pi / 180
        Theta = - np.array(Exp.roll.iloc[ori_nonzero]).T * np.pi / 180
        """notice pitch and roll are switched"""
        self.Psi = Functions.ContinuousAngle(Psi)
        self.Theta = Functions.ContinuousAngle(Theta)
        self.Phi = Functions.ContinuousAngle(Phi)
        interpulate_to_IMU_time = True
        if interpulate_to_IMU_time:
            self.Psi = np.interp(Timestamp, self.ori_time, self.Psi)
            self.Theta = np.interp(Timestamp, self.ori_time, self.Theta)
            self.Phi = np.interp(Timestamp, self.ori_time, self.Phi)

        euler_array = np.vstack([self.Psi, self.Theta, self.Phi]).T
        self.Rot = Rotation.from_euler('ZYX', euler_array, degrees=False)
        '''capital letters in the rotation sequence reffers to Intrinsic rotaions.'''

        grv_nonzero = Exp.GRV_timestamp.to_numpy().nonzero()
        self.grv_time = np.array(Exp.GRV_timestamp.iloc[grv_nonzero]).T / 1000  # ori_time != Time_IMU
        grv_qx = np.array(Exp.GRV_qx.iloc[grv_nonzero]).T
        grv_qy = np.array(Exp.GRV_qy.iloc[grv_nonzero]).T
        grv_qz = np.array(Exp.GRV_qz.iloc[grv_nonzero]).T
        grv_qw = np.array(Exp.GRV_qw.iloc[grv_nonzero]).T

        self.QuatArray = np.array([grv_qx, grv_qy, grv_qz, grv_qw]).T  # size is nX4


        # Rot = Rotation.from_quat(QuatArray)
        # EulerArray = Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        # Psi = EulerArray[:, 0]
        # Theta = EulerArray[:, 1]
        # Phi = EulerArray[:, 2]
        # Psi = Functions.ContinuousAngle(Psi)
        # Theta = Functions.ContinuousAngle(Theta)
        # Phi = Functions.ContinuousAngle(Phi)
        # self.Psi = np.interp(Timestamp, grv_time, Psi)
        # self.Theta = np.interp(Timestamp, grv_time, Theta)
        # self.Phi = np.interp(Timestamp, grv_time, Phi)
        # euler_array = np.vstack([self.Psi, self.Theta, self.Phi]).T
        # self.Rot = Rotation.from_euler('zyx', euler_array, degrees=False)
        self.Pos.x = np.zeros([self.NumberOfSamples_GT, 1])
        self.Pos.y = np.zeros([self.NumberOfSamples_GT, 1])
        self.Pos.z = np.zeros([self.NumberOfSamples_GT, 1])
        gn = np.array([0, 0, 1])
        grv_arr = Functions.transform_vectors(gn, self.Rot)
        self.Grv.x = grv_arr[:, 0]
        self.Grv.y = grv_arr[:, 1]
        self.Grv.z = grv_arr[:, 2]

        self.Gyro.x = Exp.gyroX
        self.Gyro.y = Exp.gyroY
        self.Gyro.z = Exp.gyroZ

        self.Acc.x = Exp.accX
        self.Acc.y = Exp.accY
        self.Acc.z = Exp.accZ

        # self.LinAcc.x = self.Acc.x - self.Grv.x
        # self.LinAcc.y = self.Acc.y - self.Grv.y
        # self.LinAcc.z = self.Acc.z - self.Grv.z

        self.Frame = 'ENU'


class AI_IMU_DS_Exp(AhrsExp):
    def __init__(self, path=None, initial_dir='/home/maint/Eran/AHRS/AI_dataset/DataForAiIntegration/'):
        super(AI_IMU_DS_Exp, self).__init__()
        if path is None:
            if not initial_dir == None:
                path = tkinter.filedialog.askdirectory(initialdir=initial_dir, title="Select an experiment")
            else:
                curr_directory = os.getcwd()  # will get current working directory
                path = tkinter.filedialog.askdirectory(initialdir=curr_directory, title="Select an experiment")
        self.Path = path
        # acc
        acc_file = open(join(path, 'Record_ACC.txt'), "r")
        lines = acc_file.readlines()
        acc_time = []
        acc_x = []
        acc_y = []
        acc_z = []
        num_of_valid_lines = 0
        for line in lines:
            words = line.split()
            if len(words) < 4:
                break
            num_of_valid_lines += 1
            acc_time.append(float(words[0]))
            acc_x.append(float(words[1]))
            acc_y.append(float(words[2]))
            acc_z.append(float(words[3]))
        acc_time = np.array(acc_time[:num_of_valid_lines])
        acc_time = (acc_time - acc_time[0]) * (10 ** -9)
        # gyro
        gyro_file = open(join(path, 'Record_GYRO.txt'), "r")
        lines = gyro_file.readlines()
        gyro_time = []
        gyro_x = []
        gyro_y = []
        gyro_z = []
        num_of_valid_lines = 0
        for line in lines:
            words = line.split()
            if len(words) < 4:
                break
            num_of_valid_lines += 1
            gyro_time.append(float(words[0]))
            gyro_x.append(float(words[1]))
            gyro_y.append(float(words[2]))
            gyro_z.append(float(words[3]))
        gyro_time = np.array(gyro_time[:num_of_valid_lines])
        gyro_time = (gyro_time - gyro_time[0]) * (10 ** -9)
        if acc_time.shape[0] != gyro_time.shape[0]:
            self.NumberOfSamples_IMU = min(
                [acc_time.shape[0], gyro_time.shape[0]])  # gyro and acc are asynchronized ...
            if acc_time.shape[0] > gyro_time.shape[0]:
                acc_x.pop()
                acc_y.pop()
                acc_z.pop()
                self.Time_IMU = gyro_time
            else:
                gyro_x.pop()
                gyro_y.pop()
                gyro_z.pop()
                self.Time_IMU = acc_time
        else:
            self.Time_IMU = acc_time
            self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
        self.Acc.x = np.array(acc_x[:self.NumberOfSamples_IMU])
        self.Acc.y = np.array(acc_y[:self.NumberOfSamples_IMU])
        self.Acc.z = np.array(acc_z[:self.NumberOfSamples_IMU])
        self.Acc_bias = Vec3d()
        self.Acc_bias = np.array([0.0, 0.0, 0.0])
        self.dt = np.mean(np.diff(self.Time_IMU))
        self.Gyro.x = np.array(gyro_x[:self.NumberOfSamples_IMU])
        self.Gyro.y = np.array(gyro_y[:self.NumberOfSamples_IMU])
        self.Gyro.z = np.array(gyro_z[:self.NumberOfSamples_IMU])
        # slam
        slam_file = open(join(path, 'SlamPose_Twi.txt'), "r")
        lines = slam_file.readlines()
        slam_time = []
        slam_px = []
        slam_py = []
        slam_pz = []
        slam_qx = []
        slam_qy = []
        slam_qz = []
        slam_qw = []
        for line in lines:
            words = line.split()
            slam_time.append(float(words[0]))
            slam_px.append(float(words[1]))
            slam_py.append(float(words[2]))
            slam_pz.append(float(words[3]))
            slam_qx.append(float(words[4]))
            slam_qy.append(float(words[5]))
            slam_qz.append(float(words[6]))
            slam_qw.append(float(words[7]))
        slam_time = np.array(slam_time)
        self.Time_GT = (slam_time - slam_time[0])
        self.NumberOfSamples_GT = self.Time_GT.shape[0]
        self.Pos.x = np.array(slam_px)
        self.Pos.y = np.array(slam_py)
        self.Pos.z = np.array(slam_pz)
        slam_qx = np.array(slam_qx)
        slam_qy = np.array(slam_qy)
        slam_qz = np.array(slam_qz)
        slam_qw = np.array(slam_qw)
        QuatArray = np.vstack([slam_qx, slam_qy, slam_qz, slam_qw]).T  # size is nX4 Transfered to ned
        self.Rot = Rotation.from_quat(QuatArray)
        EulerArray = self.Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        self.Psi = EulerArray[:, 0]
        self.Theta = EulerArray[:, 1]
        self.Phi = EulerArray[:, 2]
        self.SLAM_time_offset = 0.0
        # Psi = EulerArray[:, 0]
        # self.Psi = numpy.interp(x=self.Time, xp=self.Time_GT, fp=Psi)
        # Theta = EulerArray[:, 1]
        # self.Theta = numpy.interp(x=self.Time, xp=self.Time_GT, fp=Theta)
        # Phi = EulerArray[:, 2]
        # self.Phi = numpy.interp(x=self.Time, xp=self.Time_GT, fp=Phi)

        self.Frame = 'ENU'

    def load_offset(self, path='/home/maint/git/ahrs/exp_offsets.json'):
        with open(path) as f:
            offsets = json.load(f)
        offsets_item, _ = Functions.search_list_of_dict(offsets, 'path', self.Path)
        self.SLAM_time_offset = offsets_item["offset"]

    def load_static_periods(self):
        files_in_dir = listdir(self.Path)
        iterator = (file for file in files_in_dir if file[:14] == 'static_periods')
        file_name = next(iterator, None)
        if not file_name == None:
            file_path = join(self.Path, file_name)
            with open(file_path, "r") as file:
                self.QuietPeriods = json.load(file)
        else:
            print('didnt find any file')

    def load_acc_bias(self):
        """returns success/fail [True/False]"""
        files_in_dir = listdir(self.Path)
        iterator = (file for file in files_in_dir if file[:8] == 'acc bias')
        acc_bias_calc_file_name = next(iterator, None)
        acc_files = {}
        if not acc_bias_calc_file_name == None:
            acc_files[acc_bias_calc_file_name] = os.path.getmtime(join(self.Path, acc_bias_calc_file_name))
            # info_message = "found existing acc bias calculation"
            # print(info_message)
            with open(join(self.Path, acc_bias_calc_file_name), 'r') as acc_bias_file:
                lines = acc_bias_file.readlines()
                # turn string to number ...
                # todo: change format to something more convenient...
                S = ''
                for s in lines[1]:
                    if not (s == '[' or s == ']'):
                        S = S + s
            self.Acc_bias = np.array([float(S.split()[0]), float(S.split()[1]), float(S.split()[2])])
            # todo: prepare while loop and handle case of many acc files
            return True
        else:
            # print('didnt find any file')
            return False

    def writ_calib_param_to_file(self):
        calib_param = {}
        calib_param["SLAM_time_offset"] = self.SLAM_time_offset
        calib_param["acc_bias"] = self.Acc_bias
        calib_param["QuietPeriods"] = self.QuietPeriods
        now = datetime.isoformat(datetime.now())
        file_name = 'calibration_parameters' + now + '.json'
        path = join(self.Path, file_name)
        with open(path, "w") as outfile:
            json.dump(calib_param, outfile, indent=4)

    def load_calib_param_from_file(self):
        files_in_dir = listdir(self.Path)
        iterator = (file for file in files_in_dir if file[:21] == 'calibration_parameters')
        calib_param_file_name = next(iterator, None)
        acc_files = {}
        if not calib_param_file_name == None:
            with open(join(self.Path, calib_param_file_name)) as f:
                calib_param = json.load(f)
            # todo: prepare while loop and handle case of many calib files
        else:
            print('didnt find any file')

        keys_in_calib_param = list(calib_param.keys())
        iterator = (True for key in keys_in_calib_param if key == 'SLAM_time_offset')
        offset_exist = next(iterator, None)
        iterator = (True for key in keys_in_calib_param if key == 'acc_bias')
        acc_bias_exist = next(iterator, None)
        iterator = (True for key in keys_in_calib_param if key == 'QuietPeriods')
        QuietPeriods_exist = next(iterator, None)

        if offset_exist:
            self.SLAM_time_offset = calib_param["SLAM_time_offset"]
        if acc_bias_exist:
            self.Acc_bias = calib_param["acc_bias"]
        if QuietPeriods_exist:
            self.QuietPeriods = calib_param["QuietPeriods"]

    def apply_offset(self):
        self.Time_IMU = self.Time_IMU + self.SLAM_time_offset
        self.SegmentScenario([self.Time_GT[0], self.Time_GT[-1]])
        self.SLAM_time_offset = 0


class BroadExp(AhrsExp):
    def __init__(self, path=None):
        super(BroadExp, self).__init__()

        if path is None:
            curr_directory = os.getcwd()  # will get current working directory
            path = tkinter.filedialog.askopenfilename(initialdir=curr_directory, title="Select A File",
                                                      filetype=(("csv files", "*.csv"), ("all files", "*.*")))

        head, tail = ntpath.split(path)
        self.Path = head
        self.FileName = tail

        with h5py.File(path, 'r') as f:
            acc = f['imu_acc'][:]
            gyr = f['imu_gyr'][:]
            mag = f['imu_mag'][:]
            ref_quat = f['opt_quat'][:]
            pos = f['opt_pos'][:]
            self.dt = 1 / f.attrs['sampling_rate']

        self.NumberOfSamples_IMU = acc.shape[0]
        self.Time_IMU = np.linspace(0, self.dt * self.NumberOfSamples_IMU, num=self.NumberOfSamples_IMU)

        self.Pos.x = pos[:, 0]
        self.Pos.y = pos[:, 1]
        self.Pos.z = pos[:, 2]

        QuatArray = np.vstack((-ref_quat[:, 3], -ref_quat[:, 0], ref_quat[:, 1], ref_quat[:, 2])).T
        nan_idx = np.argwhere(np.isnan(QuatArray))
        att_measurement_valid_idx = [True for i in range(self.NumberOfSamples_IMU)]
        for idx in nan_idx[:, 0]:
            att_measurement_valid_idx[idx] = False

        self.valid_quat_idx = np.argwhere(att_measurement_valid_idx).squeeze()
        self.Rot = Rotation.from_quat(QuatArray[self.valid_quat_idx, :])
        EulerArray = np.zeros((self.NumberOfSamples_IMU, 3))
        EulerArray[self.valid_quat_idx] = self.Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        self.Psi = EulerArray[:, 0]
        self.Psi[self.valid_quat_idx] = Functions.ContinuousAngle(self.Psi[self.valid_quat_idx], 'rad')
        self.Theta = EulerArray[:, 1]
        self.Theta[self.valid_quat_idx] = Functions.ContinuousAngle(self.Theta[self.valid_quat_idx], 'rad')
        self.Phi = EulerArray[:, 2]
        self.Phi[self.valid_quat_idx] = Functions.ContinuousAngle(self.Phi[self.valid_quat_idx], 'rad')

        self.Gyro.x = gyr[:, 0]
        self.Gyro.y = gyr[:, 1]
        self.Gyro.z = gyr[:, 2]

        self.Acc.x = acc[:, 0]
        self.Acc.y = acc[:, 1]
        self.Acc.z = acc[:, 2]

        self.Mag.x = mag[:, 0]
        self.Mag.y = mag[:, 1]
        self.Mag.z = mag[:, 2]

        self.Frame = 'ENU'


class Euler_angles_plot():
    def __init__(self):
        # print('Euler_angles_plot instance initialized')
        super(Euler_angles_plot, self).__init__()
        self.plot_errors = True
        self.plot_heading = False
        self.plot_values = True
        self.ref = {"time": None, "pitch": None, "roll": None, "heading": None}
        self.estimates = {}  # dictionary [keys: est_names] [values: dictionaries,  [est_name][keys: time, pitch, roll, heading]]
        self.fig = None
        self.axes = []
        self.num_of_estimates = 0
        self.ref_added = False
        self.linewidth = 1

    def add_ref(self, time, roll, pitch, heading):
        self.ref["time"] = time
        self.ref["pitch"] = pitch
        self.ref["roll"] = roll
        self.ref["heading"] = heading
        self.ref_added = True

    def add_est(self, name, time, roll, pitch, heading):
        self.estimates[name] = {"time": time, "pitch": pitch, "roll": roll, "heading": heading,
                                "errors_calculated": False}
        self.num_of_estimates += 1

    def calc_est_err(self, est):
        assert self.ref_added, "can't calculate errors without references"
        # assert self.ref["roll"].shape == self.estimates[est]["roll"].shape
        # assert self.ref["pitch"].shape == self.estimates[est]["pitch"].shape
        # assert self.ref["heading"].shape == self.estimates[est]["heading"].shape
        if self.ref["time"].shape == self.estimates[est]["time"].shape:
            self.estimates[est]["roll_error"] = self.ref["roll"] - self.estimates[est]["roll"]
            self.estimates[est]["pitch_error"] = self.ref["pitch"] - self.estimates[est]["pitch"]
            self.estimates[est]["heading_error"] = self.ref["heading"] - self.estimates[est]["heading"]
            self.estimates[est]["errors_calculated"] = True
        else:
            # interpulate estimates to GT time samples
            psi = self.ref["heading"]
            psi_hat = self.estimates[est]["heading"]
            theta = self.ref["pitch"]
            theta_hat = self.estimates[est]["pitch"]
            phi = self.ref["roll"]
            phi_hat = self.estimates[est]["roll"]
            t_ref = self.ref["time"]
            t_est = self.estimates[est]["time"]

            psi_hat_interp = np.interp(x=t_ref, xp=t_est, fp=psi_hat)
            theta_hat_interp = np.interp(x=t_ref, xp=t_est, fp=theta_hat)
            phi_hat_interp = np.interp(x=t_ref, xp=t_est, fp=phi_hat)
            self.estimates[est]["roll_error"] = phi - phi_hat_interp
            self.estimates[est]["pitch_error"] = theta - theta_hat_interp
            self.estimates[est]["heading_error"] = psi - psi_hat_interp
            self.estimates[est]["errors_calculated"] = True

    def plot_fig(self):
        # plt.close('Euler Angles Plot')
        self.fig = plt.figure('Euler Angles Plot')
        assert self.plot_errors or self.plot_values, "nothing to plot"

        if self.plot_errors and self.plot_values:
            n_cols = 2
        else:
            n_cols = 1

        if self.plot_heading:
            n_rows = 3
        else:
            n_rows = 2
        for i in range(n_rows * n_cols):
            if i == 0:
                self.axes.append(self.fig.add_subplot(n_rows, n_cols, i + 1))
            else:
                self.axes.append(self.fig.add_subplot(n_rows, n_cols, i + 1, sharex=self.axes[0]))

        N = self.num_of_estimates
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        color_list = list(RGB_tuples)

        # roll values
        if self.plot_values:
            ax_idx = 0
            self.axes[ax_idx].set(ylabel=r"$\phi$ [deg]"), self.axes[ax_idx].grid(True)
            # plot ref
            assert (self.ref["time"] is not None) and (self.ref["roll"] is not None)
            self.axes[ax_idx].plot(self.ref["time"], self.ref["roll"] * 180 / np.pi,
                                   color='black', linewidth=self.linewidth, label='Ref')
            # plot estimates
            if self.num_of_estimates > 0:
                estimator_names = list(self.estimates.keys())
                for i in range(self.num_of_estimates):
                    self.axes[ax_idx].plot(self.estimates[estimator_names[i]]["time"],
                                           self.estimates[estimator_names[i]]["roll"] * 180 / np.pi,
                                           color=color_list[i], linewidth=self.linewidth, label=estimator_names[i])

            self.axes[ax_idx].legend()
        # roll errors
        if self.plot_errors:
            if self.plot_values:
                ax_idx = 1
            else:
                ax_idx = 0
            if self.plot_values:
                self.axes[ax_idx].set(title="errors"), self.axes[ax_idx].grid(True)
            else:
                self.axes[ax_idx].set(ylabel=r"$\phi$ [deg]"), self.axes[ax_idx].grid(True)
            # plot estimates errors
            if self.num_of_estimates > 0:
                estimator_names = list(self.estimates.keys())
                for i in range(self.num_of_estimates):
                    assert self.estimates[estimator_names[i]]["errors_calculated"], "errors not calculated"
                    self.axes[ax_idx].plot(self.ref["time"],
                                           self.estimates[estimator_names[i]]["roll_error"] * 180 / np.pi,
                                           color=color_list[i], linewidth=self.linewidth, label=estimator_names[i])
            self.axes[ax_idx].legend()
            # self.axes[ax_idx].plot(self.estimates[estimator_names[i]]["time"],
            #                        self.estimates[estimator_names[i]]["roll_error"] * 180 / np.pi,
            #                        color=color_list[i], linewidth=self.linewidth)
        # pitch values
        if self.plot_values:
            if self.plot_errors:
                ax_idx = 2
            else:
                ax_idx = 1
            if self.plot_heading:
                self.axes[ax_idx].set(ylabel=r"$\theta$ [deg]"), self.axes[ax_idx].grid(True)
            else:
                self.axes[ax_idx].set(xlabel="[sec]", ylabel=r"$\theta$ [deg]"), self.axes[ax_idx].grid(True)
            # plot ref
            assert (self.ref["time"] is not None) and (self.ref["pitch"] is not None)
            self.axes[ax_idx].plot(self.ref["time"], self.ref["pitch"] * 180 / np.pi, color='black',
                                   linewidth=self.linewidth)
            # plot estimates
            if self.num_of_estimates > 0:
                estimator_names = list(self.estimates.keys())
                for i in range(self.num_of_estimates):
                    self.axes[ax_idx].plot(self.estimates[estimator_names[i]]["time"],
                                           self.estimates[estimator_names[i]]["pitch"] * 180 / np.pi,
                                           color=color_list[i], linewidth=self.linewidth)
        # pitch errors
        if self.plot_errors:
            if self.plot_values:
                ax_idx = 3
            else:
                ax_idx = 1
            if self.plot_heading:
                if self.plot_values:
                    self.axes[ax_idx].grid(True)
                else:
                    self.axes[ax_idx].set(ylabel=r"$\theta$ [deg]"), self.axes[ax_idx].grid(True)
            else:
                if self.plot_values:
                    self.axes[ax_idx].set(xlabel="[sec]"), self.axes[ax_idx].grid(True)
                else:
                    self.axes[ax_idx].set(ylabel=r"$\theta$ [deg]"), self.axes[ax_idx].grid(True)
            # plot estimates errors
            for i in range(self.num_of_estimates):
                assert self.estimates[estimator_names[i]]["errors_calculated"], "errors not calculated"
                self.axes[ax_idx].plot(self.ref["time"],
                                       self.estimates[estimator_names[i]]["pitch_error"] * 180 / np.pi,
                                       color=color_list[i], linewidth=self.linewidth)
        # heading values
        if self.plot_heading:
            if self.plot_values:
                if self.plot_errors:
                    ax_idx = 4
                else:
                    ax_idx = 2
                self.axes[ax_idx].set(xlabel="[sec]", ylabel=r"$\psi$ [deg]"), self.axes[ax_idx].grid(True)
                # plot ref
                assert (self.ref["time"] is not None) and (self.ref["heading"] is not None)
                self.axes[ax_idx].plot(self.ref["time"], self.ref["heading"] * 180 / np.pi, color='black',
                                       linewidth=self.linewidth)
                # plot estimates
                if self.num_of_estimates > 0:
                    estimator_names = list(self.estimates.keys())
                    for i in range(self.num_of_estimates):
                        self.axes[ax_idx].plot(self.estimates[estimator_names[i]]["time"],
                                               self.estimates[estimator_names[i]]["heading"] * 180 / np.pi,
                                               color=color_list[i], linewidth=self.linewidth)
            # heading errors
            if self.plot_errors:
                if self.plot_values:
                    ax_idx = 5
                else:
                    ax_idx = 2
                if self.plot_values:
                    self.axes[ax_idx].set(xlabel="[sec]"), self.axes[ax_idx].grid(True)
                else:
                    self.axes[ax_idx].set(xlabel="[sec]", ylabel=r"$\psi$ [deg]"), self.axes[ax_idx].grid(True)
                # plot estimates errors
                for i in range(self.num_of_estimates):
                    assert self.estimates[estimator_names[i]]["errors_calculated"], "errors not calculated"
                    self.axes[ax_idx].plot(self.ref["time"],
                                           self.estimates[estimator_names[i]]["heading_error"] * 180 / np.pi,
                                           color=color_list[i], linewidth=self.linewidth)
        # plt.show()
        return


class WDE_performance_analysis():
    def __init__(self, segment: AhrsExp,
                 use_GT_att=False, lin_acc_b_frame_est=None, grv_est=None, Rot_est=None, Heading_est=None,
                 dl_net=None, use_GT_dl=True, arc_length_for_dl=False):
        self.use_GT_att = use_GT_att
        self.use_GT_dl = use_GT_dl
        if not self.use_GT_att:
            assert lin_acc_b_frame_est is not None and \
                   Rot_est is not None and \
                   Heading_est is not None and \
                   grv_est is not None
        super(WDE_performance_analysis, self).__init__()
        self.segment = segment
        dP_angles, dP_vectors = self.segment.calc_walking_direction(window_size=1)
        self.dP_angles = Functions.ContinuousAngle(dP_angles, units='rad')
        self.dP_angles_variance = np.std(self.dP_angles)
        self.turn_identified = None
        self.dP_vectors = dP_vectors
        pos = self.segment.Pos.arr()
        overall_d_pos = pos[-1] - pos[0]
        self.WD_vector_GT = overall_d_pos[0:2]
        self.initial_WD_angle_GT = 0
        self.WD_angle_GT = Functions.FoldAngles(np.arctan2(self.WD_vector_GT[1], self.WD_vector_GT[0]))
        self.WD_angle_est = None
        self.WD_vector_est = None
        self.end_pos_est = None
        self.WD_error = None
        self.end_pos_error = None
        self.WD_est_method = None  # PCA or SM_heading
        self.est_traj = None
        self.window_size = len(Rot_est)
        self.WDE_model_path = None
        self.WDE_res18model = None
        self.PDR_net_model_path = None
        self.PDR_net_res18model = None
        self.dL_model_path = None
        self.dL_res18model = dl_net
        self.res18RVmodel = None
        if self.use_GT_att:
            self.lin_acc_b_frame = self.segment.LinAcc.arr()
            self.grv = self.segment.Grv.arr()
            self.Rot = self.segment.Rot
            self.Heading = self.segment.Psi
        else:  # use estimated attitude and heading
            self.lin_acc_b_frame = lin_acc_b_frame_est
            self.grv = grv_est
            self.Rot = Rot_est
            self.Heading = Heading_est
        self.Gyro = segment.Gyro
        if self.use_GT_dl:
            if arc_length_for_dl:  # use to train odometry solutions
                self.dL = np.linalg.norm(dP_vectors, axis=1).sum()
            else:
                dp = segment.Pos.arr()[-1] - segment.Pos.arr()[0]
                self.dL = np.linalg.norm(dp[0:2])
        else:  # run pdr_net:
            # gyro = segment.Gyro.arr()  # 200x3
            # acc = segment.Acc.arr()  # 200x3
            # IMU = np.hstack((gyro, acc))  # 200x6
            # IMU = IMU.T  # 6x200
            # input_pdr = torch.from_numpy(IMU[None, :, :]).float()  # torch.Size([1, 6, 250])
            # flag = torch.zeros(1)
            # if next(pdr_net.parameters()).is_cuda:
            #     input_pdr = input_pdr.cuda()
            #     flag = flag.cuda()  # ?
            # net_out = pdr_net(input_pdr, flag)
            # self.dL = np.array(net_out[0][0].cpu().detach().numpy())
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            lin_acc = self.lin_acc_b_frame
            lin_acc_n_frame = Functions.transform_vectors(lin_acc, self.Rot)  # window_sizeX3
            add_quat = True
            if add_quat:
                quat_array = self.Rot.as_quat()
                batch_of_quat = quat_array[:self.window_size, :]
                batch_of_a_nav_and_quat = np.hstack([lin_acc_n_frame, batch_of_quat])
                x = batch_of_a_nav_and_quat.reshape(1, self.window_size, 7)
            else:
                x = lin_acc_n_frame.reshape(1, self.window_size, 3)

            if self.dL_res18model is None:
                net = torch.load(self.dL_model_path)
            else:
                net = self.dL_res18model
            resent_input = Functions.PrepareInputForResnet18(x)
            resent_input.to(device)  # this cannot be done in operational function
            resent_input = resent_input.cuda()
            output = net(resent_input)
            self.dL = output.cpu().detach().numpy().squeeze()


    def PCA_direction_analysis(self, plot_results=False, use_GT_to_solve_amguity=True):
        lin_acc = self.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, self.Rot)
        if isinstance(self, SbgExpRawData):
            lin_acc_n_frame[:, 0:2] = Functions.rotate_trajectory(traj=lin_acc_n_frame[:, 0:2],
                                                                  alfa=self.segment.heading_fix)
        WD_vector_est = Functions.PCA_main_direction(lin_acc_n_frame[:, 0:2], plot_res=plot_results)
        if use_GT_to_solve_amguity:
            """for now avoiding +-180 ambiguity"""
            est_projected_on_GT = np.dot(WD_vector_est / np.linalg.norm(WD_vector_est),
                                         self.WD_vector_GT / np.linalg.norm(self.WD_vector_GT))
            self.WD_vector_est = est_projected_on_GT / np.linalg.norm(est_projected_on_GT) * WD_vector_est
        else:
            self.WD_vector_est = WD_vector_est
        self.WD_angle_est = Functions.FoldAngles(np.arctan2(self.WD_vector_est[1], self.WD_vector_est[0]))

        self.end_pos_est = self.WD_vector_est / np.linalg.norm(self.WD_vector_est) * self.dL
        self.est_traj = np.vstack([[0, 0], self.end_pos_est])
        if plot_results:
            origin = [lin_acc_n_frame[:, 0].mean(),
                      lin_acc_n_frame[:, 1].mean()]
            arr_length = np.linalg.norm(
                [lin_acc_n_frame[:, 0].std(), lin_acc_n_frame[:, 1].std()])
            v0 = np.array(origin)
            v1 = np.array(origin) + arr_length * self.WD_vector_GT
            Functions.draw_vector(v0, v1, c='green')
            plt.axis('equal')

            fig = plt.figure('PCA_direction_analysis pos plot')
            ax = fig.add_subplot(111)
            ax.axis('equal')
            ax.grid(True)
            ax.plot(self.segment.Pos.arr()[:, 0] - self.segment.Pos.arr()[0, 0],
                    self.segment.Pos.arr()[:, 1] - self.segment.Pos.arr()[0, 1], label='GT')
            ax.plot(np.array([0, self.end_pos_est[0]]),
                    np.array([0, self.end_pos_est[1]]), label='est')
            ax.legend()
        self.end_pos_error = np.linalg.norm(self.end_pos_est - self.WD_vector_GT)
        self.WD_est_method = 'PCA'

    def res18_dL_pred(self, plot_results=False, data_type='LinAcc', device='cpu', add_quat=False):
        """perform one inference of resnet18 """
        t = self.segment.Time_IMU.reshape(-1, 1)
        window_size = self.window_size  # 100 samples window gives best resutls

        lin_acc = self.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, self.Rot)  # window_sizeX3
        if add_quat:
            quat_array = self.Rot.as_quat()
            batch_of_quat = quat_array[:window_size, :]
            batch_of_a_nav_and_quat = np.hstack([lin_acc_n_frame, batch_of_quat])
            x = batch_of_a_nav_and_quat.reshape(1, window_size, 7)
        else:
            x = lin_acc_n_frame.reshape(1, window_size, 3)

        if self.dL_res18model is None:
            net = torch.load(self.dL_model_path)
        else:
            net = self.dL_res18model
        resent_input = Functions.PrepareInputForResnet18(x)
        resent_input.to(device)  # this cannot be done in operational function
        resent_input = resent_input.cuda()
        output = net(resent_input)
        self.dL = output.cpu().detach().numpy().squeeze()

    def PDR_net_V2_pred(self, plot_results=False, data_type='LinAcc', device='cpu', add_quat=False, add_dim=True,
                        convert_quat_to_rot6d=False):
        """perform one inference of resnet18 """
        t = self.segment.Time_IMU.reshape(-1, 1)
        window_size = self.window_size  # 100 samples window gives best resutls
        lin_acc = self.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(lin_acc, self.Rot)  # window_sizeX3
        if add_quat:
            if convert_quat_to_rot6d:
                m = self.Rot.as_matrix()
                r6d = Functions.rotation_matrix_to_r6d(m)[:window_size, :]
                x = np.hstack([lin_acc_n_frame, r6d]).reshape(1, window_size, 9)
            else:
                quat_array = self.Rot.as_quat()
                batch_of_quat = quat_array[:window_size, :]
                x = np.hstack([lin_acc_n_frame, batch_of_quat]).reshape(1, window_size, 7)
        else:
            x = lin_acc_n_frame.reshape(1, window_size, 3)

        if self.PDR_net_res18model is None:
            assert self.PDR_net_model_path is not None
            net = torch.load(self.PDR_net_model_path)
        else:
            assert self.PDR_net_res18model is not None
            net = self.PDR_net_res18model

        resent_input = Functions.PrepareInputForResnet18(x, add_dim=add_dim)
        resent_input = resent_input.to(device)
        output = net(resent_input)
        self.WD_vector_est = output.cpu().detach().numpy().squeeze()
        self.WD_angle_est = Functions.FoldAngles(np.arctan2(self.WD_vector_est[1], self.WD_vector_est[0]))
        self.end_pos_est = self.WD_vector_est
        self.est_traj = np.vstack([[0, 0], self.end_pos_est])
        self.dL = np.linalg.norm(self.WD_vector_est)

        if plot_results:
            fig = plt.figure('PDR_net_V2_pred')
            Ax = fig.add_subplot(111)
            Ax.plot(self.segment.Pos.arr()[:, 0] - self.segment.Pos.arr()[0, 0],
                    self.segment.Pos.arr()[:, 1] - self.segment.Pos.arr()[0, 1], label='GT')
            Ax.plot(self.est_traj[:, 0], self.est_traj[:, 1], label='Est')
            Ax.grid(True)
            Ax.legend()
            Ax.axis('equal')
        self.end_pos_error = np.linalg.norm(self.end_pos_est - self.WD_vector_GT)
        self.WD_est_method = 'PDR_net_V2_resnet18-' + data_type

    def res18RV_direction_pred(self, plot_results=False, device='cpu'):
        """perform one inference of resnet18RV """
        t = self.segment.Time_IMU.reshape(-1, 1)
        DCM_vec = self.Rot.as_matrix()  # Nx3x3
        window_size = self.window_size  # 100 samples window gives best resutls
        NumOfBatches = int(len(t) / window_size)

        pth_path = self.model_path
        LinAcc = self.lin_acc_b_frame
        GRV = self.segment.QuatArray
        a_meas_nav = np.einsum('ijk,ik->ij', DCM_vec, LinAcc)  # Lin accelaration at Nav frame
        batch_of_a_nav = a_meas_nav[:window_size, :]  # x,y,z
        batch_of_RV = GRV[:window_size, :]
        batch_of_a_nav_RV = np.hstack([batch_of_a_nav, batch_of_RV])
        batch_of_a_nav_RV = batch_of_a_nav_RV[None, :, :]

        if self.res18RVmodel is not None:
            net = torch.load(pth_path)
        else:
            net = self.res18RVmodel
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # net.to(device)
        # wde_est = np.zeros((NumOfBatches, 2))  # (only x,y) , NumOfBatches is expected to be N/window_size
        resent_input = Functions.PrepareInputForResnet18RV(batch_of_a_nav_RV)
        resent_input.to(device)  # this cannot be done in operational function
        resent_input = resent_input.cuda()
        output = net(resent_input)
        self.WD_vector_est = output.cpu().detach().numpy().squeeze()
        self.WD_angle_est = Functions.FoldAngles(np.arctan2(self.WD_vector_est[1], self.WD_vector_est[0]))
        self.end_pos_est = self.WD_vector_est / np.linalg.norm(self.WD_vector_est) * self.dL
        self.est_traj = np.vstack([[0, 0], self.end_pos_est])
        if plot_results:
            fig = plt.figure('res18_direction_pred')
            Ax = fig.add_subplot(111)
            Ax.plot(self.segment.Pos.arr()[:, 0] - self.segment.Pos.arr()[0, 0],
                    self.segment.Pos.arr()[:, 1] - self.segment.Pos.arr()[0, 1], label='GT')
            Ax.plot(self.est_traj[:, 0], self.est_traj[:, 1], label='Est')
            Ax.grid(True)
            Ax.legend()
            Ax.axis('equal')
        self.end_pos_error = np.linalg.norm(self.end_pos_est - self.WD_vector_GT)
        self.WD_est_method = 'resnet18RV'

    def pca_on_raw_acc(self, plot_results=False, use_GT_to_solve_amguity=True):
        # i want first to estimate 2nd direction and then to rotate it..
        # erna first rotate the gravity clean vec and then pca
        # modified first find 2nd component, then rotate (take mean) because we have a lot of R
        acc = self.segment.Acc.arr()
        acc_n_frame = Functions.transform_vectors(acc, self.Rot)
        WD_vector_est = Functions.est_second_direction(acc_n_frame, plot_res=plot_results)
        WD_vector_est = WD_vector_est[:2]  # only xy plane
        if use_GT_to_solve_amguity:  # TODO: I think we can get the sign from the eigen-value
            """for now avoiding +-180 ambiguity"""
            est_projected_on_GT = np.dot(WD_vector_est / np.linalg.norm(WD_vector_est),
                                         self.WD_vector_GT / np.linalg.norm(self.WD_vector_GT))
            self.WD_vector_est = est_projected_on_GT / np.linalg.norm(est_projected_on_GT) * WD_vector_est
        else:
            self.WD_vector_est = WD_vector_est

        self.WD_angle_est = Functions.FoldAngles(np.arctan2(self.WD_vector_est[1], self.WD_vector_est[0]))
        # It is OK to use self.dL because we want to see only the effect of WDE error
        self.end_pos_est = self.WD_vector_est / np.linalg.norm(self.WD_vector_est) * self.dL
        self.est_traj = np.vstack([[0, 0], self.end_pos_est])
        if plot_results:
            origin = [acc_n_frame[:, 0].mean(),
                      acc_n_frame[:, 1].mean()]
            arr_length = np.linalg.norm(
                [acc_n_frame[:, 0].std(), acc_n_frame[:, 1].std()])
            v0 = np.array(origin)
            v1 = np.array(origin) + arr_length * self.WD_vector_GT
            Functions.draw_vector(v0, v1, c='green')
            plt.axis('equal')

            fig = plt.figure('pca_on_raw_acc pos plot')
            ax = fig.add_subplot(111)
            ax.axis('equal')
            ax.grid(True)
            ax.plot(self.segment.Pos.arr()[:, 0] - self.segment.Pos.arr()[0, 0],
                    self.segment.Pos.arr()[:, 1] - self.segment.Pos.arr()[0, 1], label='GT')
            ax.plot(np.array([0, self.end_pos_est[0]]),
                    np.array([0, self.end_pos_est[1]]), label='est')
            ax.legend()
        self.end_pos_error = np.linalg.norm(self.end_pos_est - self.WD_vector_GT)
        self.WD_est_method = 'raw PCA'

    def walking_direction_estimation_using_smartphone_heading(self, plot_results=False):
        """the heading angle is initialized at the exp starting point. to compare 'apples to apples' we will assume
        constant speed using the dL (up-sampling) and calculate the segment end location estimation.
        the segment estimated walking angle is calculated using the estimated Pend. Also, a position error is
        calculated"""
        vel = self.dL / self.segment.NumberOfSamples_GT  # notice that this is the average velocity on a segment
        # therefore not accurate for a long experiment
        self.end_pos_est = np.array([0.0, 0.0])
        self.est_traj = np.zeros([self.segment.NumberOfSamples_GT, 2])
        self.Heading = self.Heading - self.segment.initial_heading + self.segment.initial_WD_angle_GT
        # dl_vector = np.linalg.norm(self.dP_vectors, axis=1)
        for i in range(self.segment.NumberOfSamples_GT - 1):
            self.end_pos_est = self.end_pos_est + \
                               vel * np.array([np.cos(self.Heading[i]), np.sin(self.Heading[i])])
            # self.end_pos_est = self.end_pos_est + \
            #                    dl_vector[i] * np.array([np.cos(self.Heading[i]), np.sin(self.Heading[i])])
            self.est_traj[i + 1] = self.end_pos_est
        self.end_pos_error = np.linalg.norm(self.end_pos_est - self.WD_vector_GT)
        self.WD_angle_est = Functions.FoldAngles(np.arctan2(self.end_pos_est[1], self.end_pos_est[0]))
        if plot_results:
            fig = plt.figure('walking_direction_estimation_using_smartphone_heading')
            Ax = fig.add_subplot(111)
            Ax.plot(self.segment.Pos.arr()[:, 0] - self.segment.Pos.arr()[0, 0],
                    self.segment.Pos.arr()[:, 1] - self.segment.Pos.arr()[0, 1], label='GT')
            Ax.plot(self.est_traj[:, 0], self.est_traj[:, 1], label='Est')
            Ax.grid(True)
            Ax.legend()
            Ax.axis('equal')
            self.WD_est_method = 'SM_heading'

    def inverted_pendulum_model(self, plot_results=False):
        vel = self.dL / self.segment.NumberOfSamples_GT
        self.end_pos_est = np.array([0.0, 0.0])
        self.est_traj = np.zeros([self.segment.NumberOfSamples_GT, 2])
        al_b = self.lin_acc_b_frame
        lin_acc_n_frame = Functions.transform_vectors(al_b, self.Rot)
        axy_n = np.linalg.norm(lin_acc_n_frame[:, 0:2], axis=1)
        grv_GT = self.grv
        az = np.zeros(self.segment.NumberOfSamples_GT)
        axy = np.zeros(self.segment.NumberOfSamples_GT)
        az_vec = np.zeros([self.segment.NumberOfSamples_GT, 3])
        axy_vec = np.zeros([self.segment.NumberOfSamples_GT, 3])
        yp_n = np.zeros([self.segment.NumberOfSamples_GT - 1, 3])
        for i in range(self.segment.NumberOfSamples_GT):
            grv_GT[i] = grv_GT[i] / np.linalg.norm(grv_GT[i])
            az[i] = al_b[i].reshape([1, 3]) @ grv_GT[i].reshape([3, 1])
            az_vec[i] = az[i] * grv_GT[i]
            axy_vec[i] = al_b[i] - az_vec[i]
            axy[i] = np.linalg.norm(axy_vec[i])
        dt = self.segment.dt
        fs = 1 / dt
        filter = signal.butter(N=3, Wn=[0.6, 2], btype='band', analog=False, fs=fs, output='sos')
        az_filt = signal.sosfiltfilt(filter, az)
        axy_filt_x = signal.sosfiltfilt(filter, axy_vec[:, 0]).reshape([axy_vec.shape[0], 1])
        axy_filt_y = signal.sosfiltfilt(filter, axy_vec[:, 1]).reshape([axy_vec.shape[0], 1])
        axy_filt_z = signal.sosfiltfilt(filter, axy_vec[:, 2]).reshape([axy_vec.shape[0], 1])
        axy_filt = np.hstack([axy_filt_x, axy_filt_y, axy_filt_z])
        if False:
            fig = plt.figure('walking_direction_estimation_using_inverted_pendulum')
            Ax = fig.add_subplot(111)
            Ax.plot(self.segment.Time_GT, az, label='az')
            Ax.plot(self.segment.Time_GT, az_filt, 'k--', label='az_filt')
            # Ax.plot(self.segment.Time_GT, axy, label='axy')
            Ax.plot(self.segment.Time_GT, -lin_acc_n_frame[:, 2], '--', label='al_zn')
            # Ax.plot(self.segment.Time_GT, -lin_acc_n_frame[:, 2], '--', label='al_xyn')
            Ax.grid(True)
            Ax.legend()
        for i in range(1, self.segment.NumberOfSamples_GT):
            # if i != 0:
            dt = self.segment.Time_GT[i] - self.segment.Time_GT[i - 1]
            d_dt_az = (az_filt[i] - az_filt[i - 1]) / dt
            d_dt_axy = (axy_filt[i] - axy_filt[i - 1]) / dt
            omega = d_dt_az * axy_filt[i] - d_dt_axy * az_filt[i]
            yp_b = omega / np.linalg.norm(omega)
            yp_n[i - 1] = self.Rot[i].as_matrix() @ yp_b.reshape([3, 1]).squeeze()
            self.end_pos_est = self.end_pos_est + \
                               vel * yp_n[i - 1][0:2]
            self.est_traj[i] = self.end_pos_est
        self.end_pos_error = np.linalg.norm(self.end_pos_est - self.WD_vector_GT)
        self.WD_angle_est = Functions.FoldAngles(np.arctan2(self.end_pos_est[1], self.end_pos_est[0]))
        if plot_results:
            fig = plt.figure('walking_direction_estimation_using_inverted_pendulum')
            Ax = fig.add_subplot(111)
            Ax.plot(self.segment.Pos.arr()[:, 0] - self.segment.Pos.arr()[0, 0],
                    self.segment.Pos.arr()[:, 1] - self.segment.Pos.arr()[0, 1], label='GT')
            Ax.plot(self.est_traj[:, 0], self.est_traj[:, 1], label='Est')
            Ax.grid(True)
            Ax.legend()
            Ax.axis('equal')
            self.WD_est_method = 'inverted_pendulum'

    def calc_error(self):
        assert self.WD_angle_GT is not None
        assert self.WD_angle_est is not None
        # e_tag = np.mod(abs(self.WD_angle_GT - self.WD_angle_est), np.pi)
        # self.error = min(e_tag, abs(e_tag - (np.pi)))
        self.WD_error = Functions.FoldAngles(self.WD_angle_GT - self.WD_angle_est)
        return self.WD_error, self.end_pos_error

    def identify_turn(self, plot_results=False, threshold=1.0):
        self.turn_identified = self.dP_angles_variance > threshold
        if plot_results:
            fig = plt.figure('turn identification')
            Ax = fig.add_subplot(121)
            Ax.plot(self.dP_angles)
            plt.axhline(y=np.mean(self.dP_angles), color='k', linestyle='--')
            plt.text(x=0, y=np.mean(self.dP_angles), s='mean')
            plt.axhline(y=np.mean(self.dP_angles) + threshold, color='r', linestyle='-')
            plt.text(x=0, y=np.mean(self.dP_angles) + threshold, s='threshold')
            plt.axhline(y=np.mean(self.dP_angles) + self.dP_angles_variance, color='g', linestyle='-')
            plt.text(x=0, y=np.mean(self.dP_angles) + self.dP_angles_variance, s='var')
            Ax = fig.add_subplot(122)
            Ax.plot(self.segment.Pos.arr()[:, 0], self.segment.Pos.arr()[:, 1])
            if self.turn_identified:
                plt.text(x=self.segment.Pos.arr()[:, 0].min(), y=self.segment.Pos.arr()[:, 1].min(),
                         s='turn_identified')
            Ax.axis('equal')
            plt.show()

    def PlotPosition(self):
        self.segment.PlotPosition()

    def initialize_WD_angle(self, wind_size_for_heading_init=750):
        # if Identify_using_trj_length:
        #     dL = np.linalg.norm(self.dP_vectors, axis=1)
        #     dL_cumsum = dL.cumsum()
        #     wind_size_for_heading_init = next(i for i in range(dL.shape[0]) if dL_cumsum[i] > 1)
        overall_d_pos = self.segment.Pos.arr()[wind_size_for_heading_init] - self.segment.Pos.arr()[0]
        initial_WD_vector_GT = overall_d_pos[0:2]
        self.initial_WD_angle_GT = Functions.FoldAngles(np.arctan2(initial_WD_vector_GT[1], initial_WD_vector_GT[0]))


class AHRS_results():
    def __init__(self, t=None, lin_acc_b_frame=None, Rot=None, heading=None, grv=None, id=None):
        self.t = t
        self.lin_acc_b_frame = lin_acc_b_frame
        self.Rot = Rot
        self.heading = heading
        self.grv = grv
        self.id = id


class RoninExp(AhrsExp):
    def __init__(self, path=None):
        super(RoninExp, self).__init__()
        if path is None:
            curr_directory = os.getcwd()  # will get current working directory
            path = tkinter.filedialog.askopenfilename(initialdir=curr_directory, title="Select A File",
                                                      filetype=(("csv files", "*.csv"), ("all files", "*.*")))
        self.Path = path
        self.FileName = path.split('\\')[-1]
        with open(os.path.join(path, 'info.json')) as f:
            info = json.load(f)
        info['path'] = os.path.split(path)[-1]
        with h5py.File(os.path.join(path, 'data.hdf5')) as f:
            ori = np.copy(f['synced/game_rv'])
        with h5py.File(os.path.join(path, 'data.hdf5'), "r") as f:
            gyro_uncalib = np.array(f['synced/gyro_uncalib'])
            acce_uncalib = np.array(f['synced/acce'])
            gyro_calib = gyro_uncalib - np.array(info['imu_init_gyro_bias'])
            acc_calib = np.array(info['imu_acce_scale']) * (acce_uncalib - np.array(info['imu_acce_bias']))
            time_vector= np.copy(f['synced/time'])
            raw_time = np.copy(f['raw/imu/step'][:,0])*1e-9
            tango_pos = np.copy(f['pose/tango_pos'])
            init_tango_ori = f['pose/tango_ori'][0]
            if False:
                plt.figure()
                plt.subplot(3,1,1)
                plt.plot( f['raw']['imu']['acce'][:,0]*1e-9,f['raw']['imu']['acce'][:,1])
                plt.plot( time_vector,acce_uncalib[:,0])
                plt.plot( time_vector,acc_calib[:,0])
                plt.subplot(3,1,2)
                plt.plot( f['raw']['imu']['acce'][:,0]*1e-9,f['raw']['imu']['acce'][:,2])
                plt.plot( time_vector,acce_uncalib[:,1])
                plt.plot( time_vector,acc_calib[:,1])
                plt.subplot(3,1,3)
                plt.plot( f['raw']['imu']['acce'][:,0]*1e-9,f['raw']['imu']['acce'][:,3])
                plt.plot( time_vector,acce_uncalib[:,2])
                plt.plot( time_vector,acc_calib[:,2])

                plt.figure()
                plt.subplot(3,1,1)
                plt.plot( f['raw']['imu']['gyro'][:,0]*1e-9,f['raw']['imu']['gyro'][:,1])
                plt.plot( time_vector,gyro_uncalib[:,0])
                plt.plot( time_vector,gyro_calib[:,0])
                plt.subplot(3,1,2)
                plt.plot( f['raw']['imu']['gyro'][:,0]*1e-9,f['raw']['imu']['gyro'][:,2])
                plt.plot( time_vector,gyro_uncalib[:,1])
                plt.plot( time_vector,gyro_calib[:,1])
                plt.subplot(3,1,3)
                plt.plot( f['raw']['imu']['gyro'][:,0]*1e-9,f['raw']['imu']['gyro'][:,3])
                plt.plot( time_vector,gyro_uncalib[:,2])
                plt.plot( time_vector,gyro_calib[:,2])
                plt.show()

        time = np.array(time_vector)
        self.Time_IMU = (time - time[0]) #* 1e-9
        self.Time_GT = (time - time[0]) #* 1e-9
        self.NumberOfSamples_IMU = self.Time_IMU.shape[0]
        self.NumberOfSamples_GT = self.Time_GT.shape[0]
        self.IMU_valid_idx = list(range(self.NumberOfSamples_IMU))
        self.dt = np.mean(np.diff(self.Time_IMU))

        self.Pos.x = np.array(tango_pos)[:, 0]
        self.Pos.y = np.array(tango_pos)[:, 1]
        self.Pos.z = np.array(tango_pos)[:, 2]
        #  self.QuatArray = np.array([grv_qx, grv_qy, grv_qz, grv_qw]).T  # size is nX4
        QuatArray = np.array([ori[:, 1], ori[:, 2], ori[:, 3], ori[:,0]]).T  # size is nX4 Transfered to ned
        self.Rot = Rotation.from_quat(QuatArray)
        EulerArray = self.Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        self.Psi = Functions.ContinuousAngle(EulerArray[:, 0])
        self.initial_heading = self.Psi[0]
        self.Theta = Functions.ContinuousAngle(EulerArray[:, 1])
        self.Phi = Functions.ContinuousAngle(EulerArray[:, 2])

        self.Gyro.x = gyro_calib[:, 0]
        self.Gyro.y = gyro_calib[:, 1]
        self.Gyro.z = gyro_calib[:, 2]

        self.Acc.x = acc_calib[:, 0]
        self.Acc.y = acc_calib[:, 1]
        self.Acc.z = acc_calib[:, 2]

        self.Frame = 'ENU'
    def resample(self, new_SF):
        t = self.Time_IMU
        num_of_samples = int((t[-1] - t[0]) * new_SF)
        self.NumberOfSamples_IMU = num_of_samples
        self.NumberOfSamples_GT = num_of_samples
        new_t = np.linspace(t[0], t[-1], num=num_of_samples, endpoint=True)
        # rotation

        slerp = Slerp(self.Time_GT, self.Rot)
        self.Rot = slerp(new_t)
        EulerArray = self.Rot.as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
        self.Psi = Functions.ContinuousAngle(EulerArray[:, 0])
        self.initial_heading = self.Psi[0]
        self.Theta = Functions.ContinuousAngle(EulerArray[:, 1])
        self.Phi = Functions.ContinuousAngle(EulerArray[:, 2])

        # Euler
        # self.Psi = np.interp(new_t, self.Time_IMU, self.Psi)
        # self.Theta = np.interp(new_t, self.Time_IMU, self.Theta)
        # self.Phi = np.interp(new_t, self.Time_IMU, self.Phi)
        # euler_array = np.vstack([self.Psi, self.Theta, self.Phi]).T
        # self.Rot = Rotation.from_euler('ZYX', euler_array, degrees=False)

        # IMU
        self.Gyro.x = np.interp(new_t, self.Time_IMU, self.Gyro.x)
        self.Gyro.y = np.interp(new_t, self.Time_IMU, self.Gyro.y)
        self.Gyro.z = np.interp(new_t, self.Time_IMU, self.Gyro.z)

        self.Acc.x = np.interp(new_t, self.Time_IMU, self.Acc.x)
        self.Acc.y = np.interp(new_t, self.Time_IMU, self.Acc.y)
        self.Acc.z = np.interp(new_t, self.Time_IMU, self.Acc.z)

        # pos
        self.Pos.x = np.interp(new_t, self.Time_IMU, self.Pos.x)
        self.Pos.y = np.interp(new_t, self.Time_IMU, self.Pos.y)
        self.Pos.z = np.interp(new_t, self.Time_IMU, self.Pos.z)

        # time vectors
        self.Time_IMU = new_t
        self.Time_GT = new_t
        self.dt = np.mean(np.diff(self.Time_IMU))
