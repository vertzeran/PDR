import pandas as pd 
import os
from datetime import datetime
from utils import Functions
import numpy as np
import matplotlib.pyplot as plt


def validate_data_using_time_vectors(time_IMU, time_GPS, IMU_time_gap_th, GPS_time_gap_th, min_length_th):
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
    iterator = (idx for idx in range(time_IMU.shape[0] - 1) if
                time_IMU[idx + 1] - time_IMU[idx] > IMU_time_gap_th)
    first_idx_of_time_gap_IMU = next(iterator, None)
    if first_idx_of_time_gap_IMU is not None:
        IMU_idx = list(range(time_IMU.shape[0])[:first_idx_of_time_gap_IMU])
    else:
        IMU_idx = list(range(time_IMU.shape[0]))
    # 2. segment GPS in IMU time frame
    t_start_IMU = time_IMU[IMU_idx][0]
    t_stop_IMU = time_IMU[IMU_idx][-1]
    GPS_idx = np.where((time_GPS >= t_start_IMU) & (time_GPS <= t_stop_IMU))
    # 3. look for time gaps in GPS
    iterator = (idx for idx in range(time_GPS[GPS_idx].shape[0] - 1) if
                time_GPS[GPS_idx][idx + 1] - time_GPS[GPS_idx][idx] > GPS_time_gap_th)
    first_idx_of_time_gap_GPS = next(iterator, None)  # notice this is an idx in an idx vector
    if first_idx_of_time_gap_GPS is not None:  ##    |
        GPS_idx = GPS_idx[:first_idx_of_time_gap_GPS]  ## <------------------------------------|
    # 4. varify that IMU data is processed only in times where GPS data available
    t_start_GPS = time_GPS[GPS_idx][0]
    t_stop_GPS = time_GPS[GPS_idx][-1]
    IMU_idx_of_idx = np.where((time_IMU[IMU_idx] >= t_start_GPS) &
                              (time_IMU[IMU_idx] <= t_stop_GPS))  # notice this is an idx in an idx vector
    IMU_idx_of_idx = IMU_idx_of_idx[0].astype(int)
    IMU_idx = list(map(IMU_idx.__getitem__, IMU_idx_of_idx))  ## <--------------------------------------|
    # varify the residual that is left (after time gaps segmentation) is long enough
    if time_IMU[IMU_idx][-1] - time_IMU[IMU_idx][0] < min_length_th:
        valid = False
    else:
        valid = True

    return IMU_idx, GPS_idx, valid


def import_exp_data(path):
    Exp = pd.read_csv(path)
    GT_path = os.path.split(path)[:-1][0] + "/ascii-output.txt"
    print(os.path.split(path)[1])
    GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
    ind = range(0, len(GT))
    sec = GT['GPS Time'].values[ind]
    Timestamp = Exp['gyroTimestamp'].values / 1000
    date = datetime.strptime(GT['UTC Date'].values[0], "%Y-%m-%d")
    (gpsweek, gpsseconds) = Functions.utctoweekseconds(date, 18)
    Unixtime = Functions.gpsTimeToUnixTime(gpsweek, sec)
    t0 = Timestamp[0]  # this value is used for initializing the time vectors so it would be consistent with the AHRS results files
    IMU_valid_idx, GPS_valid_idx, valid = validate_data_using_time_vectors(time_IMU=Timestamp,
                                                                           time_GPS=Unixtime,
                                                                           IMU_time_gap_th=0.02,
                                                                           GPS_time_gap_th=0.5,
                                                                           min_length_th=10)
    Unixtime = Unixtime[GPS_valid_idx]
    x_ECEF, y_ECEF, z_ECEF = Functions.LLA2ECEF(lat=GT['Latitude'].values[GPS_valid_idx],
                                                lon=GT['Longitude'].values[GPS_valid_idx],
                                                alt=GT['Altitude MSL'].values[GPS_valid_idx])
    ECEF_arr = np.vstack([x_ECEF, y_ECEF, z_ECEF]).T
    DCM = Functions.DCM_ECEF2NED(Long=GT['Longitude'].values[ind][0],
                                 Lat=GT['Latitude'].values[ind][0])
    n_e_d = np.dot(DCM, ECEF_arr.T)
    Pn = n_e_d[0, :].squeeze()
    Pn = Pn - Pn[0]
    Pe = n_e_d[1, :].squeeze()
    Pe = Pe - Pe[0]
    Pd = n_e_d[2, :].squeeze()

    return Unixtime, Pn, Pe, Pd


def plot_position(Unixtime, Pn, Pe, Pd):
    fig = plt.figure('Position Temporal Plot')
    Ax_pos_x = fig.add_subplot(311)
    Ax_pos_x.plot(Unixtime, Pn, color='blue', linewidth=1)
    Ax_pos_x.set_xlim(Unixtime[0], Unixtime[-1])
    Ax_pos_x.set(title=r"Position", ylabel="$[m]$")
    Ax_pos_x.grid(True)

    Ax_pos_y = fig.add_subplot(312, sharex=Ax_pos_x)
    Ax_pos_y.plot(Unixtime, Pe, color='blue', linewidth=1)
    Ax_pos_y.set_xlim(Unixtime[0], Unixtime[-1])
    Ax_pos_y.set(ylabel="$[m]$")
    Ax_pos_y.grid(True)

    Ax_pos_z = fig.add_subplot(313, sharex=Ax_pos_x)
    Ax_pos_z.plot(Unixtime, Pd, color='blue', linewidth=1)
    Ax_pos_z.set_xlim(Unixtime[0], Unixtime[-1])
    Ax_pos_z.set(xlabel="Time [sec]", ylabel="$[m]$")
    Ax_pos_z.grid(True)

    fig = plt.figure('Position XY Plot')
    Ax = fig.add_subplot(111)
    Ax.plot(Pn, Pe, color='blue', linewidth=1)
    Ax.set(xlabel="$[m]$", ylabel="$[m]$")
    Ax.grid(True)


if __name__ == '__main__':
    file_path = None
    root_dir = '/home/maint/Eran/AHRS/SBG-PDR-DATA/swing/22_07_27_swing_nati_R'
    if file_path is not None:
        Unixtime, Pn, Pe, Pd = import_exp_data(file_path)
        plot_position(Unixtime, Pn, Pe, Pd )
        plt.show()
    else:
        assert root_dir is not None
        print(root_dir)
        data_list = os.listdir(root_dir)
        i = 1
        for file_name in data_list:
            file_path = os.path.join(root_dir, file_name)
            if '_AHRS_results.xlsx' not in file_path and 'ascii-output.txt' not in file_path:
                Unixtime, Pn, Pe, Pd = import_exp_data(file_path)
                plot_position(Unixtime, Pn, Pe, Pd)
                plt.show()