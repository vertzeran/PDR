from create_segments_for_WDE_SBG import get_dir_to_analyze, get_exp_list
import os
from os import listdir, mkdir
from os.path import join
import pandas as pd
import utils.Classes as Classes
import numpy as np
from datetime import datetime
import utils.Functions as Functions
import json

def find_time_gaps(time_IMU, time_GPS, IMU_time_gap_th, GPS_time_gap_th, min_length_th):
    t0 = time_IMU[0]
    # 1. look for time gaps in IMU
    idx_of_time_gap_IMU = np.where(np.diff(time_IMU) > IMU_time_gap_th)[0]
    if idx_of_time_gap_IMU.shape[0] != 0:
        time_of_gaps_in_IMU = time_IMU[idx_of_time_gap_IMU] - t0
        # idx of gap start ->  time_IMU[idx_of_time_gap_IMU[i + 1]] - time_IMU[idx_of_time_gap_IMU[i]] > IMU_time_gap_th
        gap_lengths_IMU = time_IMU[idx_of_time_gap_IMU + 1] - \
                          time_IMU[idx_of_time_gap_IMU]
        idx_of_time_gap_IMU = idx_of_time_gap_IMU.tolist()
        time_of_gaps_in_IMU = time_of_gaps_in_IMU.tolist()
        gap_lengths_IMU = gap_lengths_IMU.tolist()
    else:
        idx_of_time_gap_IMU = None
        time_of_gaps_in_IMU = None
        gap_lengths_IMU = None
    # 2. segment GPS in IMU time frame
    t_start_IMU = time_IMU[0]
    t_stop_IMU = time_IMU[-1]

    start_idx = abs(time_GPS - t_start_IMU).argmin()
    start_idx = max(0, start_idx - 1)
    stop_idx = abs(time_GPS - t_stop_IMU).argmin()
    stop_idx = min(stop_idx + 1, time_GPS.shape[0])
    GPS_idx = np.array(range(start_idx, stop_idx + 1))  # to include stop_idx

    # 3. look for time gaps in GPS
    idx_of_time_gap_GPS = np.where(np.diff(time_GPS[GPS_idx]) > GPS_time_gap_th)[0]
    # notice this is an idx in an idx vector
    if idx_of_time_gap_GPS.shape[0] != 0:
        idx_of_time_gap_GPS = GPS_idx[idx_of_time_gap_GPS]
        time_of_gaps_in_GPS = time_GPS[idx_of_time_gap_GPS] - t0
        # idx of gap start ->  time_GPS[GPS_idx][idx_of_time_gap_GPS + 1] - time_GPS[GPS_idx][idx_of_time_gap_GPS] > GPS_time_gap_th
        gap_lengths_GPS = time_GPS[idx_of_time_gap_GPS + 1] - \
                          time_GPS[idx_of_time_gap_GPS]

        idx_of_time_gap_GPS = idx_of_time_gap_GPS.tolist()
        time_of_gaps_in_GPS = time_of_gaps_in_GPS.tolist()
        gap_lengths_GPS = gap_lengths_GPS.tolist()
    else:
        idx_of_time_gap_GPS = None
        time_of_gaps_in_GPS = None
        gap_lengths_GPS = None

    res = {"idx_of_time_gap_IMU": idx_of_time_gap_IMU,
           "time_of_gaps_in_IMU": time_of_gaps_in_IMU,
           "gap_lengths_IMU": gap_lengths_IMU,
           "idx_of_time_gap_GPS": idx_of_time_gap_GPS,
           "time_of_gaps_in_GPS": time_of_gaps_in_GPS,
           "gap_lengths_GPS": gap_lengths_GPS
           }
    return res


if __name__ == '__main__':
    data_location = 'magneto'  # could be 'magneto' or 'local_machine'
    mode = 'swing'
    dir_to_analyze = get_dir_to_analyze(data_location, mode)  # get main dir
    person_list = listdir(dir_to_analyze)
    exp_path_list = get_exp_list(list_of_persons=person_list,
                                 dir_to_analyze=dir_to_analyze)
    N = len(exp_path_list)
    i = 1
    res = {}
    for person in person_list:
        person_path = join(dir_to_analyze, person)
        if os.path.isdir(person_path):
            GT_path = join(person_path, 'ascii-output.txt')
            GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
            exp_list_in_person = get_exp_list(list_of_persons=[person],
                                              dir_to_analyze=dir_to_analyze)
            for exp_path in exp_list_in_person:
                print('working on ' + exp_path)
                Exp = pd.read_csv(exp_path)
                ind = range(0, len(GT))
                sec = GT['GPS Time'].values[ind]
                Timestamp = Exp['gyroTimestamp'].values / 1000  # traeted as IMU time vector
                IMU_idx = np.array(range(Timestamp.shape[0]))
                date = datetime.strptime(GT['UTC Date'].values[0], "%Y-%m-%d")
                (gpsweek, gpsseconds) = Functions.utctoweekseconds(date, 18)
                Unixtime = Functions.gpsTimeToUnixTime(gpsweek, sec)
                res_temp = find_time_gaps(time_IMU=Timestamp,
                                               time_GPS=Unixtime,
                                               IMU_time_gap_th=0.02,
                                               GPS_time_gap_th=0.1,
                                               min_length_th=10)
                if res_temp["idx_of_time_gap_IMU"] is not None or \
                        res_temp["idx_of_time_gap_GPS"] is not None:
                    res[exp_path] = res_temp
                print(str(round(i / N * 100, 2)) + " % completed")
                i += 1
    with open(join(dir_to_analyze, 'time_gap_analysis.json'), "w") as outfile:
        json.dump(res, outfile, indent=4)