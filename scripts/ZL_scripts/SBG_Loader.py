#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 12:44:44 2023
Before using this functions you should:
    1) Run calculate_AHRS_results_on_list and make sure you have xls file with results for each csv
    2) 
Then, run this script. 
It will go over a "root of root dir", for exmple, entire swing mode,
and for each sub-dir it will load the experiment (a csv file) with their AHRS results (xls).
Then, the script will convert the LinAcc to nav frame, calculate |dl| and WDE
and save them to X,Y1,Y2
@author: zahi
"""

import pandas as pd
import numpy as np
from numpy.linalg import norm
# import os
from os import listdir
from os.path import join
from datetime import datetime,timedelta
# import sys
# cwd = os.getcwd()
# sys.path.append(cwd[:cwd.rindex('/')]+'/utils')
# import Classes
#from utils import Classes,Functions
from utils import CutByFirstDim,GetTimeValidtyCode,NeedToSkipDueToBadValidErr

################ Function from utils.Functions ######################
def utctoweekseconds(utc,leapseconds):
    """ Returns the GPS week, the GPS day, and the seconds
        and microseconds since the beginning of the GPS week """
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.strptime("1980-01-06 00:00:00",datetimeformat)
    tdiff = utc -epoch  + timedelta(seconds=leapseconds)
    gpsweek = tdiff.days // 7
    gpsdays = tdiff.days - 7*gpsweek
    gpsseconds = tdiff.seconds + 86400* (tdiff.days -7*gpsweek)
    return gpsweek, gpsseconds #,gpsdays,tdiff.microseconds

def gpsTimeToUnixTime(week, sec):
    """convert GPS week and TOW to a time in seconds since 1970"""
    epoch = 86400 * (10 * 365 + int((1980 - 1969) / 4) + 1 + 6 - 2)
    return epoch + 86400 * 7 * week + sec - 18

def LLA2ECEF(lat, lon, alt):
    # WGS84 ellipsoid constants:

    a = 6378137
    e = 8.1819190842622e-2

    # intermediate calculation
    # (prime vertical radius of curvature)
    N = a / np.sqrt(1 - np.power(e, 2) * np.power(np.sin(lat * np.pi / 180), 2))

    # results:
    x = (N + alt) * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y = (N + alt) * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
    z = ((1 - np.power(e, 2)) * N + alt) * np.sin(lat * np.pi / 180)

    return x, y, z

def Mtheta(Theta):
    Mtheta=np.array(
        [[np.cos(Theta), 0, -np.sin(Theta)],
         [0            , 1,  0            ],
         [np.sin(Theta), 0,  np.cos(Theta)]])
    return Mtheta

def Mpsi(Psi):
    Mpsi=np.array(
        [[ np.cos(Psi)  ,np.sin(Psi),0],
         [-np.sin(Psi)  ,np.cos(Psi),0],
         [    0         ,   0       ,1]])
    return Mpsi

def DCM_Ned2ECEF(Long, Lat):
    M12 = Mtheta(Lat*np.pi/180+np.pi/2)
    M01 = Mpsi(-Long*np.pi/180)
    DCM = np.dot(M01, M12)
    return DCM

def DCM_ECEF2NED(Long, Lat):
    M10 = Mpsi(Long * np.pi / 180)
    M21 = Mtheta(-Lat*np.pi/180-np.pi/2)
    DCM = np.dot(M21, M10)
    return DCM
###############################################################################

def get_AHRS_results_for_exp(exp_path):
    """
    Load results from ahrs results file
    """
    suffix = '_AHRS_results.xlsx'
    AHRS_results_file_path  = exp_path.replace('.csv',suffix)
    DF = pd.read_excel(AHRS_results_file_path)
    grv_hat = np.hstack((np.array(DF.grv_x).reshape(-1,1), np.array(DF.grv_y).reshape(-1,1), np.array(DF.grv_z).reshape(-1,1)))
    n = len(DF.grv_x)
    Rnb = np.zeros([n, 3, 3])
    Rnb[:, 0, 0] = DF.Rnb_11
    Rnb[:, 0, 0] = DF.Rnb_11
    Rnb[:, 0, 1] = DF.Rnb_12
    Rnb[:, 0, 2] = DF.Rnb_13
    Rnb[:, 1, 0] = DF.Rnb_21
    Rnb[:, 1, 1] = DF.Rnb_22
    Rnb[:, 1, 2] = DF.Rnb_23
    Rnb[:, 2, 0] = DF.Rnb_31
    Rnb[:, 2, 1] = DF.Rnb_32
    Rnb[:, 2, 2] = DF.Rnb_33
    return  grv_hat, Rnb
def PrintGTTimeGaps(t_gt):
    dt_gt = np.diff(t_gt,axis=0)
    bad_inds = np.where(dt_gt > 0.01)[0] # detecting a time gap of 10 ms
    bad_inds_neg = np.where(dt_gt < 0)[0] # detecting a negative jump
    inds_to_print = np.concatenate((bad_inds,bad_inds_neg))
    if len(inds_to_print)>0:
        for ind in inds_to_print:
            print('Detected time gap at GT timestamp at inds: ',ind, ' Time Gap: ',dt_gt[ind])
    # cut_ind = len(SBGTimeAtUnixFormat)
    # if len(inds1) > 0:
    #     print('Warning: found ',len(inds1),' GPS samples with bad time stamp')
    #     cut_ind = min(inds1[0],cut_ind)
    #     print('Detected time gap at GT timestamp at ind:',inds1[0])
    # SBGTimeAtUnixFormat = SBGTimeAtUnixFormat[:cut_ind]
    # Pos = Pos[:cut_ind,:] 
    
def LoadGT(GT_path):
    """
    Open the text file and load the GT POS with sbg time 
    I have decided not to Remove samples after detecting 20ms time gap
    because manual check shows that only one sample at a file (at half ot the times)
    is actually jumping and the jumps is about 100-200ms at most. Moreover,
    we can not know which of the part (before or after the jump) is the correct time
    """
    GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
    week_sec =  np.array(GT['GPS Time'])
    mydate = datetime.strptime(GT['UTC Date'].values[0], "%Y-%m-%d")
    gpsweek,gpsseconds = utctoweekseconds(mydate, 18)
    SBGTimeAtUnixFormat = gpsTimeToUnixTime(gpsweek, week_sec).reshape(-1,1)
    #print(Unixtime*1000 - 1658900638576)
    
    #TODO: why not taking ECEF from the GT. According to Eran it is possible
    x_ECEF, y_ECEF, z_ECEF = LLA2ECEF(lat=GT['Latitude'].values,lon=GT['Longitude'].values,alt=GT['Altitude MSL'].values)
    ECEF_arr = np.vstack([x_ECEF, y_ECEF, z_ECEF]).T
    DCM = DCM_ECEF2NED(Long=GT['Longitude'].values[0],Lat=GT['Latitude'].values[0])
    n_e_d = np.dot(DCM, ECEF_arr.T)
    Pn = n_e_d[0, :].squeeze()
    Pe = n_e_d[1, :].squeeze()
    Pd = n_e_d[2, :].squeeze()
    # Pn, Pe, Pd = Functions.LLLN2NED(GT['Latitude'].values[ind], GT['Longitude'].values[ind], GT['Altitude MSL'].values[ind],
    #                       RN, RM)
    Pos = np.array([Pe, Pn, -Pd]).T  # NED to ENU conversion
    Pos = Pos-Pos[1,:] #TODO: why ind 1 and not 0?
    
    # Report time gaps 
    PrintGTTimeGaps(SBGTimeAtUnixFormat)

    return SBGTimeAtUnixFormat , Pos

# def LoadExpAndAHRS(exp_path):
#     """
#     Old function that uses the Class structure
#     Load exp, extract LinAcc and DCM (with get_AHRS_results_for_exp)
#     and cut all of them between valid indexes    
#     """
#     exp = Classes.SbgExpRawData(exp_path) #already cutted in last ind during the exp init
#     grv_hat, Rot = get_AHRS_results_for_exp(exp_path) #already cutted in last ind during the exp init
#     exp.define_walking_start_idx()
#     first_ind = exp.index_of_walking_start   
#     if exp.first_idx_of_time_gap is not None:
#         last_ind = exp.first_idx_of_time_gap
#         print('removed samples with time gap from index',last_ind)
#     else:
#         last_ind = len(grv_hat)
#     # Pos = exp.Pos.arr()[first_ind:last_ind,:]
#     Pos = exp.Pos.arr()[first_ind:,:]
#     lin_acc_b_frame = exp.Acc.arr()[first_ind:,:] - grv_hat[first_ind:last_ind,:]
#     # lin_acc_b_frame = lin_acc_b_frame[first_ind:last_ind,:]
#     DCM_vec = Rot[first_ind:last_ind,:,:]
    
#     return Pos,lin_acc_b_frame,DCM_vec
    
def SyncData(t_exp,t_data,DataArray):
    """
    Interpulate the android rotation\magnetometer estimation 
    or gt position to the same time axis we have for the gyro and acc. 
    The function removes zeros and find the relevant inds for interpulation
    Inputs:
        1) time axis (given in the same units!!). Zeros mean non-valid data
        2) DataArray with size (num_of_samples,dims), for example 1000x3
    Outputs:
        DataArray synced to t_exp time axis 
    """
    # Remove non-valid data
    non_zero_inds = np.where(t_data>0)[0] #catch t = 0 and t =0.0
    x = t_data[non_zero_inds]
    y = DataArray[non_zero_inds,:]
    
    # Get only relevant inds for interp
    ind_start = np.argmin(np.abs(x-t_exp[0])) #Debug: ind 461315 for Pos 
    ind_stop = np.argmin(np.abs(x-t_exp[-1])) #Debug: ind 476545 for Pos 
    
    # Detrmine the validity. 
    # Note that we already dealed with Pos \ ACC \ Gyro sample rate when loaded them
    # So we assume the error now will be only bad overlaps
    # For the Mag and RV we have no control on sample rate which is around 100ms
    edges,codes = GetTimeValidtyCode()
    t_data_range = (x[ind_stop] - x[ind_start])
    missing_gt_time = t_exp[-1]-t_exp[0]-t_data_range # is negative in the ideal case
    ValidErr = codes[-1] # the "all good" code
    for k in range(len(edges)):
        if (missing_gt_time - edges[k] < 0):
            ValidErr = codes[k]
            break        
    if (ind_stop == ind_start): #The worst case 
        print('Critical Error: interpolation can not be done')
        return ValidErr,None
    
    # Work with small numbers to improve interpolation
    t0 = t_exp[0] 
    
    # note that interp is working only on 1D arrays
    ListOfVectors = []
    for dim in range(y.shape[1]):
        vec = np.interp(t_exp[:,0]-t0, x[ind_start:ind_stop,0]-t0, y[ind_start:ind_stop,dim]).reshape(-1,1)
        ListOfVectors.append(vec)
    SyncedData = np.hstack(ListOfVectors)
    return ValidErr,SyncedData
				
def GetExpData(exp_path):
    """
    Load csv with raw data from IMU (ACC + Gyro)
    Calculate the time axis and valid inds based on the "core data" (ACC+Gyro)
    Then, aquire some more data such as RV and Mag if it is possible
    """
    DF = pd.read_csv(exp_path)
    t_acc = DF['accTimestamp'].values
    t_gyro = DF['gyroTimestamp'].values
    dt_acc = np.diff(t_acc)
    dt_gyro = np.diff(t_gyro)
    ValidErr = {}
    
    # Cut after detecting a time gap of 20 ms, or timestamp diff of 5 ms
    inds1 = np.where(dt_acc > 20)[0]
    inds2 = np.where(dt_gyro > 20)[0]
    inds3 = np.where(np.abs(t_acc-t_gyro) > 5)[0]
    cut_ind = len(t_acc)
    if len(inds1) > 0:
        cut_ind = min(inds1[0],cut_ind)
        print('Detected time gap at acc timestamp at ind:',inds1[0])
    if len(inds2) > 0:
        cut_ind = min(inds2[0],cut_ind)
        print('Detected time gap at gyro timestamp at ind:',inds2[0])
    if len(inds3) > 0:
        cut_ind = min(inds3[0],cut_ind)
        print('Detected gap between acc and gyro timestamps at ind:',inds3[0])
    
    # TODO: what it the convention? NED or ENU? should we flip gyro \ acc axis?
    AndroidTimeAtUnixFormat = ((t_acc + t_gyro)/2/1000).reshape(-1,1)
    Acc = np.array([DF.accX,DF.accY,DF.accZ]).T
    Gyro = np.array([DF.gyroX,DF.gyroY,DF.gyroZ]).T
    
    # Core data and main time axis
    AndroidTimeAtUnixFormat = AndroidTimeAtUnixFormat[:cut_ind]
    Acc = Acc[:cut_ind,:]
    Gyro = Gyro[:cut_ind,:]
    
    # 4D qutarnion synced to acc and gyro time
    t_RV = np.array(DF.RV_timestamp).reshape(-1,1)/1000 #from ms to sec
    temp_RV = np.array([DF.RV_qx,DF.RV_qy,DF.RV_qz,DF.RV_qw]).T
    ValidErr['RV'],RV = SyncData(AndroidTimeAtUnixFormat,t_RV,temp_RV)
    
    # 3D magnetometer synced to acc and gyro time
    t_mag = np.array(DF.magTimestamp).reshape(-1,1)/1000 #from ms to sec
    temp_mag = np.array([DF.magX,DF.magY,DF.magZ]).T
    ValidErr['Mag'],Mag = SyncData(AndroidTimeAtUnixFormat,t_mag,temp_mag)
    
    return  AndroidTimeAtUnixFormat,Acc,Gyro,RV,Mag,cut_ind,ValidErr
            
def LoadExpAndAHRS_V2(exp_path,t_gt=None,Pos_gt=None,initial_dist_to_cut = 1):
    """
    Calls 3 sub functions:
        1) LoadGT (if the gt data was not given as input)
        2) Load exp raw measurments (csv file)
        3) Load AHRS resutls
    The GT is given in one time axis while the measurments are given in other
    so interpolation is required.
    AHRS results seems to be aligned with android time stamp ("T IMU") so interp is NOT needed for them.
    In addition we cut the first meter to avoid dirty data
    """
    # 1) Load GT. This code should be here so that the function can work as a stand alone
    if t_gt is None:
        GT_path = exp_path[:exp_path.rindex('/')+1] + 'ascii-output.txt'
        t_gt,Pos_gt = LoadGT(GT_path)
    
    # 2) Load CSV (cut_ind needed to cut AHRS data)
    t_exp,Acc,Gyro,RV,Mag,cut_ind,ValidErr = GetExpData(exp_path)
    
    # Sync position time axis
    ValidErr['Pos'],Pos = SyncData(t_exp,t_gt,Pos_gt)
        
    # 3) Load AHRS results (already synced) + cut AHRS data
    grv_hat, DCM_vec = get_AHRS_results_for_exp(exp_path) #already cutted in last ind during the exp init
    grv_hat, DCM_vec = CutByFirstDim([grv_hat, DCM_vec],0,cut_ind)
    
    # Cut the first meter (as eran does in define_walking_start_idx)
    # If the man is standing, a Pos err will be raised
    if Pos is not None: # ValidErr['Pos'] == 3
        first_ind = np.argmin(np.abs(norm(Pos-Pos[0,:],axis=1)-initial_dist_to_cut))
        if first_ind == len(t_exp):
            ValidErr['Pos'] = GetTimeValidtyCode()[1][-1] # for codes, -1 for worst code
        print('Number of removed inds due to standing man:',first_ind)
        t_exp,Acc,Gyro,grv_hat,DCM_vec,Pos,RV,Mag = CutByFirstDim([t_exp,Acc,Gyro,grv_hat,DCM_vec,Pos,RV,Mag],first_ind,len(t_exp))
    
    lin_acc_b_frame = Acc - grv_hat
    
    return t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat,ValidErr

def PrepareSequence(Pos,lin_acc_b_frame,DCM_vec,window_size,RV = None):
    """
    Convert LinAcc to nav frame, and create sequences with N=window_size samples
    Then calculate the WDE and |dl| for each sequence
    """
    Inputs = np.einsum('ijk,ik->ij', DCM_vec, lin_acc_b_frame)  # Nx3
    if RV is not None:
        Inputs = np.hstack((Inputs,RV))
    num_of_input_channels = Inputs.shape[1] #3 for ACC, 4 For RV, 7 when combined
    
    last_ind = len(Pos)-len(Pos)%window_size
    X = np.reshape(Inputs[:last_ind,:],(-1,window_size,num_of_input_channels)) #(N/200)x200x3

    Pos_sequences = np.reshape(Pos[:last_ind,:],(-1,window_size,3)) #(N/200)x200x3
    segment_dl = Pos_sequences[:,-1,:] - Pos_sequences[:,0,:] #(N/200)x3
    
    # For odometry
    Y1 = np.linalg.norm(segment_dl[:,:2], axis = 1, keepdims = True)
    
    # For WDE
    Y2 = segment_dl[:,:2]/Y1

    return X,Y1,Y2


# def PrepareSequenceFromDir(root_dir,window_size):
#     """
#     Old function that uses the Class structure 
#     """
#     X = np.zeros((0,window_size,3))
#     Y1 = np.zeros((0,1))
#     Y2 = np.zeros((0,2))
#     data_list = listdir(root_dir)
#     data_list = [item for item in data_list if '_AHRS_results.xlsx' not in item]
#     data_list = [item for item in data_list if 'ascii-output.txt' not in item]
#     for ind,file_name in enumerate(data_list):
#         exp_path = join(root_dir, file_name)               
#         print('Loading: ',file_name)
#         Pos,lin_acc_b_frame,DCM_vec = LoadExpAndAHRS(exp_path)
#         tmp_x,tmp_y1,tmp_y2 = PrepareSequence(Pos,lin_acc_b_frame,DCM_vec,window_size)
#         X = np.concatenate((X,tmp_x),axis=0)
#         Y1 = np.concatenate((Y1,tmp_y1),axis=0)
#         Y2 = np.concatenate((Y2,tmp_y2),axis=0)
#     return X,Y1,Y2

def PrepareSequenceFromDir_v2(root_dir,window_size,data_type = 'LinAcc'):
    """
    Loading the GT+CSV+AHRS for each exp in the dir. 
    Creating a big list of data
    """
    X = np.zeros((0,window_size,7 if (data_type == 'LinAccWithRV') else 3))
    Y1 = np.zeros((0,1))
    Y2 = np.zeros((0,2))
    # Remove any non-csv files that might be in the dir
    data_list = listdir(root_dir)
    data_list = [item for item in data_list if '_AHRS_results.xlsx' not in item]
    data_list = [item for item in data_list if 'ascii-output.txt' not in item]
    for ind,file_name in enumerate(data_list):
        if ind == 0: #Load the GT only if this is the first time
            GT_path = join(root_dir,'ascii-output.txt')
            t_gt,Pos_gt = LoadGT(GT_path)
        exp_path = join(root_dir, file_name)               
        print('Loading: ',file_name)
        t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=t_gt,Pos_gt=Pos_gt)
        if NeedToSkipDueToBadValidErr(ValidErr,file_name): 
            continue
        if data_type == 'LinAcc':
            tmp_x,tmp_y1,tmp_y2 = PrepareSequence(Pos,lin_acc_b_frame,DCM_vec,window_size)
        if data_type == 'LinAccWithRV':
            tmp_x,tmp_y1,tmp_y2 = PrepareSequence(Pos,lin_acc_b_frame,DCM_vec,window_size,RV = RV)
        X = np.concatenate((X,tmp_x),axis=0)
        Y1 = np.concatenate((Y1,tmp_y1),axis=0)
        Y2 = np.concatenate((Y2,tmp_y2),axis=0)
    return X,Y1,Y2

def RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,list_of_dirs,window_size,data_type = 'LinAcc'):
    X = np.zeros((0,window_size,7 if (data_type == 'LinAccWithRV') else 3))
    Y1 = np.zeros((0,1))
    Y2 = np.zeros((0,2))
    for ind,root_dir in enumerate(list_of_dirs):
        print('### Working on dir: ',root_dir,' ###')
        # tmp_x,tmp_y1,tmp_y2 = PrepareSequenceFromDir(join(root_of_roots,root_dir),window_size)
        tmp_x,tmp_y1,tmp_y2 = PrepareSequenceFromDir_v2(join(root_of_roots,root_dir),window_size,data_type = data_type)
        X = np.concatenate((X,tmp_x),axis=0)
        Y1 = np.concatenate((Y1,tmp_y1),axis=0)
        Y2 = np.concatenate((Y2,tmp_y2),axis=0)
    return X,Y1,Y2

def printshapes(header,ListOfArrays):
    print(header)
    for arr in ListOfArrays:
        print(arr.shape)

if __name__ == '__main__':
    
    window_size = 200
    ### List of bad exps to debug (only the first is normal):
    exp_path = '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R/outdoor_output_2022-07-27_08_56_16.csv'
    #exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/pocket/22_09_15_pocket_zeev/outdoor_output_2022-09-15_09_33_00.csv'
    #exp_path = '/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_firas/outdoor_output_2021-11-07_11_45_52.csv'
    #exp_path = '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_09_15_swing_nadav_R/outdoor_output_2022-09-15_11_11_37.csv'
    #exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/pocket/21_11_10_omri/outdoor_output_2021-11-10_14_57_42.csv'
    #exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_10_31_eran/outdoor_output_2021-10-31_11_02_09.csv'
    
    ### Test 1 - the old functions
    # csv_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R/outdoor_output_2022-07-27_08_49_41.csv'
    # Pos,lin_acc_b_frame,DCM_vec = LoadExpAndAHRS(csv_path)
    # X,Y1,Y2 = PrepareSequence(Pos,lin_acc_b_frame,DCM_vec,window_size)
    
    ### Test 2 - load GT: 
    GT_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R/ascii-output.txt'
    t_gt,Pos_gt = LoadGT(GT_path)
    assert(t_gt.shape[0] == Pos_gt.shape[0])
    printshapes('*** Test LoadGT: ***',[t_gt,Pos_gt])    
    print('------------------------------------------------')
    
    ### Test 3 - load exp + AHRS results
    t_exp,Acc,Gyro,RV,Mag,cut_ind,ValidErr = GetExpData(exp_path)
    grv_hat, Rot = get_AHRS_results_for_exp(exp_path)
    #first is true when no time gap detected, second is true always
    assert((t_exp.shape[0] == Rot.shape[0]) or (t_exp.shape[0]==cut_ind)) 
    print('cut_ind:',cut_ind)
    print(ValidErr)
    printshapes('*** Test GetExpData,get_AHRS_results_for_exp: ***',[t_exp,Acc,Gyro,RV,Mag,grv_hat,Rot])    
    print('------------------------------------------------')

    ### Test 4 - load both exp + AHRS results + GT with LoadExpAndAHRS_V2
    # one time with GT, second time without
    t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=None,Pos_gt=None)
    printshapes('*** Test LoadExpAndAHRS_V2: ***',[t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat]) 
    # #make sure GT path correspand exp path before activating this test
    # t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=t_gt,Pos_gt=Pos_gt)
    # printshapes('*** Test LoadExpAndAHRS_V2: ***',[t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat]) 
    assert(Pos.shape[0] < cut_ind) #cut_ind is the original time len from GetExpData, but we cut additional 1 meter
    print(ValidErr)
    print('------------------------------------------------')

    ### Test 5 - prepare X,Y from Pos GT and linear acc in nav frame
    X,Y1,Y2 = PrepareSequence(Pos,lin_acc_b_frame,DCM_vec,window_size,RV=RV)
    printshapes('*** Test PrepareSequence: ***',[X,Y1,Y2]) 
    print('------------------------------------------------')

    ### Test 6: check LoadRiDiDir function
    root_dir = '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R'
    X,Y1,Y2 = PrepareSequenceFromDir_v2(root_dir,window_size,data_type = 'LinAccWithRV')
    printshapes('*** Test PrepareSequenceFromDir_v2: ***',[X,Y1,Y2]) 
    print('------------------------------------------------')

    ### Test 7: check LoadMultyDirs function
    root_of_roots = '/data/Datasets/Navigation/SBG-PDR-DATA/swing'
    list_of_dirs = ['22_07_27_swing_zahi_R','22_07_27_swing_zahi_L']
    X,Y1,Y2 = RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,list_of_dirs,window_size,data_type = 'LinAccWithRV')
    printshapes('*** Test RunPrepareSequenceFromDirOnListOfDirs: ***',[X,Y1,Y2]) 
    print('------------------------------------------------')
