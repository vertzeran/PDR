#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:01:52 2022
Collection of loading functions:
    1) ReadRidiCSV(file_path) - to read a single csv file
    2) LoadRiDiDir(path_to_dir) - to read all files (data is stacked vertically)
    3) LoadMultyDirs(TrainOrTest) - Read the entire RIDI DS
In the end of the file there is a short code to test the functions
@author: zahi
"""
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from os import listdir
import os.path as osp

def ReadRidiCSV(file_path):
    """
    Ridi dataset was acquired with one phone, containing both slam and IMU
    the orientation is given with respect to an initial frame we dont know
    the gravity is also given with respect to this system
    This script load their excel file and return Nx3 arrays
    """
    DF = pd.read_csv(file_path)
    
    t_ns = np.array(DF.time).reshape(-1,1)
    t = t_ns*1e-9 
    # please note that altough fs=200Hz, dt is not const, and verays around 0.00495
    
    x = np.array(DF.pos_x).reshape(-1,1)
    y = np.array(-DF.pos_y).reshape(-1,1) # conversion from ENU to NED
    z = np.array(-DF.pos_z).reshape(-1,1) # conversion from ENU to NED
    Pos = np.hstack((x,y,z))
        
    ori_x = np.array(DF.ori_x).reshape(-1,1)
    ori_y = -np.array(DF.ori_y).reshape(-1,1) # conversion from ENU to NED
    ori_z = -np.array(DF.ori_z).reshape(-1,1) # conversion from ENU to NED
    ori_w = np.array(DF.ori_w).reshape(-1,1)
    Quat_vec = np.hstack((ori_x,ori_y,ori_z,ori_w))
    DCM_vec = Rotation.from_quat(Quat_vec).as_matrix() #Nx3x3
    Euler_vec = Rotation.from_quat(Quat_vec).as_euler('ZYX', degrees=False)  # ZYX is capital important!!!
    # Euler_vec is given at: Psi,Theta,Phi

    gyro_x = np.array(DF.gyro_x).reshape(-1,1)
    gyro_y = np.array(-DF.gyro_y).reshape(-1,1)  # conversion from ENU to NED
    gyro_z = np.array(-DF.gyro_z).reshape(-1,1)  # conversion from ENU to NED
    Gyro = np.hstack((gyro_x,gyro_y,gyro_z))
    
    acc_x = np.array(DF.acce_x).reshape(-1,1)
    acc_y = np.array(-DF.acce_y).reshape(-1,1)  # conversion from ENU to NED
    acc_z = np.array(-DF.acce_z).reshape(-1,1)  # conversion from ENU to NED
    Acc = np.hstack((acc_x,acc_y,acc_z))
    
    mag_x = np.array(DF.magnet_x).reshape(-1,1)
    mag_y = np.array(-DF.magnet_y).reshape(-1,1)  # conversion from ENU to NED
    mag_z = np.array(-DF.magnet_z).reshape(-1,1)  # conversion from ENU to NED
    Mag = np.hstack((mag_x,mag_y,mag_z))

    linAcc_x = np.array(DF.linacce_x).reshape(-1,1)
    linAcc_y = np.array(-DF.linacce_y).reshape(-1,1)  # conversion from ENU to NED
    linAcc_z = np.array(-DF.linacce_z).reshape(-1,1)  # conversion from ENU to NED
    LinAcc = np.hstack((linAcc_x,linAcc_y,linAcc_z))

    grv_x = np.array(DF.grav_x).reshape(-1,1)
    grv_y = np.array(-DF.grav_y).reshape(-1,1)  # conversion from ENU to NED
    grv_z = np.array(-DF.grav_z).reshape(-1,1)  # conversion from ENU to NED
    Grv = np.hstack((grv_x,grv_y,grv_z))
    
    return t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv


def find_csv_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    files_in_dir = [ filename for filename in filenames if filename.endswith( suffix ) ]
    files_in_dir.sort()
    return files_in_dir

def LoadRiDiDir(path_to_dir):
    """
    Go over entire dir and concatenate the relevant data
    I also added file_num_vec with shape of Nx1, so using this vec
    and the files_in_dir list (which is sorted) one can track the relevant exp
    """
    files_in_dir = find_csv_filenames(path_to_dir) # a sorted list
    for file_num,file_name in enumerate(files_in_dir):
        exp_path = osp.join(path_to_dir,file_name)
        t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv = ReadRidiCSV(exp_path)
        file_num_vec = np.ones((len(t),1),dtype=np.int32)*int(file_num)
        if file_num == 0: #first iteration
            t_all,Pos_all,DCM_vec_all,Gyro_all,Acc_all,Mag_all,LinAcc_all = t,Pos,DCM_vec,Gyro,Acc,Mag,LinAcc
            file_num_vec_all = file_num_vec
        else:
            t_all = np.concatenate((t_all, t), axis=0)
            Pos_all = np.concatenate((Pos_all, Pos), axis=0)
            DCM_vec_all = np.concatenate((DCM_vec_all, DCM_vec), axis=0)
            Gyro_all = np.concatenate((Gyro_all, Gyro), axis=0)
            Acc_all = np.concatenate((Acc_all, Acc), axis=0)
            Mag_all = np.concatenate((Mag_all, Mag), axis=0)
            LinAcc_all = np.concatenate((LinAcc_all, LinAcc), axis=0)
            file_num_vec_all = np.concatenate((file_num_vec_all, file_num_vec), axis=0)
    
    return t_all,Pos_all,DCM_vec_all,Gyro_all,Acc_all,Mag_all,LinAcc_all,file_num_vec_all,files_in_dir

def LoadMultyDirs(TrainOrTest):
    """
    Here we are loading the entire dataset using the previous functions (load dir, load file).
    Note that there are hard coded names specifically for RIDI in the code.
    User need to chhose train \ test.
    Actually there is no reason to use this function because we would like speific model for each mode.
    A list of modes is also given so we can track back each sample.
    """
    RIDI_path = '/data/Datasets/Navigation/RIDI_dataset_train_test'
    if TrainOrTest == 'Train':
        dirs_in_ds = ['RIDI - Bag - Train', 'RIDI - Body - Train', 'RIDI - Pocket - Train', 'RIDI - Text - Train']
    if TrainOrTest == 'Test':
        dirs_in_ds = ['RIDI - Bag - Test', 'RIDI - Body - Test', 'RIDI - Pocket - Test', 'RIDI - Text - Test']
    
    dirs_in_ds.sort() # so we can track each file
    
    # Load the 4 modes (4 dirs per train\test)
    # note that we load them in order 0,1,2,3 which is sorted so we can track each file with files_in_dir_all
    t_all_0,Pos_all_0,DCM_vec_all_0,Gyro_all_0,Acc_all_0,Mag_all_0,LinAcc_all_0,file_num_vec_all_0,files_in_dir_0 =  LoadRiDiDir(osp.join(RIDI_path,dirs_in_ds[0]))       
    dir_num_vec_0 = np.ones((len(t_all_0),1),dtype=np.int32)*int(0)
    t_all_1,Pos_all_1,DCM_vec_all_1,Gyro_all_1,Acc_all_1,Mag_all_1,LinAcc_all_1,file_num_vec_all_1,files_in_dir_1 =  LoadRiDiDir(osp.join(RIDI_path,dirs_in_ds[1]))       
    dir_num_vec_1 = np.ones((len(t_all_1),1),dtype=np.int32)*int(1)
    t_all_2,Pos_all_2,DCM_vec_all_2,Gyro_all_2,Acc_all_2,Mag_all_2,LinAcc_all_2,file_num_vec_all_2,files_in_dir_2 =  LoadRiDiDir(osp.join(RIDI_path,dirs_in_ds[2]))       
    dir_num_vec_2 = np.ones((len(t_all_2),1),dtype=np.int32)*int(2)
    t_all_3,Pos_all_3,DCM_vec_all_3,Gyro_all_3,Acc_all_3,Mag_all_3,LinAcc_all_3,file_num_vec_all_3,files_in_dir_3 =  LoadRiDiDir(osp.join(RIDI_path,dirs_in_ds[3]))       
    dir_num_vec_3 = np.ones((len(t_all_3),1),dtype=np.int32)*int(3)

    # Concatenate
    t_all = np.concatenate((t_all_0,t_all_1,t_all_2,t_all_3), axis=0)
    Pos_all = np.concatenate((Pos_all_0,Pos_all_1,Pos_all_2,Pos_all_3), axis=0)
    DCM_vec_all = np.concatenate((DCM_vec_all_0,DCM_vec_all_1,DCM_vec_all_2,DCM_vec_all_3), axis=0)
    Gyro_all = np.concatenate((Gyro_all_0,Gyro_all_1,Gyro_all_2,Gyro_all_3), axis=0)
    Acc_all = np.concatenate((Acc_all_0,Acc_all_1,Acc_all_2,Acc_all_3), axis=0)
    Mag_all = np.concatenate((Mag_all_0,Mag_all_1,Mag_all_2,Mag_all_3), axis=0)
    LinAcc_all = np.concatenate((LinAcc_all_0,LinAcc_all_1,LinAcc_all_2,LinAcc_all_3), axis=0)
    file_num_vec_all = np.concatenate((file_num_vec_all_0,file_num_vec_all_1,file_num_vec_all_2,file_num_vec_all_3), axis=0)
    files_in_dir_all = [files_in_dir_0,files_in_dir_1,files_in_dir_2,files_in_dir_3] #list with the same order as dirs_in_ds
    dir_num_vec_all = np.concatenate((dir_num_vec_0,dir_num_vec_1,dir_num_vec_2,dir_num_vec_3), axis=0)
    return t_all,Pos_all,DCM_vec_all,Gyro_all,Acc_all,Mag_all,LinAcc_all,file_num_vec_all,files_in_dir_all,dir_num_vec_all,dirs_in_ds

if __name__ == '__main__':
    file_name = 'hao_leg2.csv' 
    path_to_dir = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Test'
    TrainOrTest = 'Test'
    
    # Load arbitrary exp
    exp_path = osp.join(path_to_dir,file_name)
    t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv = ReadRidiCSV(exp_path)
    
    ### Test one: check LoadRiDiDir function
    if True:
        # Load its entire dir
        t_all,Pos_all,DCM_vec_all,Gyro_all,Acc_all,Mag_all,LinAcc_all,file_num_vec_all,files_in_dir =  LoadRiDiDir(path_to_dir)       
        
        # make sure we can track each sample
        file_num = files_in_dir.index(file_name)
        inds = np.where(file_num_vec_all==file_num)[0]
        t_reduced = t_all[inds,:]
        DCM_reduced = DCM_vec_all[inds,:,:]
    
        assert(np.all(t == t_reduced) )
        assert(np.all(DCM_vec == DCM_reduced))
    
    ### Test two: check LoadMultyDirs function
    if False:
        t_all,Pos_all,DCM_vec_all,Gyro_all,Acc_all,Mag_all,LinAcc_all, \
        file_num_vec_all,files_in_dir_all,dir_num_vec_all,dirs_in_ds = LoadMultyDirs(TrainOrTest)       
        
        # make sure we can track each sample
        dir_num = dirs_in_ds.index('RIDI - Pocket - Test')
        inds_dir = np.where(dir_num_vec_all==dir_num)[0]
        file_num = (files_in_dir_all[dir_num]).index(file_name)
        inds_file = np.where(file_num_vec_all==file_num)[0]
        inds = np.intersect1d(inds_file, inds_dir) 
        t_reduced = t_all[inds,:]
        DCM_reduced = DCM_vec_all[inds,:,:]
        
        assert(np.all(t == t_reduced))
        assert(np.all(DCM_vec == DCM_reduced))
        