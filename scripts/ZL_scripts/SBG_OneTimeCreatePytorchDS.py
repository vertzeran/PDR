#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:40:51 2023
A script to create X,y arrays for training SBG dataset
User should manual split the dirs (see for example the GetSwingTrainList function)
@author: zahi
"""
import numpy as np
import scipy.io as sio
from SBG_Loader import RunPrepareSequenceFromDirOnListOfDirs

def GetSwingTrainList():
    ListOfTrainDirs = ['22_07_12_swing_itzik',
    '22_07_27_swing_nati_L',
    '22_07_27_swing_zahi_R',
    '22_07_27_swing_zahi_L',
    '22_07_26_swing_ariel_L',
    '22_07_26_swing_ariel_R',
    '22_07_27_swing_nati_R',
    '22_08_01_swing_eran_L',
    '22_08_01_swing_eran_R',
    '22_08_02_swing_mani_R',
    '22_08_02_swing_mani_L',
    '22_08_02_swing_ofer_L',
    '22_08_02_swing_ofer_R',
    '22_08_02_swing_ran_L',
    '22_08_02_swing_ran_R',
    '22_09_15_swing_nadav_R',
    ]
    return ListOfTrainDirs

def GetSwingTestList():
    ListOfTestDirs = ['22_08_30_swing_sharon_R',
    '22_09_15_swing_yair_L',
    '22_09_15_swing_zeev_R']
    return ListOfTestDirs

def GetPocketTrainList():
    ListOfTrainDirs = ['21_11_10_omri',
    '21_11_28_eran',
    '21_11_28_itzik',
    '21_11_28_ofer',
    '21_11_28_omri',
    '22_09_15_pocket_nadav']
    return ListOfTrainDirs

def GetPocketTestList():
    ListOfTestDirs = ['22_08_30_pocket_sharon_R',
    '22_09_15_pocket_yair',
    '22_09_15_pocket_zeev']
    return ListOfTestDirs

def GetTextingTrainList():
    ListOfTrainDirs = ['21_10_31_eran',
    '21_11_07_mani',
    '21_11_07_ofer',
    '21_11_07_ran',
    '21_11_10_alex',
    '21_11_10_nati']
    return ListOfTrainDirs

def GetTextingTestList():
    ListOfTestDirs = ['21_11_10_demian',
    '21_11_07_firas',
    '21_11_10_omri']
    return ListOfTestDirs



if __name__ == '__main__':
    window_size = 200
    data_type = 'LinAcc' #'LinAcc' #'LinAccWithRV'
    root_of_roots = '/data/Datasets/Navigation/SBG-PDR-DATA/swing'
    ListOfTrainDirs = GetSwingTrainList()
    ListOfTestDirs = GetSwingTestList()
    assert(len(set(ListOfTestDirs) & set(ListOfTrainDirs)) == 0)    
    X_train, Y1_train, Y2_train = RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,ListOfTrainDirs,window_size,data_type=data_type)
    X_test, Y1_test, Y2_test = RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,ListOfTestDirs,window_size,data_type=data_type)
    sio.savemat('swing_train_'+data_type+'.mat', {'X':X_train,'Y1':Y1_train,'Y2':Y2_train})
    sio.savemat('swing_test_'+data_type+'.mat', {'X':X_test,'Y1':Y1_test,'Y2':Y2_test})
    
    root_of_roots = '/data/Datasets/Navigation/SBG-PDR-DATA/pocket'
    ListOfTrainDirs = GetPocketTrainList()
    ListOfTestDirs = GetPocketTestList()
    assert(len(set(ListOfTestDirs) & set(ListOfTrainDirs)) == 0)    
    X_train, Y1_train, Y2_train = RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,ListOfTrainDirs,window_size,data_type=data_type)
    X_test, Y1_test, Y2_test = RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,ListOfTestDirs,window_size,data_type=data_type)
    sio.savemat('pocket_train_'+data_type+'.mat', {'X':X_train,'Y1':Y1_train,'Y2':Y2_train})
    sio.savemat('pocket_test_'+data_type+'.mat', {'X':X_test,'Y1':Y1_test,'Y2':Y2_test})
    
    root_of_roots = '/data/Datasets/Navigation/SBG-PDR-DATA/texting'
    ListOfTrainDirs = GetTextingTrainList()
    ListOfTestDirs = GetTextingTestList()
    assert(len(set(ListOfTestDirs) & set(ListOfTrainDirs)) == 0)    
    X_train, Y1_train, Y2_train = RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,ListOfTrainDirs,window_size,data_type=data_type)
    X_test, Y1_test, Y2_test = RunPrepareSequenceFromDirOnListOfDirs(root_of_roots,ListOfTestDirs,window_size,data_type=data_type)
    sio.savemat('texting_train_'+data_type+'.mat', {'X':X_train,'Y1':Y1_train,'Y2':Y2_train})
    sio.savemat('texting_test_'+data_type+'.mat', {'X':X_test,'Y1':Y1_test,'Y2':Y2_test})