#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:55:28 2022
This script plot the GT data of a given experiment,
and then also plots the trajectory that will be estimated
by a perfect estimator.
Since that the estimated dl (of ~1sec movment) assums that the human 
had walked in a stright line, rather than a curved line,
we expect to see some differance as the window_size is getting bigger
@author: zahi
"""
import numpy as np
import matplotlib.pyplot as plt
from Ridi_Loader import ReadRidiCSV
from SBG_Loader import LoadExpAndAHRS_V2
### Init 
window_size_list = [100,200,400,800]
col_list = ['--b','--r','--g','--y']
t_start = 0
t_stop = 60
RIDI_or_SBG = 'SBG'

### Load RIDI
if RIDI_or_SBG == 'RIDI':
    exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Train/hang_leg1.csv'
    #exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Body - Train/tang_body1.csv'
    t,Pos,Euler_vec,DCM_vec,Gyro,ACC,Mag,LinAcc,Grv = ReadRidiCSV(exp_path)

### Load SBG
if RIDI_or_SBG == 'SBG':
    exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R/outdoor_output_2022-07-27_08_53_54.csv'
    t,Pos,RV,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv = LoadExpAndAHRS_V2(exp_path,t_gt=None,Pos_gt=None)
    
### GT Pos 
ind_start = (np.abs(t -t[0] -t_start)).argmin()
ind_stop = (np.abs(t -t[0] -t_stop)).argmin() #we are loosing one sample in the slicing
GT = Pos[ind_start:ind_stop,:] - Pos[ind_start,:] #start trajectory at 0,0,0

### plot
plt.figure()
plt.title('GT and Perfectly Estimated trajectory')
plt.plot(GT[:,0],GT[:,1],'k',label = 'GT')

### Estimated Trajectory
for k,window_size in enumerate(window_size_list):
    windows_dl = GT[window_size:-1:window_size] - GT[0:-window_size-1:window_size] 
    EstimatedPos = np.cumsum(windows_dl,axis=0)
    EstimatedPos = np.vstack((np.zeros((1,3)),EstimatedPos)) #start trajectory at 0,0,0
    plt.plot(EstimatedPos[:,0],EstimatedPos[:,1],col_list[k],label = 'windows_size = ' + str(window_size))
plt.legend()
plt.grid() 