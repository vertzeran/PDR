#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:03:30 2022
A script to learn the dataset with plots
some insights:
    1) Z axis is drifting
    2) LinAcc is not equal to ACC-Grav
    3) Euler convention here is ZYX. X is the short smartphone axis
    since it is not clear wheter we use intrinsic \ extrinsic convention - Just use the DCM!
@author: zahi
"""
import numpy as np
from Ridi_Loader import ReadRidiCSV
from utils import Plot3CordVecs,PlotCDF,PlotTrajectories

#exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Train/hang_leg1.csv'
exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Body - Train/tang_body1.csv'

t,Pos,Euler_vec,DCM_vec,Gyro,ACC,Mag,LinAcc,Grv = ReadRidiCSV(exp_path)

# some plots
# Plot3CordVecs([ACC],['b'], suptitle='ACC', ListofTitles = ['Ax','Ay','Az'], ylabel='m/sec^2')
# Plot3CordVecs([ACC,Grv],['b','k'], suptitle='ACC and Grv', ListofTitles = ['Ax','Ay','Az'], ylabel='m/sec^2')
# Plot3CordVecs([ACC-Grv,LinAcc],['b','k'], suptitle='LinAcc GT suspected error', ListofTitles = ['Ax','Ay','Az'], ylabel='m/sec^2')
Plot3CordVecs([ACC-Grv-LinAcc],['--r'], suptitle='GT suspected error: ACC-Grv-LinAcc', ListofTitles = ['Ax','Ay','Az'], ylabel='m/sec^2')
# Plot3CordVecs([Pos],['b'], suptitle='GT Location', ListofTitles = ['x','y','z'], ylabel='[m]')
# Plot3CordVecs([Euler_vec],['k'], suptitle='GT Euler', ListofTitles = ['psi','theta','phi'], ylabel='[rad]')
# Plot3CordVecs([v_gt],['b'],ListofTitles = ['Vx','Vy','Vz'], xlabel='samples', ylabel='m/sec')

# validate rotation
a_est_nav = np.einsum('ijk,ik->ij', DCM_vec, LinAcc)  # same as a_est_nav[k,:] = DCM_vec[k,:,:]@(LinAcc[k,:].T)   
dt = np.diff(t,axis=0)
#dt = np.diff(t,axis=0).mean() uncomment this to see how bad it is
v_gt = np.diff(Pos,axis=0)/dt
a_gt = np.diff(v_gt,axis=0)/dt[:-1]
Plot3CordVecs([a_gt,a_est_nav],['k','--r'],ListofTitles = ['Ax','Ay','Az'], xlabel='samples', ylabel='m/sec^2')

# Turn identification-Eran method (90% of segment have std smaller than 17[deg]):
window_size = 200
NumOfBatches = int(len(t)/window_size)
last_ind = len(t)-len(t)%window_size
v_gt_segments = np.reshape(v_gt[:last_ind],(-1,window_size,3)) #(segment num, fast time ind, 3D dims)
wd_angles = np.unwrap(np.arctan2(v_gt_segments[:,:,1],v_gt_segments[:,:,0]),axis=1)
wd_angles_std = np.std(wd_angles,axis=1,keepdims=True)
# PlotCDF([wd_angles_std*180/np.pi],['.'], title='STD of WD angles in segment', ListofLabels = ['std'], xlabel = '[deg]')
turn_inds = np.where(wd_angles_std>0.3)[0]
turn_inds_rolled = np.zeros((len(t),), dtype=bool)
for ind in turn_inds:
    turn_inds_rolled[window_size*ind:window_size*(ind+1)] = True
# Turn identification - plot trajectory
PlotTrajectories([Pos,Pos[turn_inds_rolled,:]],['b','or'], title='Turns detection',ListofLabels = ['Pos','Turns'])