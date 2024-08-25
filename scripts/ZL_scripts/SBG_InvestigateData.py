#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:20:41 2023

@author: zahi
"""
import numpy as np
from utils import Plot3CordVecs,PlotCDF,PlotTrajectories,NeedToSkipDueToBadValidErr,FixPCAamb
from SBG_Loader import LoadExpAndAHRS_V2,PrepareSequence,LoadGT
from os import listdir
from os.path import join
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

### validate trajectories and GT - PASS
# This code prints a lot of trajectories so it is disabled by default
if False:
    #root_dir = '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R'
    #root_dir =  '/data/Datasets/Navigation/SBG-PDR-DATA/pocket/22_09_15_pocket_zeev'
    #root_dir = '/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_firas' # up to 200ms gap!!
    root_dir = '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_09_15_swing_nadav_R'
    data_list = listdir(root_dir)
    data_list = [item for item in data_list if '_AHRS_results.xlsx' not in item]
    data_list = [item for item in data_list if 'ascii-output.txt' not in item]
    for ind,file_name in enumerate(data_list):
        if ind == 0: #Load the GT only if this is the first time
            GT_path = join(root_dir,'ascii-output.txt')
            t_gt,Pos_gt = LoadGT(GT_path)
            dt_gt = np.diff(t_gt,axis=0)
            print('Max dt_gt: ',dt_gt.max(),'  Min dt_gt: ',dt_gt.min())
        exp_path = join(root_dir, file_name)               
        print('Loading: ',file_name)
        t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=t_gt,Pos_gt=Pos_gt)
        if NeedToSkipDueToBadValidErr(ValidErr,file_name): 
            continue
        PlotTrajectories([Pos],['--b'],ListofLabels=['traj'],title = str(file_name))

### Load GT 
#GT_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R/ascii-output.txt'
#GT_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_09_15_swing_yair_L/ascii-output.txt'
#GT_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/pocket/21_11_10_omri/ascii-output.txt'
GT_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_firas/ascii-output.txt'
t_gt,Pos_gt = LoadGT(GT_path)
dt_gt = np.diff(t_gt,axis=0)
bad_inds = np.where(dt_gt > 0.02)[0]
bad_inds_neg = np.where(dt_gt < 0)[0]
print('GT dt unusual values:',dt_gt[bad_inds],dt_gt[bad_inds_neg])
plt.figure()
plt.plot(dt_gt,linestyle="None",marker='x')

### Load exp 
exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R/outdoor_output_2022-07-27_08_56_16.csv'
#exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_09_15_swing_yair_L/outdoor_output_2022-09-15_10_23_40.csv'
t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=None,Pos_gt=None)
print('Done loading')
if NeedToSkipDueToBadValidErr(ValidErr,'main'):
    raise SystemExit

### Acc at nav frame
lin_a_nav = np.einsum('ijk,ik->ij', DCM_vec, lin_acc_b_frame)  # Nx3
raw_a_nav = np.einsum('ijk,ik->ij', DCM_vec, grv_hat+lin_acc_b_frame)

### Comapring measured phone acc to human gt acc - FAILED!
Pos_filtered = np.zeros_like(Pos)
ker = np.ones((10,))
Pos_filtered[:,0] = np.convolve(Pos[:,0],ker, mode='same')
Pos_filtered[:,1] = np.convolve(Pos[:,1],ker, mode='same')
Pos_filtered[:,2] = np.convolve(Pos[:,2],ker, mode='same')
Pos_filtered = Pos_filtered/np.linalg.norm(ker)**2
dt = np.diff(t_exp,axis=0)
#dt = np.diff(t,axis=0).mean() uncomment this to see how bad it is
v_gt = np.diff(Pos_filtered,axis=0)/dt
a_gt = np.diff(v_gt,axis=0)/dt[:-1]
# Plot3CordVecs([a_gt,lin_a_nav],['k','--r'],ListofTitles = ['Ax','Ay','Az'], xlabel='samples', ylabel='m/sec^2')

### Comapring calculated phone Pos to human gt Pos - FAILED!
v_est = np.cumsum(lin_a_nav[:-1,:]*dt,axis=0) + 0
p_est = np.cumsum(v_est*dt,axis=0) + 0
p_est = p_est - p_est[0,:] #start trajectory at 0,0,0
# Plot3CordVecs([Pos,p_est+Pos[0,:]],['k','--r'],ListofTitles = ['X','Y','Z'], xlabel='samples', ylabel='[m]')

### Android rotation vs AHRS - pass in pocket and texting
r_ahrs = Rotation.from_matrix(DCM_vec)
r_android = Rotation.from_quat(RV)
euler_ahrs = r_ahrs.as_euler('zxy')
euler_android = r_android.as_euler('zxy')
Plot3CordVecs([euler_ahrs*180/np.pi,euler_android*180/np.pi],['b','r'],ListofTitles = ['Psi','Theta','Phi'], xlabel='samples', ylabel='deg')

### some plots
# Plot3CordVecs([Gyro],['k'], suptitle='Gyro', ListofTitles = ['Wx','Wy','Wz'], ylabel='rad/sec')
# Plot3CordVecs([lin_a_nav,lin_acc_b_frame],['b','r'], suptitle='Linear ACC at Nav and body frame', ListofTitles = ['Ax','Ay','Az'], ylabel='m/sec^2')
# Plot3CordVecs([grv_hat],['k'],ListofTitles = ['Ax','Ay','Az'], xlabel='samples', ylabel='m/sec^2')
# Plot3CordVecs([grv_hat+lin_acc_b_frame],['k'],ListofTitles = ['Ax','Ay','Az'], xlabel='samples', ylabel='m/sec^2')
# PlotTrajectories([Pos],['--b'],ListofLabels=['traj'],title = str(exp_path))

### Prepare segments
window_size=200
# X,Y1,Y2 = PrepareSequence(Pos,lin_acc_b_frame,DCM_vec,window_size)
# assert(all(np.linalg.norm(Y2,axis=1)-1.0<1e-10))
# assert(X.shape[1]==window_size)


### PCA test - Partily Pass ### THIS IS a duplicated code from EstimateOnOneTraj !!
Pos = Pos-Pos[0,:] # start trajectory at 0,0,0  
windows_dl = Pos[window_size:-1:window_size] - Pos[0:-window_size-1:window_size] #N/window_size samples
windows_dl_norm = np.linalg.norm(windows_dl[:,:2],axis=1,keepdims=True) #only x,y
t_start = 0
t_stop = 60
NumOfBatches = int(len(t_exp)/window_size)

if False:
    wde_est = np.zeros((NumOfBatches,2)) #(only x,y) , NumOfBatches is expected to be N/window_size
    wde_est_raw = np.zeros((NumOfBatches,2)) #(only x,y) , NumOfBatches is expected to be N/window_size

    for k in range(NumOfBatches):
        batch_of_a_nav = lin_a_nav[k*window_size:(k+1)*window_size,:] 
        batch_of_a_nav_raw = raw_a_nav[k*window_size:(k+1)*window_size,:] 

        # PCA:
        pca = PCA(n_components=2).fit(batch_of_a_nav[:,:2]) #only x,y
        wde_est[k,:] = pca.components_[0].reshape(1,2) #take 1st component
        wde_est[k,:] = FixPCAamb(wde_est[k,:],windows_dl[k,:2]) #fix ambiguty
        
        # RAW PCA:
        pca = PCA(n_components=3).fit(batch_of_a_nav_raw)
        wde_est_raw[k,:] = pca.components_[1,:2].reshape(1,2) #take 2nd component
        wde_est_raw[k,:] = FixPCAamb(wde_est_raw[k,:],windows_dl[k,:2]) #fix ambiguty
            
    p_est = Pos[0,:2] + np.cumsum(windows_dl_norm*wde_est/np.linalg.norm(wde_est,axis=1,keepdims=True),axis=0)
    p_est_raw = Pos[0,:2] + np.cumsum(windows_dl_norm*wde_est_raw/np.linalg.norm(wde_est_raw,axis=1,keepdims=True),axis=0)

    Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    Pos_raw_RMSE = np.linalg.norm(p_est_raw[-1,:]-Pos[-1,:2],keepdims=True)
    print('PCA RMSE:',Pos_RMSE,Pos_raw_RMSE)
    PlotTrajectories([Pos,p_est,p_est_raw],['k','--b','--r'],title='GT and PCA-estimated trajectory', ListofLabels = ['GT','PCA','Raw PCA'])