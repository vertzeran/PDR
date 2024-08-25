#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:09:41 2022
This code is too much specific for RIDI
TODO:
    1) Add documentation!
    2) Add turn detection
    3) Add Inv- pendulum
@author: zahi
"""
import torch
import numpy as np
from Ridi_Loader import ReadRidiCSV,find_csv_filenames
import os.path as osp
from utils import PlotTrajectories,CutByFirstDim,FixPCAamb,PlotCDF
from OperationalFunctions import PCAOnLinAccAtNav,PCAOnAccAtNav,PrepareInputForResnet18
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

### Init 
t_start = 0
t_stop = 60
window_size = 200 #100 samples window gives best resutls
mode = 'Pocket'
path_to_dir = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - mode - Test'
path_to_dir = path_to_dir.replace('mode', mode)

### Empty arrays to store results
wde_gt_all = np.array([]).reshape(-1,2) #2 cooardinates if you want to save (x,y)
wde_smh_all = np.array([]).reshape(-1,2) 
wde_pca_all = np.array([]).reshape(-1,2)
wde_pca_raw_all = np.array([]).reshape(-1,2)
wde_resnet_raw_all = np.array([]).reshape(-1,2)
wde_resnet_all = np.array([]).reshape(-1,2)
windows_dl_norm_all = np.array([]).reshape(-1,1)

### Resnet 18 Init (RawIMU)
if mode == 'Pocket': pth_path = 'WDE_regressor_Pocket_RawIMU_0.902.pth'
if mode == 'Text': pth_path = 'WDE_regressor_Text_RawIMU_0.902.pth'
if mode == 'Body': pth_path = 'WDE_regressor_Body_RawIMU_0.905.pth'
if mode == 'Bag': pth_path = 'WDE_regressor_Bag_RawIMU_0.951.pth'
net_raw = torch.load(pth_path)   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net_raw.to(device)  

### Resnet 18 Init (LinAcc)
if mode == 'Pocket': pth_path = 'WDE_regressor_Pocket_LinAcc_0.9.pth'
if mode == 'Text': pth_path = 'WDE_regressor_Text_LinAcc_0.902.pth'
if mode == 'Body': pth_path = 'WDE_regressor_Body_LinAcc_0.901.pth'
if mode == 'Bag': pth_path = 'WDE_regressor_Bag_LinAcc_0.95.pth'
net = torch.load(pth_path)   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)  

### Go over all files in the dir, and read_csv
files_in_dir = find_csv_filenames(path_to_dir) # a sorted list
for file_num,file_name in enumerate(files_in_dir):
    exp_path = osp.join(path_to_dir,file_name)
    t,Pos,Euler_vec,DCM_vec,Gyro,ACC,Mag,LinAcc,Grv = ReadRidiCSV(exp_path)
    print('Processing csv file:',file_name)
    
    ### Segment (note that cutting after processing is not the same as cutting before)
    ind_start = (np.abs(t -t[0] -t_start)).argmin()
    ind_stop = (np.abs(t -t[0] -t_stop)).argmin() #we are loosing one sample in the slicing
    ListOfTensors = t,Pos,Euler_vec,DCM_vec,Gyro,ACC,Mag,LinAcc,Grv
    t,Pos,Euler_vec,DCM_vec,Gyro,ACC,Mag,LinAcc,Grv = CutByFirstDim(ListOfTensors,ind_start,ind_stop)
    NumOfBatches = int(len(t)/window_size)

    ### GT and dl data (dl norm is assumed to be knon from PDRnet)
    Pos = Pos-Pos[0,:] # start trajectory at 0,0,0  
    windows_dl = Pos[window_size:-1:window_size] - Pos[0:-window_size-1:window_size] #N/window_size samples
    windows_dl_xy = windows_dl[:,:2]
    windows_dl_norm = norm(windows_dl_xy,axis=1,keepdims=True) #only x,y
    wde_gt = windows_dl_xy/windows_dl_norm
    
    ### Detect turns
    #turn_inds = DetectTurns(Pos)
    
    ### dt - dont take the mean
    dt = np.diff(t,axis=0)
    dt = np.vstack((dt[0],dt))

    ### Accelarations (raw and linear) to Nav frame
    raw_a_nav = np.einsum('ijk,ik->ij', DCM_vec, ACC)
    a_meas_nav = np.einsum('ijk,ik->ij', DCM_vec, LinAcc)

    ############################# Estimators ################################
    '''
    SM Heading (calculated in a "vectorized" manner on the entire file)
    See Doc at EstimatorOnOnetraj                                                                     
    '''
    Psi = Euler_vec[::window_size,0] #samling Psi
    NumOfWindowsToAvarage = 5 #I tuned this paramter with trial and error
    initial_Psi = np.arctan2(windows_dl[:NumOfWindowsToAvarage,1].mean(), windows_dl[:NumOfWindowsToAvarage,0].mean())
    Psi = Psi-Psi[0] + initial_Psi #Now phone Psi is the same as user Psi
    Psi = Psi[:-1].reshape(-1,1) #-1 is needed because Psi have one more sample than windows samples
    wde_est_smh = np.hstack((np.cos(Psi),np.sin(Psi))) * windows_dl_norm 


    '''
    PCA + RawPCA + resnet (this 3 have to use for loop)
    See Doc at EstimatorOnOnetraj                                                                     
    '''
    #WDE vectors to store the results per file. This WDE contain only (x,y). NumOfBatches is expected to be N/window_size
    wde_est_pca = np.zeros((NumOfBatches,2)) 
    wde_est_pca_raw = np.zeros((NumOfBatches,2))
    wde_est_resnet_raw = np.zeros((NumOfBatches,2))
    wde_est_resnet = np.zeros((NumOfBatches,2))

    for k in range(NumOfBatches):
        batch_of_a_nav = a_meas_nav[k*window_size:(k+1)*window_size,:] 
        batch_of_a_nav_raw = raw_a_nav[k*window_size:(k+1)*window_size,:] 
        
        # PCA:
        wde_est_pca[k,:] = PCAOnLinAccAtNav(batch_of_a_nav[:,:2]) #only x,y are input to PCA
        wde_est_pca[k,:] = FixPCAamb(wde_est_pca[k,:],windows_dl[k,:2]) #fix ambiguty
        
        # RAW PCA (we could have worked with LinAcc as well and the results will be the same: g is just a shift of point cloud):
        wde_est_pca_raw[k,:] = PCAOnAccAtNav(batch_of_a_nav_raw) #x,y,z, are input to PCA_raw
        wde_est_pca_raw[k,:] = FixPCAamb(wde_est_pca_raw[k,:],windows_dl[k,:2]) #fix ambiguty

        # Resnet 18:
        tmp1 = raw_a_nav[k*window_size:(k+1)*window_size,:] #x,y,z
        tmp2 = Gyro[k*window_size:(k+1)*window_size,:]
        tmp = np.hstack((tmp1,tmp2))
        tmp = tmp[None,:,:] #Adding the batch dim as pytorch expect
        resent_input = PrepareInputForResnet18(tmp) #from numpy to tensor
        resent_input = resent_input.to(device) #this cannot be done in operational function
        output = net_raw(resent_input)
        wde_est_resnet_raw[k,:] = output.cpu().detach().numpy()
        
        # Resnet 18:
        tmp = batch_of_a_nav[None,:,:] #Adding the batch dim as pytorch expect
        resent_input = PrepareInputForResnet18(tmp) #from numpy to tensor
        resent_input = resent_input.to(device) #this cannot be done in operational function
        output = net(resent_input)
        wde_est_resnet[k,:] = output.cpu().detach().numpy()
  
    ### Normalization (not necesry if we me measure the error distance with the "cosine sentence")
    wde_est_smh = wde_est_smh/norm(wde_est_smh,axis=1,keepdims=True)
    wde_est_pca = wde_est_pca/norm(wde_est_pca,axis=1,keepdims=True)
    wde_est_pca_raw = wde_est_pca_raw/norm(wde_est_pca_raw,axis=1,keepdims=True)
    wde_est_resnet_raw = wde_est_resnet_raw/norm(wde_est_resnet_raw,axis=1,keepdims=True)
    wde_est_resnet = wde_est_resnet/norm(wde_est_resnet,axis=1,keepdims=True)
    
    ### Stack wde vectors estimations from each file together 
    wde_gt_all = np.vstack((wde_gt_all,wde_gt))
    wde_smh_all = np.vstack((wde_smh_all,wde_est_smh))
    wde_pca_all = np.vstack((wde_pca_all,wde_est_pca))
    wde_pca_raw_all = np.vstack((wde_pca_raw_all,wde_est_pca_raw))
    wde_resnet_raw_all = np.vstack((wde_resnet_raw_all,wde_est_resnet_raw))
    wde_resnet_all = np.vstack((wde_resnet_all,wde_est_resnet))
    windows_dl_norm_all = np.vstack((windows_dl_norm_all,windows_dl_norm))

    ### Plot trajectories for each file (crucial to see if we have "bias estimator")
    if True:
        p_est_SMH = Pos[0,:2] + np.cumsum(wde_est_smh*windows_dl_norm,axis=0) #Assuming normlized WDE
        p_est_pca = Pos[0,:2] + np.cumsum(wde_est_pca*windows_dl_norm,axis=0) #Assuming normlized WDE
        p_est_pca_raw = Pos[0,:2] + np.cumsum(wde_est_pca_raw*windows_dl_norm,axis=0) #Assuming normlized WDE
        p_est_resnet_raw = Pos[0,:2] + np.cumsum(wde_est_resnet_raw*windows_dl_norm,axis=0) #Assuming normlized WDE
        p_est_resnet = Pos[0,:2] + np.cumsum(wde_est_resnet*windows_dl_norm,axis=0) #Assuming normlized WDE
        PlotTrajectories([Pos,p_est_SMH,p_est_pca,p_est_pca_raw,p_est_resnet_raw,p_est_resnet],['k','--b','--r','--g','--y','--c'],\
                         title='Trajectory for: '+ file_name, ListofLabels = ['GT','SM heading','PCA','Raw PCA','Resnet raw','Resnet'])
    
  
### "Distance CDF". Each direction error is "weighted" by dl norm , so we get the displacement of each time-window
displacement_smh = norm((wde_smh_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_pca = norm((wde_pca_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_pca_raw = norm((wde_pca_raw_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_resnet_raw = norm((wde_resnet_raw_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_resnet = norm((wde_resnet_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)

### Cosine Disimilarity (Calculated with the assumeion that all wde are normlized)
dism_smh = 1-np.sum((wde_smh_all*wde_gt_all),axis=1,keepdims=True)
dism_pca = 1-np.sum((wde_pca_all*wde_gt_all),axis=1,keepdims=True)
dism_pca_raw = 1-np.sum((wde_pca_raw_all*wde_gt_all),axis=1,keepdims=True)
dism_resnet_raw = 1-np.sum((wde_resnet_raw_all*wde_gt_all),axis=1,keepdims=True)
dism_resnet = 1-np.sum((wde_resnet_all*wde_gt_all),axis=1,keepdims=True)

### CDF plots on the entire dir (for example, all files in "Pocket-Test") 
PlotCDF([displacement_smh,displacement_pca,displacement_pca_raw,displacement_resnet_raw,displacement_resnet], ['k','--b','--r','--g','--c'], title='Displacement CDF', ListofLabels = ['SM heading','PCA','Raw PCA','Resnet raw','Resnet'], xlabel = 'Displacement[m]')
PlotCDF([dism_smh,dism_pca,dism_pca_raw,dism_resnet_raw,dism_resnet], ['k','--b','--r','--g','--c'], title='Cosine Loss CDF', ListofLabels = ['SM heading','PCA','Raw PCA','Resnet raw','Resnet'], xlabel = '1-Cos(gamma)')

### Prints
print('mean displacement_smh:',displacement_smh.mean().round(3))
print('mean displacement_pca:',displacement_pca.mean().round(3))
print('mean displacement_pca_raw:',displacement_pca_raw.mean().round(3))
print('mean displacement_resnet_raw:',displacement_resnet_raw.mean().round(3))
print('mean displacement_resnet:',displacement_resnet.mean().round(3))