#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on sunday 15 januar 2023
This code is duplication of the code for RIDI (may god forgive me)
because doing it generic was too clumsy (SBG load the GT only one time..)
@author: zahi
"""
import torch
import numpy as np
from SBG_Loader import LoadExpAndAHRS_V2,PrepareSequence,LoadGT
import os.path as osp
from os import listdir
from utils import PlotTrajectories,CutByFirstDim,FixPCAamb,PlotCDF,NeedToSkipDueToBadValidErr,EstimateTraj
from OperationalFunctions import PCAOnLinAccAtNav,PCAOnAccAtNav,PrepareInputForResnet18
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt
from SBG_OneTimeCreatePytorchDS import GetSwingTestList,GetPocketTestList,GetTextingTestList
from scipy.spatial.transform import Rotation
import scipy.io as sio

### Init 
t_start = 0
t_stop = 6000
window_size = 200 #100 samples window gives best resutls
mode = 'pocket' #pocket,swing,texting
root_of_roots = '/data/Datasets/Navigation/SBG-PDR-DATA/mode'
root_of_roots = root_of_roots.replace('mode', mode)
if mode == 'pocket': list_of_dirs = GetPocketTestList()
if mode == 'swing': list_of_dirs = GetSwingTestList()
if mode == 'texting': list_of_dirs = GetTextingTestList()

### Empty arrays to store results
wde_gt_all = np.array([]).reshape(-1,2) #2 cooardinates if you want to save (x,y)
wde_smh_all = np.array([]).reshape(-1,2) 
wde_pca_all = np.array([]).reshape(-1,2)
wde_pca_raw_all = np.array([]).reshape(-1,2)
wde_resnet_rv_all = np.array([]).reshape(-1,2)
wde_resnet_all = np.array([]).reshape(-1,2)
windows_dl_norm_all = np.array([]).reshape(-1,1)
path_len_all = np.array([]).reshape(-1,1)
pos_all = np.array([]).reshape(-1,2) #needed for statitics
p_est_SMH_all = np.array([]).reshape(-1,2)
p_est_pca_all = np.array([]).reshape(-1,2)
p_est_pca_raw_all = np.array([]).reshape(-1,2)
p_est_resnet_rv_all = np.array([]).reshape(-1,2)
p_est_resnet_all = np.array([]).reshape(-1,2)

### Resnet 18 Init (LinACCWithRV)
if mode == 'pocket': pth_path = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_pocket_LinAccWithRV_Displacment_0.299.pth'
if mode == 'swing': pth_path = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_swing_LinAccWithRV_Displacment_0.248.pth'
if mode == 'texting': pth_path = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_texting_LinAccWithRV_Displacment_0.199.pth'
net_rv = torch.load(pth_path)   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net_rv.to(device)  

### Resnet 18 Init (LinAcc)
if mode == 'pocket': pth_path = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_pocket_Displacment_0.295.pth'
if mode == 'swing': pth_path = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_swing_Displacment_0.348.pth'
if mode == 'texting': pth_path = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_texting_Displacment_0.25.pth'
net = torch.load(pth_path)   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)  

### Go over all files in the dir, and read_csv
for ind,root_dir in enumerate(list_of_dirs):
    print('\n ### Working on dir: ',root_dir,' ###')
    root_dir = osp.join(root_of_roots,root_dir)
    data_list = listdir(root_dir)
    data_list = [item for item in data_list if '_AHRS_results.xlsx' not in item]
    data_list = [item for item in data_list if 'ascii-output.txt' not in item]
    for ind,file_name in enumerate(data_list):
        if ind == 0: #Load the GT only if this is the first time
            GT_path = osp.join(root_dir,'ascii-output.txt')
            t_gt,Pos_gt = LoadGT(GT_path)
        exp_path = osp.join(root_dir, file_name)               
        print('Processing csv file:',file_name)
        t,Pos,RV,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=t_gt,Pos_gt=Pos_gt)
        Euler_vec = Rotation.from_quat(RV).as_euler('zxy')
        if NeedToSkipDueToBadValidErr(ValidErr,file_name): 
            continue
        
        """ From Here and downwards it is a duplication of code. sorry.. """
        
        ### Segment (note that cutting after processing is not the same as cutting before)
        ind_start = (np.abs(t -t[0] -t_start)).argmin()
        ind_stop = (np.abs(t -t[0] -t_stop)).argmin() #we are loosing one sample in the slicing
        ListOfTensors = t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv,RV
        t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv,RV = CutByFirstDim(ListOfTensors,ind_start,ind_stop)
        NumOfBatches = int(len(t)/window_size)
        if (NumOfBatches < 1):
            # probably bad record of standing user like firas 11_45_52
            print('Bad record (probably a standing man):',file_name)
            continue
        
        ### GT and dl data (dl norm is assumed to be known from PDRnet)
        Pos = Pos-Pos[0,:] # start trajectory at 0,0,0  
        windows_dl = Pos[window_size:-1:window_size] - Pos[0:-window_size-1:window_size] #N/window_size samples
        windows_dl_xy = windows_dl[:,:2]
        windows_dl_norm = norm(windows_dl_xy,axis=1,keepdims=True) #only x,y
        assert(all(windows_dl_norm>0)) #We should have cut the standing time
        wde_gt = windows_dl_xy/windows_dl_norm
        
        ### Detect turns
        #turn_inds = DetectTurns(Pos)
        
        ### dt - dont take the mean
        dt = np.diff(t,axis=0)
        dt = np.vstack((dt[0],dt))
    
        ### Accelarations (raw and linear) to Nav frame
        raw_a_nav = np.einsum('ijk,ik->ij', DCM_vec, Acc)
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
        PCA + RawPCA + resnet + resnet_rv (this 4 have to use for loop)
        See Doc at EstimatorOnOnetraj                                                                     
        '''
        #WDE vectors to store the results per file. This WDE contain only (x,y). NumOfBatches is expected to be N/window_size
        wde_est_pca = np.zeros((NumOfBatches,2)) 
        wde_est_pca_raw = np.zeros((NumOfBatches,2))
        wde_est_resnet_rv = np.zeros((NumOfBatches,2))
        wde_est_resnet = np.zeros((NumOfBatches,2))
    
        for k in range(NumOfBatches):
            batch_of_a_nav = a_meas_nav[k*window_size:(k+1)*window_size,:] 
            batch_of_a_nav_raw = raw_a_nav[k*window_size:(k+1)*window_size,:] 
            batch_of_RV = RV[k*window_size:(k+1)*window_size,:]
            # PCA:
            wde_est_pca[k,:] = PCAOnLinAccAtNav(batch_of_a_nav[:,:2]) #only x,y are input to PCA
            wde_est_pca[k,:] = FixPCAamb(wde_est_pca[k,:],windows_dl[k,:2]) #fix ambiguty
            
            # RAW PCA (we could have worked with LinAcc as well and the results will be the same: g is just a shift of point cloud):
            wde_est_pca_raw[k,:] = PCAOnAccAtNav(batch_of_a_nav_raw) #x,y,z, are input to PCA_raw
            wde_est_pca_raw[k,:] = FixPCAamb(wde_est_pca_raw[k,:],windows_dl[k,:2]) #fix ambiguty
    
            # Resnet 18 (LinACCWithRV):
            tmp = np.hstack((batch_of_a_nav,batch_of_RV))
            tmp = tmp[None,:,:] #Adding the batch dim as pytorch expect
            resent_input = PrepareInputForResnet18(tmp) #from numpy to tensor
            resent_input = resent_input.to(device) #this cannot be done in operational function
            output = net_rv(resent_input)
            wde_est_resnet_rv[k,:] = output.cpu().detach().numpy()
            
            # Resnet 18 (LinACC):
            tmp = batch_of_a_nav[None,:,:] #Adding the batch dim as pytorch expect
            resent_input = PrepareInputForResnet18(tmp) #from numpy to tensor
            resent_input = resent_input.to(device) #this cannot be done in operational function
            output = net(resent_input)
            wde_est_resnet[k,:] = output.cpu().detach().numpy()
      
        ### Normalization (not necesry if we me measure the error distance with the "cosine sentence")
        wde_est_smh = wde_est_smh/norm(wde_est_smh,axis=1,keepdims=True)
        wde_est_pca = wde_est_pca/norm(wde_est_pca,axis=1,keepdims=True)
        wde_est_pca_raw = wde_est_pca_raw/norm(wde_est_pca_raw,axis=1,keepdims=True)
        wde_est_resnet_rv = wde_est_resnet_rv/norm(wde_est_resnet_rv,axis=1,keepdims=True)
        wde_est_resnet = wde_est_resnet/norm(wde_est_resnet,axis=1,keepdims=True)
        
        ### Stack wde vectors estimations from each file together 
        wde_gt_all = np.vstack((wde_gt_all,wde_gt))
        wde_smh_all = np.vstack((wde_smh_all,wde_est_smh))
        wde_pca_all = np.vstack((wde_pca_all,wde_est_pca))
        wde_pca_raw_all = np.vstack((wde_pca_raw_all,wde_est_pca_raw))
        wde_resnet_rv_all = np.vstack((wde_resnet_rv_all,wde_est_resnet_rv))
        wde_resnet_all = np.vstack((wde_resnet_all,wde_est_resnet))
        windows_dl_norm_all = np.vstack((windows_dl_norm_all,windows_dl_norm))
    
        ### Calc position estimation for normlized error or trajectory plot
        path_len = np.cumsum(windows_dl_norm,axis=0)
        p_est_SMH = EstimateTraj(Pos,wde_est_smh,windows_dl_norm)
        p_est_pca = EstimateTraj(Pos,wde_est_pca,windows_dl_norm)
        p_est_pca_raw = EstimateTraj(Pos,wde_est_pca_raw,windows_dl_norm)
        p_est_resnet_rv = EstimateTraj(Pos,wde_est_resnet_rv,windows_dl_norm)
        p_est_resnet = EstimateTraj(Pos,wde_est_resnet,windows_dl_norm)       
        
        ### Stack normlized_error vectors estimations from each file together 
        # [1:,:] is requiered because trajectory starts at 0,0 with L = 0
        path_len_all = np.vstack((path_len_all,path_len))
        pos_all = np.vstack((pos_all,Pos[window_size::window_size,:2]))
        p_est_SMH_all = np.vstack((p_est_SMH_all,p_est_SMH[1:,:]))
        p_est_pca_all = np.vstack((p_est_pca_all,p_est_pca[1:,:]))
        p_est_pca_raw_all = np.vstack((p_est_pca_raw_all,p_est_pca_raw[1:,:]))
        p_est_resnet_rv_all = np.vstack((p_est_resnet_rv_all,p_est_resnet_rv[1:,:]))
        p_est_resnet_all = np.vstack((p_est_resnet_all,p_est_resnet[1:,:]))
        
        ### Plot trajectories for each file (crucial to see if we have "bias estimator")
        if (NumOfBatches>10) and True:
            best_estimator = p_est_resnet if mode == 'pocket' else p_est_resnet_rv
            # PlotTrajectories([Pos,p_est_SMH,p_est_pca,p_est_resnet_rv,p_est_resnet],\
            #                   ['k','--b','--r','--g','--c'],\
            #                   title = mode +'_trajectory_'+ root_dir.split('/')[-1] + '_' + file_name,\
            #                   ListofLabels = ['GT','SM heading','PCA','Resnet RV','Resnet'],\
            #                   SaveAndClose = True)
            PlotTrajectories([Pos,best_estimator],\
                              ['k','--g'],\
                              title = mode +'_trajectory_'+ root_dir.split('/')[-1] + '_' + file_name,\
                              ListofLabels = ['GT','TRC estimator'],\
                              SaveAndClose = True)
  
### "Distance CDF". Each direction error is "weighted" by dl norm , so we get the displacement of each time-window
displacement_smh = norm((wde_smh_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_pca = norm((wde_pca_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_pca_raw = norm((wde_pca_raw_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_resnet_rv = norm((wde_resnet_rv_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)
displacement_resnet = norm((wde_resnet_all-wde_gt_all)*windows_dl_norm_all,axis=1,keepdims=True)

### Cosine Disimilarity (Calculated with the assumeion that all wde are normlized)
dism_smh = 1-np.sum((wde_smh_all*wde_gt_all),axis=1,keepdims=True)
dism_pca = 1-np.sum((wde_pca_all*wde_gt_all),axis=1,keepdims=True)
dism_pca_raw = 1-np.sum((wde_pca_raw_all*wde_gt_all),axis=1,keepdims=True)
dism_resnet_rv = 1-np.sum((wde_resnet_rv_all*wde_gt_all),axis=1,keepdims=True)
dism_resnet = 1-np.sum((wde_resnet_all*wde_gt_all),axis=1,keepdims=True)

### Angle error:
ang_err_smh = np.arccos(np.sum((wde_smh_all*wde_gt_all),axis=1,keepdims=True))
ang_err_pca = np.arccos(np.sum((wde_pca_all*wde_gt_all),axis=1,keepdims=True))
ang_err_pca_raw = np.arccos(np.sum((wde_pca_raw_all*wde_gt_all),axis=1,keepdims=True))
ang_err_resnet_rv = np.arccos(np.sum((wde_resnet_rv_all*wde_gt_all),axis=1,keepdims=True))
ang_err_resnet = np.arccos(np.sum((wde_resnet_all*wde_gt_all),axis=1,keepdims=True))

### Normlized error
normed_error_smh = np.linalg.norm(pos_all - p_est_SMH_all,axis=1,keepdims=True) / path_len_all
normed_error_pca = np.linalg.norm(pos_all - p_est_pca_all,axis=1,keepdims=True) / path_len_all
normed_error_pca_raw = np.linalg.norm(pos_all- p_est_pca_raw_all,axis=1,keepdims=True) / path_len_all
normed_error_resnet_rv = np.linalg.norm(pos_all - p_est_resnet_rv_all,axis=1,keepdims=True) / path_len_all
normed_error_resnet = np.linalg.norm(pos_all - p_est_resnet_all,axis=1,keepdims=True) / path_len_all

### CDF plots on the entire dir (for example, all files in "Pocket-Test") 
PlotCDF([displacement_smh,displacement_pca,displacement_pca_raw,displacement_resnet_rv,displacement_resnet], ['--k','--b','--r','--g','--c'], title='Displacement CDF', ListofLabels = ['SM heading','PCA','Raw PCA','Resnet RV','Resnet'], xlabel = 'Displacement[m]')
PlotCDF([dism_smh,dism_pca,dism_pca_raw,dism_resnet_rv,dism_resnet], ['--k','--b','--r','--g','--c'], title='Cosine Loss CDF', ListofLabels = ['SM heading','PCA','Raw PCA','Resnet RV','Resnet'], xlabel = '1-Cos(gamma)')
PlotCDF([ang_err_smh,ang_err_pca,ang_err_pca_raw,ang_err_resnet_rv,ang_err_resnet], ['--k','--b','--r','--g','--c'], title='Angle Error CDF', ListofLabels = ['SM heading','PCA','Raw PCA','Resnet RV','Resnet'], xlabel = 'Ang[rad]')
PlotCDF([normed_error_smh,normed_error_pca,normed_error_pca_raw,normed_error_resnet_rv,normed_error_resnet], ['--k','--b','--r','--g','--c'], title='Normlized Error CDF', ListofLabels = ['SM heading','PCA','Raw PCA','Resnet RV','Resnet'], xlabel = '{Position Error / Path Len}')

### Prints
print('mean displacement_smh:',displacement_smh.mean().round(3) ,' Normed error: ',normed_error_smh.mean().round(3))
print('mean displacement_pca:',displacement_pca.mean().round(3),' Normed error: ',normed_error_pca.mean().round(3))
print('mean displacement_pca_raw:',displacement_pca_raw.mean().round(3),' Normed error: ',normed_error_pca_raw.mean().round(3))
print('mean displacement_resnet_rv:',displacement_resnet_rv.mean().round(3),' Normed error: ',normed_error_resnet_rv.mean().round(3))
print('mean displacement_resnet:',displacement_resnet.mean().round(3),' Normed error: ',normed_error_resnet.mean().round(3))

if False:
    sio.savemat(mode + '_results' + '.mat',
                {'pos_all':pos_all,'wde_gt_all':wde_gt_all,'windows_dl_norm_all':windows_dl_norm_all,'path_len_all':path_len_all,
                 'wde_smh_all':wde_smh_all,'wde_pca_all':wde_pca_all,'wde_resnet_rv_all': wde_resnet_rv_all,'wde_resnet_all':wde_resnet_all,
                 'p_est_SMH_all':p_est_SMH_all,'p_est_pca_all':p_est_pca_all,'p_est_resnet_rv_all':p_est_resnet_rv_all,'p_est_resnet_all':p_est_resnet_all})
