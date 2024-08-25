#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:59:05 2022
A collection of estimators for WDE, that we run on a single file
The puprpose it to show some in-depth plots and not only error statistics.
Assumptions (All assume ACC is measired):
    0) Integration: The orientation and therefore the DCM is known from AHRS
    1) PCA: Orientation + DCM + ||dl|| (known from PDRnet) + Gravity + GT sign for ambiguty fix
    1.5) Raw PCA: same as above without Gravity
    2) SM Heading: Orientation + DCM + dl + Initial Psy (from AHRS)
    3) CNN: Orientation + DCM + ||dl|| + Gravity
The purpose of this script is to try the algo on one file to validate
that there are no bugs - and then to use them in other scripts
@author: zahi
"""
import numpy as np
import matplotlib.pyplot as plt
from Ridi_Loader import ReadRidiCSV
from utils import Plot3CordVecs,PlotTrajectories,CutByFirstDim,FixPCAamb,NeedToSkipDueToBadValidErr
from sklearn.decomposition import PCA
import torch
from OperationalFunctions import PrepareInputForResnet18
from scipy import signal
from SBG_Loader import LoadExpAndAHRS_V2
from scipy.spatial.transform import Rotation

print('*** Warning - User should write matched resnet path manually ***')

### Init 
if False:
    #exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Train/hang_leg1.csv'
    exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Test/hao_leg2.csv'
    t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv = ReadRidiCSV(exp_path)
    pth_path_raw = '/data/Work/Navigation/TrainedModels/RIDI_WDE_regressor_Pocket_RawIMU_0.902.pth'
    pth_path = '/data/Work/Navigation/TrainedModels/RIDI_WDE_regressor_Pocket_LinAcc_0.9.pth'
if True:
    # exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_07_27_swing_zahi_R/outdoor_output_2022-07-27_08_56_16.csv'
    # exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_mani/outdoor_output_2021-11-07_12_20_21.csv'
    exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/pocket/22_08_30_pocket_sharon_R/outdoor_output_2022-08-30_09_27_32.csv'
    t,Pos,RV,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=None,Pos_gt=None)
    Euler_vec = Rotation.from_quat(RV).as_euler('zxy')
    if NeedToSkipDueToBadValidErr(ValidErr,'main'):
        raise SystemExit
    pth_path_rv = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_pocket_LinAccWithRV_Displacment_0.299.pth'
    pth_path = '/data/Work/Navigation/TrainedModels/SBG_WDE_regressor_pocket_Displacment_0.295.pth'
    print(ValidErr)
t_start = 0
t_stop = 60
window_size = 200 #100 samples window gives best resutls

### Segment (note that cutting after processing is not the same as cutting before)
ind_start = (np.abs(t -t[0] -t_start)).argmin()
ind_stop = (np.abs(t -t[0] -t_stop)).argmin() #we are loosing one sample in the slicing
ListOfTensors = t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv
t,Pos,Euler_vec,DCM_vec,Gyro,Acc,Mag,LinAcc,Grv = CutByFirstDim(ListOfTensors,ind_start,ind_stop)
NumOfBatches = int(len(t)/window_size)

### GT and dl data (dl norm is assumed to be knon from PDRnet)
Pos = Pos-Pos[0,:] # start trajectory at 0,0,0  
windows_dl = Pos[window_size:-1:window_size] - Pos[0:-window_size-1:window_size] #N/window_size samples
windows_dl_norm = np.linalg.norm(windows_dl[:,:2],axis=1,keepdims=True) #only x,y

### dt - dont take the mean
dt = np.diff(t,axis=0)
dt = np.vstack((dt[0],dt))

### Lin accelaration at Nav frame
raw_a_nav = np.einsum('ijk,ik->ij', DCM_vec, Acc)
lin_a_nav = np.einsum('ijk,ik->ij', DCM_vec, LinAcc)  # the same as the code below
# a_meas_nav = np.zeros_like(LinAcc)
# for k in range(len(LinAcc)):
#     a_meas_nav[k,:] = DCM_vec[k,:,:]@(LinAcc[k,:].T)

############################# Estimators ################################

### Double integration of the accelaration - good for ~5 sec
'''
Using standart integration on the accelaration
one can see that the results are valid only for ~500 samples
which is equal to about ~2.5 seconds
'''
if False:
    v_est = np.cumsum(lin_a_nav*dt,axis=0) + 0
    p_est = np.cumsum(v_est*dt,axis=0) + 0
    p_est = p_est - p_est #start trajectory at 0,0,0
    Plot3CordVecs([Pos,p_est],['k','--r'],ListofTitles = ['X','Y','Z'], xlabel='samples', ylabel='[m]')
    # p_est = p_est[ind_start:ind_stop,:] - p_est[ind_start,:] #start trajectory at 0,0,0
    # Plot3CordVecs([Pos[ind_start:ind_stop,:],p_est],['k','--r'],ListofTitles = ['X','Y','Z'], xlabel='samples', ylabel='[m]')


### Main direction from accelaration (PCA) + known dl_norm
'''
Here we assume we have a perfect estimation of the dl norm (called "windows_dl_norm" here)
and also perfect DCM estimation.
We use the dl norm to multply the walking direction estimation (called here wde_est),
and then integrating to get position.
We show 2 methods to calculate the wde. the first is by using PCA on the ACC (which
is raw accelarations that contains gravity) and the second is PCA on LinAcc.
When working with ACC, we take the second pca component, because the gravity
is expected to be the major component   
In both cases we first rotate those vectors to navigation frame,
and we are working only with x,y components.                                                                        
'''
if True:
    wde_est = np.zeros((NumOfBatches,2)) #(only x,y) , NumOfBatches is expected to be N/window_size
    wde_est_raw = np.zeros((NumOfBatches,2)) #(only x,y) , NumOfBatches is expected to be N/window_size

    for k in range(NumOfBatches):
        batch_of_a_nav = lin_a_nav[k*window_size:(k+1)*window_size,:] 
        batch_of_a_nav_raw = raw_a_nav[k*window_size:(k+1)*window_size,:] 

        # PCA:
        pca = PCA(n_components=2).fit(batch_of_a_nav[:,:2]) #only x,y
        wde_est[k,:] = pca.components_[0].reshape(1,2) #take 1st component
        wde_est[k,:] = FixPCAamb(wde_est[k,:],windows_dl[k,:2]) #fix ambiguty
        
        # RAW PCA (we could have worked with LinAcc as well and the results will be the same: g is just a shift of point cloud):
        pca = PCA(n_components=3).fit(batch_of_a_nav_raw)
        wde_est_raw[k,:] = pca.components_[1,:2].reshape(1,2) #take 2nd component
        wde_est_raw[k,:] = FixPCAamb(wde_est_raw[k,:],windows_dl[k,:2]) #fix ambiguty
            
    p_est = Pos[0,:2] + np.cumsum(windows_dl_norm*wde_est/np.linalg.norm(wde_est,axis=1,keepdims=True),axis=0)
    p_est_raw = Pos[0,:2] + np.cumsum(windows_dl_norm*wde_est_raw/np.linalg.norm(wde_est_raw,axis=1,keepdims=True),axis=0)

    Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    Pos_raw_RMSE = np.linalg.norm(p_est_raw[-1,:]-Pos[-1,:2],keepdims=True)
    print('PCA RMSE:',Pos_RMSE,Pos_raw_RMSE)
    PlotTrajectories([Pos,p_est,p_est_raw],['k','--b','--r'],title='GT and PCA-estimated trajectory', ListofLabels = ['GT','PCA','Raw PCA'])


### SM heading
'''
Here we assume that we have perfect Psi mesurments form AHRS,
so we are using the Psi from GT. However, even in the GT one cannot
get the initial Psi with respect to the navigation frame (since user orientation
is not the same as phone estimation?) so we have to calculate the initial Psi
from the windows_dl GT. After getting Psi from GT, we are multipling the windows_dl_norm 
by sin(Psi)/cos(Psi) and integrating.
I dont like this method because:
    1) you have to use alot of GT vectors: windows_dl,windows_dl_norm,Psi
    2) The calculation of the initial_Psi is done by tuning the number
    of windows on which we avarage ()
    3) It will not work if user moves the phone with respect to his wde
'''
if True:
    Psi = Euler_vec[::window_size,0] #samling Psi
    NumOfWindowsToAvarage = 5 #I tuned this paramter with trial and error
    initial_Psi = np.arctan2(windows_dl[:NumOfWindowsToAvarage,1].mean(), windows_dl[:NumOfWindowsToAvarage,0].mean())
    Psi = Psi-Psi[0] + initial_Psi #Now phone Psi is the same as user Psi
    Psi = Psi[:-1].reshape(-1,1) #-1 is needed because Psi have one more sample than windows samples
    dl_est = np.hstack((np.cos(Psi),np.sin(Psi))) * windows_dl_norm 
    p_est = Pos[0,:2] + np.cumsum(dl_est,axis=0)
    Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    print('SM heading RMSE:',Pos_RMSE)
    PlotTrajectories([Pos,p_est],['k','--b'],title='GT and SM heading-estimated trajectory', ListofLabels = ['GT','SM heading'])

### Resnet 18 - RawIMU
if False:
    net = torch.load(pth_path_raw)   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)  
    wde_est = np.zeros((NumOfBatches,2)) #(only x,y) , NumOfBatches is expected to be N/window_size
    for k in range(NumOfBatches):
        tmp1 = raw_a_nav[k*window_size:(k+1)*window_size,:] #x,y,z
        tmp2 = Gyro[k*window_size:(k+1)*window_size,:]
        resent_input = np.hstack((tmp1,tmp2))
        resent_input = resent_input[None,:,:] #adding batch dim 
        resent_input = PrepareInputForResnet18(resent_input)
        resent_input.to(device) #this cannot be done in operational function
        resent_input = resent_input.cuda()
        output = net(resent_input)
        wde_est[k,:] = output.cpu().detach().numpy()
    
    p_est = Pos[0,:2] + np.cumsum(windows_dl_norm*wde_est/np.linalg.norm(wde_est,axis=1,keepdims=True),axis=0)
    Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    print('Resnet-18 (RawIMU) RMSE:',Pos_RMSE)
    PlotTrajectories([Pos,p_est],['k','--b'],title='GT and CNN-estimated trajectory', ListofLabels = ['GT','ResNet18 (RawIMU)'])

### Resnet 18 - LinACCWithRV
if True:
    net = torch.load(pth_path_rv)   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)  
    wde_est = np.zeros((NumOfBatches,2)) #(only x,y) , NumOfBatches is expected to be N/window_size
    for k in range(NumOfBatches):
        tmp1 = lin_a_nav[k*window_size:(k+1)*window_size,:] #x,y,z
        tmp2 = RV[k*window_size:(k+1)*window_size,:]
        resent_input = np.hstack((tmp1,tmp2))
        resent_input = resent_input[None,:,:] #adding batch dim 
        resent_input = PrepareInputForResnet18(resent_input)
        resent_input.to(device) #this cannot be done in operational function
        resent_input = resent_input.cuda()
        output = net(resent_input)
        wde_est[k,:] = output.cpu().detach().numpy()
    
    p_est = Pos[0,:2] + np.cumsum(windows_dl_norm*wde_est/np.linalg.norm(wde_est,axis=1,keepdims=True),axis=0)
    Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    print('Resnet-18 (LinAccWithRV) RMSE:',Pos_RMSE)
    PlotTrajectories([Pos,p_est],['k','--b'],title='GT and CNN-estimated trajectory', ListofLabels = ['GT','ResNet18 (LinAccWithRV)'])

### Resnet 18 - LinAcc
if True:
    net = torch.load(pth_path)   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)  
    wde_est = np.zeros((NumOfBatches,2)) #(only x,y) , NumOfBatches is expected to be N/window_size
    for k in range(NumOfBatches):
        batch_of_a_nav = lin_a_nav[k*window_size:(k+1)*window_size,:] #x,y,z
        batch_of_a_nav = batch_of_a_nav[None,:,:] #adding batch dim 
        resent_input = PrepareInputForResnet18(batch_of_a_nav)
        resent_input.to(device) #this cannot be done in operational function
        resent_input = resent_input.cuda()
        output = net(resent_input)
        wde_est[k,:] = output.cpu().detach().numpy()
    
    p_est = Pos[0,:2] + np.cumsum(windows_dl_norm*wde_est/np.linalg.norm(wde_est,axis=1,keepdims=True),axis=0)
    Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    print('Resnet-18 (LinAcc) RMSE:',Pos_RMSE)
    PlotTrajectories([Pos,p_est],['k','--b'],title='GT and CNN-estimated trajectory', ListofLabels = ['GT','ResNet18 (LinAcc)'])

### Inverted pendulum
'''
Matematical model to calculate WDE based on 
    1) dl norm
    2) gravity
    3) DCM \ LinAcc at nav frame
Note that this model can not work on time windows, because we are estimating frequency
Therefore it is useless in my opinion 
also you need to filter the signal after you have accumulate it!!
'''
if True:
    dl = Pos[1:] - Pos[:-1] #N/window_size samples
    dl_norm = np.linalg.norm(dl[:,:2],axis=1,keepdims=True) #only x,y
    g_hat = Grv/np.linalg.norm(Grv,keepdims=True,axis=1)
    az_norm = np.sum(LinAcc * g_hat,axis=1,keepdims=True)
    az = az_norm * g_hat
    axy = LinAcc - az
    axy_norm = np.linalg.norm(axy,axis=1)       
    fs = 1/dt
    # filter params from Eran:
    my_filt = signal.butter(N=3, Wn=[0.6, 2], btype='band', analog=False, fs=fs.mean(), output='sos')
    az_filt = signal.sosfiltfilt(my_filt, az_norm.reshape(-1,)).reshape(-1,1)
    axy_filt_x = signal.sosfiltfilt(my_filt, axy[:, 0]).reshape(-1,1)
    axy_filt_y = signal.sosfiltfilt(my_filt, axy[:, 1]).reshape(-1,1)
    axy_filt_z = signal.sosfiltfilt(my_filt, axy[:, 2]).reshape(-1,1)
    axy_filt = np.hstack([axy_filt_x, axy_filt_y, axy_filt_z])
    
    d_dt_az = (az_filt[1:]-az_filt[:-1])/dt[:-1]
    d_dt_axy = (axy_filt[1:]-axy_filt[:-1])/dt[:-1]
    omega = d_dt_az * axy_filt[:-1,:] - d_dt_axy * az_filt[:-1]
    yp_b = omega / np.linalg.norm(omega,axis=1,keepdims=True)
    yp_n = np.einsum('ijk,ik->ij', DCM_vec[:-1], yp_b)
    
    p_est = Pos[0,:2] + np.cumsum(dl_norm * yp_n[:,:2],axis=0)
    Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    print('Inv pendulum RMSE:',Pos_RMSE)
    PlotTrajectories([Pos,p_est],['k','--b'],title='GT and Inv pendulum trajectory', ListofLabels = ['GT','Inv pendulum'])
    
    #### This code DOES NOT works due to sampling issues
    # vel = windows_dl_norm # what exactly is this?
    # g_hat = Grv[::window_size,:]/np.linalg.norm(Grv[::window_size,:],keepdims=True,axis=1)
    # az_norm = np.sum(LinAcc[::window_size,:]* g_hat,axis=1,keepdims=True)
    # az = az_norm * g_hat
    # axy = LinAcc[::window_size,:] - az
    # axy_norm = np.linalg.norm(axy,axis=1)       
    # #fs = dt[::window_size]**(-1)
    # # filter params from Eran:
    # # my_filt = signal.butter(N=3, Wn=[0.6, 2], btype='band', analog=False, fs=fs, output='sos')
    # az_filt = az_norm #signal.sosfiltfilt(my_filt, az_norm.reshape(-1,)).reshape(-1,1)
    # # axy_filt_x = signal.sosfiltfilt(my_filt, axy[:, 0]).reshape(-1,1)
    # # axy_filt_y = signal.sosfiltfilt(my_filt, axy[:, 1]).reshape(-1,1)
    # # axy_filt_z = signal.sosfiltfilt(my_filt, axy[:, 2]).reshape(-1,1)
    # axy_filt = axy #np.hstack([axy_filt_x, axy_filt_y, axy_filt_z])
    
    # d_dt_az = (az_filt[1:]-az_filt[:-1])/dt[::window_size][:-1]
    # d_dt_axy = (axy_filt[1:]-axy_filt[:-1])/dt[::window_size][:-1]
    # omega = d_dt_az * axy_filt[:-1,:] - d_dt_axy * az_filt[:-1]
    # yp_b = omega / np.linalg.norm(omega)
    # yp_n = np.einsum('ijk,ik->ij', DCM_vec[::window_size][:-1], yp_b)
    
    # p_est = Pos[0,:2] + np.cumsum(vel * yp_n[:,:2],axis=0)
    # Pos_RMSE = np.linalg.norm(p_est[-1,:]-Pos[-1,:2],keepdims=True)
    # print('Inv pendulum RMSE:',Pos_RMSE)
    # PlotTrajectories([Pos,p_est],['k','--b'],title='GT and Inv pendulum trajectory', ListofLabels = ['GT','Inv pendulum'])