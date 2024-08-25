#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:55:59 2022
TODO:
    1) Add another class for 6 IMU inputs
@author: zahi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
#from scipy.io import loadmat
from Ridi_Loader import LoadRiDiDir
from OperationalFunctions import PrepareInputForResnet18

class MyRidiPytorchDS(Dataset):

    def __init__(self, mode,TrainOrTest,window_size, data_type = 'LinAcc'):
        path_to_dir = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - mode - TrainOrTest'
        path_to_dir = path_to_dir.replace("mode", mode )
        path_to_dir = path_to_dir.replace("TrainOrTest", TrainOrTest )
        t_all,Pos_all,DCM_vec_all,Gyro_all,Acc_all,Mag_all,LinAcc_all,file_num_vec_all,files_in_dir = \
            LoadRiDiDir(path_to_dir)       
        
        # Find valid inds
        last_ind = len(t_all)-len(t_all)%window_size
        file_num_vec_all = np.reshape(file_num_vec_all[:last_ind,:],(-1,window_size))
        valid_inds = np.where(np.max(file_num_vec_all,axis=1)==np.min(file_num_vec_all,axis=1))[0]   
        
        if data_type =='LinAcc':
            lin_a_nav = np.einsum('ijk,ik->ij', DCM_vec_all, LinAcc_all)  # Nx3
            X = np.reshape(lin_a_nav[:last_ind,:],(-1,window_size,3)) #(N/200)x200x3
        
        if data_type =='RawIMU':
            raw_a_nav = np.einsum('ijk,ik->ij', DCM_vec_all, Acc_all)  # Nx3
            X = np.hstack((raw_a_nav,Gyro_all)) #Nx6
            X = np.reshape(X[:last_ind,:],(-1,window_size,6)) #(N/200)x200x3
            
        # output:
        windows_dl = Pos_all[window_size:-1:window_size] - Pos_all[0:-window_size-1:window_size] #N/window_size samples

        X = X[valid_inds,:,:]
        Y = windows_dl[valid_inds,:]
        self.X = PrepareInputForResnet18(X)
        self.Y = torch.tensor(Y,dtype=torch.float)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx,:,:]
        y = self.Y[idx,:2] #only x,y
        sample = x,y
        # if self.transform:
        #     sample = self.transform(sample)
        return sample
   
if __name__ == "__main__":
   ds = MyRidiPytorchDS('Pocket','Test',200)
   dl = DataLoader(ds,batch_size=3)
   x,y = next(iter(dl))
   print(x.shape)
   print(y.shape)
   