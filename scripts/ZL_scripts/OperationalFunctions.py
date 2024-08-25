#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:23:24 2022
This is the most important file - here are all the function that are requiered
for inferance. We will need to deliver this functions.
This is not a simulation script, dont write here code for plots or error anlysis
Also note that this function should work in "realtime" so we cannot use them
if we want to "vectorize" the simulation (only inside for loops..)
@author: zahi
"""

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torchvision

def PCAOnLinAccAtNav(LinAccAtNavFrame):
    '''
    Input - LinAccAtNavFrame is a time-sequance with shape (N,2). 
    It is a time window of the Linear accelaration after rotation, only at xy plane
    Output - walking direction at xy plane  with shape (2,1). 
    '''
    pca = PCA(n_components=2).fit(LinAccAtNavFrame)
    wde_est = pca.components_[0].reshape(1,2) #take 1st component
    return wde_est

def PCAOnAccAtNav(AccAtNavFrame):
    '''
    Input - AccAtNavFrame is an array with shape (N,3). 
    It is a time window of the raw accelaration after rotation (including gravity)
    Output - walking direction at xy plane  with shape (2,1). 
    '''
    pca = PCA(n_components=3).fit(AccAtNavFrame)
    wde_est = pca.components_[1,:2].reshape(1,2) #take 1st component
    return wde_est

def ResNet18_model(num_of_input_channels,num_of_outputs):
    net = torchvision.models.resnet18(pretrained=False)
    ### Add MLP head to the ResNet 
    LastLayer = nn.Sequential(
              nn.Dropout(p=0.5), # nn.Dropout(p=0.75),
              nn.Linear(512,128), # nn.Linear(512,256),
              nn.ReLU(),
              nn.Dropout(p=0.25), # nn.Dropout(p=0.75)
              nn.Linear(128,32), # nn.Linear(256,32)
              nn.ReLU(),
              nn.Dropout(p=0.1),
              nn.Linear(32,num_of_outputs)) # no activation at the end
    net.fc = LastLayer
    
    if num_of_input_channels != 3:
        net.conv1 = nn.Conv2d(num_of_input_channels, 64, kernel_size=(7, 7),
                           stride=(2, 2), padding=(3, 3), bias=False)
    return net
        
def PrepareInputForResnet18(BatchOfLinAccAtNavFrame):
    '''
    Input - LinAccAtNavFrame is an array with shape (1,200,3). 
            It is a time window of the Linear accelaration after rotation
            Please note that we assume the "batch dim" was already added
    Output - Input to be fed into resent 18 
    '''
    X = torch.tensor(BatchOfLinAccAtNavFrame,dtype=torch.float)
    X = X.permute(0, 2, 1) #instead of X = np.swapaxes(X, axis1, axis2)
    X = X[:, :, :,None] #created another dim for resnet 18
    return X

def GetFeaturesFromLinAccTimeSegments(LinAcc):
    '''
    Input - BatchOfData is an array with shape (1,200,3). 
            It is a time window of the Linear accelaration (or raw) after rotation
            Please note that we assume the "batch dim" was already added
    Output - Input to be fed into tree
    '''
    N = 12
    X = np.zeros((len(LinAcc),N))
    X[:,0] = np.mean(LinAcc[:,:,0],axis=1)
    X[:,1] = np.mean(LinAcc[:,:,1],axis=1)
    X[:,2] = np.mean(LinAcc[:,:,2],axis=1)
    X[:,4] = np.std(LinAcc[:,:,0],axis=1)
    X[:,5] = np.std(LinAcc[:,:,1],axis=1)
    X[:,6] = np.std(LinAcc[:,:,2],axis=1)
    X[:,7] = X[:,0]/X[:,2]
    X[:,8] = X[:,1]/X[:,2]
    X[:,9] = X[:,0]/X[:,1]
    XYMag = np.sum(LinAcc[:,:,:2]**2,axis=2) #batchx200
    FFTxy = np.abs(np.fft.fftshift(np.fft.fft(XYMag,axis=1),axes =1)) #batchx200
    X[:,10] = np.max(FFTxy,axis=1)
    X[:,11] = np.std(FFTxy,axis=1)
    return X

def GetFeaturesFromGyroTimeSegments(Gyro):
    '''
    Input - BatchOfData is an array with shape (1,200,3). 
            It is a time window of the Gyro measurments
            Please note that we assume the "batch dim" was already added
    Output - Input to be fed into tree
    '''
    N = 12
    X = np.zeros((len(Gyro),N))
    X[:,0] = np.mean(Gyro[:,:,0],axis=1) #11
    X[:,1] = np.mean(Gyro[:,:,1],axis=1)
    X[:,2] = np.mean(Gyro[:,:,2],axis=1)
    X[:,4] = np.std(Gyro[:,:,0],axis=1) #14
    X[:,5] = np.std(Gyro[:,:,1],axis=1)
    X[:,6] = np.std(Gyro[:,:,2],axis=1)
    X[:,7] = X[:,0]/X[:,2] #17
    X[:,8] = X[:,1]/X[:,2] #18
    X[:,9] = X[:,0]/X[:,1] #19
    XYMag = np.sum(Gyro[:,:,:2]**2,axis=2) #batchx200
    FFTxy = np.abs(np.fft.fftshift(np.fft.fft(XYMag,axis=1),axes =1)) #batchx200
    X[:,10] = np.max(FFTxy,axis=1) #22
    X[:,11] = np.std(FFTxy,axis=1) #23
    return X
    
    
    
    
