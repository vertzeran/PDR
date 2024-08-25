#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:19:14 2023

@author: zahi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
#from scipy.io import loadmat
from OperationalFunctions import PrepareInputForResnet18
import scipy.io as sio
from os.path import join
from os.path import join

class MySBGPytorchDS(Dataset):

    def __init__(self,  mode,TrainOrTest, data_type = 'LinAcc'):
        root_dir = '/data/Work/Navigation/SBGSeqForTrain'
        matfile = join(root_dir,mode +'_'+ TrainOrTest +'_'+ data_type +'.mat')
        X = sio.loadmat(matfile)['X']
        Y1 = sio.loadmat(matfile)['Y1']
        Y2 = sio.loadmat(matfile)['Y2']
        self.X = PrepareInputForResnet18(X)
        self.Y1 = torch.tensor(Y1,dtype=torch.float)
        self.Y2 = torch.tensor(Y2,dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx,:,:]
        y1 = self.Y1[idx,:]
        y2 = self.Y2[idx,:]
        sample = x,y1,y2
        # if self.transform:
        #     sample = self.transform(sample)
        return sample
    
if __name__ == "__main__":
   #root_dir = '/data/Work/Navigation/SBGSeqForTrain'
   TrainOrTest = 'test'
   mode = 'pocket'
   ds = MySBGPytorchDS(mode,TrainOrTest,data_type = 'LinAccWithRV')
   dl = DataLoader(ds,batch_size=17)
   x,y1,y2 = next(iter(dl))
   print(x.shape)
   print(y1.shape)
   print(y2.shape)