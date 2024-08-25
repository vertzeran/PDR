#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:20:24 2022
Training script.
User should choose the Hyper-Params
TODO:
    1) Add resnet with customized number of input channels and output dim
@author: zahi
"""

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from Ridi_PytorchDataset import MyRidiPytorchDS
from OperationalFunctions import ResNet18_model

### Hyper-Params
epochs = 50
batch_size = 128
window_size = 200
net = ResNet18_model(num_of_input_channels=3,num_of_outputs=2)
mode = 'Bag'
data_type = 'LinAcc' #'RawIMU' #'LinAcc'

#Constand transforms are defined in the dataset
# RandomTransforms = transforms.Compose([transforms.RandomRotation(degrees=15),
#                                        transforms.RandomHorizontalFlip(p=0.5),
#                                        transforms.RandomVerticalFlip(p=0.25),
#                                        transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0)])

### Dataset split
train_ds = MyRidiPytorchDS(mode,'Train',window_size,data_type=data_type)
val_ds = MyRidiPytorchDS(mode,'Test',window_size,data_type=data_type)
train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
val_dl = DataLoader(val_ds,len(val_ds),shuffle=False)

### Criterion and output
criterion = nn.CosineEmbeddingLoss()

### Optimizer (make sure this line is after the "net.fc = LastLayer")
optimizer = optim.Adam(net.parameters(), lr=0.0003) 

### Prepare val data (both saving time and make sure augmentation is done only once)
LinAccNav_val,WDE_GT_val = next(iter(val_dl))
#WDE_GT_val = WDE_GT_val.cpu().detach().numpy()

### To cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)  
LinAccNav_val = LinAccNav_val.to(device)
WDE_GT_val = WDE_GT_val.to(device)

### Training loop
CostTr = []
CostVal = []
Max_Similarity = 0.85

for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(train_dl, 0):
        #if i == 3:#for debug
        #    break
        LinAcc, WDE = data[0].to(device) , data[1].to(device)
        #imgs = RandomTransforms(imgs) #only on the train set
        optimizer.zero_grad()
        outputs = net(LinAcc)
        targets = (torch.ones(len(WDE))).cuda()
        loss = criterion(outputs,WDE,targets)
        loss.backward()
        optimizer.step()
       
        # Val loss
        net.eval()
        with torch.no_grad():
            outputs_val = net(LinAccNav_val)
            targets = (torch.ones(len(WDE_GT_val))).cuda()
            loss_val = criterion(outputs_val,WDE_GT_val,targets)
            Sim = 1-loss_val.item()
            if (Sim > Max_Similarity):
                max_name = 'WDE_regressor_' + mode + '_' + data_type + '_' + str(round(Sim,3)) + '.pth'
                #torch.save(net.state_dict(),max_name)
                torch.save(net,max_name)
                Max_Similarity += 0.05
        net.train()

        # print statistics
        CostTr.append(loss.item())
        CostVal.append(loss_val.item())
        print('epoch: ' + str(epoch) + ' , batch: ' + str(i) + ' , Train Loss: ' + str(round(loss.item(),3)) + ' Val Loss: ' + str(round(loss_val.item(),3)) + ' Similarity:' +str(round(Sim,3)))

### Save net weights
print('Finished Training')