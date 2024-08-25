#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:20:24 2022
Training script.
User should choose the Hyper-Params

Please refer to the "cosine rule" for isosceles triangle:
    C^2 = a^2 + b^2 -2ab*cos(gamma)
        = 2|dl|^2 * (1-cos(gamme))
the second term is exactly pytorch embedding loss, while C is the displacement

@author: zahi
"""

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from SBG_PytorchDataset import MySBGPytorchDS
from OperationalFunctions import ResNet18_model

### Hyper-Params
epochs = 50 # 300
batch_size = 1024 # 256
mode = 'texting' #swing,pocket,texting
data_type = 'LinAccWithRV' #'LinAcc' #'LinAccWithRV'
num_of_input_channels = 7 if data_type == 'LinAccWithRV' else 3
net = ResNet18_model(num_of_input_channels=num_of_input_channels,num_of_outputs=2)
LossType  = 'Displacment' # 'UnWeighted', 'Displacment'

#Constand transforms are defined in the dataset
# RandomTransforms = transforms.Compose([transforms.RandomRotation(degrees=15),
#                                        transforms.RandomHorizontalFlip(p=0.5),
#                                        transforms.RandomVerticalFlip(p=0.25),
#                                        transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0)])

### Dataset split
train_ds = MySBGPytorchDS(mode,'train',data_type=data_type)
val_ds = MySBGPytorchDS(mode,'test',data_type=data_type)
train_dl = DataLoader(train_ds,batch_size=batch_size,shuffle=True)
val_dl = DataLoader(val_ds,len(val_ds),shuffle=False)

### Criterion and output
cos_loss = nn.CosineEmbeddingLoss(reduction='mean')
def GetLoss(cos_loss, outputs, WDE_GT, segment_dl_norm, LossType = 'UnWeighted', device='cuda'):
    targets = (torch.ones(len(WDE_GT))).to(device)
    if LossType == 'UnWeighted':
        loss = torch.mean(cos_loss(outputs,WDE_GT,targets))
        return loss
    
    if LossType == 'Displacment': #wheight with |dl|
        loss = torch.norm(outputs-WDE_GT,dim=1)*segment_dl_norm  
        loss = torch.mean(loss)
        return loss
        # loss = cos_loss(outputs,WDE_GT,targets)
        # sim = 1-(torch.mean(loss)).item()
        # loss = loss * 2*(segment_dl_norm**2)
        # #loss = loss / torch.max(loss) #torch.norm(2*(segment_dl_norm**2)) #to get something roughly between 0-1
        # loss = torch.mean(loss)
        # loss = torch.sqrt(loss)

    
    
### Optimizer (make sure this line is after the "net.fc = LastLayer")
optimizer = optim.Adam(net.parameters(), lr=0.0003) 

### Prepare val data (both saving time and make sure augmentation is done only once)
LinAccNav_val,segment_dl_norm_val,WDE_GT_val = next(iter(val_dl))
#WDE_GT_val = WDE_GT_val.cpu().detach().numpy()

### To cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)  
LinAccNav_val = LinAccNav_val.to(device)
segment_dl_norm_val = segment_dl_norm_val.to(device)
WDE_GT_val = WDE_GT_val.to(device)

### Training loop
CostTr = []
CostVal = []
#Max_Similarity = 0.8
if LossType == 'UnWeighted':
    Min_Loss = 0.2 #1-cosine angle
if LossType == 'Displacment':
    Min_Loss = 0.4 #displacement in meters 
# i = 0
# data = next(iter(train_dl))
for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(train_dl, 0):
        # if i == 3:#for debug
        #     break
        LinAccNav, segment_dl_norm, WDE_GT = data[0].to(device) , data[1].to(device) , data[2].to(device)
        #imgs = RandomTransforms(imgs) #only on the train set
        optimizer.zero_grad()
        outputs = net(LinAccNav)
        loss = GetLoss(cos_loss,outputs,WDE_GT,segment_dl_norm,LossType=LossType)
        loss.backward()
        optimizer.step()
       
        # Val loss
        net.eval()
        with torch.no_grad():
            outputs_val = net(LinAccNav_val)
            targets = (torch.ones(len(WDE_GT_val))).cuda()
            loss_val = GetLoss(cos_loss,outputs_val,WDE_GT_val,segment_dl_norm_val,LossType=LossType)
            loss_val_item = loss_val.item() 
            if (loss_val_item < Min_Loss):
                #max_name = 'WDE_regressor_' + mode + '_' + LossType + '_' + str(round(Sim,3)) + '.pth'
                max_name = 'SBG_WDE_regressor_' + mode + '_' + data_type + '_' + LossType + '_' + str(round(loss_val_item,3)) + '.pth'
                #torch.save(net.state_dict(),max_name)
                torch.save(net,max_name)
                Min_Loss -= 0.05
        net.train()

        # print statistics
        CostTr.append(loss.item())
        CostVal.append(loss_val.item())
        #print('epoch: ' + str(epoch) + ' , batch: ' + str(i) + ' , Train Loss: ' + str(round(loss.item(),3)) + ' Val Loss: ' + str(round(loss_val.item(),3)) + ' Similarity:' +str(round(Sim,3)))
        print('epoch: ' + str(epoch) + ' , batch: ' + str(i) + ' , Train Loss: ' + str(round(loss.item(),3)) + ' Val Loss: ' + str(round(loss_val.item(),3)) )

### Save net weights
print('Finished Training')