import os

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from ZL_scripts.SBGPytorchDataset import MySBGPytorchDS
# from ZL_scripts.OperationalFunctions import ResNet18_model
from os.path import join
import pickle
from utils.Classes import WDE_performance_analysis
import scipy.io as sio
from utils.Functions import PrepareInputForResnet18
from datetime import datetime
import json
import matplotlib.pyplot as plt


def ResNet18_model(num_of_input_channels, num_of_outputs):
    net = torchvision.models.resnet18(pretrained=False)
    ### Add MLP head to the ResNet
    LastLayer = nn.Sequential(
        nn.Dropout(p=0.75),#0.5
        nn.Linear(512, 256),#(512, 128)
        nn.ReLU(),
        nn.Dropout(p=0.75),# 0.25
        nn.Linear(256, 32),# 128, 32
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(32, num_of_outputs))  # no activation at the end
    net.fc = LastLayer

    if num_of_input_channels != 3:
        net.conv1 = nn.Conv2d(num_of_input_channels, 64, kernel_size=(7, 7),
                              stride=(2, 2), padding=(3, 3), bias=False)
    return net


class MySBGPytorchDS(Dataset):

    def __init__(self, matfile, data_type = 'LinAcc'):
        X = sio.loadmat(matfile)['X']
        Y1 = sio.loadmat(matfile)['Y1']
        Y2 = sio.loadmat(matfile)['Y2']
        self.X = PrepareInputForResnet18(X)
        self.Y1 = torch.tensor(Y1,dtype=torch.float)
        self.Y2 = torch.tensor(Y2,dtype=torch.float)
        self.window_size = X.shape[1]
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx,:,:]
        y1 = self.Y1[idx,:]
        y2 = self.Y2[idx, :]
        sample = x, y1, y2
        # if self.transform:
        #     sample = self.transform(sample)
        return sample


def train(params):
    # root_dir, comment, epochs=300, batch_size=1024, lr=0.0005, mode=None,
    #                   add_quat=False, heading_fix=False,
    #                   use_validation=False, dataset=None):
    ### Hyper-Params
    # epochs = 300 # 50
    # batch_size = 1024 # 512
    if params["add_quat"]:
        net = ResNet18_model(num_of_input_channels=7, num_of_outputs=2)
    else:
        net = ResNet18_model(num_of_input_channels=3, num_of_outputs=2)
    data_type = 'LinAcc' #'RawIMU' #'LinAcc'
    loss_type = 'position_error' # could be:'position_error' ,  'vector_cosine'

    ### Dataset split
    # root_dir = '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-01-22T19:51:40.411866TRC#1 swing wind_size 200swing left right mix'
    root_dir = params["data_folder"]
    train_ds = MySBGPytorchDS(join(root_dir, 'train.mat'))

    if params["create_validation_set"]:
        val_ds = MySBGPytorchDS(join(root_dir, 'validation.mat'))
    test_ds = MySBGPytorchDS(join(root_dir, 'test.mat'))

    ### output folder
    main_wd_path = os.getcwd()#'/'
    now = datetime.isoformat(datetime.now())
    # comment = ' swing left right mix'
    description = 'optimization results on ' + params["mode"] + ', window size: ' + str(train_ds.window_size) + params["comment"]
    print(description)
    outputfolder = join(main_wd_path, 'data', 'optimization_results', now + description)
    os.mkdir(outputfolder)

    ### info file
    info_dict = params
    info_dict['data_type'] = data_type
    info_dict['training_window_size'] = train_ds.window_size
    info_dict['comment_training'] = ''
    info_dict['optimization_folder'] = outputfolder


    train_dl = DataLoader(train_ds,batch_size=params["batch_size"],shuffle=True)
    if params["create_validation_set"]:
        val_dl = DataLoader(val_ds,len(val_ds),shuffle=False)
    test_dl = DataLoader(test_ds,len(test_ds),shuffle=False)
    ### Criterion and output
    if loss_type == 'vector_cosine':
        criterion = nn.CosineEmbeddingLoss()
    elif loss_type == 'position_error':
        criterion = nn.L1Loss(reduction='mean')
        # criterion = position_error_loss()

    ### Optimizer (make sure this line is after the "net.fc = LastLayer")
    optimizer = optim.Adam(net.parameters(), lr=params["LR"])

    ### Prepare val data (both saving time and make sure augmentation is done only once)
    if params["create_validation_set"]:
        LinAccNav_val, dL_GT_val, WDE_GT_val = next(iter(val_dl))
    LinAccNav_test, dL_GT_test, WDE_GT_test = next(iter(test_dl))
    #WDE_GT_val = WDE_GT_val.cpu().detach().numpy()

    ### To cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    if params["create_validation_set"]:
        LinAccNav_val = LinAccNav_val.to(device)
        dL_GT_val = dL_GT_val.to(device)
        WDE_GT_val = WDE_GT_val.to(device)
    LinAccNav_test = LinAccNav_test.to(device)
    dL_GT_test = dL_GT_test.to(device)
    WDE_GT_test = WDE_GT_test.to(device)
    ### Training loop
    CostTr = []
    CostVal = []
    CostTest = []
    Min_loss = 0.36
    regressor_saved = False
    # i = 0
    # data = next(iter(train_dl))
    for epoch in range(params["epochs"]):  # loop over the dataset multiple times
        for i, data in enumerate(train_dl, 0):
            LinAcc, dL, WDE = data[0].to(device) , data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            outputs = net(LinAcc)
            if loss_type == 'vector_cosine':
                targets = (torch.ones(len(WDE))).cuda()
                loss = criterion(outputs, WDE, targets)
            elif loss_type == 'position_error':
                WDE = WDE / torch.linalg.norm(WDE, dim=1).unsqueeze(1)
                outputs = outputs / torch.linalg.norm(outputs, dim=1).unsqueeze(1)
                # loss = criterion(outputs * dL, WDE * dL)
                loss = torch.linalg.norm(outputs * dL - WDE * dL, dim=1).mean()
            loss.backward()
            optimizer.step()

            # Val loss
            net.eval()
            with torch.no_grad():
                if params["create_validation_set"]:
                    outputs_val = net(LinAccNav_val)
                outputs_test = net(LinAccNav_test)
                if loss_type == 'vector_cosine':
                    targets = (torch.ones(len(WDE_GT_val))).cuda()
                    if params["create_validation_set"]:
                        loss_val = criterion(outputs_val, WDE_GT_val, targets)
                    targets = (torch.ones(len(WDE_GT_test))).cuda()
                    loss_test = criterion(outputs_test, WDE_GT_test, targets)
                elif loss_type == 'position_error':
                    if params["create_validation_set"]:
                        WDE_GT_val = WDE_GT_val / torch.linalg.norm(WDE_GT_val, dim=1).unsqueeze(1)
                        outputs_val = outputs_val / torch.linalg.norm(outputs_val, dim=1).unsqueeze(1)
                        # loss_val = criterion(outputs_val * dL_GT_val, WDE_GT_val * dL_GT_val)
                        loss_val = torch.linalg.norm(outputs_val * dL_GT_val - WDE_GT_val * dL_GT_val, dim=1).mean()

                    WDE_GT_test = WDE_GT_test / torch.linalg.norm(WDE_GT_test, dim=1).unsqueeze(1)
                    outputs_test = outputs_test / torch.linalg.norm(outputs_test, dim=1).unsqueeze(1)
                    # loss_test = criterion(outputs_test * dL_GT_test, WDE_GT_test * dL_GT_test)
                    loss_test = torch.linalg.norm(outputs_test * dL_GT_test - WDE_GT_test * dL_GT_test, dim=1).mean()

                # Sim = 1-loss_test.item()
                if params["create_validation_set"]:
                    Sim = loss_val.item()
                else:
                    Sim = loss_test.item()
                if (Sim < Min_loss):
                    if regressor_saved:
                        os.remove(join(outputfolder, min_name))
                    else:
                        regressor_saved = True
                    min_name = 'WDE_regressor_' + params["mode"] + '_' + data_type + '_' + str(round(Sim,3)) + '.pth'
                    torch.save(net, join(outputfolder, min_name))
                    Min_loss -= 0.01

            net.train()

            # print statistics
            CostTr.append(loss.item())
            if params["create_validation_set"]:
                CostVal.append(loss_val.item())
            CostTest.append(loss_test.item())
            if params["create_validation_set"]:
                print('epoch: ' + str(epoch) + ' , batch: ' + str(i) + \
                      ' , Train Loss: ' + str(round(loss.item(), 3)) + \
                      ' Val Loss: ' + str(round(loss_val.item(), 3)) + \
                      ' test Loss: ' + str(round(loss_test.item(), 3))
                      )
            else:
                print('epoch: ' + str(epoch) + ' , batch: ' + str(i) + \
                      ' , Train Loss: ' + str(round(loss.item(), 3)) + \
                      ' test Loss: ' + str(round(loss_test.item(), 3))
                      )
        plt.figure()
        plt.plot(CostTr, label='CostTr')
        plt.plot(CostVal, label='loss_val')
        plt.plot(CostTest, label='loss_test')
        plt.grid(True)
        plt.legend()
        plt.savefig(join(outputfolder, 'losses.png'))
        plt.close()

    plt.figure()
    plt.plot(CostTr, label='CostTr')
    if params["create_validation_set"]:
        plt.plot(CostVal, label='loss_val')
    plt.plot(CostTest, label='loss_test')
    plt.grid(True)
    plt.legend()
    plt.savefig(join(outputfolder, 'losses.png'))
    # plt.show()
    info_dict['best_saved_model'] = min_name
    with open(join(outputfolder, 'info_file.json'), "w") as outfile:
        json.dump(info_dict, outfile, indent=4)
    print('Finished Training')
    return outputfolder, min_name


if __name__ == '__main__':
    params = {}
    outputfolder, min_name = train(params)