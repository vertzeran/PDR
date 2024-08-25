import os
from functools import partial
import sys
from datetime import datetime
import json

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from os.path import join
import scipy.io as sio
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from scripts.a13_TRC_dx_dy.losses import LOSSES
from utils.Functions import PrepareInputForResnet18, convert_model_2d_to_1d, rotation_matrix_to_r6d


def resnet18_model(num_of_input_channels, num_of_outputs):
    net = torchvision.models.resnet18(pretrained=False)
    # Add MLP head to the ResNet
    last_layer = nn.Sequential(
        nn.Dropout(p=0.75),  # 0.5
        nn.Linear(512, 256),  # (512, 128)
        nn.ReLU(),
        nn.Dropout(p=0.75),  # 0.25
        nn.Linear(256, 32),  # 128, 32
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(32, num_of_outputs))  # no activation at the end
    net.fc = last_layer

    if num_of_input_channels != 3:
        net.conv1 = nn.Conv2d(num_of_input_channels, 64, kernel_size=(7, 7),
                              stride=(2, 2), padding=(3, 3), bias=False)
    return net


def resnet50_model(num_of_input_channels, num_of_outputs):
    net = torchvision.models.resnet50()
    # Add MLP head to the ResNet
    last_layer = nn.Sequential(
        nn.Dropout(p=0.75),  # 0.5
        nn.Linear(2048, 256),  # (512, 128)
        nn.ReLU(),
        nn.Dropout(p=0.75),  # 0.25
        nn.Linear(256, 32),  # 128, 32
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(32, num_of_outputs))  # no activation at the end
    net.fc = last_layer

    if num_of_input_channels != 3:
        net.conv1 = nn.Conv2d(num_of_input_channels, 64, kernel_size=(7, 7),
                              stride=(2, 2), padding=(3, 3), bias=False)
    return net


class MySBGPytorchDS(Dataset):
    def __init__(self, matfile, add_dim=True, convert_quat_to_rot6d=False, max_dl=None):
        data = sio.loadmat(matfile)
        x = data['X']

        if convert_quat_to_rot6d and x.shape[2] >= 7:
            n_samples, seq_len, n_features = x.shape
            q = x[:, :, 3:7].reshape(-1, 4)
            m = Rotation.from_quat(q).as_matrix()
            r6d = rotation_matrix_to_r6d(m).reshape((n_samples, seq_len, 6))
            x = np.dstack((x[:, :, :3], r6d, x[:, :, 7:]))

        if 'Y1' in data:
            y1 = data['Y1']  # dl
            y2 = data['Y2']  # wd
        elif 'Y' in data:
            y = data['Y']
            y1 = np.linalg.norm(y, axis=1, keepdims=True)
            y2 = y / y1
        else:
            raise KeyError(list(data.keys()))

        self.X = PrepareInputForResnet18(x, add_dim=add_dim)
        self.Y1 = torch.tensor(y1, dtype=torch.float)
        self.Y2 = torch.tensor(y2, dtype=torch.float)

        if max_dl is not None:
            valid_idx = self.Y1.flatten() <= max_dl
            self.X = self.X[valid_idx]
            self.Y1 = self.Y1[valid_idx]
            self.Y2 = self.Y2[valid_idx]

        self.window_size = x.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx, :, :]
        y1 = self.Y1[idx, :]
        y2 = self.Y2[idx, :]
        sample = x, y1, y2

        return sample


def train(params):
    device = torch.device(params["device"] if torch.cuda.is_available() else 'cpu')

    if params["resnet"] == 50:
        resnet_func = resnet50_model
    elif params["resnet"] == 18:
        resnet_func = resnet18_model
    else:
        raise ValueError(f'resnet: {params["resnet"]}')

    if params["add_quat"]:
        num_of_input_channels = 7
        if params["convert_quat_to_rot6d"]:
            num_of_input_channels = 9
    else:
        num_of_input_channels = 3

    num_of_outputs = params.get("num_of_outputs", 2)
    net = resnet_func(num_of_input_channels=num_of_input_channels, num_of_outputs=num_of_outputs)
    if params["resnet1d"]:
        convert_model_2d_to_1d(net)
        print('converted to 1d resnet')

    data_type = 'LinAcc'  # 'RawIMU' #'LinAcc'

    loss_name = params["loss"].get("name")
    if loss_name == 'separate_length_and_direction_loss':
        length_loss = LOSSES[params["loss"]["length"]]
        direction_loss = LOSSES[params["loss"]["direction"]]
        loss_func = partial(LOSSES[loss_name], length_loss=length_loss, direction_loss=direction_loss)
        loss_name = f'{loss_name}_{length_loss}_{direction_loss}'
    elif num_of_outputs == 3:
        length_loss = LOSSES[params["loss"]["length"]]
        direction_loss = LOSSES[params["loss"]["direction"]]
    else:
        loss_func = LOSSES[loss_name]
    test_loss_func = LOSSES['norm_loss']

    # Dataset split
    root_dir = params["TRC_params"]["processed_data_folder"]
    max_dl = params["TRC_params"].get("max_dl")
    train_ds = MySBGPytorchDS(join(root_dir, 'train.mat'), add_dim=not params["resnet1d"],
                              convert_quat_to_rot6d=params["convert_quat_to_rot6d"], max_dl=max_dl)
    params["create_validation_set"] = False
    test_ds = MySBGPytorchDS(join(root_dir, 'test.mat'), add_dim=not params["resnet1d"],
                             convert_quat_to_rot6d=params["convert_quat_to_rot6d"], max_dl=max_dl)

    # detects debug mode
    num_workers = 4
    create_output_folder = True
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is not None and gettrace():
        print('DEBUG mode detected, using num_workers = 0')
        num_workers = 0
        create_output_folder = False

    # output folder
    main_wd_path = params["main_dir"]  # os.getcwd()#'/'
    now = datetime.isoformat(datetime.now())
    description = params["training_title"]
    print(description)
    outputfolder = join(main_wd_path, 'data', 'optimization_results', now + '_' + description)
    if create_output_folder:
        os.makedirs(outputfolder)

    # info file
    info_dict = params
    info_dict['data_type'] = data_type
    info_dict['training_window_size'] = train_ds.window_size
    info_dict['optimization_folder'] = outputfolder
    if create_output_folder:
        with open(join(outputfolder, 'info_file.json'), "w") as outfile:
            json.dump(info_dict, outfile, indent=4)

    train_dl = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test_ds, len(test_ds), shuffle=False)
    if params["create_validation_set"]:
        val_ds = MySBGPytorchDS(join(root_dir, 'validation.mat'), add_dim=not params["resnet1d"],
                                convert_quat_to_rot6d=params["convert_quat_to_rot6d"], max_dl=max_dl)
        val_dl = DataLoader(val_ds, len(val_ds), shuffle=False)

    # Optimizer (make sure this line is after the "net.fc = LastLayer")
    optimizer = optim.Adam(net.parameters(), lr=params["LR"], weight_decay=params.get("weight_decay", 0))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params["milestones"])

    x_test, dl_gt_test, wde_gt_test = next(iter(test_dl))
    x_test = x_test.to(device)
    dl_gt_test = dl_gt_test.to(device)
    dxdy_gt_test = dl_gt_test * wde_gt_test.to(device)

    # Prepare val data (both saving time and make sure augmentation is done only once)
    if params["create_validation_set"]:
        x_val, dl_gt_val, wde_gt_val = next(iter(val_dl))
        x_val = x_val.to(device)
        dl_gt_val = dl_gt_val.to(device)
        dxdy_gt_val = dl_gt_val * wde_gt_val.to(device)

    net.to(device)

    # Training loop
    cost_train = []
    cost_val = []
    cost_test = []
    dl_diff_test = []
    min_loss = 100000000
    min_name = None

    p_bar = tqdm(range(params["epochs"]), total=params["epochs"], unit='epoch')
    for epoch in p_bar:  # loop over the dataset multiple times
        avg_loss = []
        for data in train_dl:
            x, dl, wde = data
            x = x.to(device)
            dl = dl.to(device)
            wde = wde.to(device)

            optimizer.zero_grad()
            outputs = net(x)

            if num_of_outputs == 2:
                dxdy_gt = (dl * wde)
                loss = loss_func(outputs, dxdy_gt)
            elif num_of_outputs == 3:
                dl_outputs = outputs[:, 0:1]
                wde_outputs = outputs[:, 1:3]
                dl_loss = length_loss(dl_outputs, dl)
                weight = None
                if params["loss"].get("direction_weight") == 'dl':
                    weight = dl.detach()
                wde_loss = direction_loss(wde_outputs, wde, weight=weight)
                loss = dl_loss + wde_loss
            else:
                raise ValueError(f'num_of_outputs: {num_of_outputs}')

            if loss.item() > 100:  # invalid batch
                print(f'invalid batch with loss: {loss.item()}')
                # continue
            loss.backward()
            optimizer.step()

            avg_loss.extend([loss.item()] * x.shape[0])

        scheduler.step()
        avg_loss = sum(avg_loss) / len(avg_loss)

        # Val loss
        net.eval()
        with torch.no_grad():
            outputs_test = net(x_test)
            if num_of_outputs == 3:
                outputs_test_dl = outputs_test[:, 0:1]
                outputs_test = outputs_test_dl * outputs_test[:, 1:]
            elif num_of_outputs == 2:
                outputs_test_dl = torch.linalg.norm(outputs_test, dim=1, keepdim=True)

            test_mean_dl_diff = (outputs_test_dl - dl_gt_test).mean().item()
            loss_test = test_loss_func(outputs_test, dxdy_gt_test).item()
            sim = loss_test

            if params["create_validation_set"]:
                outputs_val = net(x_val)
                if num_of_outputs == 3:
                    outputs_val_dl = outputs_val[:, 0:1]
                    outputs_val = outputs_val_dl * outputs_val[:, 1:]

                loss_val = test_loss_func(outputs_val, dxdy_gt_val).item()
                sim = loss_val

            if sim < min_loss:
                if create_output_folder:
                    if min_name is not None:
                        os.remove(join(outputfolder, min_name))
                    min_name = f'dx_dy_regressor_epoch{epoch}_loss_{loss_name}_{sim:.3f}.pth'
                    torch.save(net, join(outputfolder, min_name))
                min_loss = sim
                # print(f'Checkpoint saved: {min_name}')
        net.train()

        # print statistics
        cost_train.append(avg_loss)
        cost_test.append(loss_test)
        dl_diff_test.append(test_mean_dl_diff)

        loss_dict = {'train loss': avg_loss, 'test loss': loss_test, 'test dl diff': test_mean_dl_diff}
        if params["create_validation_set"]:
            cost_val.append(loss_val)
            loss_dict['val loss'] = loss_val
        p_bar.set_postfix(loss_dict)

        plt.figure()
        plt.plot(cost_train, label='loss_train')
        plt.plot(cost_test, label='loss_test')
        plt.plot(dl_diff_test, label='dl_diff_test')
        if params["create_validation_set"]:
            plt.plot(cost_val, label='loss_val')
        plt.grid(True)
        plt.legend()
        if create_output_folder:
            plt.savefig(join(outputfolder, 'losses.png'))
        plt.close()

    print('Finished Training')
    return outputfolder, min_name


if __name__ == '__main__':
    info_file_path = 'params.json'
    with open(info_file_path, "r") as f:
        params = json.loads(f.read())

    print('training')
    optimization_folder, model_file_name = train(params)
