import os
import sys
from datetime import datetime
import json

import torch
from torch.utils.data import DataLoader
from torch import optim
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt

from scripts.a13_TRC_dx_dy.losses import LOSSES
from scripts.a13_TRC_dx_dy.train_pdr_net import resnet50_model, resnet18_model, MySBGPytorchDS
from utils.Functions import convert_model_2d_to_1d


def normalize_wd_output(wd_output):
    return wd_output / torch.linalg.norm(wd_output.detach(), dim=1, keepdim=True)


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

    dl_net = resnet_func(num_of_input_channels=num_of_input_channels, num_of_outputs=1)
    wd_net = resnet_func(num_of_input_channels=num_of_input_channels, num_of_outputs=2)
    if params["resnet1d"]:
        convert_model_2d_to_1d(dl_net)
        convert_model_2d_to_1d(wd_net)
        print('converted to 1d resnet')

    data_type = 'LinAcc'  # 'RawIMU' #'LinAcc'

    dl_loss_name = params["dl_loss"]
    wd_loss_name = params["wd_loss"]
    dl_loss_func = LOSSES[dl_loss_name]
    wd_loss_func = LOSSES[wd_loss_name]
    dl_test_loss_func = LOSSES['mse_loss']
    wd_test_loss_func = LOSSES['norm_loss']

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
    dl_optimizer = optim.Adam(dl_net.parameters(), lr=params["LR"], weight_decay=params.get("weight_decay", 0))
    wd_optimizer = optim.Adam(wd_net.parameters(), lr=params["LR"], weight_decay=params.get("weight_decay", 0))
    dl_scheduler = optim.lr_scheduler.MultiStepLR(dl_optimizer, milestones=params["milestones"])
    wd_scheduler = optim.lr_scheduler.MultiStepLR(wd_optimizer, milestones=params["milestones"])

    x_test, dl_gt_test, wd_gt_test = next(iter(test_dl))
    x_test = x_test.to(device)
    dl_gt_test = dl_gt_test.to(device)
    wd_gt_test = wd_gt_test.to(device)

    dl_net.to(device)
    wd_net.to(device)

    # Training loop
    cost_dl_train = []
    cost_wd_train = []
    cost_dl_test = []
    cost_wd_test = []
    dl_diff_test = []
    dl_min_loss = 100000000
    wd_min_loss = 100000000
    wd_min_name = dl_min_name = None

    p_bar = tqdm(range(params["epochs"]), total=params["epochs"], unit='epoch')
    for epoch in p_bar:  # loop over the dataset multiple times
        avg_dl_loss = []
        avg_wd_loss = []
        for data in train_dl:
            x, dl, wd = data
            x = x.to(device)
            dl = dl.to(device)
            wd = wd.to(device)

            dl_optimizer.zero_grad()
            wd_optimizer.zero_grad()

            dl_output = dl_net(x)
            wd_output = wd_net(x)
            wd_output = normalize_wd_output(wd_output)

            dl_loss = dl_loss_func(dl_output, dl)
            wd_weight = None
            if params["wd_loss_weight"] == 'dl':
                wd_weight = dl.detach()
            wd_loss = wd_loss_func(wd_output, wd, weight=wd_weight)

            dl_loss.backward()
            wd_loss.backward()

            dl_optimizer.step()
            wd_optimizer.step()

            avg_dl_loss.extend([dl_loss.item()] * x.shape[0])
            avg_wd_loss.extend([wd_loss.item()] * x.shape[0])

        dl_scheduler.step()
        wd_scheduler.step()

        avg_dl_loss = sum(avg_dl_loss) / len(avg_dl_loss)
        avg_wd_loss = sum(avg_wd_loss) / len(avg_wd_loss)

        # Val loss
        dl_net.eval()
        wd_net.eval()
        with torch.no_grad():
            outputs_test_dl = dl_net(x_test)
            outputs_test_wd = wd_net(x_test)
            outputs_test_wd = normalize_wd_output(outputs_test_wd)
            test_mean_dl_diff = (outputs_test_dl - dl_gt_test).mean().item()

            dl_loss_test = dl_test_loss_func(outputs_test_dl, dl_gt_test).item()
            if dl_loss_test < dl_min_loss:
                if create_output_folder:
                    if dl_min_name is not None:
                        os.remove(join(outputfolder, dl_min_name))
                    dl_min_name = f'dl_regressor_epoch{epoch}_dl_loss_{dl_loss_name}_{dl_min_loss:.3f}.pth'
                    torch.save(dl_net, join(outputfolder, dl_min_name))
                dl_min_loss = dl_loss_test

            wd_loss_test = wd_test_loss_func(outputs_test_wd, wd_gt_test).item()
            if wd_loss_test < wd_min_loss:
                if create_output_folder:
                    if wd_min_name is not None:
                        os.remove(join(outputfolder, wd_min_name))
                    wd_min_name = f'wd_regressor_epoch{epoch}_wd_loss_{wd_loss_name}_{wd_min_loss:.3f}.pth'
                    torch.save(wd_net, join(outputfolder, wd_min_name))
                wd_min_loss = wd_loss_test

        dl_net.train()
        wd_net.train()

        # print statistics
        cost_dl_train.append(avg_dl_loss)
        cost_wd_train.append(avg_wd_loss)
        cost_dl_test.append(dl_loss_test)
        cost_wd_test.append(wd_loss_test)
        dl_diff_test.append(test_mean_dl_diff)

        loss_dict = {
            'dl train': avg_dl_loss,
            'wd train': avg_wd_loss,
            'dl test': dl_loss_test,
            'wd test': wd_loss_test,
            'test dl diff': test_mean_dl_diff
        }
        p_bar.set_postfix(loss_dict)

        plt.figure()
        plt.plot(cost_dl_train, label='dl_train')
        plt.plot(cost_wd_train, label='wd_train')
        plt.plot(cost_dl_test, label='dl_test')
        plt.plot(cost_wd_test, label='wd_test')
        plt.plot(dl_diff_test, label='dl_diff_test')
        plt.grid(True)
        plt.legend()
        if create_output_folder:
            plt.savefig(join(outputfolder, 'losses.png'))
        plt.close()

    print('Finished Training')
    return outputfolder, dl_min_name


if __name__ == '__main__':
    info_file_path = 'params_2_models.json'
    with open(info_file_path, "r") as f:
        params = json.loads(f.read())

    print('training')
    optimization_folder, model_file_name = train(params)
