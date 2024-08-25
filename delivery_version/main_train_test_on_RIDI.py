from create_segments_for_WDE_RIDI import create_segments, get_tain_test_dir
from test_on_full_exp_RIDI import test_on_list
from train_WDE_on_TRC1 import train_on_SBG_data
from os.path import join
from os import listdir
import json
import torch
import os

if __name__ == '__main__':
    params_path = join(os.getcwd(), 'params.json')
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    prepare_data_for_training = False
    performe_training = False
    calculate_performance = True
    if prepare_data_for_training:
        data_folder = create_segments(params)
    else:
        data_folder = '/home/maint/git/walking_direction_estimation/delivery_version/data/XY_pairs/2023-02-21T15:08:09.773703RIDI_pocket_test'
        info_file_path = join(data_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        mode = info["mode"]
        window_size = info["window_size"]
        add_quat = info["add_quat"]
    print('data ready, start train')
    if performe_training:
        optimization_folder, model_file_name = train_on_SBG_data(data_folder, comment=params["comment"],
                                                                 batch_size=params["batch_size"], epochs=params["epochs"],
                                                                 lr=params["LR"], mode=params["mode"], add_quat=params["add_quat"])
    elif calculate_performance:
        optimization_folder = '/home/maint/git/walking_direction_estimation/delivery_version/data/optimization_results/2023-02-21T15:09:51.745391optimization results on pocket, window size: 200test'
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        mode = info["mode"]
        model_file_name = info["best_saved_model"]

    if calculate_performance:
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        data_folder = info["root_dir"]
        model_name = info["best_saved_model"]
        model_path = join(optimization_folder, model_name)
        res18model = torch.load(model_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        res18model.to(device)
        _, test_dir = get_tain_test_dir(mode=info["mode"])
        exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                         'AHRS_results' not in item]
        traj_folder = join(optimization_folder, 'traj')
        if not os.path.exists(traj_folder):
            os.mkdir(traj_folder)
        test_on_list(exp_list=exp_list, info=info, traj_folder=traj_folder,
                     res18model=res18model, dataset='RIDI')