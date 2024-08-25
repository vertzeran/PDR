# from algo_performance_evaluation import calculate_performance
from os.path import join
from os import listdir
import json
import torch
import os
from scripts.a08_training_on_RIDI_ENU.create_segments_for_WDE_RIDI_ENU import create_segments, get_train_test_dir
from scripts.a08_training_on_RIDI_ENU.test_on_full_exp_RIDI_ENU import get_exp_list, test_on_person_list, test_on_list
from scripts.a08_training_on_RIDI_ENU.train_WDE import train


if __name__ == '__main__':
    main_dir = "/home/maint/git/walking_direction_estimation/"
    params_path = join(main_dir, 'scripts/a8_training_on_RIDI_ENU/params.json')#'scripts/a7_training_on_SZ_dataset/params.json'
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    print(params["comment"])
    prepare_data_for_training = True
    performe_training = True
    calculate_performance_on_full_exp = True
    if prepare_data_for_training:
        print('preparing data')
        data_folder = create_segments(params)
        print('data ready')
    elif performe_training:
        data_folder = '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-03-14T17_02_56.295302RIDI_ENU_mixed_wind_size_200limit-traj_length_150_epochs'
        info_file_path = join(data_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())
    if performe_training:
        print('training')
        optimization_folder, model_file_name = train(params)
    elif calculate_performance_on_full_exp:
        optimization_folder = ''
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())

    if calculate_performance_on_full_exp:
        print('testing')
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())
        data_folder = params["data_folder"]
        model_name = params["best_saved_model"]
        model_path = join(optimization_folder, model_name)
        res18model = torch.load(model_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        res18model.to(device)
        _, test_dir = get_train_test_dir(dataset=params["dataset"], mode=params["mode"])
        exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                         'AHRS_results' not in item]
        traj_folder = join(optimization_folder, 'traj')
        if not os.path.exists(traj_folder):
            os.mkdir(traj_folder)
        test_on_list(exp_list=exp_list, params=params, traj_folder=traj_folder, res18model=res18model)