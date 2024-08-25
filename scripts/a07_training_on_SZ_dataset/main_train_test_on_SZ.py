# from algo_performance_evaluation import calculate_performance
from os.path import join
from os import listdir
import json
import torch
import os
from scripts.a07_training_on_SZ_dataset.create_segments_for_WDE_SZ import create_segments, get_tain_test_dir
from scripts.a07_training_on_SZ_dataset.test_on_full_exp_SZ import get_exp_list, test_on_person_list, test_on_list
from scripts.a07_training_on_SZ_dataset.train_WDE import train


if __name__ == '__main__':
    main_dir = "/home/maint/git/walking_direction_estimation/"
    params_path = join(main_dir, 'scripts/a7_training_on_SZ_dataset/params.json')#'scripts/a7_training_on_SZ_dataset/params.json'
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    comment = '_SZ_mixed_modes'
    print(comment)
    # add_quat = True
    # heading_fix = False
    # allign_traj_at_test = False
    # traj_length_limit = 250
    prepare_data_for_training = True
    performe_training = True
    calculate_performance_on_full_exp = True
    if prepare_data_for_training:
        data_folder = create_segments(params)
    elif performe_training:
        data_folder = '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-03-13T10_36_14.465081AI_PDR mixed wind_size 250_SZ_mixed_modes'
        info_file_path = join(data_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())

    print('data ready, start train')
    if performe_training:
        optimization_folder, model_file_name = train(params)
    elif calculate_performance_on_full_exp:
        optimization_folder = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-03-13T11:54:13.910513optimization results on mixed, window size: 250_SZ_mixed_modes'
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())

    if calculate_performance_on_full_exp:
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())
        data_folder = params["data_folder"]
        model_name = params["best_saved_model"]
        model_path = join(optimization_folder, model_name)
        res18model = torch.load(model_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        res18model.to(device)
        _, test_dir = get_tain_test_dir(mode=params["mode"])
        exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                         'AHRS_results' not in item]
        traj_folder = join(optimization_folder, 'traj')
        if not os.path.exists(traj_folder):
            os.mkdir(traj_folder)
        test_on_list(exp_list=exp_list, params=params, traj_folder=traj_folder, res18model=res18model)