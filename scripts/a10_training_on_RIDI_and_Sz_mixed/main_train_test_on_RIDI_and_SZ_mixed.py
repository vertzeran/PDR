# from algo_performance_evaluation import calculate_performance
from os.path import join
from os import listdir
import json
import torch
import os
from scripts.a10_training_on_RIDI_and_Sz_mixed.process_data_for_training import prepare_data_for_training, get_train_test_dir
from scripts.a10_training_on_RIDI_and_Sz_mixed.test_RIDI_SZ_mixed import analyze_inference_results, test_on_list
from scripts.a10_training_on_RIDI_and_Sz_mixed.train_WDE import train
from utils import Functions
import numpy as np


if __name__ == '__main__':
    main_dir = "/home/maint/git/walking_direction_estimation/"
    params_path = join(main_dir, 'scripts/a10_training_on_RIDI_and_Sz_mixed/params.json')#'scripts/a7_training_on_SZ_dataset/params.json'
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    print(params["comment"])
    prepare_data = False
    performe_training = False
    calculate_performance_on_full_exp = True
    if prepare_data:
        print('preparing data')
        data_folder = prepare_data_for_training(params)
        print('data ready')
    elif performe_training:
        data_folder = '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-03-16T11_31_49.194521RIDI & SZ combined'
        info_file_path = join(data_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())
    if performe_training:
        print('training')
        optimization_folder, model_file_name = train(params)
    elif calculate_performance_on_full_exp:
        optimization_folder = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-03-16T14:49:12.177262optimization results on RIDI & SZ combined'
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
        traj_folder = join(optimization_folder, 'traj')
        if not os.path.exists(traj_folder):
            os.mkdir(traj_folder)

        # RIDI
        _, test_dir = get_train_test_dir(dataset=params["RIDI_params"]["dataset"], mode=params["RIDI_params"]["mode"])
        exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                    'AHRS_results' not in item]
        PCA_tarj_lengths, PCA_tarj_errors, PCA_errors, PCA_errors_normalized, \
        SP_heading_tarj_lengths, SP_heading_tarj_errors, SP_heading_errors, SP_heading_errors_normalized, \
        resnet18_tarj_lengths, resnet18_tarj_errors, resnet18_errors, resnet18_errors_normalized, \
        file_names= test_on_list(exp_list=exp_list, params=params['RIDI_params'],
                                 traj_folder=traj_folder, res18model=res18model)
        # SZ
        _, test_dir = get_train_test_dir(dataset=params["SZ_params"]["dataset"], mode=params["SZ_params"]["mode"])
        exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                    'AHRS_results' not in item]
        PCA_tarj_lengths_SZ, PCA_tarj_errors_SZ, PCA_errors_SZ, PCA_errors_normalized_SZ, \
        SP_heading_tarj_lengths_SZ, SP_heading_tarj_errors_SZ, SP_heading_errors_SZ, SP_heading_errors_normalized_SZ, \
        resnet18_tarj_lengths_SZ, resnet18_tarj_errors_SZ, resnet18_errors_SZ, resnet18_errors_normalized_SZ, \
        file_names_SZ = test_on_list(exp_list=exp_list, params=params['SZ_params'],
                                  traj_folder=traj_folder, res18model=res18model)
        PCA_tarj_lengths.extend(PCA_tarj_lengths_SZ)
        PCA_tarj_errors.extend(PCA_tarj_errors_SZ)
        PCA_errors.extend(PCA_errors_SZ)
        PCA_errors_normalized.extend(PCA_errors_normalized_SZ)
        SP_heading_tarj_lengths.extend(SP_heading_tarj_lengths_SZ)
        SP_heading_tarj_errors.extend(SP_heading_tarj_errors_SZ)
        SP_heading_errors.extend(SP_heading_errors_SZ)
        SP_heading_errors_normalized.extend(SP_heading_errors_normalized_SZ)
        resnet18_tarj_lengths.extend(resnet18_tarj_lengths_SZ)
        resnet18_tarj_errors.extend(resnet18_tarj_errors_SZ)
        resnet18_errors.extend(resnet18_errors_SZ)
        resnet18_errors_normalized.extend(resnet18_errors_normalized_SZ)
        file_names.extend(file_names_SZ)

        data_to_save = {}
        data_to_save["file_names"] = file_names
        data_to_save["PCA_errors"] = PCA_errors
        data_to_save["PCA_errors_normalized"] = PCA_errors_normalized
        data_to_save["SP_heading_errors"] = SP_heading_errors
        data_to_save["SP_heading_errors_normalized"] = SP_heading_errors_normalized
        data_to_save["resnet18_errors"] = resnet18_errors
        data_to_save["resnet18_errors_normalized"] = resnet18_errors_normalized
        file_path = join(traj_folder, 'performance summary.csv')
        Functions.save_csv(path=file_path, dic=data_to_save, print_message=True)
        print('PCA mean normalized errors: ' + str(np.array(PCA_errors_normalized).mean()))
        print('SP_heading mean normalized errors: ' + str(np.array(SP_heading_errors_normalized).mean()))
        print('resnet18 mean normalized errors: ' + str(np.array(resnet18_errors_normalized).mean()))

        analyze_inference_results(PCA_tarj_lengths, PCA_tarj_errors, PCA_errors, PCA_errors_normalized, \
           SP_heading_tarj_lengths, SP_heading_tarj_errors, SP_heading_errors, SP_heading_errors_normalized, \
           resnet18_tarj_lengths, resnet18_tarj_errors, resnet18_errors, resnet18_errors_normalized, \
           file_names, traj_folder)
