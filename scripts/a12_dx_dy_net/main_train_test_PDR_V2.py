from os.path import join
from os import listdir
import json
import torch
import os
from scripts.a12_dx_dy_net.process_data_for_training import prepare_data_for_training, get_train_test_dir
from scripts.a12_dx_dy_net.test_PDR_net import analyze_inference_results, test_on_list
from scripts.a12_dx_dy_net.train_PDR_net import train as train_WDE
from utils import Functions
import numpy as np


if __name__ == '__main__':
    main_dir = "/home/maint/git/walking_direction_estimation/"
    params_path = join(main_dir, 'scripts/a11_train_odometry_net/params.json')#'scripts/a7_training_on_SZ_dataset/params.json'
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    print(params["preprocess_comment"])
    prepare_data = True
    performe_training = True
    calculate_performance_on_full_exp = True
    if prepare_data:
        print('preparing data')
        data_folder = prepare_data_for_training(params)
        print('data ready')
    elif performe_training:
        data_folder = ''
        info_file_path = join(data_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())
    if performe_training:
        print('training WDE')
        optimization_folder, model_file_name = train_WDE(params)
    elif calculate_performance_on_full_exp:
        optimization_folder = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-03-26T15:53:37.473138_testing effect of window size'
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())
    if calculate_performance_on_full_exp:
        print('testing')
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            params = json.loads(f.read())
        model_name = params["best_saved_model"]
        model_path = join(optimization_folder, model_name)
        print(model_path)
        PDR_net_model = torch.load(model_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        PDR_net_model.to(device)

        traj_folder = join(optimization_folder, 'traj')
        if not os.path.exists(traj_folder):
            os.mkdir(traj_folder)
        dL_source = 'GT'
        PCA_tarj_lengths = []
        PCA_tarj_errors = []
        PCA_errors = []
        PCA_errors_normalized = []
        SP_heading_tarj_lengths = []
        SP_heading_tarj_errors = []
        SP_heading_errors = []
        SP_heading_errors_normalized = []
        resnet18_tarj_lengths = []
        resnet18_tarj_errors = []
        resnet18_errors = []
        resnet18_errors_normalized = []
        file_names = []
        if params["test_on_RIDI"]:
            # RIDI
            _, test_dir = get_train_test_dir(dataset=params["RIDI_params"]["dataset"],
                                             mode=params["RIDI_params"]["mode"])
            exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                        'AHRS_results' not in item]
            PCA_tarj_lengths_RIDI, PCA_tarj_errors_RIDI, PCA_errors_RIDI, PCA_errors_normalized_RIDI, \
            SP_heading_tarj_lengths_RIDI, SP_heading_tarj_errors_RIDI, SP_heading_errors_RIDI, SP_heading_errors_normalized_RIDI, \
            resnet18_tarj_lengths_RIDI, resnet18_tarj_errors_RIDI, resnet18_errors_RIDI, resnet18_errors_normalized_RIDI, \
            file_names_RIDI = test_on_list(exp_list=exp_list, params=params['RIDI_params'],
                                           traj_folder=traj_folder, PDR_net_model=PDR_net_model)
            PCA_tarj_lengths.extend(PCA_tarj_lengths_RIDI)
            PCA_tarj_errors.extend(PCA_tarj_errors_RIDI)
            PCA_errors.extend(PCA_errors_RIDI)
            PCA_errors_normalized.extend(PCA_errors_normalized_RIDI)
            SP_heading_tarj_lengths.extend(SP_heading_tarj_lengths_RIDI)
            SP_heading_tarj_errors.extend(SP_heading_tarj_errors_RIDI)
            SP_heading_errors.extend(SP_heading_errors_RIDI)
            SP_heading_errors_normalized.extend(SP_heading_errors_normalized_RIDI)
            resnet18_tarj_lengths.extend(resnet18_tarj_lengths_RIDI)
            resnet18_tarj_errors.extend(resnet18_tarj_errors_RIDI)
            resnet18_errors.extend(resnet18_errors_RIDI)
            resnet18_errors_normalized.extend(resnet18_errors_normalized_RIDI)
            file_names.extend(file_names_RIDI)
        if params["test_on_SZ"]:
            # SZ
            _, test_dir = get_train_test_dir(dataset=params["SZ_params"]["dataset"], mode=params["SZ_params"]["mode"])
            exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                        'AHRS_results' not in item]
            PCA_tarj_lengths_SZ, PCA_tarj_errors_SZ, PCA_errors_SZ, PCA_errors_normalized_SZ, \
            SP_heading_tarj_lengths_SZ, SP_heading_tarj_errors_SZ, SP_heading_errors_SZ, SP_heading_errors_normalized_SZ, \
            resnet18_tarj_lengths_SZ, resnet18_tarj_errors_SZ, resnet18_errors_SZ, resnet18_errors_normalized_SZ, \
            file_names_SZ = test_on_list(exp_list=exp_list, params=params['SZ_params'],
                                         traj_folder=traj_folder, PDR_net_model=PDR_net_model)
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
        file_path = join(traj_folder, 'performance_on_test_scenarios.csv')
        Functions.save_csv(path=file_path, dic=data_to_save, print_message=True)
        print('PCA mean normalized errors: ' + str(np.array(PCA_errors_normalized).mean()))
        print('SP_heading mean normalized errors: ' + str(np.array(SP_heading_errors_normalized).mean()))
        print('resnet18 mean normalized errors: ' + str(np.array(resnet18_errors_normalized).mean()))
        del data_to_save
        data_to_save = {}
        data_to_save['PCA_mean_errors'] = np.array(PCA_errors).mean()
        data_to_save['SP_heading_mean_errors'] = np.array(SP_heading_errors).mean()
        data_to_save['PDR_net_V2_mean_errors'] = np.array(resnet18_errors).mean()
        data_to_save['PCA_mean_normalized_errors'] = np.array(PCA_errors_normalized).mean()
        data_to_save['SP_heading_mean_normalized_errors'] = np.array(SP_heading_errors_normalized).mean()
        data_to_save['PDR_net_V2_mean_normalized_errors'] = np.array(resnet18_errors_normalized).mean()
        with open(join(traj_folder, 'performance_summery.json'), "w") as outfile:
            json.dump(data_to_save, outfile, indent=4)
        print('Finished Training')
        analyze_inference_results(PCA_tarj_lengths, PCA_tarj_errors, PCA_errors, PCA_errors_normalized, \
                                  SP_heading_tarj_lengths, SP_heading_tarj_errors, SP_heading_errors,
                                  SP_heading_errors_normalized, \
                                  resnet18_tarj_lengths, resnet18_tarj_errors, resnet18_errors,
                                  resnet18_errors_normalized, \
                                  file_names, traj_folder)
