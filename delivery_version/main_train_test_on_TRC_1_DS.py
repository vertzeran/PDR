from delivery_version.create_segments_for_WDE_SBG import create_segments, get_dir_to_analyze_TRC_1
from delivery_version.test_on_full_exp_TRC_1 import get_exp_list, test_on_person_list
from delivery_version.train_WDE_on_TRC1 import train_on_SBG_data
from os.path import join
import json
import torch
import os

if __name__ == '__main__':
    params_path = join(os.getcwd(), 'params.json')
    with open(params_path, "r") as f:
        params = json.loads(f.read())
    prepare_data_for_training = False
    performe_training = True
    perform_local_performance_calculation = False
    calculate_performance = False
    if prepare_data_for_training:
        data_folder = create_segments(params)
    else:
        data_folder = '/home/maint/git/walking_direction_estimation/delivery_version/data/XY_pairs/swing'
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
    else:
        optimization_folder = '/home/maint/git/walking_direction_estimation/delivery_version/data/optimization_results/swing'
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
        with open(join(data_folder, 'train_test_division_list.json'), "r") as f:
            train_test_division_list = json.loads(f.read())
        test_persons = train_test_division_list["test"]
        dir_to_analyze = get_dir_to_analyze_TRC_1(dataset_location=params["TRC_1_dataset_location"], mode=mode)
        exp_list = get_exp_list(list_of_persons=test_persons, dir_to_analyze=dir_to_analyze)
        traj_folder = join(optimization_folder, 'traj')
        if not os.path.exists(traj_folder):
            os.mkdir(traj_folder)
        test_on_person_list(person_list=test_persons,
                            dir_to_analyze=dir_to_analyze,
                            info=info,
                            traj_folder=traj_folder,
                            res18model=res18model,
                            predict_dL_with_PDRnet=False,
                            pdr_net=None,
                            add_quat=params["add_quat"]
                            )