from create_segments_for_WDE_RIDI import create_segments, get_tain_test_dir
from test_on_full_exp_RIDI import get_exp_list, test_on_person_list, test_on_list
from train_WDE_on_TRC1 import train_on_SBG_data
# from algo_performance_evaluation import calculate_performance
from os.path import join
from os import listdir
import json
import torch
import os

if __name__ == '__main__':
    comment = '_RIDI_mixed_modes'
    print(comment)
    add_quat = True
    heading_fix = False
    allign_traj_at_test = False
    traj_length_limit = 200
    prepare_data_for_training = False
    performe_training = True
    perform_local_performance_calculation = False
    calculate_performance_on_full_exp = True
    if prepare_data_for_training:
        mode = 'mixed'
        window_size = 200
        data_folder = create_segments(comment=comment, window_size=window_size, mode=mode, add_quat=add_quat, heading_fix=heading_fix,
                                      traj_length_limit=traj_length_limit, heading_fix_initialization_time=None)
    else:
        data_folder = '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-02-28T16:19:19.890326RIDI mixed wind_size 200_RIDI_mixed_modes'
        info_file_path = join(data_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        mode = info["mode"]
        window_size = info["window_size"]
        add_quat = info["add_quat"]
    print('data ready, start train')
    if performe_training:
        optimization_folder, model_file_name = train_on_SBG_data(data_folder, comment, batch_size=512, epochs=150,
                                                                 lr=0.0001, mode=mode, add_quat=add_quat, heading_fix=heading_fix)
    else:
        optimization_folder = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-02-21T14:12:18.179100optimization results on text, window size: 200_test_RIDI_text_with_quat'
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        mode = info["mode"]
        model_file_name = info["best_saved_model"]

    # if perform_local_performance_calculation:
    #     print('calculate local performance')
    #     calculate_performance(
    #         data_location=data_folder,
    #         res18_optimization_results_location=optimization_folder,
    #         model_name=model_file_name,
    #         WD_est_method='compare_all_methods',
    #         add_quat=add_quat)
    if calculate_performance_on_full_exp:
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        data_folder = info["root_dir"]
        model_name = info["best_saved_model"]
        model_path = join(optimization_folder, model_name)
        res18model = torch.load(model_path)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        res18model.to(device)
        _, test_dir = get_tain_test_dir(mode=mode)
        exp_list = [join(test_dir, item) for item in listdir(test_dir) if
                         'AHRS_results' not in item]
        traj_folder = join(optimization_folder, 'traj')
        if not os.path.exists(traj_folder):
            os.mkdir(traj_folder)
        test_on_list(exp_list=exp_list, info=info, traj_folder=traj_folder,
                     res18model=res18model, add_quat=add_quat, dataset='RIDI',
                     allign_traj=allign_traj_at_test)