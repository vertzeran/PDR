from scripts.a05_train_on_TRC_1.create_segments_for_WDE_SBG import create_segments, get_dir_to_analyze
from scripts.a05_train_on_TRC_1.test_on_full_exp_TRC_1 import get_exp_list, test_on_person_list
from scripts.a05_train_on_TRC_1.train_WDE_on_TRC1 import train_on_SBG_data
from algo_performance_evaluation import calculate_performance
from os.path import join
import json
import torch
import os

if __name__ == '__main__':
    comment = '_add_quat'
    print(comment)
    add_quat = True
    heading_fix = False
    traj_length_limit = 200
    prepare_data_for_training = False
    performe_training = True
    perform_local_performance_calculation = False
    calculate_performance_on_full_exp = True
    if prepare_data_for_training:
        mode = 'text'
        window_size = 200
        data_folder = create_segments(comment=comment, window_size=window_size, mode=mode, add_quat=add_quat, heading_fix=heading_fix,
                                      traj_length_limit=traj_length_limit, heading_fix_initialization_time=None)
    else:
        data_folder = '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-01-30T04:39:45.396622TRC#1 swing wind_size 200 add quat to input'
        # '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-02-13T18:20:16.494803TRC#1 text wind_size 200no_quat'
        # '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-01-30T04:39:45.396622TRC#1 swing wind_size 200 add quat to input'
        # '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-02-09T18:00:08.334343TRC#1 swing wind_size 200no quat no heading fix'

        info_file_path = join(data_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        mode = info["mode"]
        window_size = info["window_size"]
        add_quat = info["add_quat"]
        # data_folder = '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-02-06T19:26:25.254241TRC#1 swing wind_size 200_heading_angle_fix_initialization_using_full_traj'
        # '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-02-05T12:27:58.902312TRC#1 swing wind_size 200_heading_angle_fix'
        # '/home/maint/git/walking_direction_estimation/data/XY_pairs/2023-01-30T04:39:45.396622TRC#1 swing wind_size 200 add quat to input'
    print('data ready, start train')
    if performe_training:
        optimization_folder, model_file_name = train_on_SBG_data(data_folder, comment, batch_size=512, epochs=100,
                                                                 lr=0.0001, mode=mode, add_quat=add_quat, heading_fix=heading_fix)
    else:
        optimization_folder = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-02-13T18:34:44.226565optimization results on text, window size: 200no_quat'
        # '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-01-30T10:28:31.654615optimization results on Swing, window size: 200 add quaternion to input'
        # '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-02-09T18:27:26.623140optimization results on swing, window size: 200no quat no heading fix'
        # ('/home/maint/git/walking_direction_estimation/data/optimization_results/2023-01-30T10:28:31.654615optimization results on Swing, window size: 200 add quaternion to input'
        #                             'WDE_regressor_Swing_LinAcc_0.34.pth')
        info_file_path = join(optimization_folder, 'info_file.json')
        with open(info_file_path, "r") as f:
            info = json.loads(f.read())
        mode = info["mode"]
        model_file_name = info["best_saved_model"]

    if perform_local_performance_calculation:
        print('calculate local performance')
        calculate_performance(
            data_location=data_folder,
            res18_optimization_results_location=optimization_folder,
            model_name=model_file_name,
            WD_est_method='compare_all_methods',
            add_quat=add_quat)
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
        with open(join(data_folder, 'train_test_division_list.json'), "r") as f:
            train_test_division_list = json.loads(f.read())
        test_persons = train_test_division_list["test"]
        dir_to_analyze = get_dir_to_analyze(data_location='magneto', mode=mode)
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
                            add_quat=add_quat
                            )