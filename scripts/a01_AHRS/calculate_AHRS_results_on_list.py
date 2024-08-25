import argparse
from os.path import join, exists
from utils.Classes import RidiExp, SbgExpRawData, AI_PDR_exp_w_SP_GT, RidiExp_ENU
from utils.AAE import AtitudeEstimator
from os import listdir
import pandas as pd
import glob
"""
A script to calculate the AHRS results for one or more experiments
to run (example):
python calculate_AHRS_results_on_list.py 
--root_dir /data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_ran
--dataset TRC#1
"""


def CalcResultsOnFile(file_path, dataset, override, GT=None):
    # Skip if file exist (without loading the exp)
    suffix_to_identify_results = '_AHRS_results.xlsx'
    possible_result_file = file_path[:-4] + suffix_to_identify_results
    if exists(possible_result_file) and not override:
        print('found AHRS results for: \n' + file_path)
        return

    print('Loading exp: \n',file_path)
    if dataset == 'RIDI':
        exp = RidiExp(file_path)
        Ka = 0.00026096
        AAE_AHRS = AtitudeEstimator(Ka=Ka)
    if dataset == 'AI_PDR':
        exp = AI_PDR_exp_w_SP_GT(file_path)
        Ka = 0.00026096
        AAE_AHRS = AtitudeEstimator(Ka=Ka, coor_sys_convention=exp.Frame)
    if dataset == 'TRC#1':
        if GT is not None:
            exp = SbgExpRawData(file_path, GT=GT)
        else:
            exp = SbgExpRawData(file_path)
        Ka = 0.005
        AAE_AHRS = AtitudeEstimator(Ka=Ka, coor_sys_convention=exp.Frame)
        if exp.first_idx_of_time_gap_IMU.shape[0] != 0:
            print('---Found time gap in IMU')
        if exp.first_idx_of_time_gap_GPS.shape[0] != 0:
            print('---Found time gap in GPS')
    if dataset == 'RIDI_ENU':
        exp = RidiExp_ENU(file_path)
        Ka = 0.00026096
        AAE_AHRS = AtitudeEstimator(Ka=Ka, coor_sys_convention=exp.Frame)

    AHRS_results_files = exp.check_for_AHRS_results()
    if len(AHRS_results_files) == 0 or override:
        print('performing AHRS analysis on: \n' + exp.FileName)
        _, _, _, _, _, _, _, _, = AAE_AHRS.run_exp(exp=exp, return_grv=True, return_euler=True,save_results_to_file=True, visualize=False)
    # Skip if file exist (after loading the exp). we need both checks becasue suffix_to_identify_results
    # might be changed in the future and we will not know this until loading exp
    else:
        print('found AHRS results for: \n' + file_path)


def CalcResultsOnRootDir(root_dir,dataset,override):
    data_list = glob.glob(root_dir + '/*.csv')
    # data_list = listdir(root_dir)
    # data_list = [item for item in data_list if '_AHRS_results.xlsx' not in item]
    # data_list = [item for item in data_list if 'ascii-output.txt' not in item]
    # data_list = [item for item in data_list if '.json' not in item]
    # data_list = [item for item in data_list if '.png' not in item]
    if dataset == 'TRC#1':
        GT_path = join(root_dir, 'ascii-output.txt')
        GT = pd.read_csv(GT_path, sep='\t', skiprows=28)
    else:
        GT = None
    for ind,file_path in enumerate(data_list):
        # file_path = join(root_dir, file_name)
        print(str(ind) + ' out of ' + str(len(data_list)))
        CalcResultsOnFile(file_path, dataset, override, GT=GT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_list', type=str, default=None, help='if not given work on the whole directory')
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--root_of_roots', type=str, default=None, help='root of a root folders')
    parser.add_argument('--dataset', type=str, default='RIDI', help='could be RIDI, TRC#1')
    parser.add_argument('--override', type=str, default='False', help='override current results to make sure no bug was made')
    args = parser.parse_args()
    dataset = args.dataset
    override = args.override
    # args.root_dir = 'C:/Eran/Onebox Sync Folder/Nav Projects/AI IMU/Datasets/RIDI dataset Train-Test/RIDI - Bag - Test'
    ###########################################################################
    if args.exp_list is not None:
        assert(args.root_dir is None)
        assert(args.root_of_roots is None)
        with open(args.exp_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
        for ind,file_name in enumerate(data_list):
            file_path = join(args.root_dir, file_name+'.csv')
            print(str(ind) + ' out of ' + str(len(data_list)))
            CalcResultsOnFile(file_path,dataset,override)

    ###########################################################################
    if args.root_dir is not None:
        assert(args.exp_list is None)
        assert(args.root_of_roots is None)
        CalcResultsOnRootDir(args.root_dir,dataset,override)

    ###########################################################################
    if args.root_of_roots is not None:
        assert(args.exp_list is None)
        assert(args.root_dir is None)
        for root in listdir(args.root_of_roots):
            print('########################')
            print('Processing root dir: ', root)
            CalcResultsOnRootDir(join(args.root_of_roots, root),dataset,override)
