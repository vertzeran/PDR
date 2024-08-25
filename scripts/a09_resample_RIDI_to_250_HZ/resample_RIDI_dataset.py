import os
from os.path import join
from utils.Classes import RidiExp_ENU
from os import listdir
import ntpath
from scripts.a09_resample_RIDI_to_250_HZ.resample_RIDI_exp import resample_RIDI_exp
import glob

if __name__ == '__main__':
    dataset_location = r"/data/Datasets/Navigation/RIDI_dataset_train_test_ENU"
    root, dir_name = ntpath.split(dataset_location)
    new_dir_name = 'RIDI_dataset_train_test_ENU_250_Hz'
    new_dataset_location = join(root, new_dir_name)
    if not os.path.exists(new_dataset_location):
        os.mkdir(new_dataset_location)
    for sub_dir in listdir(dataset_location):
        exp_path_list = glob.glob(join(dataset_location, sub_dir, "*.csv"))
        for exp_path in exp_path_list:
            exp = RidiExp_ENU(path=exp_path)
            exp = resample_RIDI_exp(exp, new_SF=250)
            sub_dir_in_new_location = join(new_dataset_location, sub_dir)
            if not os.path.exists(sub_dir_in_new_location):
                os.mkdir(sub_dir_in_new_location)
            exp.save_csv(new_path=join(sub_dir_in_new_location, exp.FileName))
    print('finished')