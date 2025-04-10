import os
import utils.Classes
from os.path import join
from utils.Classes import RidiExp_ENU
import numpy as np
import matplotlib.pyplot as plt
import utils.Functions as Functions
from utils import Classes, AAE
import ntpath
from scripts.a01_AHRS.calculate_AHRS_results_on_list import CalcResultsOnFile
from scipy import signal
from scripts.a08_training_on_RIDI_ENU.create_segments_for_WDE_RIDI_ENU import get_AHRS_results_for_exp
from scipy.spatial.transform import Rotation as Rotation
from create_segments import get_segments_from_exp_no_gt, get_segments_from_exp

if __name__ == '__main__':
    data_path = 'C:\\Users\\EranVertzberger\\PHD\\ronin_dataset\\dataset\\train_dataset_1\\a001_1'
    exp = Classes.RoninExp(data_path)
    exp.SegmentScenario([0, 500])
    exp_resampled = exp.clone()
    print('resampling...')
    exp_resampled.resample(new_SF=250)

    segment_list, ahrs_results_list = get_segments_from_exp_no_gt(
        exp=exp_resampled, window_size=200, use_gt_att=False,
        win_size_for_heading_init=1000, AHRS_results_path=data_path)
    print("num segments = " + str(len(segment_list)))
    print("num AHRS segments = " + str(len(ahrs_results_list)))
