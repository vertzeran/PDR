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

if __name__ == '__main__':
    data_path = 'C:\\Users\\EranVertzberger\\PHD\\ronin_dataset\\dataset\\train_dataset_1\\a000_10'
    exp = Classes.RoninExp(data_path)
    exp.SegmentScenario([0, 10])
    exp_resampled = exp.clone()
    print('resampling')
    exp_resampled.resample(new_SF=250)
    AAE_AHRS = AAE.AtitudeEstimator(Ka=0.005, coor_sys_convention=exp_resampled.Frame)

    print('performing AHRS analysis on: \n' + exp_resampled.Path)
    AAE_AHRS.run_exp(exp=exp_resampled, return_grv=True, return_euler=True,
                                         save_results_to_file=True, visualize=True)
    plt.show()
