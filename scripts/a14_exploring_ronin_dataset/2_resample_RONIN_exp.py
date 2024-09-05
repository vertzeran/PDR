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
    data_path = 'C:\\Users\\EranVertzberger\\PHD\\ronin_dataset\\dataset\\train_dataset_1\\a001_1'
    exp = Classes.RoninExp(data_path)
    exp.SegmentScenario([0, 60])
    exp_resampled = exp.clone()
    exp_resampled.resample(new_SF=250)
