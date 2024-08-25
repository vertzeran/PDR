from utils.Classes import AI_PDR_exp_w_SP_GT
from matplotlib import pyplot as plt
from os.path import join

if __name__ =='__main__':
    dataset_folder = '/data/Datasets/Navigation/Shenzhen_datasets/dataset-ShenZhen'
    train_test = 'test'
    exp_name = 'texting-0029-circle.csv'
    exp_path = join(dataset_folder, train_test, exp_name)
    exp = AI_PDR_exp_w_SP_GT(exp_path)
    exp.limit_traj_length(limit=200)
    exp.define_walking_start_idx(th=2)
    start_time = exp.Time_IMU[exp.index_of_walking_start]
    stop_time = exp.Time_IMU[-1]
    exp.SegmentScenario([start_time, stop_time])
    # exp.PlotAngles()
    # exp.PlotSensors()
    exp.PlotPosition(XY=True, temporal=False)
    plt.show()