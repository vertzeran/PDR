import os
import scipy.io as sio
import numpy as np
import shutil


data_roots = [
    '/home/adam/git/walking_direction_estimation/data/XY_pairs/2023-05-31T11:51:49.350143TRC#1 mixed wind_size 200test',
    '/home/adam/git/walking_direction_estimation/data/XY_pairs/2023-03-30T17_31_56.289009_RIDI_ENU_and_SZ_WS_200',
]

out_root = '/home/adam/git/walking_direction_estimation/data/XY_pairs/RIDI_ENU_and_SZ_WS_and_TRC#1 mixed wind_size 200test'
os.makedirs(out_root, exist_ok=True)


def load_data(matfile):
    data = sio.loadmat(matfile)
    x = data['X']
    if 'Y1' in data:
        y1 = data['Y1']  # dl
        y2 = data['Y2']  # wd
    elif 'Y' in data:
        y = data['Y']
        y1 = np.linalg.norm(y, axis=1, keepdims=True)
        y2 = y / y1
    else:
        raise 'ahhh'

    return x, y1, y2


x_test_list = []
y1_test_list = []
y2_test_list = []
x_train_list = []
y1_train_list = []
y2_train_list = []
for data_root in data_roots:
    test_file = os.path.join(data_root, 'test.mat')
    train_file = os.path.join(data_root, 'train.mat')

    x_test, y1_test, y2_test = load_data(test_file)
    x_train, y1_train, y2_train = load_data(train_file)

    x_test_list.append(x_test)
    y1_test_list.append(y1_test)
    y2_test_list.append(y2_test)

    x_train_list.append(x_train)
    y1_train_list.append(y1_train)
    y2_train_list.append(y2_train)

x_test = np.vstack(x_test_list)
y1_test = np.vstack(y1_test_list)
y2_test = np.vstack(y2_test_list)
x_train = np.vstack(x_train_list)
y1_train = np.vstack(y1_train_list)
y2_train = np.vstack(y2_train_list)

sio.savemat(os.path.join(out_root, 'test.mat'), {'X': x_test, 'Y1': y1_test, 'Y2': y2_test})
sio.savemat(os.path.join(out_root, 'train.mat'), {'X': x_train, 'Y1': y1_train, 'Y2': y2_train})
shutil.copyfile(os.path.join(data_roots[0], 'info_file.json'), os.path.join(out_root, 'info_file.json'))
