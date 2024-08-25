import os
import utils.Classes
from utils.AAE import AdaptiveAtitudeEstimator
from numpy.linalg import norm
import numpy as np
from scipy.optimize import minimize ,Bounds
from data_handling.create_training_data_RIDI import \
    get_samples_from_dir, get_batches_from_samples
from os.path import join
from datetime import datetime
import json
import argparse
from os import listdir
from utils.Classes import RidiExp


def att_error_sample(ka_vec, sample: utils.Classes.AhrsExp):
    # print('sample id: ' + str(sample.id) + '; errors : ' + str(sample.ePhi0) + ', ' + str(sample.eTheta0))
    AAE_AHRS = AdaptiveAtitudeEstimator(num_of_sample_points=ka_vec.shape[0])
    AAE_AHRS.gain_map.gain_vector = ka_vec
    phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e = \
        AAE_AHRS.run_exp(exp=sample, add_errors=True, visualize=False)
    att_err = norm([np.mean(np.abs(phi_e)), np.mean(np.abs(theta_e))]) * 180 / np.pi
    return att_err


def att_err_batch(ka_vec, batch):
    err_list = []
    for sample in batch:
        err_list.append(att_error_sample(ka_vec, sample))
    return np.mean(np.array(err_list))


def att_err_batch_statistics(ka_vec, batch):
    err_list = []
    for sample in batch:
        err_list.append(att_error_sample(ka_vec, sample))
    mean_err = np.mean(np.array(err_list))
    std_err = np.std(np.array(err_list))
    return err_list, mean_err, std_err


def loss(ka_vec):
    # print('optimization batch ids:')
    # print_batch_sample_ids(batches[batch_idx_for_loss])
    # print(ka_vec)
    global current_func_value
    l = att_err_batch(ka_vec, batches[batch_idx_for_loss])
    current_func_value = l
    print(l)
    return l


def opt_callback(x):
    global batches, batch_idx_for_loss, opt_trj_path, opt_trj, iteration_idx
    batch_idx_for_loss += 1
    if batch_idx_for_loss == len(batches):
        batches = get_batches_from_samples(samples, batch_size, use_residual=True)
        batch_idx_for_loss = 0
        print('shuffle performed')
    iteration_data = {"iteration": iteration_idx, "x": list(x), "loss": current_func_value}
    with open(opt_trj_path, 'w') as f:
        opt_trj.append(iteration_data)
        json.dump(opt_trj, f, indent=4)
        f.close()
    # print("iteration: " + str(state.nit) + " ?= " + str(iteration_idx + 1))
    # print("function value: " + str(state.fun) + " ?= " + str(current_func_value))
    iteration_idx += 1
    # print(state.nit, ' ', x[0], ' ', state.fun)
    print('iter: ' + str(iteration_data["iteration"]) + ' loss = ' + str(iteration_data["loss"]))


def test(opt_res_path):
    for file in os.listdir(opt_res_path):
        if 'info' in file:
            info_file = file
        elif 'trj_file' in file:
            trj_file = file
        elif 'test_losses' in file:
            test_losses_file = file
    trj_file_path = join(opt_res_path,trj_file)
    info_file_path = join(opt_res_path, info_file)
    if 'test_losses_file' in locals():
        test_losses_file_found = True
    else:
        test_losses_file_found = False
    if test_losses_file_found:
        test_losses_file_path = join(opt_res_path, test_losses_file)
        with open(test_losses_file_path, 'r') as file:
            test_losses = json.load(file)
        first_iteration_to_calculate = len(test_losses)
    else:
        test_losses = []
        first_iteration_to_calculate = 0

    with open(trj_file_path, 'r') as file:
        traj = json.load(file)
    with open(info_file_path, 'r') as file:
        info = json.load(file)
    sample_size = info["sample_size"]
    batch_size = info["batch_size"]
    att_err_amp = info["att_err_amp"]
    data_set_location = info["data_set_location"]
    train_set_subfolder = info["train_set_subfolder"]
    test_set_subfolder = info["test_set_subfolder"]
    validation_set_subfolder = info["validation_set_subfolder"]
    test_path = join(data_set_location, test_set_subfolder)
    test_samples = get_samples_from_dir(test_path, sample_size, att_err_amp)
    i = 0
    train_losses = []
    update_test_losses_file = False
    for iteration in traj:
        train_losses.append(iteration["loss"])
        print(str(i + 1) + ' out of ' + str(len(traj)))
        if i >= first_iteration_to_calculate:
            ka_vec = np.array(iteration["x"])
            test_losses.append(att_err_batch(ka_vec, test_samples))
            if not update_test_losses_file:
                update_test_losses_file = True
        i += 1
    if update_test_losses_file:
        with open(join(opt_res_path, 'test_losses.json.json.json'), 'w') as file:
            json.dump(test_losses, file, indent=4)
            file.close()


def select_optimal_gain_map(opt_res_path):
    for file in os.listdir(opt_res_path):
        if 'info' in file:
            info_file = file
        elif 'trj_file' in file:
            trj_file = file
        elif 'test_losses.json' in file:
            test_losses_file = file
        elif 'optimal_gain' in file:
            optimal_gain_file = file
    trj_file_path = join(opt_res_path, trj_file)
    info_file_path = join(opt_res_path, info_file)
    if 'test_losses_file' in locals():
        test_losses_file_found = True
    else:
        raise 'didnt find the test losses. run analyze_opt_traj.py and try again'

    test_losses_file_path = join(opt_res_path, test_losses_file)
    with open(test_losses_file_path, 'r') as file:
        test_losses = json.load(file)
    idx_of_best_iteration = list(test_losses).index(min(list(test_losses)))
    with open(trj_file_path, 'r') as file:
        traj = json.load(file)
    best_iteration = traj[idx_of_best_iteration]
    optimal_gain = best_iteration["x"]
    optimal_gain_file_name = 'optimal_gain.json'
    optimal_gain_file_path = join(opt_res_path, optimal_gain_file_name)
    with open(optimal_gain_file_path, 'w') as file:
        json.dump(optimal_gain, file, indent=4)
        file.close()


def validate(opt_res_path):
    for file in os.listdir(opt_res_path):
        if 'info' in file:
            info_file = file
        elif 'trj_file' in file:
            trj_file = file
        elif 'test_losses.json' in file:
            test_losses_file = file
        elif 'optimal_gain' in file:
            optimal_gain_file = file
    trj_file_path = join(opt_res_path,trj_file)
    info_file_path = join(opt_res_path, info_file)

    if 'optimal_gain_file' not in locals():
        raise 'didnt find optimal gain file'
    optimal_gain_file_path = join(opt_res_path, optimal_gain_file)
    with open(optimal_gain_file_path, 'r') as file:
        optimal_gain = np.array(json.load(file))

    with open(info_file_path, 'r') as file:
        info = json.load(file)
    sample_size_loc = info["sample_size"]
    batch_size_loc = info["batch_size"]
    att_err_amp_loc = info["att_err_amp"]
    data_set_location = info["data_set_location"]
    train_set_subfolder = info["train_set_subfolder"]
    test_set_subfolder = info["test_set_subfolder"]
    validation_set_subfolder = info["validation_set_subfolder"]
    test_path = join(data_set_location, test_set_subfolder)
    test_samples = get_samples_from_dir(test_path, sample_size_loc, att_err_amp_loc)

    test_err_list, test_err_mean, test_err_std = att_err_batch_statistics(optimal_gain, test_samples)

    validation_samples = []
    for item in listdir(test_path):
        item_path = join(test_path, item)  # exp dir
        sample = RidiExp(item_path)
        sample.eTheta0 = 0.0
        sample.ePhi0 = 0.0
        validation_samples.append(sample)

    validation_err_list, validation_err_mean, validation_err_std = att_err_batch_statistics(optimal_gain, validation_samples)

    validation_results = {"test_err_list": test_err_list,
                          "test_err_mean": test_err_mean,
                          "test_err_std":  test_err_std,
                          "validation_err_list": validation_err_list,
                          "validation_err_mean": validation_err_mean,
                          "validation_err_std": validation_err_std}
    validation_results_file_name = 'validation_results.json'
    validation_results_file_path = join(opt_res_path, validation_results_file_name)
    with open(validation_results_file_path, 'w') as file:
        json.dump(validation_results, file, indent=4)
        file.close()


if __name__ == '__main__':
    """
       to run from terminal:
       conda activate AHRS
       export PYTHONPATH=$PYTHONPATH:/home/maint/git/ahrs
       cd /home/maint/git/ahrs
       python ./AAE/AAE_batch_optimization_on_RIDI.py
       """
    parser = argparse.ArgumentParser()
    parser.add_argument('--description', type=str, required=False)
    parser.add_argument('--data_set_location', type=str,
                        default='/home/maint/Eran/AHRS/RIDI_dataset_train_test')
    parser.add_argument('--train_set_subfolder', type=str, required=True)
    parser.add_argument('--test_set_subfolder', type=str, required=True)
    parser.add_argument('--validation_set_subfolder', type=str, required=False)
    parser.add_argument('--optimization_log_file_path', type=str,
                        default='/home/maint/git/ahrs/logs')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--number_of_acc_sample_points', type=int, default=25)
    parser.add_argument('--att_err_amp', type=float, default=0.1)
    parser.add_argument('--finite_diff_rel_step', type=float, default=0.005)
    parser.add_argument('--initial_cond', type=float, default=0.01)

    args = parser.parse_args()
    comment = args.description
    data_set_path = args.data_set_location
    train_subdirectory = args.train_set_subfolder
    sample_size = args.sample_size  # GT_samples
    initial_cond = args.initial_cond
    finite_diff_rel_step = args.finite_diff_rel_step
    path = join(data_set_path, train_subdirectory)
    global samples, batches, batch_idx_for_loss, batch_size, \
        opt_trj_path, sample_id, opt_trj, att_err_amp, iteration_idx, current_func_value
    iteration_idx = 0
    att_err_amp = args.att_err_amp
    opt_trj = []
    sample_id = 0
    batch_size = args.batch_size

    optimization_information = {}
    optimization_information['description'] = 'stochastic batch otimization'
    optimization_information['solver'] = 'L-BFGS-B' # 'TNC', 'L-BFGS-B','trust-constr','SLSQP'
    optimization_information['sample_size'] = args.sample_size
    optimization_information['batch_size'] = args.batch_size
    optimization_information['data_set_location'] = args.data_set_location
    optimization_information['train_set_subfolder'] = args.train_set_subfolder
    optimization_information['test_set_subfolder'] = args.test_set_subfolder
    if args.validation_set_subfolder is None:
        optimization_information['validation_set_subfolder'] = ''
    else:
        optimization_information['validation_set_subfolder'] = args.validation_set_subfolder
    optimization_information['initial_cond'] = args.initial_cond
    optimization_information['att_err_amp'] = args.att_err_amp

    now = datetime.isoformat(datetime.now())
    info_file_name = 'opt_info_' + now + '.json'
    optimization_folder_name = 'optimization_' + now
    optimization_folder_path = join(args.optimization_log_file_path, optimization_folder_name)
    os.mkdir(optimization_folder_path)
    info_file_path = join(optimization_folder_path, info_file_name)
    with open(info_file_path, 'w') as f:
        json.dump(optimization_information, f, indent=4)
        f.close()

    opt_trj_file_name = 'opt_trj_file ' + now + '.json'
    opt_trj_path = join(optimization_folder_path, opt_trj_file_name)

    print('gathering samples...')
    samples = get_samples_from_dir(path, sample_size, att_err_amp)
    # samples = [samples[0]]
    # print('initial sampl ids')
    # print_batch_sample_ids(samples)
    batches = get_batches_from_samples(samples, batch_size, use_residual=True)
    # batches = [batches[0]]
    # print('batches sample ids')
    # print_batches_sample_ids(batches)
    batch_idx_for_loss = 0
    opt_trj = []
    number_of_acc_sample_points = args.number_of_acc_sample_points
    x0 = np.ones(number_of_acc_sample_points) * initial_cond
    bounds = Bounds(np.zeros(number_of_acc_sample_points), np.ones(number_of_acc_sample_points))
    print('starting optimization ...')
    current_func_value = loss(x0)
    opt_callback(x0)
    i_max = 25
    i = 1
    while iteration_idx < i_max:
        res = minimize(
            loss, x0,
            method=optimization_information['solver'],
            jac='3-point',
            tol=1e-8,
            options={
                'ftol': 1e-8,
                'gtol': 1e-8,
                'disp': True,
                'maxiter': 100,
                'eps': 1e-8,
                'finite_diff_rel_step': np.ones(number_of_acc_sample_points) * finite_diff_rel_step,
                'iprint': 100
            },
            bounds=bounds,
            callback=opt_callback
        )
        x0 = res.x
        i += 1
    test(optimization_folder_path)
    select_optimal_gain_map(optimization_folder_path)
    validate(optimization_folder_path)