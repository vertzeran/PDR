#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:35:01 2023
A script to plot nice figures for presentations
It is not written good because I wanted controll on colors, markers etc
We assume that resnet without rv is the best estimator for pocket mode
@author: zahi
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from scipy.optimize import curve_fit
from utils import PlotMeanWithStdFils

### Load
root_dir = '/data/Work/Navigation/SBGResults/'
mode = 'texting'
path_len_all = sio.loadmat(root_dir + mode + '_results' + '.mat')['path_len_all']
pos_all = sio.loadmat(root_dir + mode + '_results' + '.mat')['pos_all']
p_est_trc_all = sio.loadmat(root_dir + mode + '_results' + '.mat')['p_est_resnet_all' if mode == 'pocket' else 'p_est_resnet_rv_all' ]
p_est_pca_all = sio.loadmat(root_dir + mode + '_results' + '.mat')['p_est_pca_all' ]
p_est_smh_all = sio.loadmat(root_dir + mode + '_results' + '.mat')['p_est_SMH_all' ]

### Pos Error
calc_err_norm = lambda p_est : np.linalg.norm(pos_all - p_est,axis=1,keepdims=True)
trc_pos_err_norm_all = calc_err_norm(p_est_trc_all)
pca_pos_err_norm_all = calc_err_norm(p_est_pca_all)
smh_pos_err_norm_all = calc_err_norm(p_est_smh_all)

### Calc Pos erorr mean + std for quantized path len (we must group path lens to create statistics) 
quantized_path_len = (np.round((path_len_all/25),0)*25).astype(int)
path_len_edges = np.sort(np.unique(quantized_path_len)).reshape(-1,1)
path_len_axis = np.arange(10,path_len_edges.max(),10) # create a smoother plot with samples every 10[m]
trc_mean,trc_std,pca_mean,pca_std,smh_mean,smh_std = np.array([]).reshape(6,-1,1)

for p_len in path_len_edges:
    inds = np.where(quantized_path_len==p_len)[0]
    
    trc_mean = np.append(trc_mean,trc_pos_err_norm_all[inds].mean())
    pca_mean = np.append(pca_mean,pca_pos_err_norm_all[inds].mean())
    smh_mean = np.append(smh_mean,smh_pos_err_norm_all[inds].mean())
    
    trc_std = np.append(trc_std,trc_pos_err_norm_all[inds].std())
    pca_std = np.append(pca_std,pca_pos_err_norm_all[inds].std())
    smh_std = np.append(smh_std,smh_pos_err_norm_all[inds].std())

### Create a smoother plot with samples every 10[m]
interp_to_path_len_axis = lambda y : np.interp(path_len_axis,path_len_edges.reshape(-1,),y.reshape(-1,))
trc_mean,pca_mean,smh_mean = [interp_to_path_len_axis(x) for x in [trc_mean,pca_mean,smh_mean]]
trc_std,pca_std,smh_std = [interp_to_path_len_axis(x) for x in [trc_std,pca_std,smh_std]]

### Plot Pos error mean + std, as a function of path len
PlotMeanWithStdFils(path_len_axis,[trc_mean,pca_mean,smh_mean],
                                  [trc_std,pca_std,smh_std],
                                  ['g','b','r'],
                                  ['TRC','PCA','Smartphone heading'])
PlotMeanWithStdFils(path_len_axis,100*np.array([trc_mean,pca_mean,smh_mean])/path_len_axis,
                                  100*np.array([trc_std,pca_std,smh_std])/path_len_axis,
                                  ['g','b','r'],
                                  ['TRC','PCA','Smartphone heading'])
plt.xlim([0,200]) # there are little trajectories with such len, so the statistic is not correct
plt.ylim([0,50]) # for better visualization