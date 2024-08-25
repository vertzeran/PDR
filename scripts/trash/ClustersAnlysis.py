#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:08:12 2022
A simple clustering script to check if our data has some structure 
(for example, one can say it is a combination of steps and flying)
Unfortunately, it does not have a structure
@author: zahi
"""

import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
#from sklearn.preprocessing import StandardScaler # I dont think we need to scale, because all acc_lin are around 0
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import utils.Classes as Classes

def Plot3DClusters(X,labels,interval=1,title=''):
    """
    X - the data (each row is a sample)
    labels - the clustering resutls 
    interval - plot only 1 out of 10 samples, due to RAM consumption
    """
    fig = plt.figure(title)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[::interval,0], X[::interval,1], X[::interval,2],c=labels[::interval], cmap=matplotlib.colors.ListedColormap(colors))
    
# Load a csv file (change this manually)
exp_path = '/data/Datasets/Navigation/RIDI_dataset_train_test/RIDI - Pocket - Train/hang_leg1.csv'
Exp = Classes.RidiExp(exp_path)
acc = Exp.Acc.arr() # raw accelometer readings
acc_lin = Exp.LinAcc.arr() # accelometer readings after substracting gravity vector

#TODO: Add turn identification to remove segments that can confuse our clustering

# Clustering operators
colors = ['red','green','blue','purple']
spectral_2 = cluster.SpectralClustering(n_clusters=2,eigen_solver="arpack",affinity="nearest_neighbors")
spectral_3 = cluster.SpectralClustering(n_clusters=3,eigen_solver="arpack",affinity="nearest_neighbors")

### Clustering on entire experiment (can we see "step" + "flying"?)
if False:
    X = acc_lin
    spectral_2.fit(X)
    spectral_3.fit(X)
    Plot3DClusters(X,spectral_2.labels_,interval=10,title='LinAcc - 2 clusters')
    Plot3DClusters(X,spectral_3.labels_,interval=10,title='LinAcc - 3 clusters')


### Clustering on segment
if True:
    ind_start = 1111
    ind_stop = 1111 + 250
    X = acc_lin[ind_start:ind_stop,:]
    spectral_2.fit(X)
    spectral_3.fit(X)
    Plot3DClusters(X,spectral_2.labels_,interval=1,title='LinAcc - 2 clusters')
    Plot3DClusters(X,spectral_3.labels_,interval=1,title='LinAcc - 3 clusters')
