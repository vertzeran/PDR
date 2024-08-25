#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:01:18 2022
Collection of auxillary function that are not required for delivery,
but are used in the  development process for anlysis etc
Some of them are very "RIDI oriented"
Therefore, they will not be documented
@author: zahi
"""

import numpy as np
import matplotlib.pyplot as plt

def CutByFirstDim(ListOfTensors,ind_start,ind_stop):
    NewListOfTensors = []
    for tens in ListOfTensors:
        NewListOfTensors.append(tens[ind_start:ind_stop,:])
    return NewListOfTensors
        
def Plot3CordVecs(ListOfMats,ListofColors,suptitle='', ListofTitles = None, xlabel='', ylabel=''):
    plt.figure()
    plt.suptitle(suptitle)
    for cord_ind in range(3):
        plt.subplot(3,1,cord_ind+1)
        if ListofTitles is not None: plt.title(ListofTitles[cord_ind])
        for mat_ind,mat in enumerate(ListOfMats):
            plt.plot(mat[:,cord_ind],ListofColors[mat_ind])
        plt.grid();plt.xlabel(xlabel);plt.ylabel(ylabel)

def PlotTrajectories(ListOfMats,ListofColors,ListofLabels,title='',SaveAndClose = False):
    plt.figure()
    plt.title(title)
    for mat_ind,mat in enumerate(ListOfMats):
        plt.plot(mat[:,0],mat[:,1],ListofColors[mat_ind],label = ListofLabels[mat_ind])
    plt.legend()
    plt.grid();plt.xlabel('X[m]');plt.ylabel('Y[m]')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    if SaveAndClose:
        plt.savefig('figures/'+title+'.png')
        plt.close()
    
def MyCDF(data):
    # getting data of the histogram
    count, bins_count = np.histogram(data, bins=100)
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return bins_count,cdf 

def PlotCDF(ListOf1DArrays, ListofColors, title='', ListofLabels = None, xlabel = ''):
    plt.figure()
    for mat_ind,arr in enumerate(ListOf1DArrays):
        bins_count,cdf = MyCDF(arr)
        plt.plot(bins_count[1:], cdf, ListofColors[mat_ind], label=ListofLabels[mat_ind])   
    plt.title(title);plt.legend()
    plt.grid();plt.xlabel(xlabel);plt.ylabel('CDF')
    
def FixPCAamb(est_vec,gt_vec):
    if np.dot(est_vec,gt_vec)<0:
        est_vec = -est_vec
    return est_vec  

def GetTimeValidtyCode():
    edges = [0.05,0.01,0.02,np.inf]
    codes = [0,1,2,3]
    assert(len(edges)==len(codes))
    return edges,codes

def NeedToSkipDueToBadValidErr(ValidErr,file_name):
    skip = False
    print('ValidErr:',ValidErr)
    best_code = (GetTimeValidtyCode()[1][0]) # for codes, 0 for best code
    if ValidErr['Pos'] > best_code:
        # probably bad record of standing user like firas 11_45_52
        print('Bad record (probably a standing man):',file_name)
        skip = True #The code is working even without the continue sometimes (it crete a sqeuence with size 0x200x3)
    return skip

def EstimateTraj(Pos,wde_est,windows_dl_norm):
    p_est = np.cumsum(wde_est*windows_dl_norm,axis=0) #Assuming normlized WDE
    p_est = np.vstack((np.zeros((1,2)),p_est)) + Pos[0,:2] 
    return p_est

def CalcNormliizedError(Pos,p_est,path_len):
    err = np.linalg.norm(Pos[:,:2] - p_est,axis=1,keepdims=True)
    normed_err =  err / path_len
    return normed_err

def PlotMeanWithStdFils(x,ListOfMeans,ListOfSTD,ListOfCol,ListOfLabels,alpha = 0.2,linewidth=4):
    plt.figure()
    for ind,y_mean in enumerate(ListOfMeans):
        y_std = ListOfSTD[ind]
        col = ListOfCol[ind]
        plt.plot(x,y_mean,col,label=ListOfLabels[ind])
        plt.fill_between(x.reshape(-1,), y_mean+y_std/2, y_mean-y_std/2,
            alpha=alpha, facecolor=col, antialiased=True)
        plt.legend();plt.grid()