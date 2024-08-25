#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 10:41:41 2022
Using Gradient boosting regressor to predict the wde
@author: zahi
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from RidiPytorchDataset import MyRidiPytorchDS
from OperationalFunctions import GetFeaturesFromLinAccTimeSegments,GetFeaturesFromGyroTimeSegments

###################################################################
window_size = 200
mode = 'Body'
data_type = 'RawIMU' #'RawIMU' #'LinAcc'
###################################################################
n_estimators = 20
learning_rate = 0.1
max_depth = 5
###################################################################
train_ds = MyRidiPytorchDS(mode,'Train',window_size,data_type=data_type)
val_ds = MyRidiPytorchDS(mode,'Test',window_size,data_type=data_type)
segments_train = train_ds.X.squeeze().permute(0, 2, 1).numpy() #(1372, 200, 3)
segments_val = val_ds.X.squeeze().permute(0, 2, 1).numpy()
targets_train = train_ds.Y.numpy() #(1372, 3)
targets_val = val_ds.Y.numpy()
print('Done loading')
###################################################################
if data_type == 'LinAcc':
    X_train = GetFeaturesFromLinAccTimeSegments(segments_train)
    X_val = GetFeaturesFromLinAccTimeSegments(segments_val)
if data_type == 'RawIMU':
    X_train = np.hstack((GetFeaturesFromLinAccTimeSegments(segments_train[:,:3]),
                         GetFeaturesFromGyroTimeSegments(segments_train[:,3:])))
    X_val = np.hstack((GetFeaturesFromLinAccTimeSegments(segments_val[:,:3]),
                       GetFeaturesFromGyroTimeSegments(segments_val[:,3:])))
# y_train = np.arctan2(targets_train[:,1],targets_train[:,0])
# y_val = np.arctan2(targets_val[:,1],targets_val[:,0])
y_train = (targets_train[:,1]**2+targets_train[:,0]**2)**0.5
y_val = (targets_val[:,1]**2+targets_val[:,0]**2)**0.5
model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                  loss = 'absolute_error',criterion='friedman_mse',random_state=0)
model.fit(X_train, y_train)
print('RMSE: ',round(mean_squared_error(y_val, model.predict(X_val))**0.5,3))
###################################################################
test_score = np.zeros((n_estimators,), dtype=np.float64)
for i, y_pred in enumerate(model.staged_predict(X_val)):
    test_score[i] = model.loss_(y_val, y_pred)
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.plot(np.arange(n_estimators) + 1,model.train_score_,"b-",label="Train")
plt.plot(np.arange(n_estimators) + 1,test_score,"r-", label="Test")
plt.title("");plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations");plt.ylabel("friedman_mse")
fig.tight_layout()
plt.show();plt.grid()
###################################################################
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos,sorted_idx)
plt.title("Feature Importance (MDI)")