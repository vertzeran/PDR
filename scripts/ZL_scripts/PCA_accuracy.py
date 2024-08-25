import numpy as np
from utils import Plot3CordVecs,PlotCDF,PlotTrajectories,NeedToSkipDueToBadValidErr,FixPCAamb
from SBG_Loader import LoadExpAndAHRS_V2,PrepareSequence,LoadGT
from os import listdir
from os.path import join
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# GT_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/texting/21_11_07_firas/ascii-output.txt'
# t_gt,Pos_gt = LoadGT(GT_path)
# dt_gt = np.diff(t_gt,axis=0)
# bad_inds = np.where(dt_gt > 0.02)[0]
# bad_inds_neg = np.where(dt_gt < 0)[0]
# print('GT dt unusual values:',dt_gt[bad_inds],dt_gt[bad_inds_neg])
# plt.figure()
# plt.plot(dt_gt,linestyle="None",marker='x')

### Load exp
exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_09_15_swing_yair_L/outdoor_output_2022-09-15_10_41_56.csv'
#exp_path =  '/data/Datasets/Navigation/SBG-PDR-DATA/swing/22_09_15_swing_yair_L/outdoor_output_2022-09-15_10_23_40.csv'
t_exp,Pos,RV,DCM_vec,Gyro,Acc,Mag,lin_acc_b_frame,grv_hat,ValidErr = LoadExpAndAHRS_V2(exp_path,t_gt=None,Pos_gt=None)
print('Done loading')
if NeedToSkipDueToBadValidErr(ValidErr,'main'):
    raise SystemExit

### Acc at nav frame
lin_a_nav = np.einsum('ijk,ik->ij', DCM_vec, lin_acc_b_frame)  # Nx3
raw_a_nav = np.einsum('ijk,ik->ij', DCM_vec, grv_hat+lin_acc_b_frame)



window_size=200


Pos = Pos-Pos[0,:] # start trajectory at 0,0,0
windows_dl = Pos[window_size:-1:window_size] - Pos[0:-window_size-1:window_size] #N/window_size samples
windows_dl_norm = np.linalg.norm(windows_dl[:,:2],axis=1,keepdims=True) #only x,y
t_start = 0
t_stop = 60
NumOfBatches = int(len(t_exp)/window_size)


wde_est = np.zeros((NumOfBatches, 2))  # (only x,y) , NumOfBatches is expected to be N/window_size
wde_est_raw = np.zeros((NumOfBatches, 2))  # (only x,y) , NumOfBatches is expected to be N/window_size

for k in range(NumOfBatches):
    batch_of_a_nav = lin_a_nav[k * window_size:(k + 1) * window_size, :]
    batch_of_a_nav_raw = raw_a_nav[k * window_size:(k + 1) * window_size, :]

    # PCA:
    pca = PCA(n_components=2).fit(batch_of_a_nav[:, :2])  # only x,y
    wde_est[k, :] = pca.components_[0].reshape(1, 2)  # take 1st component
    wde_est[k, :] = FixPCAamb(wde_est[k, :], windows_dl[k, :2])  # fix ambiguty

    # RAW PCA:
    pca = PCA(n_components=3).fit(batch_of_a_nav_raw)
    wde_est_raw[k, :] = pca.components_[1, :2].reshape(1, 2)  # take 2nd component
    wde_est_raw[k, :] = FixPCAamb(wde_est_raw[k, :], windows_dl[k, :2])  # fix ambiguty

p_est = Pos[0, :2] + np.cumsum(windows_dl_norm * wde_est / np.linalg.norm(wde_est, axis=1, keepdims=True), axis=0)
p_est_raw = Pos[0, :2] + np.cumsum(windows_dl_norm * wde_est_raw / np.linalg.norm(wde_est_raw, axis=1, keepdims=True),
                                   axis=0)

Pos_RMSE = np.linalg.norm(p_est[-1, :] - Pos[-1, :2], keepdims=True)
Pos_raw_RMSE = np.linalg.norm(p_est_raw[-1, :] - Pos[-1, :2], keepdims=True)
print('PCA RMSE:', Pos_RMSE, Pos_raw_RMSE)
PlotTrajectories([Pos, p_est, p_est_raw], ['k', '--b', '--r'], title='GT and PCA-estimated trajectory',
                 ListofLabels=['GT', 'PCA', 'Raw PCA'])
plt.show()