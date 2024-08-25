import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.optimize import minimize
from os import listdir
from scipy.spatial.transform import Rotation as Rotation
from sklearn.decomposition import PCA
import datetime
import torch
import pandas as pd
from torch import nn


def FoldAngles(u):
    """angles are folded to +-pi"""
    u = np.squeeze(u)
    if len(u.shape) == 0:
        u = u.reshape([1, 1])

    NumOfSamples = u.shape[0]
    y = np.zeros(u.shape)
    for i in range(NumOfSamples):
        if u[i] >= 0:
            y[i] = np.mod(u[i], 2 * np.pi)
            if y[i] > np.pi:
                y[i] = y[i] - 2 * np.pi
        else:
            y[i] = np.mod(u[i], -2 * np.pi)
            if y[i] < -np.pi:
                y[i] = y[i] + 2 * np.pi
    return y.squeeze()


def ContinuousAngle(U, units='rad'):
    n = len(U)
    Y = U.copy()
    Counter = 0
    if units == 'deg':
        Cycle = 360
    elif units == 'rad':
        Cycle = 2 * np.pi
    else:
        print("ContinuousAngle:wrong units")
    for i in range(1, n - 1):
        # if abs(U[i] - U[i - 1]) > Cycle / 10:
        #     print(U[i] - U[i - 1])
        if (U[i] - U[i - 1]) > Cycle / 2:
            Counter = Counter - 1
        elif (U[i] - U[i - 1]) < - Cycle / 2:
            Counter = Counter + 1
        Y[i] = U[i] + Cycle * Counter
    return Y


def RotMat2euler(R):
    # one = torch.from_numpy(np.array([1])).float()
    phi = np.arctan2(R[2, 1], R[2, 2])
    theta = -np.arctan(R[2, 0] / np.sqrt(1 - R[2, 0] ** 2))
    psi = np.arctan2(R[1, 0], R[0, 0])

    Euler = [phi, theta, psi]
    return Euler


def Vec2SkewSimetric(Vec):
    SSM = np.array([[0, -Vec[2], Vec[1]],
                    [Vec[2], 0, -Vec[0]],
                    [-Vec[1], Vec[0], 0]])
    return SSM


def OrthonormalizeRotationMatrix(R):
    Rn = R.dot(np.linalg.inv(scipy.linalg.sqrtm(R.T.dot(R))))  # Rnb_m*(Rnb_m'*Rnb_m)^-0.5
    return Rn


def PlotEulerAngles(t, Phi, PhiHat, Theta, ThetaHat, Psi, PsiHat):
    plt.close('Euler Angles Plot')
    fig = plt.figure('Euler Angles Plot')
    Ax1 = fig.add_subplot(321)
    Ax1.plot(t, Phi * 180 / np.pi, color='blue', linewidth=1)
    Ax1.plot(t, PhiHat * 180 / np.pi, color='red', linewidth=1)
    Ax1.set(title=r"$\phi$", xlabel="", ylabel="[deg]"), Ax1.grid(True)

    ePhi = Phi - PhiHat
    Ax2 = fig.add_subplot(322, sharex=Ax1)
    Ax2.plot(t, ePhi * 180 / np.pi, color='black', linewidth=1)
    Ax2.set(title=r"$e_{\Phi}$", xlabel="", ylabel="[deg]"), Ax2.grid(True)

    Ax3 = fig.add_subplot(323, sharex=Ax1)
    Ax3.plot(t, Theta * 180 / np.pi, color='blue', linewidth=1)
    Ax3.plot(t, ThetaHat * 180 / np.pi, color='red', linewidth=1)
    Ax3.set(title=r"$\theta$", xlabel="", ylabel="[deg]"), Ax3.grid(True)

    eTheta = Theta - ThetaHat
    Ax4 = fig.add_subplot(324, sharex=Ax1)
    Ax4.plot(t, eTheta * 180 / np.pi, color='black', linewidth=1)
    Ax4.set(title=r"$e_{\theta}$", xlabel="", ylabel="[deg]"), Ax4.grid(True)

    Ax5 = fig.add_subplot(325, sharex=Ax1)
    Ax5.plot(t, Psi * 180 / np.pi, color='blue', linewidth=1)
    Ax5.plot(t, PsiHat * 180 / np.pi, color='red', linewidth=1)
    Ax5.set(title=r"$\psi$", xlabel="", ylabel="[deg]"), Ax5.grid(True)

    ePsi = FoldAngles(Psi - PsiHat)
    Ax6 = fig.add_subplot(326, sharex=Ax1)
    Ax6.plot(t, ePsi * 180 / np.pi, color='black', linewidth=1)
    Ax6.set(title=r"$e_{\psi}$", xlabel="", ylabel="[deg]"), Ax6.grid(True)

    plt.tight_layout()
    plt.show()
    print('RMS(ePhi) = ', RMS(ePhi) * 180 / np.pi)
    print('RMS(eTheta) = ', RMS(eTheta) * 180 / np.pi)
    print('RMS(ePsi) = ', RMS(ePsi) * 180 / np.pi)


def TRIAD(fb, mb, fn, mn):
    W1 = fb / np.linalg.norm(fb)
    W2 = mb / np.linalg.norm(mb)

    V1 = fn
    V2 = mn

    Ou1 = W1
    Ou2 = np.cross(W1, W2) / np.linalg.norm(np.cross(W1, W2))
    Ou3 = np.cross(W1, np.cross(W1, W2)) / np.linalg.norm(np.cross(W1, W2))

    R1 = V1
    R2 = np.cross(V1, V2) / np.linalg.norm(np.cross(V1, V2))
    R3 = np.cross(V1, np.cross(V1, V2)) / np.linalg.norm(np.cross(V1, V2))

    Mou = np.vstack([Ou1, Ou2, Ou3]).T
    Mr = np.vstack([R1, R2, R3]).T

    Cbn = Mr.dot(Mou.T)
    return Cbn


def RMS(u):
    return np.sqrt(np.mean(u ** 2))


def find_graph_offset(x1, y1, x2, y2, initial_cond=[0.0], plot_results=False):
    """
    calculate a constant offset to be added to x2 to minimize the error between y1 and y2

    :return: offset
    """

    def graph_error(o, args=(x1, y1, x2, y2)):
        """objective function"""
        y2_interp = np.interp(x=x1, xp=x2 + o, fp=y2)
        e = y2_interp - y1
        return np.linalg.norm(e)

    opt_res = []
    for x0 in initial_cond:
        opt_res_i = minimize(graph_error, x0, method='Powell', options={'xatol': 1e-8, 'disp': True})
        opt_res.append(opt_res_i)

    def sorting_criteria(item):
        return item.fun

    opt_res.sort(reverse=False, key=sorting_criteria)
    offset = opt_res[0].x
    final_error = opt_res[0].fun
    if plot_results:
        test = False
        if test:
            # offset = -1
            # print("test mode, e = " + str(graph_error(offset)))
            x = np.linspace(-2, 2, num=50)
            y = np.zeros(x.shape)
            for i in range(x.shape[0]):
                y[i] = graph_error(x[i])
            fig = plt.figure('optimization function')
            ax = fig.add_subplot(111)
            ax.plot(x, y, label='J')
            ax.set(xlabel="offset")
            ax.grid(True)
            ax.legend()

        fig = plt.figure('offset')
        ax = fig.add_subplot(111)
        ax.plot(x1, y1, label='f1')
        ax.plot(x2, y2, label='f2')
        ax.plot(x2 + offset, y2, label='f2 with offset')
        ax.grid(True)
        ax.legend()
    return offset, final_error


def att_error(phi_e, theta_e):
    loss = np.linalg.norm([np.mean(np.abs(phi_e)), np.mean(np.abs(theta_e))])
    return loss


def search_list_of_dict(list_of_dict, key_to_search_by, desired_value):
    """
    search list_of_dict for  item[key_to_search_by] == desired_value

    :return: index
    """
    # iterator = (item for item in list_of_dict if item[key_to_search_by] == desired_value)
    # next(iterator, None)
    # if only return the list componenet
    iterator = (i for i in range(len(list_of_dict)) if list_of_dict[i][key_to_search_by] == desired_value)
    indx = next(iterator, None)
    return list_of_dict[indx], indx


def files_in_dir_containing_string(path, string):
    files_in_dir = listdir(path)
    iterator = (file for file in files_in_dir if string in file)
    file_list = []
    file_name = next(iterator, None)
    while file_name is not None:
        file_list.append(file_name)
        file_name = next(iterator, None)
    return file_list


def limit_value(input, min_value, max_value):
    return max(min(input, max_value), min_value)


def transform_vectors(va: np.array, Rot: Rotation):
    """
    transform va to vb using Rba
    :param va: numpy array with size n_samples X 3 or 1X3
    :param Rot: rotation object with n_samples instances
    :return: vb numpy array with size n_samples X 3
    """
    if len(va.shape) == 2:
        assert va.shape[1] == 3
        assert va.shape[0] == len(Rot)
        n_samples = len(Rot)
        vb = np.zeros(va.shape)
        for i in range(n_samples):
            Rba = Rot[i].as_matrix()
            vb[i, :] = Rba.dot(va[i, :].reshape(3, 1)).squeeze()
    elif len(va.shape) == 1:
        # to convert same vector by varrying rotations
        assert va.shape[0] == 3
        n_samples = len(Rot)
        vb = np.zeros([n_samples, 3])
        for i in range(n_samples):
            Rba = Rot[i].as_matrix()
            vb[i, :] = Rba.dot(va.reshape(3, 1)).squeeze()
    return vb


def construct_traj(dl=None, walking_angle=None, dx_dy=None , plot_result=False, pos_gt=None, method='WDE_dL'):
    """
    This function sums the walking vectors (dl+headings)
    into a single trajectory
    method: can be 'WDE_dL' or dx_dy
    """
    if method == 'WDE_dL':
        dl = dl.squeeze()
        walking_angle = walking_angle.squeeze()
        assert dl.squeeze().shape == walking_angle.squeeze().shape
        # vector with dx or dy elements along the trajectory:
        dx_vector = np.cos(walking_angle) * dl
        dy_vector = np.sin(walking_angle) * dl
    elif method == 'dx_dy':
        dx_vector = dx_dy[:, 0]
        dy_vector = dx_dy[:, 1]
    # summing dx, adding the initial condition x=0,y=0 and reshaping to 2D vector and
    x = np.insert(np.cumsum(dx_vector), obj=0, values=0).reshape(-1, 1)
    y = np.insert(np.cumsum(dy_vector), obj=0, values=0).reshape(-1, 1)

    traj = np.hstack((x, y))
    if plot_result:
        fig = plt.figure('construct_traj plot')
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlabel=r"$x [m]$", ylabel=r"$y [m]$", title="position GT"), ax.grid(True)
        ax.plot(traj[:, 0], traj[:, 1], color='red', linestyle='-', linewidth=2, label='')
        if pos_gt is not None:
            ax.plot(pos_gt[:, 0], pos_gt[:, 1], color='black', linestyle='--', linewidth=2, label='')
        plt.show()
    return traj


def PCA_main_direction(data, n_dim=2, plot_res=False):
    """
    fitting PCA on data, to find the main component, and then plotting the main componnets
    Note that there is no dimention reduction since the input is 2D vector
    aquiered by accelometer vector after substracting the gravity
    """
    pca = PCA(n_components=n_dim)  # keep 2 components
    pca.fit(data)
    if plot_res:
        fig = plt.figure('acc WDE Plot')
        Ax = fig.add_subplot(111)
        Ax.scatter(data[:, 0], data[:, 1], alpha=0.2)
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)  # 1 sigma to 3 sigma
            draw_vector(pca.mean_, pca.mean_ + v, ax=Ax)
        Ax.axis('equal')
    return pca.components_[0]


def draw_vector(v0, v1, ax=None, c='blue'):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0,
                      color=c)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def MyCDF(data):
    # getting data of the histogram
    count, bins_count = np.histogram(data, bins=10)
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
    return bins_count, cdf


def est_second_direction(data, n_dim=3, plot_res=False):
    """
    fitting PCA on data, to find the main component, and then plotting the main componnets
    Note that there is no dimention reduction since the input is 2D vector
    aquiered by accelometer vector after substracting the gravity
    """
    pca = PCA(n_components=n_dim)  # keep 3 components
    pca.fit(data)
    if plot_res:
        fig = plt.figure('acc WDE Plot')
        Ax = fig.add_subplot(111)
        Ax.scatter(data[:, 0], data[:, 1], alpha=0.2)
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)  # 1 sigma to 3 sigma
            draw_vector(pca.mean_, pca.mean_ + v, ax=Ax)
        Ax.axis('equal')
    return pca.components_[1]  # take 2nd component


def utctoweekseconds(utc, leapseconds):
    """ Returns the GPS week, the GPS day, and the seconds
        and microseconds since the beginning of the GPS week """
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00", datetimeformat)
    tdiff = utc - epoch + datetime.timedelta(seconds=leapseconds)
    gpsweek = tdiff.days // 7
    gpsdays = tdiff.days - 7 * gpsweek
    gpsseconds = tdiff.seconds + 86400 * (tdiff.days - 7 * gpsweek)
    return gpsweek, gpsseconds  # ,gpsdays,tdiff.microseconds


def gpsTimeToUnixTime(week, sec):
    """convert GPS week and TOW to a time in seconds since 1970"""
    epoch = 86400 * (10 * 365 + int((1980 - 1969) / 4) + 1 + 6 - 2)
    return epoch + 86400 * 7 * week + sec - 18


def Radius(Lat):
    R0 = 6.378388e6
    Rp = 6.356912e6
    e = np.sqrt(1 - (Rp / R0) ** 2)
    RN = R0 / np.sqrt(1 - e ** 2 * np.sin(Lat) ** 2)
    RM = R0 * (1 - e ** 2) / (1 - e ** 2 * np.sin(Lat) ** 2) ** (3 / 2)
    Re = R0 * (1 - e * np.sin(Lat) ** 2)

    return RN, RM, Re


def LLLN2NED(Lat, Long, Alt, RN, RM):
    # omri's function
    dLat = Lat - Lat[0]
    dLong = Long - Long[0]
    dAlt = Alt - Alt[0]
    Pn = dLat * (RM + Alt)
    Pe = dLong * (RN + Alt) * np.cos(Lat)
    Pd = -dAlt

    return Pn, Pe, Pd


def LLA2ECEF(lat, lon, alt):
    # WGS84 ellipsoid constants:

    a = 6378137
    e = 8.1819190842622e-2

    # intermediate calculation
    # (prime vertical radius of curvature)
    N = a / np.sqrt(1 - np.power(e, 2) * np.power(np.sin(lat * np.pi / 180), 2))

    # results:
    x = (N + alt) * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    y = (N + alt) * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
    z = ((1 - np.power(e, 2)) * N + alt) * np.sin(lat * np.pi / 180)

    return x, y, z


def Mphi(Phi):
    Mphi = np.array(
        [[1, 0, 0],
         [0, np.cos(Phi), np.sin(Phi)],
         [0, -np.sin(Phi), np.cos(Phi)]])
    return Mphi


def Mtheta(Theta):
    Mtheta = np.array(
        [[np.cos(Theta), 0, -np.sin(Theta)],
         [0, 1, 0],
         [np.sin(Theta), 0, np.cos(Theta)]])
    return Mtheta


def Mpsi(Psi):
    Mpsi = np.array(
        [[np.cos(Psi), np.sin(Psi), 0],
         [-np.sin(Psi), np.cos(Psi), 0],
         [0, 0, 1]])
    return Mpsi


def DCM_Ned2ECEF(Long, Lat):
    M12 = Mtheta(Lat * np.pi / 180 + np.pi / 2)
    M01 = Mpsi(-Long * np.pi / 180)
    DCM = np.dot(M01, M12)
    return DCM


def DCM_ECEF2NED(Long, Lat):
    M10 = Mpsi(Long * np.pi / 180)
    M21 = Mtheta(-Lat * np.pi / 180 - np.pi / 2)
    DCM = np.dot(M21, M10)
    return DCM


def ECEF2LLA(x, y, z):
    # WGS84 ellipsoid constants:
    a = float(6378137)
    e = 8.1819190842622e-2
    # calculations:
    b = np.sqrt(np.power(a, 2) * (1 - np.power(e, 2)))
    ep = np.sqrt((np.power(a, 2) - np.power(b, 2)) / np.power(b, 2))
    p = np.sqrt(np.power(x, 2) + np.power(y, 2))
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + np.power(ep, 2) * b * np.power(np.sin(th), 3),
                     p - np.power(e, 2) * a * np.power(np.cos(th), 3))
    N = a / np.sqrt(1 - np.power(e, 2) * np.power(np.sin(lat), 2))
    alt = p / np.cos(lat) - N
    # return lon in range [0,360)
    lon = np.mod(lon, 2 * np.pi)
    # correct for numerical instability in altitude near exact poles:
    # (after this correction, error is about 2 millimeters, which is about
    # the same as the numerical precision of the overall function)
    if abs(x) < 1 and abs(y) < 1:
        alt = abs(z) - b

    return lat, lon, alt


def LocalNav2Geo(p0, L, Psi):
    """Find LLA location from azimuth distance"""
    # 1. calc p0 in ECEF
    [Xecef, Yecef, Zecef] = LLA2ECEF(p0[0], p0[1], p0[2])
    P0_ECEF = np.array([Xecef, Yecef, Zecef])
    # 2. calculate DCM NED->ECEF at p0
    R_ECEF_NED = DCM_Ned2ECEF(p0[0], p0[1])
    # 3.calc p1 in LLLN
    p1_NED = L * np.array([np.cos(Psi), np.sin(Psi), 0])
    # 4. calc 2nd point in ECEF
    p1_ECEF = np.dot(R_ECEF_NED, p1_NED) + P0_ECEF
    # 5. convert ECEF 2 LLA
    [Lat, Long, Alt] = ECEF2LLA(p1_ECEF[0], p1_ECEF[1], p1_ECEF[2])
    p1_LLA = np.array([Lat * 180 / np.pi, Long * 180 / np.pi, Alt])
    return p1_LLA


def LLA2NED(lat, long, alt):
    x_ECEF, y_ECEF, z_ECEF = LLA2ECEF(lat, long, alt)


def PrepareInputForResnet18(x, add_dim=True):
    '''
    Input - LinAccAtNavFrame is an array with shape (1,200,3).
            It is a time window of the Linear accelaration after rotation
            Please note that we assume the "batch dim" was already added
    Output - Input to be fed into resent 18
    '''
    X = torch.tensor(x, dtype=torch.float)
    X = X.permute(0, 2, 1)  # instead of X = np.swapaxes(X, axis1, axis2)
    if add_dim:
        X = X[:, :, :, None]  # created another dim for resnet 18
    return X


def PrepareInputForResnet18RV(BatchOfLinAccAtNavFrameRV):
    '''
    Input - LinAccAtNavFrameRV is an array with shape (1,200,7).
            It is a time window of the Linear accelaration after rotation
            Please note that we assume the "batch dim" was already added
    Output - Input to be fed into resent 18
    '''
    X = torch.tensor(BatchOfLinAccAtNavFrameRV, dtype=torch.float)
    X = X.permute(0, 2, 1)  # instead of X = np.swapaxes(X, axis1, axis2)
    X = X[:, :, :, None]  # created another dim for resnet 18
    return X


def save_to_excel_file(path, dic, print_message=False):
    df = pd.DataFrame(dic)
    df.to_excel(path, index=False, header=True)
    if print_message:
        print('results saved to ' + path)


def load_from_excel_file(path):
    df = pd.read_excel(path, engine='openpyxl')
    return df


def save_csv(dic, path, print_message=False):
    df = pd.DataFrame(data=dic)
    df.to_csv(path)
    if print_message:
        print('results saved to ' + path)


def read_AHRS_results(path):
    AHRS_results = load_from_excel_file(path)
    t = np.array(AHRS_results.time_IMU)
    phi_hat = AHRS_results.phi_hat
    theta_hat = AHRS_results.theta_hat
    psi_hat = AHRS_results.psi_hat
    phi_e = AHRS_results.phi_e
    theta_e = AHRS_results.theta_e
    psi_e = AHRS_results.psi_e
    grv = np.vstack([AHRS_results.grv_x, AHRS_results.grv_y, AHRS_results.grv_z]).T
    n = len(AHRS_results.grv_x)
    Rnb = np.zeros([n, 3, 3])
    Rnb[:, 0, 0] = AHRS_results.Rnb_11
    Rnb[:, 0, 1] = AHRS_results.Rnb_12
    Rnb[:, 0, 2] = AHRS_results.Rnb_13
    Rnb[:, 1, 0] = AHRS_results.Rnb_21
    Rnb[:, 1, 1] = AHRS_results.Rnb_22
    Rnb[:, 1, 2] = AHRS_results.Rnb_23
    Rnb[:, 2, 0] = AHRS_results.Rnb_31
    Rnb[:, 2, 1] = AHRS_results.Rnb_32
    Rnb[:, 2, 2] = AHRS_results.Rnb_33
    Rot = Rotation.from_matrix(Rnb)
    return t, phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e, grv, Rot


def rotate_trajectory(traj, alfa):
    """
    traj = vector of xy positions size  n X 2
    alfa = rotation angle in radians
    """
    assert traj.shape[1] == 2
    T = Rotation.from_euler('ZYX', [alfa, 0, 0], degrees=False).as_matrix()  # 3 X 3
    T = T[0:2, 0:2]
    rot_traj = (T @ traj.T).T  # (2X2 @ 2Xn).T = (2 X n).T = n X 2
    # T = Mpsi(alfa)[0:2, 0:2]
    # n = traj.shape[0]
    # rot_traj = np.zeros(traj.shape)
    # for i in range(n):
    #     rot_traj[i] = T @ traj[i]
    return rot_traj


def traj_error(traj_est, traj_gt):
    traj_error = np.linalg.norm(traj_est - traj_gt, axis=1).mean()
    return traj_error


def convert_model_2d_to_1d(model: nn.Module):
    """
    converts model's 2D modules (Conv2d, BatchNorm2d, etc.) into 1D modules (Conv1d, BatchNorm1d, etc.)
    :param model: model
    :raises NotImplementedError when found unsupported 2D module
    """
    for name, child in model.named_children():
        if len(list(child.children())):  # module has children - call recursively
            convert_model_2d_to_1d(child)

        elif isinstance(child, nn.Conv2d):
            new_child = nn.Conv1d(
                in_channels=child.in_channels, out_channels=child.out_channels, kernel_size=child.kernel_size[0],
                stride=child.stride[0], bias=child.bias is not None, padding=child.padding[0],
                dilation=child.dilation[0]
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight[..., child.weight.shape[3] // 2])
                if child.bias is not None:
                    new_child.bias.copy_(child.bias)
            setattr(model, name, new_child)

        elif isinstance(child, nn.BatchNorm2d):
            new_child = nn.BatchNorm1d(
                num_features=child.num_features, eps=child.eps, momentum=child.momentum, affine=child.affine
            )
            with torch.no_grad():
                new_child.weight.copy_(child.weight)
                if child.bias is not None:
                    new_child.bias.copy_(child.bias)
            setattr(model, name, new_child)

        elif isinstance(child, nn.MaxPool2d):
            new_child = nn.MaxPool1d(
                kernel_size=child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation
            )
            setattr(model, name, new_child)

        elif isinstance(child, nn.AdaptiveAvgPool2d):
            new_child = nn.AdaptiveAvgPool1d(output_size=child.output_size[0])
            setattr(model, name, new_child)

        elif child.__class__.__name__.endswith('2d'):
            raise NotImplementedError(f'found unsupported 2d module: {child}')


def rotation_matrix_to_r6d(r):
    """
    Turn rotation matrices into 6D vectors

    :param r: Rotation matrix that can reshape to [batch_size, 3, 3].
    :return: 6D vector array of shape [batch_size, 6].
    """
    return r.reshape(-1, 3, 3)[:, :, :2].transpose((0, 2, 1)).reshape(-1, 6)
