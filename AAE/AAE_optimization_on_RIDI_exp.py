import utils.Classes as Classes
from utils.AAE import AtitudeEstimator
from numpy.linalg import norm
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

def att_error(ka, Exp):
    AAE_AHRS = AtitudeEstimator(Ka=ka)
    phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e = AAE_AHRS.run_exp(exp=Exp, visualize=False)
    loss = norm([np.mean(np.abs(phi_e)), np.mean(np.abs(theta_e))])
    # loss = np.mean([norm(phi_e), norm(theta_e)])
    return loss


def opt_callback(x, state):
    print(state.nit, ' ', x[0], ' ', state.fun)


if __name__ == '__main__':
    Exp = Classes.RidiExp('/home/maint/git/ahrs/AAE/hang_leg1.csv')
    Exp.SegmentScenario([0, 5])
    x0 = 0.1
    att_error_init = att_error(x0, Exp)
    print(att_error_init)
    bounds = Bounds(0, 1)
    res = minimize(att_error, x0, args=Exp, method='trust-constr',
                   options={'verbose': 1, 'disp': True}, bounds=bounds, callback=opt_callback)
    ka_opt = res.x
    print(ka_opt) #[0.00026096]
    AAE_AHRS = AtitudeEstimator(Ka=ka_opt)
    phi_hat, phi_e, theta_hat, theta_e, psi_hat, psi_e = AAE_AHRS.run_exp(exp=Exp, visualize=True)
    att_error_final = np.mean([norm(phi_e), norm(theta_e)])
    print(att_error_final)