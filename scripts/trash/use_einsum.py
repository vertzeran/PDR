import numpy as np
from utils.Functions import transform_vectors
from scipy.spatial.transform import Rotation as Rotation


if __name__ == '__main__':
    n = 3
    dim = 2
    A_list = list(np.arange(n * dim).reshape((n, dim)))
    T = Rotation.from_euler('ZYX', [60, 0, 0], degrees=True).as_matrix() # 3 X 3
    T = T[0:2, 0:2]
    A_rot = []
    for a in A_list:
        A_rot.append((T @ a.reshape(dim, 1).squeeze()))

    # A = np.arange(n * dim).reshape((n, dim)) # 3 X 5
    # A_list_0_rot = np.einsum('ij,jk->ik', T, A_list[0].reshape(3,1)) #

    AA = np.arange(n * dim).reshape(n, dim).T
    # A_rot2 = np.einsum('ij,jkl->jkl', T, AA)  #
    A_rot3 = T @ AA
    print(AA)
    print(A_rot)
    print(A_rot3)