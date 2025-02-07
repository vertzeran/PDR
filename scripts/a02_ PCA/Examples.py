import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
from sklearn.decomposition import PCA


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


if __name__ == '__main__':
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.axis('equal')

    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.components_) # main directions
    print(pca.explained_variance_) # direction variances
    # plot data
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length) # 1 sigma to 3 sigma
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal')
    plt.show()

