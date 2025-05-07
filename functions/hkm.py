import numpy as np
from .balanced_km import balanced_km
from .euclid import euclid_dist2
np.random.seed(0)
def hkm(X: np.ndarray, idx0: np.ndarray, k: int, count: int):
    n = X.shape[1]
    C_sub, F_sub, y_sub = balanced_km(X[:, idx0].T)
    ys = 2*count + 1 - y_sub
    unique_sum = np.sum(np.unique(ys))
    ys = unique_sum - ys
    centers = C_sub
    if k > 1:
        id1 = np.where(y_sub == 1)[0]
        idx1 = idx0[id1]
        ys1, c1 = hkm(X, idx1, k-1, 2*count-1)
        id2 = np.where(y_sub == 2)[0]
        idx2 = idx0[id2]
        ys2, c2 = hkm(X, idx2, k-1, 2*count)
        ys[id1] = ys1
        ys[id2] = ys2
        centers = np.hstack((c1, c2))
    return ys, centers