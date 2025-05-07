import numpy as np
from .hkm import hkm
from .euclid import euclid_dist2
from .my_construct_a_np import my_construct_a_np
np.random.seed(0)

def my_ulge(X: np.ndarray, num_anchor: int, num_nearest_anchor: int):
    n, _ = X.shape
    idx0 = np.arange(n)
    flag, centers = hkm(X.T, idx0, num_anchor, 1)
    M = centers.T
    Dis = euclid_dist2(M, M)
    new_idx = np.argsort(Dis, axis=1)[:, :10 * num_nearest_anchor]
    Z = my_construct_a_np(X, M, num_nearest_anchor, new_idx, flag)
    return Z, M