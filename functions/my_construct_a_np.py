import numpy as np
from .euclid import euclid_dist2

def my_construct_a_np(X: np.ndarray,
                      anchors: np.ndarray,
                      k: int,
                      new_idx: np.ndarray,
                      flag: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    p = anchors.shape[0]
    m = new_idx.shape[1]
    Dis = np.zeros((n, m))
    for i in range(p):
        mask = (flag == i+1)
        Dis[mask] = euclid_dist2(X[mask], anchors[new_idx[i]])
    idx_sorted = np.argsort(Dis, axis=1)
    idx1 = idx_sorted[:, :k+1]
    A = np.zeros((n, p))
    for i in range(n):
        ids = idx1[i]
        di = Dis[i, ids]
        base = new_idx[flag[i]-1]
        weights = (di[k] - di[:k]) / (k*di[k] - np.sum(di[:k]) + np.finfo(float).eps)
        A[i, base[ids[:k]]] = weights
    return A