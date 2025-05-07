import numpy as np
from .l2_distance import l2_distance_1
from .transform_l import transform_l
import numpy as np
np.random.seed(0)

def balanced_km(X: np.ndarray, ratio: float = 0.5, max_iter: int = 100):
    """
    Balanced k-means: splits data into two clusters of size roughly ratio*n and (1-ratio)*n.
    Args:
        X: np.ndarray of shape (n_samples, n_features)
        ratio: float in [0, 0.5], fraction for first cluster size
        max_iter: maximum iterations
    Returns:
        C: np.ndarray shape (n_features, 2) cluster centers
        F: np.ndarray shape (n_samples, 2) indicator matrix
        y: np.ndarray shape (n_samples,) labels in {1,2}
    """
    n, d = X.shape
    class_num = 2
    # enforce ratio bounds as in MATLAB
    if ratio > 0.5:
        raise ValueError("ratio should not be larger than 0.5")
    if ratio < 0:
        ratio = 0.0
    a = int(np.floor(n * ratio))
    b = int(np.floor(n * (1 - ratio)))

    # initialize indicator F ensuring no empty cluster
    rng = np.random.RandomState(0)
    while True:
        start_ind = rng.randint(1, class_num+1, size=n)
        F = transform_l(start_ind, class_num)
        if not (F.sum(axis=0) == 0).any():
            break

    aa = np.sum(X * X, axis=1)
    last = F[:, 0].copy()

    for _ in range(max_iter):
        # compute cluster centers C
        C = X.T.dot(F).dot(
            np.linalg.inv(F.T.dot(F) + np.eye(class_num) * np.finfo(float).eps)
        )
        # distances Q
        Q = l2_distance_1(X, C, aa)
        q = Q[:, 0] - Q[:, 1]
        idx_sort = np.argsort(q)

        # determine cp with bounds
        nn = np.sum(q < 0)
        if a <= nn <= b:
            cp = nn
        elif nn < a:
            cp = a
        else:
            cp = b

        # update F
        F_new = np.zeros_like(F)
        F_new[idx_sort[:cp], 0] = 1
        F_new[:, 1] = 1 - F_new[:, 0]
        if np.array_equal(F_new[:, 0], last):
            break
        last = F_new[:, 0].copy()
        F = F_new

    # labels in {1,2}
    y = np.argmax(F, axis=1) + 1
    return C, F, y