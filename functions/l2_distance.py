import numpy as np

def l2_distance_1(X: np.ndarray, C: np.ndarray, aa: np.ndarray) -> np.ndarray:
    sum_C = np.sum(C**2, axis=0)
    dp = X.dot(C)
    return aa[:, None] + sum_C - 2 * dp