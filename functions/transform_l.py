
import numpy as np

def transform_l(labels: np.ndarray, class_num: int) -> np.ndarray:
    n = labels.shape[0]
    F = np.zeros((n, class_num))
    for i, lab in enumerate(labels):
        F[i, int(lab)-1] = 1
    return F