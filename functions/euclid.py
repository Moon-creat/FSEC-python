import numpy as np
def euclid_dist2(A, B):
    AA = np.sum(A**2, axis=1, keepdims=True)
    BB = np.sum(B**2, axis=1, keepdims=True).T
    D = AA + BB - 2 * A.dot(B.T)
    D[D < 0] = 0
    return D
