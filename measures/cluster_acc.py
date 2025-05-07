import numpy as np
from scipy.optimize import linear_sum_assignment

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cost = np.zeros((labels.size, labels.size), dtype=int)
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            cost[i, j] = np.sum((y_true == l1) & (y_pred == l2))
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    return cost[row_ind, col_ind].sum() / y_true.size