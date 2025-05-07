from sklearn.metrics import adjusted_rand_score

def rand_index(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)