from sklearn.metrics import normalized_mutual_info_score

def clustering_measure(true_labels, pred_labels):
    """Return (nmi_score, None) for compatibility"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='max')
    return nmi, None