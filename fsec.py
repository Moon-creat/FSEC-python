import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from collections import Counter
from functions.my_ulge import my_ulge
from consensus_function import consensus_function
from measures.clustering_measure import clustering_measure
from measures.cluster_acc import cluster_acc
from measures.rand_index import rand_index
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Debug utilities
def debug_B(B, n, K):
    nonzeros = np.count_nonzero(B)
    avg_nnz = nonzeros / n
    sums = B.sum(axis=1)
    print(f"[DEBUG] B shape: {B.shape}")
    print(f"[DEBUG] Total nonzeros in B: {nonzeros}, avg per row: {avg_nnz:.2f}")
    print(f"[DEBUG] Row sums min/max/mean: {sums.min():.4f}/{sums.max():.4f}/{sums.mean():.4f}")


def debug_svd(B):
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    print(f"[DEBUG] Top 10 singular values: {S[:10]}")
    return U


def debug_base_clusters(U, Y, base_cls):
    n, m = base_cls.shape
    print(f"[DEBUG] Generated {m} base clusterings, each up to {U.shape[1]} dims")
    for j in range(m):
        labels_j = base_cls[:, j]
        nmi_j, _ = clustering_measure(Y, labels_j)
        acc_j = cluster_acc(Y, labels_j)
        ari_j = rand_index(Y, labels_j)
        print(f"[DEBUG] Base {j:02d}: clusters={len(np.unique(labels_j))}, ACC={acc_j:.4f}, NMI={nmi_j:.4f}, ARI={ari_j:.4f}")


def debug_consensus(base_cls, K):
    """
    Implements the Transfer Cut consensus step for debugging:
    1. Build bipartite graph Bp (N x C)
    2. Compute Dx, Dy degrees
    3. Form W_y = Bp^T D_x^{-1} Bp
    4. Normalize nWy = D_y^{-1/2} W_y D_y^{-1/2}
    5. Extract top-K eigenvalues for inspection
    """
    # 1) Build bipartite graph: samples as rows, clusters as columns
    n, m = base_cls.shape
    clusters = [np.unique(base_cls[:, j]) for j in range(m)]
    C = sum(len(c) for c in clusters)
    rows, cols = [], []
    for j in range(m):
        offset = sum(len(clusters[k]) for k in range(j))
        for i, label in enumerate(base_cls[:, j]):
            rows.append(i)
            cols.append(offset + list(clusters[j]).index(label))
    Bp = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, C))

    # 2) Compute degree matrices
    Dx = np.array(Bp.sum(axis=1)).ravel()      # sample degrees (m)
    Dy = np.array(Bp.sum(axis=0)).ravel()      # cluster degrees

    # 3) Form raw Wy = Bp^T * D_x^{-1} * Bp
    invDx = 1.0 / Dx
    Wy = Bp.T.dot(Bp.multiply(invDx[:, None]))

    # 4) Normalize Wy to nWy = D_y^{-1/2} * Wy * D_y^{-1/2}
    inv_sqrt_Dy = 1.0 / np.sqrt(Dy + 1e-10)
    D_inv_sqrt = csr_matrix((inv_sqrt_Dy, (np.arange(C), np.arange(C))), shape=(C, C))
    nWy = D_inv_sqrt.dot(Wy.dot(D_inv_sqrt))
    print(f"[DEBUG] Step3 - nWy shape: {nWy.shape}, nnz: {nWy.nnz}")

    # 5) Extract top K eigenvalues for debugging
    vals = eigsh(nWy, k=min(K, C-1), which='LA', return_eigenvectors=False)
    print(f"[DEBUG] Step3 - Top {K} eigenvalues of nWy: {np.sort(vals)[::-1]}")
    # No return: this is for debug only
    
    print(f"[DEBUG] Top 5 eigenvalues of nWy: {np.sort(vals)[::-1]}")

def FSEC(X, Y, num_nearest_anchor: int, num_base: int, exponent_p: int = None):
    if torch.is_tensor(X): X = X.detach().cpu().numpy()
    if torch.is_tensor(Y): Y = Y.detach().cpu().numpy().flatten()
    n, d = X.shape
    K = len(np.unique(Y))
    p = exponent_p if exponent_p is not None else int(np.floor(np.log2(np.sqrt(n * K))))

    # Anchor graph
    X_norm = MinMaxScaler().fit_transform(X)
    print(X_norm.max(), X_norm.min())
    B, anchors = my_ulge(X_norm, p, num_nearest_anchor)
    debug_B(B, n, num_nearest_anchor)
    B = B / (np.sqrt(np.sum(B, axis=0)) + 1e-10)

    # SVD embedding
    U = debug_svd(B)

    # Base clusterings (original random sampling)
    rng = np.random.RandomState(0)
    max_k = min(50, int(np.round(np.sqrt(n))))
    ks = rng.randint(K, max_k + 1, size=num_base)
    base_cls = np.zeros((n, num_base), dtype=int)
    for j, k_j in enumerate(ks):
        Uj = U[:, :k_j]
        Uj = Uj / (np.linalg.norm(Uj, axis=1, keepdims=True) + 1e-10)
        base_cls[:, j] = KMeans(n_clusters=k_j, tol=1e-4, max_iter=100, init='random', n_init=1, random_state=rng).fit_predict(Uj)
    debug_base_clusters(U, Y, base_cls)

    # Consensus
    debug_consensus(base_cls, K)
    final_labels = consensus_function(base_cls, K)
    dist = Counter(final_labels)
    print(f"[DEBUG] Final labels shape: {final_labels.shape}")
    print(f"[DEBUG] Final cluster counts: {dict(dist)}")

    # Metrics
    acc = cluster_acc(Y, final_labels)
    nmi, _ = clustering_measure(Y, final_labels)
    ari = rand_index(Y, final_labels)
    print(f"[DEBUG] Final ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")

    return final_labels, anchors, B, acc, nmi, ari

