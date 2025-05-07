import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


def consensus_function(base_cls: np.ndarray,
                       k: int,
                       max_tcut_km_iters: int = 100,
                       cnt_tcut_km_reps: int = 3) -> np.ndarray:
    """
    原MATLAB版 Transfer Cut 共识函数复现（接受0-based输入）
    Args:
        base_cls: np.ndarray, shape (N, M)，各基聚类标签（0-based整型）
        k: 最终聚类簇数
        max_tcut_km_iters: k-means 最大迭代次数
        cnt_tcut_km_reps: k-means 重启次数
    Returns:
        labels: np.ndarray, shape (N,), 最终聚类标签（0-based）
    """
    # 1) 构建二部图索引偏移
    base = base_cls.astype(int)
    N, M = base.shape
    counts = np.max(base, axis=0) + 1
    offsets = np.cumsum(counts)
    for j in range(1, M):
        base[:, j] += offsets[j-1]
    cntCls = offsets[-1]

    # 2) 构建稀疏二部图 B (N x cntCls)
    rows = np.repeat(np.arange(N), M)
    cols = base.flatten()
    data = np.ones(N * M, dtype=np.float32)
    B = csr_matrix((data, (rows, cols)), shape=(N, cntCls))
    # 删除空列
    col_sum = np.array(B.sum(axis=0)).flatten()
    B = B[:, col_sum > 0]

    # 3) Transfer Cut 分割
    labels = _tcut_for_bipartite_graph(B, k, max_tcut_km_iters, cnt_tcut_km_reps)
    return labels


def _tcut_for_bipartite_graph(B: csr_matrix,
                               k: int,
                               max_km_iters: int,
                               cnt_reps: int) -> np.ndarray:
    """
    MATLAB Tcut_for_bipartite_graph 逻辑复现
    Args:
        B: csr_matrix, shape (Nx, Ny)
        k: 聚类簇数
        max_km_iters: k-means 最大迭代次数
        cnt_reps: k-means 重启次数
    Returns:
        labels: np.ndarray, shape (Nx,), 聚类标签
    """
    # 样本度矩阵
    dx = np.array(B.sum(axis=1)).flatten()
    dx[dx == 0] = 1e-10
    invDx = 1.0 / dx

    # Wy = B^T * Dx^{-1} * B
    Wy = B.T.dot(B.multiply(invDx[:, None]))

    # 归一化 Wy -> nWy = D_y^{-1/2} * Wy * D_y^{-1/2}
    dy = np.array(Wy.sum(axis=1)).flatten()
    inv_sqrt_dy = 1.0 / np.sqrt(dy + 1e-10)
    D_inv_sqrt = csr_matrix((inv_sqrt_dy, (np.arange(len(inv_sqrt_dy)), np.arange(len(inv_sqrt_dy)))), shape=Wy.shape)
    nWy = D_inv_sqrt.dot(Wy.dot(D_inv_sqrt))
    # 对称化
    nWy = (nWy + nWy.T) * 0.5

    # 特征分解，取前 k 个最大特征向量
    evals, evecs = eigsh(nWy, k=k, which='LA', return_eigenvectors=True)
    Uc = D_inv_sqrt.dot(evecs)

    # 样本嵌入 emb = Dx^{-1} * B * Uc
    emb = B.dot(Uc)
    emb = invDx[:, None] * emb
    # 行归一化
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)

    # k-means 聚类，随机初始化模拟 MATLAB 'Start','sample'
    labels = KMeans(
        n_clusters=k,
        init='random',
        n_init=cnt_reps,
        max_iter=max_km_iters,
        random_state=0
    ).fit_predict(emb)
    return labels