import os
import time
from typing import List, Optional, Tuple, Union

import hnswlib
import leidenalg
import numpy as np
from igraph import Graph
from sklearn.neighbors import NearestNeighbors

from CompleteBin.logger import get_logger

logger = get_logger()

FAISS_MAX_NODES = 350_000
HNSW_M = 200
MIN_DEGREE = 1


def faiss_available() -> bool:
    """faiss 是否可用。不可用时调用方应回退到 HNSW。"""
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def canonicalize_undirected_edges(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    return_mutual: bool = False,
):
    """Return sorted unique undirected edges, retaining the maximum weight.

    KNN 是**不对称**的（i 的邻居表里有 j，不代表 j 的邻居表里有 i），
    而 igraph 的图是无向的，所以必须先把有向邻居条目归并成无序对：
      · 取 ``min``/``max`` 作为端点、去掉自环；
      · 同一对端点只保留权重最大的一条；
      · 按 ``(lo, hi)`` 排序返回，使结果与输入顺序无关。

    ``return_mutual=True`` 时额外返回一个布尔数组，标记每个无序对是否为**互近邻**
    （两个方向都存在）。调用方用它做「单向边降权」（``alpha``）。

    ⚠️ 判定互近邻时，反向 key 必须写成 ``cols * N + rows``。
    若写成 ``max * N + min``，当 source > target 时算出来的正好是这条边**自己**的 key，
    会把每条边都判成互近邻 —— 这个坑踩过一次，务必保持现在的写法。
    """
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (len(sources) == len(targets) == len(weights)):
        raise ValueError("edge endpoint and weight arrays must have equal length")
    if not (np.isfinite(sources).all() and np.isfinite(targets).all() and np.isfinite(weights).all()):
        raise ValueError("edge endpoints and weights must be finite")
    if np.any(sources < 0) or np.any(targets < 0) or np.any(weights < 0):
        raise ValueError("edge endpoints must be nonnegative and weights must be nonnegative")
    lo = np.minimum(sources, targets)
    hi = np.maximum(sources, targets)

    # 互近邻：对每条有向边 (r, c)，检查反向边 (c, r) 是否也在输入里。
    if return_mutual and len(sources):
        n_vocab = int(max(sources.max(), targets.max())) + 1
        directed = sources * n_vocab + targets
        directed.sort()
        reverse = targets * n_vocab + sources
        pos = np.searchsorted(directed, reverse)
        pos_c = np.clip(pos, 0, len(directed) - 1)
        mutual = directed[pos_c] == reverse
    else:
        mutual = np.zeros(len(sources), dtype=bool)

    keep = lo != hi
    lo, hi, weights, mutual = lo[keep], hi[keep], weights[keep], mutual[keep]
    if len(lo) == 0:
        if return_mutual:
            return lo, hi, weights, mutual
        return lo, hi, weights
    order = np.lexsort((-weights, hi, lo))
    lo, hi, weights, mutual = lo[order], hi[order], weights[order], mutual[order]
    pair = np.stack((lo, hi), axis=1)
    first = np.r_[True, np.any(pair[1:] != pair[:-1], axis=1)]
    # 同一对的两个方向一定给出相同的 mutual 结论，所以取 first 是安全的
    if return_mutual:
        return lo[first], hi[first], weights[first], mutual[first]
    return lo[first], hi[first], weights[first]


def build_undirected_weighted_edges(
    ann_neighbor_indices: np.ndarray,
    ann_distances: np.ndarray,
    vcount: int,
    max_edges: int,
    partgraph_ratio: int,
    bandwidth: float,
    lmode: str,
    alpha: float = 1.0,
    min_degree: int = 0,
):
    """把 KNN 邻居表变成**无向加权边** ``(a, b, w)``。

    两条 Leiden 路径（第一轮 ``leiden_adaptive._build_leiden_graph``、
    第二轮 ``run_leiden``）**共用这一个函数**，保证建图口径不会各自漂移。

    步骤：
      1. 摊平成有向邻居条目 ``(source, target, distance)``；
      2. 按**全局距离百分位** ``partgraph_ratio`` 筛边（``100`` = 不筛）；
      3. 距离→相似度：``l1`` 先开方，然后统一 ``exp(-d / bandwidth)``；
      4. ``canonicalize_undirected_edges`` 归并成无序对（同端点保留相似度最大的一条），
         同时拿到「是否互近邻」；
      5. **单向边降权**：非互近邻的对权重乘 ``alpha``（``alpha=1`` 即不降权）；
      6. ``min_degree > 0`` 时，给每个点补上它最近的 ``min_degree`` 个邻居那些边
         （**保底边不打折**），使每个点的度数 ≥ ``min_degree``，消除孤立点。

    实现细节（两处容易踩坑，改动前先读完）：
      · ``canonicalize`` 的互近邻判定必须用反向 key ``cols*N + rows``；
        写成 ``max*N + min`` 会让每条边把自己判成互近邻。
      · 保底边合并时按**最终权重** ``(-w, pair_key)`` 定序，
        与离线评测（`evaluate_undirected_fixes.py`）的口径一致；
        换定序键会改变 Leiden 结果（边序会影响划分）。
    """
    sources = np.repeat(np.arange(vcount, dtype=np.int64), max_edges)
    targets_idx = np.asarray(ann_neighbor_indices)[:, 1:]
    targets = targets_idx.flatten().astype(np.int64)
    wei = np.asarray(ann_distances)[:, 1:].flatten().astype(np.float64)

    # ── 非法索引自检（否则会以很难懂的方式炸在很深的地方）───────────────
    # faiss 在「要求的邻居数 > 点数」时用 -1 填充，hnswlib 直接抛 RuntimeError。
    # 若这里出现 -1 或越界下标，真正的原因是 max_edges 相对点数太大，
    # 而不是"边端点非负"这种表面症状，所以在这里给出可操作的报错。
    if targets.size:
        tmin, tmax = int(targets.min()), int(targets.max())
        if tmin < 0 or tmax >= vcount:
            raise ValueError(
                f"KNN 邻居表含非法索引（min={tmin}, max={tmax}，点数={vcount}）。"
                f"最常见原因：max_edges({max_edges}) + 1 > 点数({vcount})，"
                f"faiss 会用 -1 填充不足的邻居。请在调用前把 max_edges 夹到 "
                f"min(max_edges, vcount - 1)。"
            )
        if sources.shape[0] != targets.shape[0]:
            raise ValueError(
                f"邻居表的列数({targets_idx.shape[1]})与 max_edges({max_edges})不一致："
                f"摊平后 source {sources.shape[0]} 条 vs target {targets.shape[0]} 条。"
            )

    if len(wei):
        cutoff = np.percentile(wei, partgraph_ratio)
        keep = wei <= cutoff
        sources, targets, wei = sources[keep], targets[keep], wei[keep]
    else:
        sources = targets = wei = np.asarray([], dtype=np.float64)

    if lmode == 'l1':
        wei = np.sqrt(wei)
        wei = np.exp(-wei / bandwidth)
    elif lmode == 'l2':
        wei = np.exp(-wei / bandwidth)
    else:
        raise ValueError(f"lmode '{lmode}' is invalid. Use 'l1' or 'l2'.")

    lo, hi, w, mutual = canonicalize_undirected_edges(
        sources, targets, wei, return_mutual=True)
    # 5) 单向边降权（互近邻对保持原权重）
    if alpha != 1.0:
        w = w * np.where(mutual, 1.0, alpha)

    # 6) 保底最小度
    if min_degree and vcount:
        d = int(min(min_degree, max_edges, ann_neighbor_indices.shape[1] - 1))
        if d > 0:
            rr = np.repeat(np.arange(vcount, dtype=np.int64), d)
            # ⚠️ 邻居表的第 0 列是**节点自身**，真正的最近邻从第 1 列开始，
            #    所以这里必须写 [:, 1:1+d]。写成 [:, :d] 时 d=1 取到的是自己，
            #    会被下面的 rr != cc 全部过滤掉 —— 保底边等于没加（踩过这个坑）。
            cc = np.asarray(ann_neighbor_indices)[:, 1:1 + d].flatten().astype(np.int64)
            dd = np.asarray(ann_distances)[:, 1:1 + d].flatten().astype(np.float64)
            if lmode == 'l1':
                dd = np.exp(-np.sqrt(dd) / bandwidth)
            else:
                dd = np.exp(-dd / bandwidth)
            ok = rr != cc
            a_all = np.concatenate([lo, np.minimum(rr[ok], cc[ok])])
            b_all = np.concatenate([hi, np.maximum(rr[ok], cc[ok])])
            w_all = np.concatenate([w, dd[ok]])
            key = a_all * np.int64(vcount) + b_all
            order = np.lexsort((-w_all, key))
            key, a_all, b_all, w_all = (key[order], a_all[order],
                                        b_all[order], w_all[order])
            uniq = np.r_[True, key[1:] != key[:-1]]
            lo, hi, w = a_all[uniq], b_all[uniq], w_all[uniq]

    return lo, hi, w


def get_KNN_nodes_faiss(
    emb_mat,
    max_edges,
    num_workers,
    print_log=True
):
    """精确 KNN（faiss IndexFlatL2）。

    返回与 :func:`get_KNN_nodes_hnsw` 完全相同的结构，方便直接替换：

        ann_neighbor_indices : (N, max_edges+1) uint64，第 0 列是查询点自身，按距离升序
        ann_distances        : (N, max_edges+1) float32，**平方** L2 距离

    两点说明：
      · faiss 的 IndexFlatL2 返回的也是平方 L2，与 hnswlib 的 space='l2' 一致；
      · faiss 返回 int64 索引，这里转成 uint64，与 hnswlib 的输出 dtype 保持一致
        （下游 run_leiden 里有 `sources > targets` 的整数比较）。
    """
    import faiss

    if print_log:
        logger.info("--> Exact KNN Nodes Method (faiss IndexFlatL2).")
    mat = np.ascontiguousarray(emb_mat, dtype=np.float32)
    num_elements, dim = mat.shape
    faiss.omp_set_num_threads(int(num_workers))

    time_start = time.time()
    index = faiss.IndexFlatL2(dim)
    index.add(mat)
    time_med = time.time()
    if print_log:
        logger.info(f"--> faiss index time cost: {time_med - time_start:.2f}s")

    D, I = index.search(mat, max_edges + 1)
    time_end = time.time()
    if print_log:
        logger.info(f"--> faiss knn query time cost: {time_end - time_med:.2f}s")
    return I.astype(np.uint64), D


def fit_hnsw_index(
    logger,
    features,
    num_threads,
    ef: int,
    M: int = 16,
    space: str = 'l2',
    save_index_file: bool = False,
    print_log = True
) -> hnswlib.Index:
    time_start = time.time()
    num_elements = len(features)
    labels_index = np.arange(num_elements)
    EMBEDDING_SIZE = len(features[0])
    # Declaring index
    # possible space options are l2, cosine or ip
    p = hnswlib.Index(space=space, dim=EMBEDDING_SIZE)
    # Initing index - the maximum number of elements should be known
    p.init_index(max_elements=num_elements, ef_construction=ef, M=M)
    # Element insertion
    int_labels = p.add_items(features, labels_index, num_threads=num_threads)
    # Controlling the recall by setting ef
    # ef should always be > k
    p.set_ef(ef)
    # If you want to save the graph to a file
    if save_index_file:
        p.save_index(save_index_file)
    time_end = time.time()
    if print_log:
        logger.info(f"--> HNSW index time cost: {time_end - time_start:.2f}s")
    return p


def get_KNN_nodes_hnsw(
    embMat,
    max_edges,
    space,
    num_workers,
    print_log = True,
    M: int = 200
):
    if print_log:
        logger.info(f"--> Approximate KNN Nodes Method (HNSW, M={M}).")
    p = fit_hnsw_index(logger, embMat, num_workers, ef=max_edges * 10, space=space,
                       M=M, print_log=print_log)
    time_start = time.time()
    ann_neighbor_indices, ann_distances = p.knn_query(embMat, max_edges + 1, num_threads=num_workers)
    # hnswlib 的 space='l2' 返回的是**平方** L2 距离（faiss 的 IndexFlatL2 亦然）
    time_end = time.time()
    if print_log:
        logger.info(f"--> knn query time cost: {time_end - time_start:.2f}s")
    return ann_neighbor_indices, ann_distances


def get_KNN_nodes_scikit(
    emb_mat,
    max_edges,
    num_workers = -1,
):
    max_edges += 1
    logger.info(f"--> Exact KNN Nodes Method.")
    knn_obj = NearestNeighbors(n_neighbors=max_edges, n_jobs=num_workers)
    time_start = time.time()
    knn_obj = knn_obj.fit(emb_mat)
    time_med = time.time()
    logger.info(f"--> knn index time cost: {time_med - time_start:.2f}s")
    ann_distances, ann_neighbor_indices = knn_obj.kneighbors(emb_mat, max_edges, return_distance=True)
    time_end = time.time()
    logger.info(f"--> knn query time cost: {time_end - time_med:.2f}s")
    return ann_neighbor_indices, ann_distances


def run_leiden(
    cur_i,
    totol_n,
    output_file: str,
    contig_name_list: List[str],
    ann_neighbor_indices: np.ndarray,
    ann_distances: np.ndarray,
    length_weight: List[float],
    max_edges: int,
    norm_embeddings: np.ndarray,
    bandwidth: float = 0.1,
    lmode: str = 'l2',
    initial_list: Optional[List[Union[int, None]]] = None,
    partgraph_ratio: int = 50,
    resolution: float = 1.0,
    is_membership_fixed: List[bool] = None,
    n_iterations=-1,
    write_file = True,
    rng_seed: int = 2048,
    coverage_values: Optional[np.ndarray] = None,
    pretrain_embeddings: Optional[np.ndarray] = None,
    edge_weight_mode: str = "none",
    coverage_alpha: float = 0.0375,
    pretrain_alpha: float = 0.05,
    expected_pretrain_checkpoint_sha256=None,
    # ---- 建图规则（2026-10 与第一轮统一；默认值 = 历史行为）----
    alpha: float = 1.0,
    min_degree: int = 0):
    """用 Leiden 对一个 bin 的子集重新分箱（第二轮 SCG 重聚类用）。

    ``rng_seed`` 是 leidenalg ``Optimiser`` 的随机种子，必须显式设置：
    ``seed_everything`` 管不到 leidenalg 自己的 RNG，不设置的话同一份输入
    每次都会得到不同的分区。默认 2048（与训练用的全局 seed 是两个独立参数）。

    ``alpha``     单向边权重折扣，1.0 = 不降权（历史行为）。
    ``min_degree`` 每个点保底保留最近几条边，0 = 不补（历史行为）。

    ⚠️ 第二轮按计划 **保持历史取值**（`k=100, partgraph_ratio=100, bandwidth=0.1`，
    `alpha=1.0`）；见 `PLAN_GRID_REFORM.md` 5.6：
    实测把第一轮的 `k=10` 传进来会让唯一真基因组的回收率从 0.500 掉到 0.388。
    """
    vcount = len(norm_embeddings)
    sources, targets, wei = build_undirected_weighted_edges(
        ann_neighbor_indices, ann_distances, vcount, max_edges,
        partgraph_ratio, bandwidth, lmode,
        alpha=alpha, min_degree=min_degree)

    edgelist = list(zip(sources, targets))
    if write_file:
        method = os.path.split(output_file)[-1]
    else:
        method = output_file
    if write_file:
        logger.info(f"--> Start Leiden algorithm with: {vcount} nodes and {len(wei)} edges."
                    f" max edges for each node: {max_edges}, part graph ratio: {partgraph_ratio}."
                    f" output method is {method}. {cur_i} / {totol_n}")
    graph = Graph(vcount, edgelist, directed=False)
    # graph.community_infomap()
    assert len(wei) == len(edgelist), ValueError(f"wei len is {len(wei)}, edgelist len is {len(edgelist)}.")
    partition = leidenalg.RBERVertexPartition(
        graph,
        initial_membership=initial_list,
        weights=wei,
        node_sizes=length_weight,
        resolution_parameter=resolution)
    optimiser = leidenalg.Optimiser()
    # 必须固定 leidenalg 自己的 RNG：seed_everything 只管 random / numpy / torch。
    # 不加这一行，同一份输入每次运行都会给出不同的分区。
    optimiser.set_rng_seed(int(rng_seed))
    optimiser.optimise_partition(partition, n_iterations, is_membership_fixed)
    
    # cluster res
    part = list(partition)
    contig_labels_dict = {}
    # dict of communities
    rang = []
    for ci in range(len(part)):
        rang.append(ci)
        for id in part[ci]:
            contig_labels_dict[contig_name_list[id]] = 'group_' + str(ci)
    # output
    if write_file:
        logger.info(f"--> End Clustering with output path: {output_file}. {cur_i} / {totol_n}")
        f = open(output_file, 'w')
        for contigIdx in range(len(contig_labels_dict)):
            f.write(contig_name_list[contigIdx] + "\t" + str(contig_labels_dict[contig_name_list[contigIdx]]) + "\n")
        f.close()
    
    clu2contigs = {}
    for contigIdx in range(len(contig_labels_dict)):
        cur_contig_name = contig_name_list[contigIdx]
        cur_contig_group = str(contig_labels_dict[contig_name_list[contigIdx]])
        if cur_contig_group not in clu2contigs:
            clu2contigs[cur_contig_group] = set([cur_contig_name])
        else:
            clu2contigs[cur_contig_group].add(cur_contig_name)
    return clu2contigs
