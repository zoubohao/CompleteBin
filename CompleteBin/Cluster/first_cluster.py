
import multiprocessing
import os
from collections import OrderedDict

import numpy as np
from scipy.stats import percentileofscore

from CompleteBin.Cluster.first_cluster_utils import (
    FAISS_MAX_NODES,
    HNSW_M,
    MIN_DEGREE,
    faiss_available,
    get_KNN_nodes_faiss,
    get_KNN_nodes_hnsw,
)
from CompleteBin.Cluster.leiden_adaptive import run_leiden_adaptive as run_leiden
from CompleteBin.Utils.mp_utils import make_spawn_pool
from CompleteBin.logger import get_logger


logger = get_logger()


def get_remove_precentile(ann_distances, out_of_nodes=0):
    wei = ann_distances[:, 1:]
    sorted_wei = np.sort(wei, axis=1)
    if out_of_nodes == 0:
        max_edges = len(wei[0])
        max_wei = np.max(sorted_wei[:, 0])
        wei = wei.flatten()
        precentile = percentileofscore(wei, max_wei, kind='rank')
        precentile = float(f"{precentile:.2f}")
        if precentile < 75.:
            precentile = 75.
        return precentile
    assert out_of_nodes >= 1
    for i in range(out_of_nodes - 1):
        ind = np.argmax(sorted_wei[:, 0])
        sorted_wei = np.delete(sorted_wei, ind, axis=0)
    max_wei = np.max(sorted_wei[:, 0])
    wei = wei.flatten()
    precentile = percentileofscore(wei, max_wei, kind='rank')
    precentile = float(f"{precentile:.2f}")
    if precentile < 75.:
        precentile = 75.
    return precentile


def first_cluster(
    contig_name_list: np.ndarray,
    simclr_embMat: np.ndarray,
    length_weight: np.ndarray,
    output_path: str,
    min_contig_len: int,
    num_workers: int,
    seed_path,
    clustering_stages,
    leiden_iter_mode,  # ="accurate" or "fast"
    leiden_seed: int = 42,
):
    logger.info("--> Start clustering.")
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    length_weight_array = np.array(length_weight)
    contig_name_list = np.array(contig_name_list)
    # filter
    simclr_embMat = simclr_embMat[length_weight_array >= min_contig_len]
    contig_name_list = contig_name_list[length_weight_array >= min_contig_len]
    length_weight_array = length_weight_array[length_weight_array >= min_contig_len]

    # transform
    length_weight = list(length_weight_array)
    initial_list = []
    contig2id = OrderedDict()
    contig2seqlength = {}
    for i, contig_name in enumerate(contig_name_list):
        contig2id[contig_name] = i
        initial_list.append(i)
        contig2seqlength[contig_name] = length_weight[i]

    seed_list = []
    with open(seed_path) as rh:
        for line in rh:
            if ">" + line.strip('\n') in contig2id:
                seed_list.append(">" + line.strip('\n'))
    # name_map = dict(zip(contig_id_list, range(len(contig_id_list))))
    seed_idx = set([contig2id[seed_name] for seed_name in seed_list if seed_name in contig2id])
    # initial_list = list(np.arange(len(namelist)))
    is_membership_fixed = [i in seed_idx for i in initial_list]
    logger.info(f"--> Fix {len(seed_idx)} contigs. seed_index is {seed_idx}")

    n_iter = -1
    if len(contig_name_list) >= 1000000:
        n_iter = 24
    elif 950000 <= len(contig_name_list) < 1000000:
        n_iter = 28
    elif 900000 <= len(contig_name_list) < 950000:
        n_iter = 30

    logger.info(f"--> Leiden mode: {leiden_iter_mode} (n_iter={n_iter}, used only in accurate mode).")
    logger.info(f"--> Leiden rng_seed: {leiden_seed} (independent from the global training seed).")
    logger.info(f"--> Num_workers: {num_workers}.")
    # gride search
    for e, embMat in enumerate([simclr_embMat]):
        if clustering_stages == 1:
            parameter_list = [1, 3, 5, 10]
            bandwidth_list = [0.1, 0.15, 0.2]
            partgraph_ratio_list = [80, 50]
            max_edges_list = [10, 50, 100]
            alpha_list = [0.1, 0.5, 1.0]
            min_degree = MIN_DEGREE
        else:
            parameter_list = [1, 3, 5, 10]
            bandwidth_list = [0.1, 0.15, 0.2]
            partgraph_ratio_list = [80, 50]
            max_edges_list = [100]
            alpha_list = [1.0]
            min_degree = MIN_DEGREE
        max_edges_ann_list = []
        max_edges_eff_list = []      # 实际用于建图的 k（极小样本会被夹小）
        space = "l2"
        n_nodes = len(embMat)
        for max_edges in max_edges_list:
            # ── KNN 要求 max_edges + 1 <= n_nodes ───────────────────────────
            # 邻居数不够时：faiss 会用 **-1** 填充索引（转 uint64 后变成巨大值），
            # hnswlib 直接抛 RuntimeError("Cannot return the results in a
            # contiguous 2D array")。真实样本点数远大于 100，但极小样本
            # （第二轮只有几十个 contig 时）会崩。这里按 `cluster_split` 的既有
            # 做法夹住 —— 只可能把「本来就会崩」变成「能跑」，不会改变任何
            # 现在能跑通的输入。
            if max_edges >= n_nodes:
                fixed = max(n_nodes - 1, 1)
                logger.warning(
                    f"--> contig 数 {n_nodes} <= max_edges {max_edges}，KNN 取不到这么多"
                    f"邻居，夹到 {fixed}（否则 faiss 返回 -1 索引 / hnswlib 报错）")
                max_edges = fixed
            max_edges_eff_list.append(max_edges)
            logger.info(f"--> Start to calculate KNN graph with max_edges: {max_edges} and space: {space}.")
            # ── 后端选择 ──────────────────────────────────────────────────────
            # N < FAISS_MAX_NODES  -> faiss 精确检索（多线程下结果确定）
            # N >= FAISS_MAX_NODES -> HNSW，M=HNSW_M
            if n_nodes < FAISS_MAX_NODES and faiss_available():
                logger.info(f"--> KNN backend: faiss (exact), n_nodes={n_nodes} "
                            f"< FAISS_MAX_NODES={FAISS_MAX_NODES}")
                ann_neighbor_indices, ann_distances = get_KNN_nodes_faiss(
                    embMat, max_edges, num_workers)
            else:
                if n_nodes < FAISS_MAX_NODES and not faiss_available():
                    logger.warning("--> faiss 不可用，回退到 HNSW")
                logger.info(f"--> KNN backend: HNSW M={HNSW_M}, n_nodes={n_nodes}")
                ann_neighbor_indices, ann_distances = get_KNN_nodes_hnsw(
                    embMat, max_edges, space=space, num_workers=num_workers, M=HNSW_M)
            max_edges_ann_list.append((ann_neighbor_indices, ann_distances))
        # clustering
        pro_list = []
        total_n = (len(parameter_list) * len(bandwidth_list)
                   * len(partgraph_ratio_list) * len(max_edges_list)
                   * len(alpha_list))
        cur_i = 0
        # Use "spawn" context to avoid fork-based COW memory explosion.
        # The parent process holds large arrays (embMat ~128 MB,
        # KNN graph ~256 MB, contigname2seq dict ~several GB).
        # Forking num_workers (30) children would inherit the full address
        # space, triggering COW page duplication and potential OOM.
        ctx = multiprocessing.get_context("spawn")
        with make_spawn_pool(ctx, num_workers, label="first_cluster") as multiprocess:
            # leiden
            for m, item in enumerate(max_edges_ann_list):
                max_edges = max_edges_eff_list[m]
                for bandwidth in bandwidth_list:
                    for partgraph_ratio in partgraph_ratio_list:
                        for alpha in alpha_list:
                            for resolution in parameter_list:
                                # ⚠️ 文件名必须带 alpha：sec_cluster.py:191 用前 8 段
                                # （maxedges/partgraphRatio/resolution）做分组键，
                                # alpha 与 bandwidth 一起作为「组内互相竞争」的维度。
                                # 也正因如此，bandwidth_list × alpha_list 必须 ≥ 3，
                                # 否则 sec_cluster.py:199 的 `assert len(...) > 2` 会炸。
                                output_file = os.path.join(output_path, 'Leiden_embMat0_maxedges_' + str(max_edges) +
                                                                        '_partgraphRatio_' + str(partgraph_ratio) +
                                                                        '_resolution_' + str(resolution) +
                                                                        "_bandwidth_" + str(bandwidth) +
                                                                        '_alpha_' + str(alpha) + '.tsv')
                                if not os.path.exists(output_file):
                                    p = multiprocess.apply_async(
                                        run_leiden,
                                        args=(cur_i,
                                              total_n,
                                              output_file,
                                              contig_name_list,
                                              item[0],
                                              item[1],
                                              length_weight,
                                              max_edges,
                                              embMat,
                                              bandwidth,
                                              space,
                                              initial_list,
                                              partgraph_ratio,
                                              resolution,
                                              is_membership_fixed,
                                              n_iter,
                                              True,
                                              leiden_iter_mode),
                                        # 用 kwds 传，避免与 mode/early_stop_threshold 等
                                        # 位置参数撞位
                                        kwds={"rng_seed": leiden_seed,
                                              "alpha": alpha,
                                              "min_degree": min_degree})
                                    pro_list.append(p)
                                cur_i += 1
            multiprocess.close()
            for p in pro_list:
                p.get()
    logger.info('--> First Clustering Done.')
