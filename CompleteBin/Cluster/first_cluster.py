
import multiprocessing
import os
from collections import OrderedDict

import numpy as np
from scipy.stats import percentileofscore

from CompleteBin.Cluster.first_cluster_utils import get_KNN_nodes_hnsw
from CompleteBin.Cluster.leiden_adaptive import run_leiden_adaptive as run_leiden
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
    leiden_iter_mode  # ="accurate" or "fast"
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
    logger.info(f"--> Num_workers: {num_workers}.")
    # gride search
    for e, embMat in enumerate([simclr_embMat]):
        if clustering_stages == 1:
            parameter_list = [1, 3, 5, 8, 10, 15, 20]
            bandwidth_list = [0.05, 0.10, 0.15, 0.20, 0.25]
            partgraph_ratio_list = [100, 80, 50]
        else:
            parameter_list = [1, 3, 5, 8, 10, 15]
            bandwidth_list = [0.1, 0.15, 0.2]
            partgraph_ratio_list = [100, 50]
        max_edges_list = [100]
        max_edges_ann_list = []
        space = "l2"
        for max_edges in max_edges_list:
            logger.info(f"--> Start to calculate KNN graph with max_edges: {max_edges} and space: {space}.")
            ann_neighbor_indices, ann_distances = get_KNN_nodes_hnsw(embMat, max_edges, space=space, num_workers=num_workers)
            max_edges_ann_list.append((ann_neighbor_indices, ann_distances))
        # clustering
        pro_list = []
        total_n = len(parameter_list) * len(bandwidth_list) * len(partgraph_ratio_list) * len(max_edges_list)
        cur_i = 0
        # Use "spawn" context to avoid fork-based COW memory explosion.
        # The parent process holds large arrays (embMat ~128 MB,
        # KNN graph ~256 MB, contigname2seq dict ~several GB).
        # Forking num_workers (30) children would inherit the full address
        # space, triggering COW page duplication and potential OOM.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(num_workers) as multiprocess:
            # leiden
            for m, item in enumerate(max_edges_ann_list):
                max_edges = max_edges_list[m]
                for bandwidth in bandwidth_list:
                    for partgraph_ratio in partgraph_ratio_list:
                        for resolution in parameter_list:
                            output_file = os.path.join(output_path, 'Leiden_embMat0_maxedges_' + str(max_edges) +
                                                                    '_partgraphRatio_' + str(partgraph_ratio) +
                                                                    '_resolution_' + str(resolution) +
                                                                    "_bandwidth_" + str(bandwidth) + '.tsv')
                            if not os.path.exists(output_file):
                                p = multiprocess.apply_async(run_leiden,
                                                             (cur_i,
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
                                                              leiden_iter_mode,))
                                pro_list.append(p)
                            cur_i += 1
            multiprocess.close()
            for p in pro_list:
                p.get()
    logger.info('--> First Clustering Done.')
