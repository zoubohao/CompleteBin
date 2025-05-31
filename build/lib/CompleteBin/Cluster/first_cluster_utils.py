import os
import time
from typing import List, Optional, Union

import hnswlib
import leidenalg
import numpy as np
from igraph import Graph
from sklearn.neighbors import NearestNeighbors

from CompleteBin.logger import get_logger

logger = get_logger()


def fit_hnsw_index(
    logger,
    features,
    num_threads,
    ef: int,
    M: int = 16,
    space: str = 'l2',
    save_index_file: bool = False
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
    logger.info('--> HNSW index time cost:\t' + str(time_end - time_start) + "s")
    return p


def get_KNN_nodes_hnsw(
    embMat,
    max_edges,
    space,
    num_workers
):
    logger.info(f"--> Approximate KNN Nodes Method.")
    p = fit_hnsw_index(logger, embMat, num_workers, ef=max_edges * 10, space=space)
    time_start = time.time()
    ann_neighbor_indices, ann_distances = p.knn_query(embMat, max_edges + 1, num_threads=num_workers)
    # ann_distances is cosine distance's square
    time_end = time.time()
    logger.info('--> knn query time cost:\t' + str(time_end - time_start) + "s")
    return ann_neighbor_indices, ann_distances


def get_KNN_nodes_scikit(
    emb_mat,
    max_edges,
    num_workers = -1,
):
    max_edges += 1
    logger.info(f"Exact KNN Nodes Method.")
    knn_obj = NearestNeighbors(n_neighbors=max_edges, n_jobs=num_workers)
    time_start = time.time()
    knn_obj = knn_obj.fit(emb_mat)
    time_med = time.time()
    logger.info('--> knn index time cost:\t' + str(time_med - time_start) + "s")
    ann_distances, ann_neighbor_indices = knn_obj.kneighbors(emb_mat, max_edges, return_distance=True)
    time_end = time.time()
    logger.info('--> knn query time cost:\t' + str(time_end - time_med) + "s")
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
    n_iterations=-1):
    vcount = len(norm_embeddings)
    sources = np.repeat(np.arange(vcount), max_edges)
    targets_indices = ann_neighbor_indices[:, 1:]
    targets = targets_indices.flatten()
    wei = ann_distances[:, 1:]
    wei = wei.flatten()
    # filter some edges
    dist_cutoff = np.percentile(wei, partgraph_ratio)
    save_index = wei <= dist_cutoff
    sources = sources[save_index]
    targets = targets[save_index]
    wei = wei[save_index]
    if lmode == 'l1':
        wei = np.sqrt(wei)
        wei = np.exp(- wei / bandwidth)
    elif lmode == 'l2':
        wei = np.exp(- wei / bandwidth)
    else:
        raise ValueError("lmodel is invalid.")
    # upper diagnal matrix
    index = sources > targets
    sources = sources[index]
    targets = targets[index]
    wei = wei[index]
    edgelist = list(zip(sources, targets))
    method = os.path.split(output_file)[-1]
    logger.info(f"-->  Start Leiden algorithm with: {vcount} nodes and {len(wei)} edges." +
                f" max edges for each node: {max_edges}, part graph ratio: {partgraph_ratio}." +
                f" output method is {method}. {cur_i} / {totol_n}")
    graph = Graph(vcount, edgelist, directed=False)
    assert len(wei) == len(edgelist), ValueError(f"wei len is {len(wei)}, edgelist len is {len(edgelist)}.")
    partition = leidenalg.RBERVertexPartition(
        graph,
        initial_membership=initial_list,
        weights=wei,
        node_sizes=length_weight,
        resolution_parameter=resolution)
    optimiser = leidenalg.Optimiser()
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
    logger.info(f"--> End Clustering with output path: {output_file}. {cur_i} / {totol_n}")
    f = open(output_file, 'w')
    for contigIdx in range(len(contig_labels_dict)):
        f.write(contig_name_list[contigIdx] + "\t" + str(contig_labels_dict[contig_name_list[contigIdx]]) + "\n")
    f.close()
