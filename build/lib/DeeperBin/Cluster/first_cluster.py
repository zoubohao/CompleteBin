
import multiprocessing
import os
from collections import OrderedDict

import numpy as np

from DeeperBin.Cluster.first_cluster_utils import get_KNN_nodes, run_leiden
from DeeperBin.logger import get_logger
from DeeperBin.Seqs.seq_utils import getGeneWithLongestLength

logger = get_logger()


def first_cluster(
    all_contigname2seq,
    contig_name_list: np.ndarray,
    simclr_embMat,
    length_weight: np.ndarray,
    output_path: str,
    min_contig_len: int,
    num_workers: int,
    bac_gene2contigNames,
    arc_gene2contigNames,
    intersect_accs: set,
):
    """_summary_

    Args:
        embMat (_type_): _description_
        length_weight (_type_): _description_
        output_path (_type_): _description_
    """
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
    
    gene_name_b, summed_b, _ = getGeneWithLongestLength(bac_gene2contigNames, all_contigname2seq, intersect_accs)
    ecoMarker2contigNames = bac_gene2contigNames
    summed_val = summed_b
    geneName = gene_name_b
    
    logger.info(f"--> Fixing the contigs with {geneName} gene. The summed length of these contigs is {summed_val}.")
    is_membership_fixed = [False for _ in range(len(contig_name_list))]
    for contig_name in ecoMarker2contigNames[geneName]:
        if contig_name in contig2id:
            is_membership_fixed[contig2id[contig_name]] = True
    
    if len(contig_name_list) >= 1000000:
        n_iter = 25
    elif 950000 <= len(contig_name_list) < 1000000:
        n_iter = 30
    elif 900000 <= len(contig_name_list) < 950000:
        n_iter = 35
    elif 800000 <= len(contig_name_list) < 900000:
        n_iter = 40
    elif 700000 <= len(contig_name_list) < 800000:
        n_iter = 45
    elif 600000 < len(contig_name_list) < 700000:
        n_iter = 50
    else:
        n_iter = -1
    
    logger.info(f"--> The number of iterations for leiden is {n_iter}.")
    logger.info(f"--> Num_workers: {num_workers}.")
    # gride search
    for e, embMat in enumerate([simclr_embMat]):
        parameter_list = [1., 2., 4., 6., 8., 10., 12.]
        bandwidth_list = [0.05, 0.1, 0.15, 0.2]
        partgraph_ratio_list = [50, 75, 100]
        max_edges_list = [80, 100]
        max_edges_ann_list = []
        space = "l2"
        for max_edges in max_edges_list:
            logger.info(f"--> Start to calculate KNN graph with max_edges: {max_edges} and space: {space}.")
            ann_neighbor_indices, ann_distances = get_KNN_nodes(embMat, max_edges, space=space, num_workers=num_workers)
            max_edges_ann_list.append((ann_neighbor_indices, ann_distances))
        pro_list = []
        total_n = 168
        cur_i = 0
        with multiprocessing.Pool(num_workers) as multiprocess:
            for m, item in enumerate(max_edges_ann_list):
                max_edges = max_edges_list[m]
                for partgraph_ratio in partgraph_ratio_list:
                    for resolution in parameter_list:
                        for bandwidth in bandwidth_list:
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
                                                            n_iter))
                                pro_list.append(p)
                            cur_i += 1
            multiprocess.close()
            for p in pro_list:
                p.get()
    logger.info('--> First Clustering Done.')
