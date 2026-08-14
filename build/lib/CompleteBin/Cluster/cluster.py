import os

import numpy as np

from CompleteBin.Cluster.first_cluster import first_cluster
from CompleteBin.Cluster.sec_cluster import second_cluster
from CompleteBin.logger import get_logger

logger = get_logger()


def combine_two_cluster_steps(
    all_contigname2seq,
    all_simclr_contigname2emb_norm_array,
    markerset_path,
    min_contig_length,
    cpu_num,
    clustering_all_folder,
    seed_path,
    bac_contigName2_gene2num,
    arc_contigName2_gene2num,
    gmm_flspp,
    mag_length_threshold,
    clustering_stages:int,
    leiden_iter_mode: str,
    coverage_profile_path=None,
):
    if os.path.exists(clustering_all_folder) is False:
        os.mkdir(clustering_all_folder)
    simclr_emb_list = []
    length_list = []
    sub_contigname_list = []
    logger.info(f"--> The minimum contig length setting is {min_contig_length} bps.")
    for contigname, seq in all_contigname2seq.items():
        length = len(seq)
        if length < min_contig_length:
            continue
        sub_contigname_list.append(contigname)
        simclr_emb_list.append(all_simclr_contigname2emb_norm_array[contigname])
        length_list.append(length)
    cluster_folder = os.path.join(clustering_all_folder, "leiden_cluster_results")
    if os.path.exists(cluster_folder) is False:
        os.mkdir(cluster_folder)
    first_cluster(
        sub_contigname_list,
        np.stack(simclr_emb_list, axis=0),
        length_list,
        cluster_folder,
        min_contig_length,
        cpu_num,
        seed_path,
        clustering_stages,
        leiden_iter_mode
    )
    # ### re-cluster
    temp_bin_output = os.path.join(clustering_all_folder, f"temp_binning_results_{gmm_flspp}")
    if os.path.exists(temp_bin_output) is False:
        os.mkdir(temp_bin_output)
    # ## re-cluster
    ensemble_list = second_cluster(
        clustering_all_folder,
        temp_bin_output,
        cluster_folder,
        all_contigname2seq,
        all_simclr_contigname2emb_norm_array,
        markerset_path,
        cpu_num,
        bac_contigName2_gene2num=bac_contigName2_gene2num,
        arc_contigName2_gene2num=arc_contigName2_gene2num,
        gmm_flspp=gmm_flspp,
        min_contig_len=float(min_contig_length),
        mag_length_threshold=mag_length_threshold,
        seed_path = seed_path,
        coverage_profile_path=coverage_profile_path,
    )
    return ensemble_list
