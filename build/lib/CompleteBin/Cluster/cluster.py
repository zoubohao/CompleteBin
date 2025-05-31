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
    phy2accs,
    min_contig_length,
    cpu_num,
    clustering_all_folder,
    bac_gene2contigNames,
    arc_gene2contigNames,
    bac_contigName2_gene2num,
    arc_contigName2_gene2num,
    gmm_flspp,
):
    if os.path.exists(clustering_all_folder) is False:
        os.mkdir(clustering_all_folder)
    simclr_emb_list = []
    length_list = []
    sub_contigname_list = []
    logger.info(f"--> The minimum contig length settting is {min_contig_length} bps.")
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
        all_contigname2seq,
        sub_contigname_list,
        np.stack(simclr_emb_list, axis=0),
        length_list,
        cluster_folder,
        min_contig_length,
        cpu_num,
        bac_gene2contigNames,
        arc_gene2contigNames,
        phy2accs["intersect"],
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
        min_contig_len=float(min_contig_length)
    )
    return ensemble_list
