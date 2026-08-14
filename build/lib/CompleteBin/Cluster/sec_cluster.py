

import multiprocessing as mp
import os
from random import shuffle
from typing import Dict

from CompleteBin.Cluster.sec_cluster_utils import re_cluster_procedure_for_one_method
from CompleteBin.IO import readClusterResult, readMarkerSets, readPickle, writePickle
from CompleteBin.logger import get_logger

logger = get_logger()

PENALTY_WEIGHT_NC = 5.0    # num_905 penalty coefficient (num_905 typically 5-50)
PENALTY_WEIGHT_AS = 10.0   # all_summed penalty coefficient (all_summed typically 50-200)


def compute_penalty_rate(temp_bin_folder_path, c_file_prefix):
    """Read penalty info from pickle and return normalized penalty rate."""
    pkl_path = os.path.join(temp_bin_folder_path, f"{c_file_prefix}_penalty.pkl")
    try:
        info = readPickle(pkl_path)
        tp, nc = info["total_penalty"], info["n_clusters"]
        return tp / nc if nc > 0 else 0.0
    except (FileNotFoundError, KeyError):
        raise ValueError("No pkl_path or it is broken.")


def compute_adjusted_scores(num_905, all_summed, penalty_rate):
    """Apply penalty to both num_905 and all_summed."""
    adj_nc = num_905 - PENALTY_WEIGHT_NC * penalty_rate
    adj_as = all_summed - PENALTY_WEIGHT_AS * penalty_rate
    return adj_nc, adj_as


def stat_quality(
    quality_record: dict,
):
    num_5010 = 0
    num_7010 = 0
    num_9010 = 0
    num_505 = 0
    num_705 = 0
    num_905 = 0
    for k, v in quality_record.items():
        comp = v[0]
        cont = v[1]
        if comp > 50 and cont < 10:
            num_5010 += 1
        if comp > 70 and cont < 10:
            num_7010 += 1
        if comp > 90 and cont < 10:
            num_9010 += 1
        if comp > 50 and cont < 5:
            num_505 += 1
        if comp > 70 and cont < 5:
            num_705 += 1
        if comp >= 90 and cont <= 5:
            num_905 += 1
    all_summed = num_505 + num_705 + num_905 + num_5010 + num_7010 + num_9010
    return num_905, all_summed


def summed_len(name2seq: dict):
    summed_v = 0
    for _, seq in name2seq.items():
        summed_v += len(seq)
    return summed_v


def change_name(
    qv_wh,
    best_method,
    temp_bin_folder_path,
    mag_length_threshold,
    index
):
    best_c_file_prefix = best_method[0]
    best_quality_record = best_method[1]
    best_num_905 = best_method[2]
    best_all_summed = best_method[3]
    adj_nc = best_method[4] if len(best_method) > 4 else None
    adj_as = best_method[5] if len(best_method) > 5 else None
    penalty_rate = best_method[6] if len(best_method) > 6 else None
    logger.info(
        f"--> Current parameters are {best_c_file_prefix}, "
        f"Num_905: {best_num_905}, Num_all_sum: {best_all_summed}"
        + (f", Adj_NC: {adj_nc:.2f}, Adj_AS: {adj_as:.2f}, Penalty_rate: {penalty_rate:.4f}" if adj_nc is not None else "")
    )
    cur_input_folder = os.path.join(temp_bin_folder_path, best_c_file_prefix)
    for k, v in best_quality_record.items():
        cur_bin_path = os.path.join(cur_input_folder, k)
        if v[3] >= mag_length_threshold:
            new_name = os.path.join(cur_input_folder, f"CompleteBin_selected_{index}.fasta")
            qv_wh.write(f"CompleteBin_selected_{index}.fasta" + "\t" + str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\n")
            index += 1
        else:
            new_name = os.path.join(cur_input_folder, f"del_{k}.NotInclude")
        os.rename(cur_bin_path, new_name)
    return index


def second_cluster(
    clustering_all_folder,
    temp_bin_folder_path: str,
    cluster_folder: str,
    contigname2seq: Dict,
    all_simclr_contigname2emb_norm_array,
    ms_path: str,
    num_workers: int,
    bac_contigName2_gene2num: dict,
    arc_contigName2_gene2num: dict,
    gmm_flspp: str,
    min_contig_len: int,
    mag_length_threshold,
    seed_path=None,
    coverage_profile_path=None,
):
    logger.info(f"--> Start to re-cluster with {num_workers} num_workers.")
    cluster_files = os.listdir(cluster_folder)
    shuffle(cluster_files)
    tname2markerset = readMarkerSets(ms_path)
    p_list = []
    filenameclu_res = []
    # Use "spawn" context to avoid fork-based COW memory explosion.
    # The parent process holds contigname2seq (dict with all sequences,
    # several GB) and all_simclr_contigname2emb_norm_array (~128 MB).
    # Forking num_workers (30) children would inherit the full address
    # space, triggering COW page duplication and potential OOM.
    # (The previous fix attempt was commented out; now enabled.)
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers) as pool_h:
        for cur_i, c_file in enumerate(cluster_files):
            if "embMat0" in c_file:
                contigname2emb_norm_array = all_simclr_contigname2emb_norm_array
            else:
                raise ValueError("No such embedding.")
            try:
                clu2contignames = readClusterResult(os.path.join(cluster_folder, c_file),
                                                    contigname2seq,
                                                    5000)
            except:
                raise ValueError(f"file {os.path.join(cluster_folder, c_file)} not match contigname2seq with {len(contigname2seq)} contigs.")
            prefix, _ = os.path.splitext(c_file)
            filenameclu_res.append(prefix)
            cur_output_folder = os.path.join(temp_bin_folder_path, prefix)
            if os.path.exists(cur_output_folder) is False:
                os.mkdir(cur_output_folder)
            if os.path.exists(os.path.join(temp_bin_folder_path, f"{prefix}_quality_record.pkl")) is False \
                    or os.path.exists(os.path.join(temp_bin_folder_path, f"{prefix}_penalty.pkl")) is False:
                p = pool_h.apply_async(
                    re_cluster_procedure_for_one_method,
                    args=(
                        cur_i,
                        len(cluster_files),
                        temp_bin_folder_path,
                        clu2contignames,
                        contigname2seq,
                        contigname2emb_norm_array,
                        tname2markerset,
                        cur_output_folder,
                        prefix,
                        bac_contigName2_gene2num,
                        arc_contigName2_gene2num,
                        gmm_flspp,
                        min_contig_len,
                        mag_length_threshold,
                        seed_path,
                        coverage_profile_path,
                    )
                )
                p_list.append(p)
            # break
        pool_h.close()
        for p in p_list:
            p.get()
    ###
    ensemble_list = []
    logger.info("--> Start to select results.")
    edges_grah_resolution2quality_list = {}
    for c_file_prefix in filenameclu_res:
        qr_path = os.path.join(temp_bin_folder_path, f"{c_file_prefix}_quality_record.pkl")
        quality_record = readPickle(qr_path)
        num_905, all_summed = stat_quality(quality_record)
        penalty_rate = compute_penalty_rate(temp_bin_folder_path, c_file_prefix)
        adj_nc, adj_as = compute_adjusted_scores(num_905, all_summed, penalty_rate)
        leiden_split_info = "_".join(c_file_prefix.split("_")[0: 8])
        entry = (c_file_prefix, quality_record, num_905, all_summed, adj_nc, adj_as, penalty_rate)
        if leiden_split_info not in edges_grah_resolution2quality_list:
            edges_grah_resolution2quality_list[leiden_split_info] = [entry]
        else:
            edges_grah_resolution2quality_list[leiden_split_info].append(entry)
    ensemble_list = []
    for _, quality_list in edges_grah_resolution2quality_list.items():
        assert len(quality_list) > 2, ValueError(f"quality list {quality_list} contains error.")
        # Sort by adj_nc (index 4), then adj_as (index 5)
        sorted_list = list(sorted(quality_list, key=lambda x: (x[4], x[5]), reverse=True))
        cur_best_nc = sorted_list[0]
        cur_best_as = list(sorted(quality_list, key=lambda x: (x[5], x[4]), reverse=True))[0]
        if cur_best_nc[0] == cur_best_as[0]:
            ensemble_list.append(cur_best_nc)
        else:
            ensemble_list.append(cur_best_nc)
            ensemble_list.append(cur_best_as)

    writePickle(os.path.join(clustering_all_folder, f"ensemble_methods_list_{gmm_flspp}.pkl"), ensemble_list)
    # change name
    index = 0
    qv_wh = open(os.path.join(clustering_all_folder, f"quality_record_{gmm_flspp}.tsv"), "w")
    for best_method in ensemble_list:
        index = change_name(qv_wh, best_method, temp_bin_folder_path, mag_length_threshold, index)
    qv_wh.close()
    return ensemble_list
