

import multiprocessing as mp
import os
from multiprocessing.pool import Pool
from random import shuffle
from typing import Dict

from CompleteBin.Cluster.sec_cluster_utils import re_cluster_procedure_for_one_method
from CompleteBin.IO import readClusterResult, readMarkerSets, readPickle, writePickle
from CompleteBin.logger import get_logger

logger = get_logger()


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
    best_c_file_prefix, best_quality_record, best_num_905, best_all_summed = best_method
    logger.info(f"--> Current parameters are {best_c_file_prefix}, Num_905: {best_num_905}, Num_all_summed: {best_all_summed}")
    cur_input_folder = os.path.join(temp_bin_folder_path, best_c_file_prefix)
    for k, v in best_quality_record.items():
        cur_bin_path = os.path.join(cur_input_folder, k)
        if v[3] >= mag_length_threshold:
            new_name = os.path.join(cur_input_folder, f"CompleteBin_selected_{index}.fasta" )
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
    mag_length_threshold=100000
):
    logger.info(f"--> Start to re-cluster with {num_workers} num_workers.")
    cluster_files = os.listdir(cluster_folder)
    shuffle(cluster_files)
    tname2markerset = readMarkerSets(ms_path)
    p_list = []
    filenameclu_res = []
    mp.set_start_method("spawn", force=True) 
    with Pool(num_workers) as pool_h:
        for cur_i, c_file in enumerate(cluster_files):
            if "embMat0" in c_file:
                contigname2emb_norm_array = all_simclr_contigname2emb_norm_array
            else:
                raise ValueError("No such embedding.")
            clu2contignames = readClusterResult(os.path.join(cluster_folder, c_file),
                                                contigname2seq,
                                                mag_length_threshold // 4)
            prefix, _ = os.path.splitext(c_file)
            filenameclu_res.append(prefix)
            cur_output_folder = os.path.join(temp_bin_folder_path, prefix)
            if os.path.exists(cur_output_folder) is False:
                os.mkdir(cur_output_folder)
            if os.path.exists(os.path.join(temp_bin_folder_path, f"{prefix}_quality_record.pkl")) is False:
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
                    )
                )
                p_list.append(p)
            # break
        pool_h.close()
        for p in p_list:
            p.get()
    ###
    logger.info("--> Start to select results.")
    edges_grah_resolution2quality_list = {}
    for c_file_prefix in filenameclu_res:
        qr_path = os.path.join(temp_bin_folder_path, f"{c_file_prefix}_quality_record.pkl")
        quality_record = readPickle(qr_path)
        leiden_split_info = "_".join(c_file_prefix.split("_")[0: 8])
        if leiden_split_info not in edges_grah_resolution2quality_list:
            edges_grah_resolution2quality_list[leiden_split_info] = [(c_file_prefix, quality_record, *stat_quality(quality_record))]
        else:
            edges_grah_resolution2quality_list[leiden_split_info].append((c_file_prefix, quality_record, *stat_quality(quality_record)))
    ensemble_list = []
    for _, quality_list in edges_grah_resolution2quality_list.items():
        assert len(quality_list) > 2, ValueError(f"quality list {quality_list} contains error.")
        cur_best_905 = list(sorted(quality_list, key=lambda x: x[2], reverse=True))[0]
        cur_best_all = list(sorted(quality_list, key=lambda x: x[-1], reverse=True))[0]
        if cur_best_905[0] == cur_best_all[0]:
            ensemble_list.append(cur_best_905)
        else:
            ensemble_list.append(cur_best_905)
            ensemble_list.append(cur_best_all)
    
    writePickle(os.path.join(clustering_all_folder, f"ensemble_methods_list_{gmm_flspp}.pkl"), ensemble_list)
    ## change name
    index = 0
    qv_wh = open(os.path.join(clustering_all_folder, f"quality_record_{gmm_flspp}.tsv"), "w")
    for best_method in ensemble_list:
        index = change_name(qv_wh, best_method, temp_bin_folder_path, mag_length_threshold, index)
    qv_wh.close()
    mp.set_start_method("fork", force=True)
    return ensemble_list


