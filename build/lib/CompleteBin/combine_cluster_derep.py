

import os
from shutil import copy
from typing import Dict


from CompleteBin.logger import get_logger
from CompleteBin.Cluster.cluster import combine_two_cluster_steps
from CompleteBin.Dereplication.galah_utils import process_galah_with_SCGs, reorder_ensemble_for_galah
from CompleteBin.IO import readFasta, readMarkerSets, readPickle, readMetaInfo, writePickle


logger = get_logger()


def summedLengthCal(name2seq: Dict[str, str]) -> int:
    return sum(len(seq) for seq in name2seq.values())


def clustering_and_dereplication(
    temp_file_folder_path: str,
    clustering_folder_name: str,
    contigname2seq: dict,
    simclr_contigname2emb_norm_array: dict,
    markerset_path: str,
    min_contig_length: int,
    cpu_workers: int,
    seed_path: str,
    bac_contigName2_gene2num: dict,
    arc_contigName2_gene2num: dict,
    mag_length_threshold: int,
    bin_output_folder_path: str,
    leiden_iter_mode,  # ="accurate" or "fast"
    clustering_stages=1,
    reuse_contig=True,
    coverage_profile_path=None,
):
    von_flspp_mix = "leiden"
    clustering_all_folder = os.path.join(temp_file_folder_path, clustering_folder_name)
    if os.path.exists(bin_output_folder_path) is False:
        os.mkdir(bin_output_folder_path)
    if os.path.exists(clustering_all_folder) is False:
        os.mkdir(clustering_all_folder)

    if os.path.exists(os.path.join(clustering_all_folder, f"ensemble_methods_list_{von_flspp_mix}.pkl")) is False:
        ensemble_list = combine_two_cluster_steps(
            contigname2seq,
            simclr_contigname2emb_norm_array,
            markerset_path,
            min_contig_length,
            cpu_num=cpu_workers,
            clustering_all_folder=clustering_all_folder,
            seed_path=seed_path,
            bac_contigName2_gene2num=bac_contigName2_gene2num,
            arc_contigName2_gene2num=arc_contigName2_gene2num,
            gmm_flspp=von_flspp_mix,
            mag_length_threshold=mag_length_threshold,
            clustering_stages=clustering_stages,
            leiden_iter_mode=leiden_iter_mode,
            coverage_profile_path=coverage_profile_path,
        )
    # ensemble the grouped results by galah.
    ensemble_list = readPickle(os.path.join(clustering_all_folder, f"ensemble_methods_list_{von_flspp_mix}.pkl"))
    temp_flspp_bin_output = os.path.join(clustering_all_folder, f"temp_binning_results_{von_flspp_mix}")
    ensemble_list = reorder_ensemble_for_galah(ensemble_list, nc_first=True) ###### try to figure out if nc_first is better than no
    scg_quality_report_path = os.path.join(clustering_all_folder, f"quality_record_{von_flspp_mix}.tsv")
    process_galah_with_SCGs(
        clustering_all_folder,
        temp_flspp_bin_output,
        ensemble_list,
        bin_output_folder_path,
        scg_quality_report_path,
        von_flspp_mix,
        mag_length_threshold=mag_length_threshold,
        markerset_path=markerset_path,
        bac_contigName2_gene2num=bac_contigName2_gene2num,
        arc_contigName2_gene2num=arc_contigName2_gene2num,
        reuse_contig=reuse_contig,
        cpus=cpu_workers,
    )


def selected_recluster_contigs(
    all_contigname2seq: dict,
    bin_output_folder_path: str,
    include_state_list: list,
    comp_thre: float = 50.
):
    # ["HighQuality", "MediumQuality", "LowQuality"]
    binned_contigs = set()
    all_contigs = set(all_contigname2seq.keys())
    files = os.listdir(bin_output_folder_path)
    copy_list = []
    bins_metainfo, h, m, l = readMetaInfo(os.path.join(bin_output_folder_path, "MetaInfo.tsv"), 2, 3)
    for file in files:
        prefix, suffix = os.path.splitext(file)
        if suffix == ".fasta":
            comp, conta, state = bins_metainfo[file]
            if state in include_state_list and comp > comp_thre:
                cur_contigname2seq = readFasta(os.path.join(bin_output_folder_path, file))
                copy_list.append((os.path.join(bin_output_folder_path, file), comp, conta, state))
                for key in cur_contigname2seq.keys():
                    binned_contigs.add(key)
    other_contigs = set()
    for name in all_contigs:
        if name not in binned_contigs:
            other_contigs.add(name)
    writePickle(os.path.join(bin_output_folder_path, "bin_copy_list.pkl"), copy_list)
    logger.info(f"--> All contigs: {len(all_contigs)}, {len(other_contigs)} contigs will be used for next clustering.")
    return other_contigs
