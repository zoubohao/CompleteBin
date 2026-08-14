from collections import OrderedDict
from copy import deepcopy
import math
import os

from typing import Dict, List, Set, Tuple

import numpy as np

from CompleteBin.Cluster.polish_bin import compute_cohens_d, should_trigger_polish
from CompleteBin.Cluster.polish_bin_improved import (
    clean_bin_by_embedding as clean_bin_by_embedding_with_switch,
)
from CompleteBin.CallGenes.hmm_utils import process_subset
from CompleteBin.IO import writeFasta, writePickle
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_info import calculateN50
from CompleteBin.Cluster.first_cluster_utils import get_KNN_nodes_hnsw, run_leiden


logger = get_logger()


def summedLengthCal(name2seq: Dict[str, str]) -> int:
    return sum(len(seq) for seq in name2seq.values())


def allocate(
    splitContigSetList: List[Set[str]],
    splitRecordGenes: List[Dict[str, int]],
    info: Tuple[str, Dict[str, int]],
    replication_times_threshold: int,
) -> None:
    if len(splitContigSetList) == 0:
        curSet = set()
        curSet.add(info[0])
        splitContigSetList.append(curSet)
        curDict = dict()
        curDict.update(info[1])
        splitRecordGenes.append(curDict)
    else:
        insertIndex = None
        for i, record in enumerate(splitRecordGenes):
            if_insert = True
            for gene, num in info[1].items():
                if gene in record:
                    recordNum = record[gene]
                    if (recordNum + num) > replication_times_threshold:
                        if_insert = False
                        break
            if if_insert is True:
                insertIndex = i
                break
        if insertIndex is not None:
            splitContigSetList[insertIndex].add(info[0])
            curRecord = splitRecordGenes[insertIndex]
            for gene, num in info[1].items():
                if gene not in curRecord:
                    curRecord[gene] = num
                else:
                    curRecord[gene] += num
        else:
            curSet = set()
            curSet.add(info[0])
            splitContigSetList.append(curSet)
            curDict = dict()
            curDict.update(info[1])
            splitRecordGenes.append(curDict)


# original split
def cluster_split_no_learning(
    sub_contigName2seq: Dict[str, str],
    contigName2RepNormV,
    gene2contigNames: Dict[str, List[str]],
    contigName2_gene2num: Dict[str, Dict[str, int]],
) -> List[Dict[str, str]]:
    contigSeqPair = [(contigName, len(seq)) for contigName, seq in sub_contigName2seq.items()]
    if len(contigSeqPair) <= 3:
        return [sub_contigName2seq]
    exist_contigs = [contig for contig, _ in sorted(contigSeqPair, key=lambda x: x[1], reverse=True)]
    existGene2contigNames = {}  # subset of gene2contigNames
    existcontig2_gene2num = []
    existContig2RepNormV = {}
    notExistGeneContig = set()
    notExistGeneContig2seq = {}
    # find the exist genes in those input contigs
    for contig in exist_contigs:
        if contigName2RepNormV is not None:
            existContig2RepNormV[contig] = contigName2RepNormV[contig]
        if contig in contigName2_gene2num:
            curExistGenes2num = contigName2_gene2num[contig]
            existcontig2_gene2num.append((contig, deepcopy(curExistGenes2num)))
            for gene, _ in curExistGenes2num.items():
                if gene not in existGene2contigNames:
                    cur_set = set()
                else:
                    cur_set = existGene2contigNames[gene]
                for cur_contigName in gene2contigNames[gene]:
                    assert cur_contigName in sub_contigName2seq, ValueError(f"cur_contigName {cur_contigName} not in subset of contigs")
                    if cur_contigName in sub_contigName2seq:
                        cur_set.add(cur_contigName)
                existGene2contigNames[gene] = cur_set
        else:
            notExistGeneContig.add(contig)
            notExistGeneContig2seq[contig] = deepcopy(sub_contigName2seq[contig])
    # go through contigs one by one
    splitContigSetList = []
    splitRecordGenes = []
    for info in existcontig2_gene2num:
        allocate(splitContigSetList, splitRecordGenes, info, 1)
    bin_cluster_num = len(splitContigSetList)
    if bin_cluster_num == 0:
        return [notExistGeneContig2seq]

    # cluster part #
    totalN = len(existGene2contigNames)
    filteredContigList = []
    for i in range(len(splitContigSetList)):
        curNumGenes = len(splitRecordGenes[i])
        curSet = splitContigSetList[i].union(notExistGeneContig)
        curContig2seq = {}
        summedLength = 0.0
        for contigName in curSet:
            curContig2seq[contigName] = deepcopy(sub_contigName2seq[contigName])
            summedLength += len(sub_contigName2seq[contigName])
        ratio = curNumGenes / totalN + 0.0
        score = curNumGenes / totalN + 0.0 + math.log(summedLength) / 20.0
        if i == 0 or summedLength >= 25000:
            filteredContigList.append((curContig2seq, ratio, score))
    filteredContigList = sorted(filteredContigList, key=lambda x: x[-1], reverse=True)
    return [infoPair[0] for i, infoPair in enumerate(filteredContigList)]


# original split
def cluster_split(
    sub_contigName2seq: Dict[str, str],
    contigName2RepNormV,
    seed_path,
    tname2markerset,
    bac_contigName2_gene2num,
    arc_contigName2_gene2num,
) -> List[Dict[str, str]]:
    num_contigs = len(sub_contigName2seq)
    # cluster part #
    # build can not link paris and X array

    length_weights = []
    contig_name_list = []
    simclr_embMat = []
    for j, (contig_name, seq) in enumerate(sub_contigName2seq.items()):
        length_weights.append(len(seq))
        contig_name_list.append(contig_name)
        simclr_embMat.append(contigName2RepNormV[contig_name])

    simclr_embMat = np.stack(simclr_embMat, axis=0)
    contig_name_list = np.array(contig_name_list)

    # transform
    initial_list = []
    contig2id = OrderedDict()
    for i, contig_name in enumerate(contig_name_list):
        contig2id[contig_name] = i
        initial_list.append(i)

    # fix seed
    seed_list = []
    with open(seed_path) as rh:
        for line in rh:
            if ">" + line.strip('\n') in contig2id:
                seed_list.append(">" + line.strip('\n'))
    seed_idx = set([contig2id[seed_name] for seed_name in seed_list if seed_name in contig2id])
    is_membership_fixed = [i in seed_idx for i in initial_list]

    parameter_list = [1, 3, 5]  # 1 3 5
    space = "l2"
    max_edges = 100
    if num_contigs <= max_edges:
        max_edges = num_contigs - 1
    ann_neighbor_indices, ann_distances = get_KNN_nodes_hnsw(simclr_embMat, max_edges, space=space, num_workers=32, print_log=False)
    partgraph_ratio = 100
    bandwidth = 0.1

    # start cluster
    all_clu2contigs = []
    for resolution in parameter_list:
        output_file = 'Leiden_embMat0_maxedges_' + str(max_edges) + \
            '_partgraphRatio_' + str(partgraph_ratio) + \
            '_resolution_' + str(resolution) + \
            "_bandwidth_" + str(bandwidth)
        cur_clu2contigs = run_leiden(
            0,
            0,
            output_file,
            contig_name_list,
            ann_neighbor_indices,
            ann_distances,
            length_weights,
            max_edges,
            simclr_embMat,
            bandwidth,
            space,
            initial_list,
            partgraph_ratio,
            resolution,
            is_membership_fixed,
            -1,
            False,
        )
        all_clu2contigs.append(cur_clu2contigs)

    # eval each cluster result
    clu_eval_stat_list = []
    for cur_clu2contigs in all_clu2contigs:
        # cur evaluation statistics
        num_5010 = 0
        num_7010 = 0
        num_9010 = 0
        num_505 = 0
        num_705 = 0
        num_905 = 0
        for _, cluster_contignames in cur_clu2contigs.items():
            _, _, _, comp, cont = determine_domain(
                tname2markerset,
                cluster_contignames,
                bac_contigName2_gene2num,
                arc_contigName2_gene2num,
            )
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
        score_all = num_5010 * 0.6 + num_7010 * 1. + num_9010 * 2. + \
            num_505 * 0.8 + num_705 * 1.5 + num_905 * 3.
        clu_eval_stat_list.append((score_all, cur_clu2contigs))
    best_clu2contigs = list(sorted(clu_eval_stat_list, key=lambda x: x[0], reverse=True))[0][1]

    # ##### result collection
    res = []
    for _, cluster_contignames in best_clu2contigs.items():
        cur_contigname2seq = {}
        for name in cluster_contignames:
            cur_contigname2seq[name] = sub_contigName2seq[name]
        res.append(cur_contigname2seq)
    return res


def genomeCheck(contigName2_gene2num, markerSet: List[Set]):
    """Calculate genome completeness and contamination."""
    gene2count = {}
    for _, gene2num in contigName2_gene2num.items():
        for gene, num in gene2num.items():
            if gene in gene2count:
                gene2count[gene] += num
            else:
                gene2count[gene] = num
    comp = 0.0
    cont = 0.0
    for ms in markerSet:
        present = 0
        multiCopy = 0
        for marker in ms:
            if marker in gene2count:
                count = gene2count[marker]
            else:
                count = 0
            # count = len(hits.get(marker, []))
            if count == 1:
                present += 1
            elif count > 1:
                present += 1
                multiCopy += (count - 1)
        comp += float(present) / len(ms)
        cont += float(multiCopy) / len(ms)
    percComp = 100 * comp / len(markerSet)
    percCont = 100 * cont / len(markerSet)
    return percComp, percCont


def genomeCheckCheckm1(contigName2_gene2num, markerSet):
    """Calculate genome completeness and contamination with CheckM1 formula.

    Differs from ``genomeCheck`` in using a global counting formula instead
    of per-set averaging:

        completeness = sum(present) / total_markers * 100
        contamination = sum(extra_copies) / total_markers * 100

    This matches CheckM1's ``checkm qa`` output format (``-o 2``).
    """
    gene2count = {}
    for _, gene2num in contigName2_gene2num.items():
        for gene, num in gene2num.items():
            gene2count[gene] = gene2count.get(gene, 0) + num

    total_markers = sum(len(ms) for ms in markerSet)
    total_present = 0
    total_extra = 0
    for ms in markerSet:
        for marker in ms:
            count = gene2count.get(marker, 0)
            if count > 0:
                total_present += 1
                if count > 1:
                    total_extra += (count - 1)

    comp = 100.0 * total_present / total_markers if total_markers > 0 else 0.0
    cont = 100.0 * total_extra / total_markers if total_markers > 0 else 0.0
    return comp, cont


def determine_domain(
    tname2markerset,
    sub_contigNames,
    bac_contigName2_gene2num,
    arc_contigName2_gene2num,
    dom=None
):
    if dom is None:
        b_marker_set = tname2markerset["d__Bacteria"]
        a_marker_set = tname2markerset["d__Archaea"]
        # bacteria
        sub_gene2contig_list_b, sub_contigName2_gene2num_b = process_subset(
            sub_contigNames,
            bac_contigName2_gene2num
        )
        bac_comp, bac_cont = genomeCheck(sub_contigName2_gene2num_b, b_marker_set)
        # archaea
        sub_gene2contig_list_a, sub_contigName2_gene2num_a = process_subset(
            sub_contigNames,
            arc_contigName2_gene2num
        )
        ar_comp, ar_cont = genomeCheck(sub_contigName2_gene2num_a, a_marker_set)
        if bac_comp + bac_cont > ar_comp + ar_cont:
            return "bac", sub_gene2contig_list_b, sub_contigName2_gene2num_b, bac_comp, bac_cont
        return "arc", sub_gene2contig_list_a, sub_contigName2_gene2num_a, ar_comp, ar_cont
    elif dom == "bac":
        b_marker_set = tname2markerset["d__Bacteria"]
        # bacteria
        sub_gene2contig_list_b, sub_contigName2_gene2num_b = process_subset(
            sub_contigNames,
            bac_contigName2_gene2num
        )
        bac_comp, bac_cont = genomeCheck(sub_contigName2_gene2num_b, b_marker_set)
        return "bac", sub_gene2contig_list_b, sub_contigName2_gene2num_b, bac_comp, bac_cont
    else:
        a_marker_set = tname2markerset["d__Archaea"]
        # archaea
        sub_gene2contig_list_a, sub_contigName2_gene2num_a = process_subset(
            sub_contigNames,
            arc_contigName2_gene2num
        )
        ar_comp, ar_cont = genomeCheck(sub_contigName2_gene2num_a, a_marker_set)
        return "arc", sub_gene2contig_list_a, sub_contigName2_gene2num_a, ar_comp, ar_cont


def eval_qualities_for_cluster_split(
    sub_split_contigname2seq_list,
    tname2markerset,
    bac_contigName2_gene2num,
    arc_contigName2_gene2num,
):
    res = []
    for sub_split_contigname2seq in sub_split_contigname2seq_list:
        sub_split_contignames = []
        for contigName in sub_split_contigname2seq.keys():
            sub_split_contignames.append(contigName)
        assert len(sub_split_contignames) != 0
        _, sub_split_gene2contig_list, sub_split_contigName2_gene2num, comp, cont = determine_domain(
            tname2markerset,
            sub_split_contignames,
            bac_contigName2_gene2num,
            arc_contigName2_gene2num)
        res.append((sub_split_contigname2seq, sub_split_gene2contig_list, sub_split_contigName2_gene2num, comp, cont))
    return res


def re_cluster_procedure_for_one_method(
    cur_i,
    tol_n,
    temp_bin_folder_path: str,
    clu2contignames: dict,
    contigname2seq: dict,
    contigname2repNormVector: dict,
    tname2markerset: dict,
    output_folder: str,
    c_file: str,
    bac_contigName2_gene2num: dict,
    arc_contigName2_gene2num: dict,
    gmm_flspp: str,
    min_contig_len: int,
    mag_length_threshold: int,
    seed_path,
    coverage_profile_path=None,
):
    # start
    quality_record = {}
    index = 0
    n_clusters = len(clu2contignames)
    total_penalty = 0.0
    logger.info(f"--> Start current method: {c_file}. {cur_i} / {tol_n}.")
    # n_cluster = len(clu2contignames)
    for i,  (_, first_cluster_contignames) in enumerate(clu2contignames.items()):
        # progressBar(i, n_cluster)
        first_cluster_contigName2seq = {}
        for contigName in first_cluster_contignames:
            first_cluster_contigName2seq[contigName] = contigname2seq[contigName]
        first_dom, first_cluster_gene2contig_list, first_cluster_contigName2_gene2num, first_comp, first_cont = determine_domain(
            tname2markerset,
            first_cluster_contignames,
            bac_contigName2_gene2num,
            arc_contigName2_gene2num,
        )
        if first_cont > 12:
            total_penalty += 1.0
            secd_cluster_contigname2seq_list = cluster_split(
                first_cluster_contigName2seq,
                contigname2repNormVector,
                seed_path,
                tname2markerset,
                bac_contigName2_gene2num,
                arc_contigName2_gene2num,
            )
            secd_cluster_bins_qualites = eval_qualities_for_cluster_split(
                secd_cluster_contigname2seq_list,
                tname2markerset,
                bac_contigName2_gene2num,
                arc_contigName2_gene2num,
            )
            for secd_cluster_contigname2seq, _, _, secd_comp, secd_cont in secd_cluster_bins_qualites:
                size = summedLengthCal(secd_cluster_contigname2seq)
                if size >= mag_length_threshold:
                    writeFasta(secd_cluster_contigname2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
                    n50 = np.log(calculateN50(secd_cluster_contigname2seq))
                    quality_record[f"CompleteBin_cand_{index}.fasta"] = (secd_comp, secd_cont, n50, size)
                    index += 1
        elif first_cont > 5.0:
            # 明确污染：必定触发 + SCG 拆分
            total_penalty += 0.5
            index, quality_record = clean_bin_by_embedding_with_switch(
                first_cluster_contigName2seq,
                contigname2repNormVector,
                first_cluster_gene2contig_list,
                first_cluster_contigName2_gene2num,
                first_dom,
                tname2markerset,
                bac_contigName2_gene2num,
                arc_contigName2_gene2num,
                output_folder, index, quality_record,
                mag_length_threshold,
                coverage_profile_path=coverage_profile_path,
            )
            secd_cluster_contigname2seq_list = cluster_split_no_learning(
                first_cluster_contigName2seq,
                contigname2repNormVector,
                first_cluster_gene2contig_list,
                first_cluster_contigName2_gene2num
            )
            for secd_cluster_contigname2seq in secd_cluster_contigname2seq_list:
                sub_split_contignames = list(secd_cluster_contigname2seq.keys())
                assert len(sub_split_contignames) != 0
                _, _, _, comp, cont = determine_domain(
                    tname2markerset,
                    sub_split_contignames,
                    bac_contigName2_gene2num,
                    arc_contigName2_gene2num,
                    first_dom)
                size = summedLengthCal(secd_cluster_contigname2seq)
                if size >= mag_length_threshold:
                    writeFasta(secd_cluster_contigname2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
                    n50 = np.log(calculateN50(secd_cluster_contigname2seq))
                    quality_record[f"CompleteBin_cand_{index}.fasta"] = (comp, cont, n50, size)
                    index += 1
        else:
            # SCG + 嵌入双信号判断（灰区 & 轻度污染）
            d, _, _, _ = compute_cohens_d(
                list(first_cluster_contigName2seq.keys()),
                contigname2repNormVector,
                first_cluster_contigName2_gene2num,
                first_cluster_gene2contig_list,
            )
            should_trigger, zone = should_trigger_polish(first_cont, d)
            if should_trigger:
                total_penalty += 0.1
                if zone == "gray":
                    tolerance, trigger = 5.0, 2.0
                elif zone == "light":
                    tolerance, trigger = 7.0, 3.45
                else:  # clear
                    tolerance, trigger = 10.0, 5.0
                index, quality_record = clean_bin_by_embedding_with_switch(
                    first_cluster_contigName2seq,
                    contigname2repNormVector,
                    first_cluster_gene2contig_list,
                    first_cluster_contigName2_gene2num,
                    first_dom,
                    tname2markerset,
                    bac_contigName2_gene2num,
                    arc_contigName2_gene2num,
                    output_folder, index, quality_record,
                    mag_length_threshold,
                    contamination_trigger=trigger,
                    completeness_drop_tolerance=tolerance,
                    coverage_profile_path=coverage_profile_path,
                )
            else:
                size = summedLengthCal(first_cluster_contigName2seq)
                if size >= mag_length_threshold:
                    writeFasta(first_cluster_contigName2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
                    n50 = np.log(calculateN50(first_cluster_contigName2seq))
                    quality_record[f"CompleteBin_cand_{index}.fasta"] = (first_comp, first_cont, n50, size)
                    index += 1
        size = summedLengthCal(first_cluster_contigName2seq)
        if size >= mag_length_threshold:
            writeFasta(first_cluster_contigName2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
            n50 = np.log(calculateN50(first_cluster_contigName2seq))
            quality_record[f"CompleteBin_cand_{index}.fasta"] = (first_comp, first_cont, n50, size)
            index += 1
    logger.info(f"--> End of second cluster with current method: {c_file}. {cur_i} / {tol_n}")
    writePickle(os.path.join(temp_bin_folder_path, f"{c_file}_quality_record.pkl"), quality_record)
    writePickle(
        os.path.join(temp_bin_folder_path, f"{c_file}_penalty.pkl"),
        {"total_penalty": total_penalty, "n_clusters": n_clusters}
    )
    return c_file, quality_record
