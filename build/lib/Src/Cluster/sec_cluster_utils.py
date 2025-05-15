import math
import os
from copy import deepcopy
from typing import Dict, List, Set, Tuple

import numpy as np
from flspp.core import FLSpp
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from Src.CallGenes.hmm_utils import process_subset
from Src.IO import writeFasta, writePickle
from Src.logger import get_logger
from Src.Seqs.seq_info import calculateN50

from .von_mises_fisher_mixture import VonMisesFisherMixture

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
def cluster_split(
    sub_contigName2seq: Dict[str, str],
    contigName2RepNormV,
    gene2contigNames: Dict[str, List[str]],
    contigName2_gene2num: Dict[str, Dict[str, int]],
    no_learning_method = False,
    gmm_flspp = "flspp",
    min_contig_len: int = 768,
    the_last_time = False,
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
    if no_learning_method:
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

    # cluster part #
    # build can not link paris and X array
    X = []
    div = np.log(min_contig_len + 0.)
    length_weights = []
    contigName2index = {}
    index2contigName = {}
    for j, (contigName, repNormVec) in enumerate(existContig2RepNormV.items()):
        X.append(repNormVec)
        contigName2index[contigName] = j
        index2contigName[j] = contigName
        length_weights.append(np.log(len(sub_contigName2seq[contigName])) / div)
        # length_weights.append(len(sub_contigName2seq[contigName]))
    X = np.array(X, dtype=np.float64)
    
    bin_cluster_num_set = set()
    for i in range(-2, 3):
        cur_bin_cluster_num = bin_cluster_num + i
        if len(X) <= cur_bin_cluster_num:
            cur_bin_cluster_num = len(X) - 1
        if cur_bin_cluster_num <= 1:
            cur_bin_cluster_num = 2
        bin_cluster_num_set.add(cur_bin_cluster_num)
    bin_cluster_num_list = list(bin_cluster_num_set)
    model_list = []
    ss_score = True
    if len(X) >= 90000:
        ss_score = False
    
    run_von = False
    if gmm_flspp == "mix":
        if 150 <= len(X) <= 1850:
            run_von = True
    
    not_flspp = True
    if len(X) > 10000:
        not_flspp = False
    
    for i, cur_bin_cluster_num in enumerate(bin_cluster_num_list):
        if (gmm_flspp == "mix" and run_von) or (gmm_flspp =="von" and not_flspp):
            # print(f"VonMiss has been applied {len(X)}")
            cur_model = VonMisesFisherMixture(n_clusters=cur_bin_cluster_num, 
                                              n_jobs=1, 
                                              init="k-means++", 
                                              random_state=3407, 
                                              n_init=1)
            cur_model = cur_model.fit(X)
            if ss_score:
                model_list.append((cur_model.labels_, cur_model.cluster_centers_, silhouette_score(X, cur_model.labels_)))
            else:
                model_list.append((cur_model.labels_, cur_model.cluster_centers_, -cur_model.inertia_))
        else:
            # print(f"FLSpp has been applied {len(X)}")
            cur_model = FLSpp(n_clusters=cur_bin_cluster_num, 
                        max_iter=600, 
                        local_search_iterations=60, 
                        random_state=3407)
            cur_model = cur_model.fit(X, sample_weight=length_weights)
            if ss_score:
                model_list.append((cur_model.labels_, cur_model.cluster_centers_, silhouette_score(X, cur_model.labels_)))
            else:
                model_list.append((cur_model.labels_, cur_model.cluster_centers_, -cur_model.inertia_))
    sorted_model_list = list(sorted(model_list, key = lambda x: x[-1], reverse=True))
    labels_ = sorted_model_list[0][0]
    
    cluster_out = {}
    for i, label in enumerate(labels_):
        contigName = index2contigName[i]
        if label not in cluster_out:
            cur_name2seq = {}
            cur_name2seq[contigName] = sub_contigName2seq[contigName]
            cluster_out[label] = cur_name2seq
        else:
            cur_name2seq = cluster_out[label]
            cur_name2seq[contigName] = sub_contigName2seq[contigName]
    
    # ##### result collection
    res = []
    for _, name2seq in cluster_out.items():
        res.append((name2seq, summedLengthCal(name2seq)))
    res_ord = []
    for name2seq, _ in list(sorted(res, key=lambda x: x[-1], reverse=True)):
        res_ord.append(name2seq)
    return res_ord


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
    min_contig_len: int
):
    # start
    quality_record = {}
    index = 0
    logger.info(f"--> Start current method: {c_file}. {cur_i} / {tol_n}.")
    n_cluster = len(clu2contignames)
    for i,  (_, first_cluster_contignames) in enumerate(clu2contignames.items()):
        # progressBar(i, n_cluster)
        first_cluster_contigName2seq = {}
        for contigName in first_cluster_contignames:
            first_cluster_contigName2seq[contigName] = contigname2seq[contigName]
        first_dom, first_cluster_gene2contig_list, first_cluster_contigName2_gene2num, comp, cont = determine_domain(
            tname2markerset,
            first_cluster_contignames,
            bac_contigName2_gene2num,
            arc_contigName2_gene2num,
            )
        if cont > 10:
            secd_cluster_contigname2seq_list = cluster_split(
                first_cluster_contigName2seq,
                contigname2repNormVector,
                first_cluster_gene2contig_list,
                first_cluster_contigName2_gene2num,
                gmm_flspp=gmm_flspp,
                min_contig_len = min_contig_len,
                the_last_time=False
            )
            secd_cluster_bins_qualites = eval_qualities_for_cluster_split(
                secd_cluster_contigname2seq_list,
                tname2markerset,
                bac_contigName2_gene2num,
                arc_contigName2_gene2num,
            )
            for secd_cluster_contigname2seq, secd_cluster_gene2contig_list, secd_cluster_contigName2_gene2num, \
                    comp, cont in secd_cluster_bins_qualites:
                if cont > 5:
                    no_learn_method = True
                    if cont > 10:
                        no_learn_method = False
                    third_cluster_contigname2seq_list = cluster_split(
                        secd_cluster_contigname2seq,
                        contigname2repNormVector,
                        secd_cluster_gene2contig_list,
                        secd_cluster_contigName2_gene2num,
                        no_learn_method,
                        gmm_flspp = gmm_flspp,
                        min_contig_len=min_contig_len,
                        the_last_time=True)
                    third_cluster_bins_qualites = eval_qualities_for_cluster_split(
                        third_cluster_contigname2seq_list,
                        tname2markerset,
                        bac_contigName2_gene2num,
                        arc_contigName2_gene2num
                    )
                    for third_cluster_contigname2seq, _, _, comp, cont in third_cluster_bins_qualites:
                        writeFasta(third_cluster_contigname2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
                        size = summedLengthCal(third_cluster_contigname2seq)
                        n50 = np.log(calculateN50(third_cluster_contigname2seq))
                        quality_record[f"CompleteBin_cand_{index}.fasta"] = (comp, cont, n50, size)
                        index += 1
                else:
                    writeFasta(secd_cluster_contigname2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
                    size = summedLengthCal(secd_cluster_contigname2seq)
                    n50 = np.log(calculateN50(secd_cluster_contigname2seq))
                    quality_record[f"CompleteBin_cand_{index}.fasta"] = (comp, cont, n50, size)
                    index += 1
        elif cont > 5:
            secd_cluster_contigname2seq_list = cluster_split(
                first_cluster_contigName2seq,
                contigname2repNormVector,
                first_cluster_gene2contig_list,
                first_cluster_contigName2_gene2num,
                True,
                gmm_flspp=gmm_flspp,
                min_contig_len=min_contig_len,
                the_last_time=True
            )
            for secd_cluster_contigname2seq in secd_cluster_contigname2seq_list:
                sub_split_contignames = []
                for contigName in secd_cluster_contigname2seq.keys():
                    sub_split_contignames.append(contigName)
                assert len(sub_split_contignames) != 0
                _, _, _, comp, cont = determine_domain(
                    tname2markerset,
                    sub_split_contignames,
                    bac_contigName2_gene2num,
                    arc_contigName2_gene2num,
                    first_dom)
                writeFasta(secd_cluster_contigname2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
                size = summedLengthCal(secd_cluster_contigname2seq)
                n50 = np.log(calculateN50(secd_cluster_contigname2seq))
                quality_record[f"CompleteBin_cand_{index}.fasta"] = (comp, cont, n50, size)
                index += 1
        else:
            writeFasta(first_cluster_contigName2seq, os.path.join(output_folder, f"CompleteBin_cand_{index}.fasta"))
            size = summedLengthCal(first_cluster_contigName2seq)
            n50 = np.log(calculateN50(first_cluster_contigName2seq))
            quality_record[f"CompleteBin_cand_{index}.fasta"] = (comp, cont, n50, size)
            index += 1
    logger.info(f"--> End of second cluster with current method: {c_file}. {cur_i} / {tol_n}")
    writePickle(os.path.join(temp_bin_folder_path, f"{c_file}_quality_record.pkl"), quality_record)
    return c_file, quality_record