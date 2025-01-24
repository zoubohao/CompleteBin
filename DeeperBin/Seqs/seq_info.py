import multiprocessing
import os
from random import seed, shuffle
from typing import List

import numpy as np
import psutil

from DeeperBin.CallGenes.gene_utils import callMarkerGenes
from DeeperBin.IO import readHMMFileReturnDict, writeFasta, writePickle
from DeeperBin.logger import get_logger
from DeeperBin.Seqs.seq_utils import base_pair_coverage_calculate

logger = get_logger()


def calculateN50(seqLens):
    if isinstance(seqLens, dict):
        contig_len = []
        for _, seq in seqLens.items():
            contig_len.append(len(seq))
        seqLens = contig_len
    thresholdN50 = sum(seqLens) / 2.0
    seqLens.sort(reverse=True)
    testSum = 0
    N50 = 0
    for seqLen in seqLens:
        testSum += seqLen
        if testSum >= thresholdN50:
            N50 = seqLen
            break
    return N50


def prepare_sequences_coverage(
    contigname2seq_ori: str,
    sorted_bam_file_list: List[str],
    temp_file_folder_path: str,
    min_contig_length: int,
    hmm_model_path,
    num_workers = None
):
    contigname2seq = contigname2seq_ori
    seq_len = []
    contigname2seq_new = {}
    for contigname, seq in contigname2seq.items():
        seq_len.append(len(seq))
        if len(seq) >= min_contig_length:
            contigname2seq_new[contigname] = seq.upper()
    
    N50 = calculateN50(seq_len)
    logger.info(f"--> The number of {len(contigname2seq_new)} contigs are longer than {min_contig_length}. The N50 of processed contigs is {N50}.")
    contigname2seq = contigname2seq_new
    split_input_folder = os.path.join(temp_file_folder_path, "random_split_contigs")
    if os.path.exists(split_input_folder) is False:
        os.mkdir(split_input_folder)
    index = 0
    temp_contigs = {}
    contignames_list = list(contigname2seq.keys())
    seed(7)
    shuffle(contignames_list)
    seed(None)
    for contigname in contignames_list:
        temp_contigs[contigname] = contigname2seq[contigname]
        if len(temp_contigs) >= 1000:
            writeFasta(temp_contigs, os.path.join(split_input_folder, f"{index}.fasta"))
            index += 1
            temp_contigs = {}
    if len(temp_contigs) > 0:
        writeFasta(temp_contigs, os.path.join(split_input_folder, f"{index}.fasta"))
        index += 1
        temp_contigs = {}
    ###
    if num_workers is None: num_workers = psutil.cpu_count()
    logger.info("--> Start to calculate the base pair coverage for each contig.")
    if os.path.exists(os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl")) is False:
        pro_list = []
        pro_num = len(sorted_bam_file_list)
        name2bparray_single_list = []
        with multiprocessing.Pool(pro_num) as multiprocess:
            for i, sorted_bam_file in enumerate(sorted_bam_file_list):
                p = multiprocess.apply_async(base_pair_coverage_calculate,
                                            (contigname2seq,
                                            sorted_bam_file,
                                            os.path.join(temp_file_folder_path, f"contigname2bpcover_nparray_{i}.pkl"),
                                            min_contig_length,
                                            num_workers,
                                            False,
                                            ))
                pro_list.append(p)
            multiprocess.close()
            for p in pro_list:
                name2bparray_single_list.append(p.get())
        
        name2bpcover_nparray_list = {}
        cov_val_list = []
        var_val_list = []
        logger.info(f"--> Start to collect the coverage information from {pro_num} bam files.")
        for name, cur_dna_seq in contigname2seq.items():
            cur_bp_array_list = []
            for cur_name2bp_array in name2bparray_single_list:
                if name not in cur_name2bp_array:
                    cur_bp_array_list.append(np.zeros(shape=[len(cur_dna_seq)], dtype=np.int64))
                else:
                    cur_bp_array_list.append(cur_name2bp_array[name])
                    cov_val_list.append(np.mean(cur_name2bp_array[name]))
                    var_val_list.append(np.sqrt(np.var(cur_name2bp_array[name], dtype=np.float32)))
            name2bpcover_nparray_list[name] = cur_bp_array_list
        logger.info(f"--> Start to write coverage information")
        for name, bp_list in name2bpcover_nparray_list.items():
            n = len(bp_list)
            assert n == pro_num, ValueError(f"There are number of {pro_num} bam files, but contig {name} only have {n} coverage info.")
        writePickle(os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl"), name2bpcover_nparray_list)
        mean_val = np.percentile(np.array(cov_val_list, dtype=np.float32), 99)
        var_val = np.percentile(np.array(var_val_list, dtype=np.float32), 99)
        logger.info(f"--> The 99 percentil of coverage mean value is {mean_val}, the sqrt var is {var_val}.")
        writePickle(os.path.join(temp_file_folder_path, "mean_var.pkl"), (mean_val, var_val))
    
    logger.info("--> Start to Call 40 Marker Genes.")
    call_genes_folder = os.path.join(temp_file_folder_path, "call_genes_random")
    if os.path.exists(call_genes_folder) is False:
        os.mkdir(call_genes_folder)
    callMarkerGenes(split_input_folder,
                    call_genes_folder,
                    num_workers,
                    hmm_model_path,
                    "fasta")
    ##
    logger.info("--> Start to Collect 40 Marker Genes.")
    contigname2hits = {}
    # gene file build
    for file in os.listdir(call_genes_folder):
        _, suffix = os.path.splitext(file)
        if suffix[1:] == "txt":
            contigname2hits.update(
                readHMMFileReturnDict(os.path.join(call_genes_folder, file))
            )
        elif suffix[1:] not in ["faa", "gff"]:
            raise ValueError(f"ERROR in the output folder: {call_genes_folder}, the file is {file}, suffix is {suffix[1:]}")
    writePickle(os.path.join(temp_file_folder_path, "contigname2hmmhits_list.pkl"), contigname2hits)
    writePickle(os.path.join(temp_file_folder_path, "contigname2seq_str.pkl"), contigname2seq)

