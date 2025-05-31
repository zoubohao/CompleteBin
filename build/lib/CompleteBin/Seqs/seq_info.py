
import os
from random import seed, shuffle
from shutil import rmtree
from typing import List
import multiprocessing

import numpy as np
import psutil

from CompleteBin.CallGenes.gene_utils import callMarkerGenes
from CompleteBin.IO import readHMMFileReturnDict, readPickle, writeFasta, writePickle
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_utils import base_pair_coverage_calculate

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
    num_workers = None,
    remove_temp_files = True
):
    contigname2seq = contigname2seq_ori
    seq_len = []
    contigname2seq_new = {}
    for contigname, seq in contigname2seq.items():
        seq_len.append(len(seq))
        if len(seq) >= min_contig_length:
            contigname2seq_new[contigname] = seq.upper()
    
    logger.info(f"--> The number of {len(contigname2seq_new)} contigs are longer than {min_contig_length}.")
    contigname2seq = contigname2seq_new
    ###
    if num_workers is None: num_workers = psutil.cpu_count()
    pro_num = len(sorted_bam_file_list)
    logger.info(f"--> Start to calculate the base pair coverage for each contig from {pro_num} bam file.")
    if os.path.exists(os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl")) is False:
        pro_list = []
        with multiprocessing.Pool(pro_num) as multiprocess:
            for i, sorted_bam_file in enumerate(sorted_bam_file_list):
                p = multiprocess.apply_async(base_pair_coverage_calculate,
                                            (contigname2seq,
                                            sorted_bam_file,
                                            os.path.join(temp_file_folder_path, f"contigname2bpcover_nparray_{i}.pkl"),
                                            min_contig_length,
                                            num_workers,
                                            True,
                                            ))
                pro_list.append(p)
            multiprocess.close()
            for p in pro_list:
                p.get()
        
        name2bparray_single_list = []
        for i in range(pro_num):
            name2bparray_single_list.append(readPickle(os.path.join(temp_file_folder_path, f"contigname2bpcover_nparray_{i}.pkl")))
        
        name2bpcover_nparray_list = {}
        cov_val_list = [[] for _ in range(pro_num)]
        var_val_list = [[] for _ in range(pro_num)]
        logger.info(f"--> Start to collect the coverage information from {pro_num} bam files.")
        for name, cur_dna_seq in contigname2seq.items():
            cur_bp_array_list = []
            for k, cur_name2bp_array in enumerate(name2bparray_single_list):
                if name not in cur_name2bp_array:
                    cur_bp_array_list.append(np.zeros(shape=[len(cur_dna_seq)], dtype=np.int64))
                    cov_val_list[k].append(0.)
                    var_val_list[k].append(0.)
                else:
                    cur_bp_array_list.append(cur_name2bp_array[name])
                    ## cal max for different bam files
                    cov_val_list[k].append(sum(cur_name2bp_array[name]) / len(cur_name2bp_array[name]))
                    var_val_list[k].append(np.std(cur_name2bp_array[name] + 1e-5, dtype=np.float32))
            name2bpcover_nparray_list[name] = cur_bp_array_list
        logger.info(f"--> Start to write coverage information")
        for name, bp_list in name2bpcover_nparray_list.items():
            n = len(bp_list)
            assert n == pro_num, ValueError(f"There are number of {pro_num} bam files, but contig {name} only have {n} coverage info.")
        writePickle(os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl"), name2bpcover_nparray_list)
        mean_val = np.max(np.array(cov_val_list, dtype=np.float32), axis=1, keepdims=False) + 1e-5
        var_val = np.max(np.array(var_val_list, dtype=np.float32), axis=1, keepdims=False) + 1e-5
        if max(mean_val) > 1000:
            mean_val /= 2.
        if max(var_val) > 1000:
            var_val /= 2.
        logger.info(f"--> The max of coverage mean value is {mean_val}.")
        logger.info(f"--> The max of coverage std value is {var_val}.")
        writePickle(os.path.join(temp_file_folder_path, "mean_var.pkl"), (mean_val, var_val))
    
    logger.info("--> Start to Call 40 Marker Genes.")
    split_input_folder = os.path.join(temp_file_folder_path, "split_contigs_random")
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
    call_genes_folder = os.path.join(temp_file_folder_path, "call_genes_random")
    if os.path.exists(call_genes_folder) is False:
        os.mkdir(call_genes_folder)
    callMarkerGenes(split_input_folder,
                    call_genes_folder,
                    num_workers,
                    hmm_model_path,
                    "fasta")
    ##
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
    if remove_temp_files:
        rmtree(call_genes_folder, ignore_errors=True)
        rmtree(split_input_folder, ignore_errors=True)

