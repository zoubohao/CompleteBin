

import os
import random
from itertools import product
from typing import Dict

import numpy as np
import pysam
from numpy.random import choice, shuffle

from Src.CallGenes.gene_utils import callMarkerGenes
from Src.IO import readFasta, readHMMFileReturnDict, writePickle
from Src.logger import get_logger

logger = get_logger()

# def reject_outliers(data: np.ndarray, min_i = -100,  max_i = 100):
#     """
#     You can adjust your cut-off for outliers by adjusting argument m in function call. 
#     The larger it is, the less outliers are removed. 
#     This function seems to be more robust to various types of outliers compared to other outlier removal techniques.
#     Args:
#         data (_type_): _description_
#         min_i (_type_, optional): _description_. Defaults to -100
#         max_i (_type_, optional): _description_. Defaults to 100
#     Returns:
#         _type_: _description_
#     """
    
#     p1 = np.percentile(data, 1)
#     p99 = np.percentile(data, 99)
#     d = data - np.median(data)
#     mdev = np.median(d)
#     if mdev != 0:
#         s = d / mdev
#     else:
#         s = d
#     print(s, mdev)
#     data[s >= max_i] = p99
#     data[s <= min_i] = p1
#     return data


def base_pair_coverage_calculate(
    name2seq: Dict[str, str],
    bam_file_path: str,
    output_path: str,
    min_contig_length: int = 600,
    num_worker=64,
    write_pickle = False
):
    logger.info("--> Start to calculate coverage for each contig.")
    name2numpyarray = {}
    for name, seq in name2seq.items():
        if len(seq) >= min_contig_length:
            name2numpyarray[name] = np.zeros(shape=[len(seq)], dtype=np.int64)
    bamfile = pysam.AlignmentFile(bam_file_path, "rb", threads=num_worker)
    index = 0
    jump_out_count = 0
    for reads in bamfile:
        if reads.reference_name is None:
            continue
        if jump_out_count >= 100000000:
            break
        name = ">" + reads.reference_name
        # print(f"the ref name is {name}")
        if name not in name2numpyarray:
            jump_out_count += 1
            continue
        cur_array = name2numpyarray[name]
        if reads.reference_start is not None and reads.reference_end is not None:
            s = reads.reference_start
            e = reads.reference_end
            cur_array[s: e] += 1
            index += 1
            jump_out_count = 0
    name2numpyarray_new = {}
    for name, base_pair_arrary in name2numpyarray.items():
        ## try to remove some outliers
        cutoff = np.percentile(base_pair_arrary, q = 97.5)
        del_index = base_pair_arrary > cutoff
        base_pair_arrary[del_index] = cutoff
        name2numpyarray_new[name] = base_pair_arrary
    # if write_pickle:
    writePickle(output_path, name2numpyarray_new)
    return name2numpyarray_new


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sampleSeqFromFasta(fasta_path: str, seq_min_len, seq_max_len, short_prob = 0.25, fixed = False):
    """
    Args:
        fasta_path (str): _description_
        seq_max_len (int): _description_
        seq_min_len (int): _description_

    Returns:
        _type_: _description_
    """
    if fixed:
        random.seed(1024)
        np.random.seed(1024)
    else:
        random.seed(None)
        np.random.seed(None)
    
    contig2seq = readFasta(fasta_path)
    contigs_list = list(contig2seq.values())
    shuffle(contigs_list)

    contigs_num = len(contigs_list)
    if contigs_num == 1 or fixed:
        seq = contigs_list[0]
    else:
        length = []
        for contig in contigs_list:
            length.append(len(contig))
        p = np.array(length, dtype=np.float32) / sum(length)
        l = len(p) * 0.25
        if l < 1:
            l = 1
        elif l > 16:
            l = 16
        p = softmax(p * l)
        index = choice(contigs_num, None, p=p)
        seq = contigs_list[index]
    
    n = len(seq)
    rand = np.random.rand()
    if rand <= 0.5:
        l = random.randint(seq_min_len, seq_max_len)
    elif 0.5 < rand <= 0.5 + short_prob:
        l = random.randint(seq_min_len, 2000)
    else:
        l = random.randint(2000, seq_max_len)

    if (n - l) > 0:
        s = random.randint(0, n - l)
    else:
        s = 0

    cur_seq = seq[s: s + l]
    return cur_seq


def random_generate_view(
    seq: str,
    min_contig_len: int,
    seed=None
):
    if seed is None:
        random.seed()
    else:
        random.seed(seed)
    n = len(seq)
    sim_len = random.randint(min_contig_len - 1, n)
    start = random.randint(0, n - sim_len)
    end = start + sim_len
    random.seed()
    return seq[start: end], start, end


def seqSimulateSNV(seq: str, vRatio=0.05) -> str:
    nt2ntList = {"A": ["T", "C", "G"], "T": ["A", "C", "G"], "C": ["T", "A", "G"], "G": ["T", "C", "A"]}
    nt = ["T", "C", "G", "A"]
    newSeq = []
    for c in seq:
        if random.random() >= vRatio:
            newSeq.append(c)
        else:
            index = np.random.randint(0, 3, dtype=np.int64)
            if c in nt2ntList:
                newSeq.append(nt2ntList[c][index])
            else:
                index = np.random.randint(0, 4, dtype=np.int64)
                newSeq.append(nt[index])
    return "".join(newSeq)


def generateNoisySeq(g_len: int) -> str:
    index2nt = {0: "A", 1: "T", 2: "C", 3: "G"}
    intSeq = np.random.randint(0, 4, size=[g_len], dtype=np.int64)
    return "".join(map(lambda x: index2nt[x], intSeq))


def seqInsertion(seq: str, iRatio=0.05, scatter=False) -> str:
    if not scatter:
        n = len(seq)
        g_len = int(n * iRatio) + 1
        noisy_seq = generateNoisySeq(g_len)
        s = random.randint(1, n - 1)
        return seq[0: s] + noisy_seq + seq[s:]
    nt2ntList = {"A": ["T", "C", "G"], "T": ["A", "C", "G"], "C": ["T", "A", "G"], "G": ["T", "C", "A"]}
    nt = ["T", "C", "G", "A"]
    newSeq = []
    for c in seq:
        newSeq.append(c)
        if random.random() < iRatio:
            index = np.random.randint(0, 3, dtype=np.int64)
            if c in nt2ntList:
                newSeq.append(nt2ntList[c][index])
            else:
                index = np.random.randint(0, 4, dtype=np.int64)
                newSeq.append(nt[index])
    return "".join(newSeq)


def seqDeletion(seq: str, dRatio=0.05, scatter=False) -> str:
    if not scatter:
        n = len(seq)
        d_len = int(n * dRatio) + 1
        s = random.randint(1, n - 1 - d_len)
        return seq[0: s] + seq[s + d_len:]
    newSeq = []
    for c in seq:
        if random.random() >= dRatio:
            newSeq.append(c)
    return "".join(newSeq)


def sequence_data_augmentation(seq: str, dRatio = 0.005, vRatio = 0.005, iRatio = 0.005):
    rand_v = random.random()
    if rand_v <= 0.333:
        return seqSimulateSNV(seq, vRatio)
    elif 0.333 < rand_v <= 0.666:
        if random.random() <= 0.5:
            return seqDeletion(seq, dRatio, scatter=True)
        else:
            return seqDeletion(seq, dRatio, scatter=False)
    else:
        if random.random() <= 0.5:
            return seqInsertion(seq, iRatio, scatter=True)
        else:
            return seqInsertion(seq, iRatio, scatter=False)




## kmer functions
def generate_feature_mapping_reverse(kmer_len):
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC", repeat=kmer_len):
        kmer = "".join(kmer)
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            rev_compl = "".join(rev_compl)
            kmer_hash[rev_compl] = counter
            counter += 1
    return kmer_hash, counter


def generate_feature_mapping_whole_tokens(kmer_len):
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC", repeat=kmer_len):
        kmer = "".join(kmer)
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            counter += 1
    return kmer_hash, counter


def getGeneWithLongestLength(gene2contigNames: dict, contigname2seq: dict, intersect_accs = None):
    gene2count = []
    for gene_name, contigs_with_this_gene in gene2contigNames.items():
        if intersect_accs is not None:
            if gene_name in intersect_accs:
                summed_length = 0
                for cur_contigname in contigs_with_this_gene:
                    summed_length += len(contigname2seq[cur_contigname])
                gene2count.append((gene_name, len(contigs_with_this_gene), summed_length))
        else:
            summed_length = 0
            for cur_contigname in contigs_with_this_gene:
                summed_length += len(contigname2seq[cur_contigname])
            gene2count.append((gene_name, len(contigs_with_this_gene), summed_length))
    gene2count.sort(key=lambda x: x[-1], reverse=True)
    gene_name = gene2count[0][0]
    summed_length = gene2count[0][2]
    return gene_name, summed_length, gene2contigNames[gene_name]


def getGeneWithLargestCount(gene2contigNames: dict, contigname2seq: dict, intersect_accs):
    gene2count = []
    for gene_name, contigs_with_this_gene in gene2contigNames.items():
        if intersect_accs is not None and gene_name in intersect_accs:
            gene2count.append((gene_name, len(contigs_with_this_gene)))
        else:
            gene2count.append((gene_name, len(contigs_with_this_gene)))
    gene2count.sort(key=lambda x: x[-1], reverse=True)
    gene_name = gene2count[0][0]
    count = gene2count[0][1]
    return gene_name, count, gene2contigNames[gene_name]


def callGenesForKmeans(
    temp_file_folder_path,
    input_bins_folder,
    num_workers,
    hmm_model_path,
    ):
    logger.info("--> Start to Call 40 Marker Genes.")
    call_genes_folder = os.path.join(temp_file_folder_path, "call_genes_initial_kmeans")
    if os.path.exists(call_genes_folder) is False:
        os.mkdir(call_genes_folder)
    callMarkerGenes(input_bins_folder,
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
    writePickle(os.path.join(temp_file_folder_path, "contigname2hmmhits_list_initial_kmeans.pkl"), contigname2hits)
