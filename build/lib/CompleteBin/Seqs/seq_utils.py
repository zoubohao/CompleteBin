

import os
import random
from itertools import product
import time
from typing import Dict

import numpy as np
import pysam
from numpy.random import choice, shuffle

from CompleteBin.IO import readFasta, writePickle
from CompleteBin.logger import get_logger

logger = get_logger()


def base_pair_coverage_calculate(
    name2seq: Dict[str, str],
    bam_file_path: str,
    output_path: str,
    min_contig_length: int = 0,
    num_worker: int = 64,
    write_pickle: bool = False,
):
    """Calculate per-base-pair coverage for each contig from a sorted BAM file.

    Iterates through every aligned read in the BAM, counting coverage at each
    reference position.  Unlike the original implementation this version:

    - Counts **all** aligned reads regardless of MAPQ (matching ``bedtools
      genomecov`` behaviour).  The original MAPQ >= 30 filter discarded reads
      with MAPQ = 0 (typically multi-mapped or repetitive-region reads),
      causing complete coverage loss for ~9 % of contigs >= 1000 bp.

    - Removes the ``jump_out_count >= 100_000_000`` early-exit guard that
      could prematurely terminate the BAM scan on large metagenomic datasets
      with many short (filtered-out) contigs.  Replaced by a progress log
      emitted every 50 M skipped reads.

    - Applies 97.5th-percentile clipping to per-base coverage on each contig
      to dampen outlier spikes (unchanged from the original).
    """
    logger.info("--> Start to calculate coverage for each contig.")
    name2numpyarray = {}
    for name, seq in name2seq.items():
        if len(seq) >= min_contig_length:
            name2numpyarray[name] = np.zeros(shape=[len(seq)], dtype=np.int64)

    n_target = len(name2numpyarray)
    logger.info(f"--> {n_target} contigs to track (>= {min_contig_length} bp).")

    bamfile = pysam.AlignmentFile(bam_file_path, "rb", threads=num_worker)

    counted_reads = 0        # reads that contributed to coverage
    skipped_reads = 0        # reads mapping to filtered-out contigs
    unmapped_reads = 0       # reads with reference_name is None
    total_reads = 0
    progress_interval = 50_000_000

    t_start = time.time()
    for reads in bamfile:
        total_reads += 1
        if reads.reference_name is None:
            unmapped_reads += 1
            continue

        name = ">" + reads.reference_name
        if name not in name2numpyarray:
            skipped_reads += 1
            if skipped_reads % progress_interval == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f"--> Scanned {total_reads // 1_000_000}M reads, "
                    f"{skipped_reads // 1_000_000}M mapped to filtered contigs "
                    f"(elapsed {elapsed:.0f}s, continuing...)"
                )
            continue

        cur_array = name2numpyarray[name]
        if reads.reference_start is not None and reads.reference_end is not None:
            s = reads.reference_start
            e = reads.reference_end
            cur_array[s:e] += 1
            counted_reads += 1

    bamfile.close()
    elapsed = time.time() - t_start
    pct_useful = 100.0 * counted_reads / max(total_reads, 1)
    logger.info(
        f"--> BAM scan finished: {total_reads:,} total reads, "
        f"{counted_reads:,} counted ({pct_useful:.1f}%), "
        f"{skipped_reads:,} skipped (filtered contig), "
        f"{unmapped_reads:,} unmapped, "
        f"{elapsed:.0f}s elapsed"
    )

    # 97.5th-percentile clipping — dampen outlier coverage spikes per contig
    name2numpyarray_new = {}
    for name, base_pair_array in name2numpyarray.items():
        cutoff = np.percentile(base_pair_array, q=97.5)
        exceed_mask = base_pair_array > cutoff
        base_pair_array[exceed_mask] = cutoff
        name2numpyarray_new[name] = base_pair_array

    # Count contigs with zero coverage and warn
    zero_cov_contigs = [name for name, arr in name2numpyarray_new.items() if arr.sum() == 0]
    logger.info(
        f"--> {len(zero_cov_contigs)} / {len(name2numpyarray_new)} contigs have zero coverage "
        f"({100.0 * len(zero_cov_contigs) / max(len(name2numpyarray_new), 1):.1f}%). "
        f"These contigs will have no signal for binning."
    )

    if write_pickle:
        writePickle(output_path, name2numpyarray_new)

    return name2numpyarray_new


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sampleSeqFromFasta(fasta_path: str, seq_min_len, seq_max_len, short_prob=0.25, fixed=False):
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
    if seed:
        random.seed(seed)
    n = len(seq)
    sim_len = random.randint(min_contig_len - 1, n)
    start = random.randint(0, n - sim_len)
    end = start + sim_len
    random.seed(2048)
    return seq[start: end], start, end


def random_generate_view_with_range(
    seq: str,
    min_contig_len: int,
    part_range: float,
    seed=None
):
    assert part_range <= 1
    n = len(seq)
    sim_len = max(min_contig_len - 10, int(n * part_range))
    start = random.randint(0, n - sim_len)
    end = start + sim_len
    return seq[start: end], start, end


def generate_view_with_fixed_len(
    seq: str,
    fix_len: int,
    seed=None
):
    n = len(seq)
    # if fix_len >= n: fix_len = n - 2
    fix_len = min(n - 10, fix_len)
    # print(fix_len)
    start = random.randint(0, n - fix_len)
    end = start + fix_len
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


def sequence_data_augmentation(seq: str, dRatio=0.005, vRatio=0.005, iRatio=0.005):
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


# kmer functions
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


def generate_feature_mapping_protein(kmer_len):
    kmer_hash = {}
    counter = 0
    for kmer in product("ACDEFGHIKLMNPQRSTVWY", repeat=kmer_len):
        kmer = "".join(kmer)
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
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


def getGeneWithLongestLength(gene2contigNames: dict, contigname2seq: dict, intersect_accs=None):
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


def filter_short_contigs(
    faa_folder,
    contig_file_path,
    contigName2_gene2num,
):
    contigname2seq = readFasta(contig_file_path)
    contigname2orf_num = {}
    for name, _ in contigname2seq.items():
        contigname2orf_num[name] = 0
    for file in os.listdir(faa_folder):
        if ".faa" in file:
            cur_longname2seq = readFasta(os.path.join(faa_folder, file))
            for longname, _ in cur_longname2seq.items():
                if "partial=00" in longname or "partial=01" in longname or "partial=10" in longname:
                    real_contigname = "_".join(longname.split()[0].split("_")[0:-1])
                    contigname2orf_num[real_contigname] += 1
    final_contig_set = set()
    for name, orf_num in contigname2orf_num.items():
        if orf_num > 0:
            final_contig_set.add(name)
    for name, _ in contigName2_gene2num.items():
        if name not in final_contig_set:
            final_contig_set.add(name)
    output = {}
    for name in final_contig_set:
        output[name] = contigname2seq[name]
    return output
