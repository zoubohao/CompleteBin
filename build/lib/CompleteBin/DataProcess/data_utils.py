

import multiprocessing
import os
import random

import numpy as np
import psutil

from CompleteBin.CallGenes.gene_utils import splitListEqually
from CompleteBin.IO import readPickle
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_utils import (generate_feature_mapping_reverse,
                                        random_generate_view_with_range)

logger = get_logger()


BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


def get_tuple_kmer(kmer: str):
    rev_kmer = "".join([BASE_COMPLEMENT[x] for x in reversed(kmer)])
    return tuple(sorted([kmer, rev_kmer]))


# ── Vectorised k-mer counting (same API, 20-50× faster) ────────────────────

_B2B = np.zeros(256, dtype=np.uint8)          # byte -> base index
_B2B[65] = _B2B[97] = 0                       # A, a
_B2B[67] = _B2B[99] = 1                       # C, c
_B2B[71] = _B2B[103] = 2                      # G, g
_B2B[84] = _B2B[116] = 3                      # T, t

_PACKED_CACHE = {}                             # id(kmer_dict) -> lookup array


def _build_lookup(kmer_dict, kmer_len):
    """(4**kmer_len,) int32 array: packed k-mer index -> canonical index."""
    t = np.full(4 ** kmer_len, -1, dtype=np.int32)
    for s, ci in kmer_dict.items():
        bi = 0
        for ch in s:
            bi = bi * 4 + _B2B[ord(ch)]
        t[bi] = ci
    return t


def get_normlized_count_vec_of_seq_fast(
        seq: str,
        kmer_dict: dict,
        nr_features: int,
        kmer_len: int,
        bparray_list,
        cal_bp_tnf=False,
):
    """Vectorised drop-in replacement for get_normlized_count_vec_of_seq.

    Same signature, same return value.  Eliminates the per-position Python
    for-loop — uses numpy integer encoding, packed indexing, and bincount.
    """
    seq = seq.upper()
    N = len(seq)

    bam_num = None
    if cal_bp_tnf:
        bam_num = len(bparray_list)
        if N != len(bparray_list[0]):
            raise ValueError(
                f"The len of seq is: {N}, "
                f"but its bparray's length is {len(bparray_list[0])}"
            )

    n_kmers = N - kmer_len + 1
    if n_kmers <= 0:
        composition_v = np.zeros(nr_features, dtype=np.float32)
        bp_cov_tnf_array = None
        if cal_bp_tnf:
            bp_cov_tnf_array = bparray_list.copy()
            len_val = N // nr_features * nr_features
            if len_val > 0:
                bp_cov_tnf_array = np.reshape(bp_cov_tnf_array[:, :len_val], (bam_num, nr_features, N // nr_features))
                bp_cov_tnf_array = np.mean(bp_cov_tnf_array, axis=-1)
            else:
                bp_cov_tnf_array = np.zeros((bam_num, nr_features), dtype=np.float32)
        return composition_v, bp_cov_tnf_array

    # Integer-encode the whole sequence in one shot
    arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    idx = _B2B[arr]                               # (N,)  values in 0..3

    # Packed k-mer indices  (vectorised over all positions)
    packed = np.zeros(n_kmers, dtype=np.int32)
    for offset in range(kmer_len):
        packed = packed * 4 + idx[offset:offset + n_kmers]

    # Map to canonical via cached lookup
    key = id(kmer_dict)
    if key not in _PACKED_CACHE:
        _PACKED_CACHE[key] = _build_lookup(kmer_dict, kmer_len)
    canonical = _PACKED_CACHE[key][packed]

    # Count
    valid = canonical >= 0
    composition_v = np.bincount(canonical[valid], minlength=nr_features)

    total = composition_v.sum()
    if total == 0:
        composition_v = np.ones(nr_features, dtype=np.float32) / nr_features
    else:
        composition_v = composition_v.astype(np.float32) / total

    # bp-coverage (unchanged logic)
    bp_cov_tnf_array = None
    if cal_bp_tnf:
        bp_cov_tnf_array = bparray_list.copy()
        len_val = N // nr_features * nr_features
        bp_cov_tnf_array = np.reshape(bp_cov_tnf_array[:, :len_val], (bam_num, nr_features, N // nr_features))
        bp_cov_tnf_array = np.mean(bp_cov_tnf_array, axis=-1)

    return composition_v, bp_cov_tnf_array


def split_seq_equally(seq: str, num_parts: int, min_len: int):
    res = []
    N = len(seq) - min_len
    if N < 0:
        N = 0
    if num_parts == 1:
        return [(seq, 0, len(seq))]
    gap = N // (num_parts - 1)
    if gap == 0:
        for _ in range(num_parts):
            res.append((seq, 0, len(seq)))
        return res
    sub_seq = gap
    if sub_seq < min_len:
        sub_seq = min_len
    for i in range(0, N, gap):
        cur_seq = seq[i: i + sub_seq]
        res.append((cur_seq, i, i + sub_seq))
    if len(res) >= num_parts:
        return res[0: num_parts]
    else:
        for i in range(num_parts - len(res)):
            res.append((seq, 0, len(seq)))
        return res


def split_seq_with_overlap(seq: str, num_parts: int, min_len: int, overlap: int = None):
    """
    Split a sequence into num_parts overlapping sub-sequences.

    Adjacent sub-sequences share overlap bases. For long contigs this
    produces the overlap the original split_seq_equally only achieves
    for short contigs.

    If overlap is None, it defaults to min_len // 2.
    """
    L = len(seq)
    if num_parts <= 1:
        return [(seq, 0, L)]

    if overlap is None:
        overlap = max(1, min_len // 2)

    N = L - min_len
    if N < 0:
        N = 0

    gap = N // (num_parts - 1)
    if gap == 0:
        res = []
        for _ in range(num_parts):
            res.append((seq, 0, L))
        return res

    sub_len = gap + overlap
    if sub_len < min_len:
        sub_len = min_len
    # Enforce minimum segment length for reliable k-mer statistics.
    # Shorter segments get more overlap instead of degenerating.
    sub_len = max(sub_len, 1500)
    if sub_len > L:
        sub_len = L
    # Recompute gap based on the (possibly clamped) sub_len so that
    # segments are evenly distributed across the contig.
    gap = (L - sub_len) // (num_parts - 1)

    res = []
    for i in range(num_parts):
        start = i * gap
        if start + sub_len > L:
            start = L - sub_len
        start = max(0, start)
        end = start + sub_len
        res.append((seq[start:end], start, end))

    return res


def split_seq_randomly(seq: str, min_len: int, n_views: int = 5, seed=None):
    L = len(seq)

    # Adaptive lower bound: for long contigs, allow fragments as short as min_len
    # L=1000:    eff_start ≈ 2.0 → 20% (unchanged)
    # L=10000:   eff_start ≈ 0.85 → 8.5%
    # L=100000:  eff_start ≈ 0.1 → 1% (floor)
    lower_start = min_len / L * 10.0
    eff_start = min(2.0, max(lower_start, 0.1))

    # Spread n_views views across [eff_start, 9.6] (upper bound 96%)
    hi_end = 9.6
    new_step = (hi_end - eff_start) / n_views

    res = []
    for i in range(n_views):
        start_pos = eff_start + i * new_step
        min_range = start_pos / 10.
        part_range = random.random() * (new_step / 10.) + min_range
        if part_range > 1.:
            part_range = 1.
        sub_seq, start, end = random_generate_view_with_range(seq, min_len, part_range, seed)
        res.append((sub_seq, start, end))
    random.shuffle(res)
    return res


def get_features_of_one_seq(seq: str,
                            bp_nparray_list,
                            count_kmer,
                            count_kmer_dict,
                            count_nr_features,
                            subparts_list,
                            min_len,
                            ):
    seq = seq.upper().replace("N", "A")
    cal_tnf_bp = False
    if bp_nparray_list is not None:
        cal_tnf_bp = True
    whole_composition_v, whole_bp_cov_tnf_array = get_normlized_count_vec_of_seq_fast(seq, count_kmer_dict, count_nr_features, count_kmer,
                                                                                      bp_nparray_list, cal_tnf_bp)
    seq_rad_tokens = [whole_composition_v]
    mean_tokens = None
    std_tokens = None
    if bp_nparray_list is not None:
        mean_tokens = [np.mean(bp_nparray_list[:, 75:-75], axis=1, keepdims=False)]
        std_tokens = [np.std(bp_nparray_list[:, 75:-75], axis=1, keepdims=False)]
    sub_seqs_list = split_seq_with_overlap(seq, 5, min_len)
    for sub_seq, start, end in sub_seqs_list:
        sub_composition_v, _ = get_normlized_count_vec_of_seq_fast(sub_seq, count_kmer_dict, count_nr_features, count_kmer,
                                                                   bp_nparray_list, False)
        seq_rad_tokens.append(sub_composition_v)
        if bp_nparray_list is not None:
            mean_tokens.append(np.mean(bp_nparray_list[:, start + 75: end - 75], axis=1, keepdims=False))
            std_tokens.append(np.std(bp_nparray_list[:, start + 75: end - 75], axis=1, keepdims=False))
    seq_rad_tokens = np.stack(seq_rad_tokens, axis=0)
    if bp_nparray_list is not None:
        mean_tokens = np.stack(mean_tokens, axis=0)
        std_tokens = np.stack(std_tokens, axis=0)
    return seq_rad_tokens, mean_tokens, std_tokens, whole_bp_cov_tnf_array


def process_data_one_thread_return_list(
    contigname2seq,
    contigname2bp_nparray_list,
    count_kmer,
    min_len
):
    j = 0
    n = len(contigname2seq)
    output_list = []
    count_kmer_dict, count_nr_features = generate_feature_mapping_reverse(count_kmer)
    for contigname, seq in contigname2seq.items():
        # progressBar(j, n)
        seq_rad_tokens, mean_tokens, std_tokens, whole_bp_cov_tnf_array = get_features_of_one_seq(
            seq,
            contigname2bp_nparray_list[contigname],
            count_kmer,
            count_kmer_dict,
            count_nr_features,
            [1, 5],
            min_len)
        cur_tuple = (seq,
                     contigname2bp_nparray_list[contigname],
                     seq_rad_tokens,
                     mean_tokens,
                     std_tokens,
                     whole_bp_cov_tnf_array
                     )
        ###
        output_list.append((contigname[1:], cur_tuple))
        j += 1
    return output_list


def sub_process_generate_data(split_list, count_kmer, min_len):
    pro_list = []
    res = []
    with multiprocessing.Pool(len(split_list)) as multiprocess:
        for i, item in enumerate(split_list):
            p = multiprocess.apply_async(process_data_one_thread_return_list,
                                         (item[0],
                                          item[1],
                                          count_kmer,
                                          min_len,
                                          ))
            pro_list.append(p)
        multiprocess.close()
        for p in pro_list:
            res.append(p.get())

    save_list = []
    for cur_thread_list in res:
        for item in cur_thread_list:
            save_list.append(item)
    return save_list


def build_training_seq_data_numpy_save(
    contigname2seq_path: str,
    contigname2bp_nparray_list_path: str,
    data_output_path: str,
    count_kmer,
    min_len,
    num_workers: int = None
):
    contigname2seq = readPickle(contigname2seq_path)
    contigname2bp_array_list = readPickle(contigname2bp_nparray_list_path)
    if os.path.exists(data_output_path) is False:
        os.mkdir(data_output_path)
    if num_workers is None:
        num_workers = psutil.cpu_count() // 10 + 2
    # num_workers=1
    contignames = list(contigname2seq.keys())
    random.shuffle(contignames)
    contignames_list = splitListEqually(contignames, num_workers)
    split_list = []
    for names in contignames_list:
        c2s = {}
        c2b = {}
        for one_name in names:
            c2s[one_name] = contigname2seq[one_name]
            c2b[one_name] = contigname2bp_array_list[one_name]
        split_list.append((c2s, c2b))

    logger.info("--> Start to generate data for training time 1.")  # len(split_list)
    save_list = sub_process_generate_data(split_list, count_kmer, min_len)
    np.save(os.path.join(data_output_path, "training_data.npy"), np.array(save_list, dtype=object), allow_pickle=True)
