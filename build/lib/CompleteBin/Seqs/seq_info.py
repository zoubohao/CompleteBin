
import os
from typing import List
import multiprocessing

import numpy as np
import psutil


from CompleteBin.IO import readFasta, readPickle, writePickle
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_utils import base_pair_coverage_calculate
from CompleteBin.Seqs.generate_seed import gen_seed

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
    contig_file_path: str,
    sorted_bam_file_list: List[str],
    temp_file_folder_path: str,
    min_contig_length: int,
    hmm_40_model_path,
    marker_perl_path,
    num_workers = None
):
    contigname2seq = readFasta(contig_file_path)
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
    p_num = 1
    if pro_num > 2:
        p_num = 2
    logger.info(f"--> {p_num} Processes start to run.")
    if os.path.exists(os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl")) is False:
        pro_list = []
        with multiprocessing.Pool(p_num) as multiprocess:
            for i, sorted_bam_file in enumerate(sorted_bam_file_list):
                if os.path.exists(os.path.join(temp_file_folder_path, f"contigname2bpcover_nparray_{i}.pkl")) is False:
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
        logger.info(f"--> Start to read the coverage information from {pro_num} bam files.")
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
                    cov_val_list[k].append(sum(cur_name2bp_array[name][75:-75]) / (len(cur_name2bp_array[name]) - 150))
                    var_val_list[k].append(np.std(cur_name2bp_array[name][75:-75], dtype=np.float32))
            name2bpcover_nparray_list[name] = np.array(cur_bp_array_list, dtype=np.float32)
        logger.info(f"--> Start to write coverage information")
        for name, bp_list in name2bpcover_nparray_list.items():
            n = len(bp_list)
            assert n == pro_num, ValueError(f"There are number of {pro_num} bam files, but contig {name} only have {n} coverage info.")
        writePickle(os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl"), name2bpcover_nparray_list)
        mean_val = np.max(np.array(cov_val_list, dtype=np.float32), axis=1, keepdims=False)
        var_val = np.max(np.array(var_val_list, dtype=np.float32), axis=1, keepdims=False)
        ##################
        ## Changed Here ##
        ##################
        # mean_val = np.log(mean_val + 1.)
        # var_val = np.log(var_val + 1.)
        logger.info(f"--> The max of coverage mean value is {mean_val}.")
        logger.info(f"--> The max coverage std value is {var_val}.")
        writePickle(os.path.join(temp_file_folder_path, "mean_var.pkl"), (mean_val, var_val))
        # writePickle(os.path.join(temp_file_folder_path, "std.pkl"), var_val)

    # gen_seed 调用已移至 binning_with_all_steps() 中，在 coverage outlier 处理之后
    # 确保种子 contig 不会落在已移除的 outlier 上

    writePickle(os.path.join(temp_file_folder_path, "contigname2seq_str.pkl"), contigname2seq)

    for i, sorted_bam_file in enumerate(sorted_bam_file_list):
        if os.path.exists(os.path.join(temp_file_folder_path, f"contigname2bpcover_nparray_{i}.pkl")):
            os.remove(os.path.join(temp_file_folder_path, f"contigname2bpcover_nparray_{i}.pkl"))


def clip_coverage_outliers(
    contigname2seq_path: str,
    contigname2bp_nparray_list_path: str,
    mean_var_path: str,
    gap_threshold_mean: float = 1.15,
    gap_threshold_std: float = 1.265,
    min_length: int = 2500,
    k_max: int = 30,
) -> dict:
    """Gap-based coverage outlier detection, removal (short) and clipping (long).

    For each BAM, sorts contigs by mean (and std) coverage descending,
    then finds the first "natural break" where the ratio c_i / c_{i+1}
    exceeds the gap threshold.  Contigs before the break are flagged as
    outliers.  Short outliers (< min_length) are removed; long outliers
    have their per-base coverage values clipped to the breakpoint value.

    After processing, rewrites the three pkl files and recomputes
    max_cov_mean / max_cov_var.

    Args:
        contigname2seq_path: path to contigname2seq_str.pkl (rewritten in-place).
        contigname2bp_nparray_list_path: path to per-base coverage pkl (rewritten in-place).
        mean_var_path: path to mean_var.pkl (rewritten in-place).
        gap_threshold_mean: minimum ratio c_i/c_{i+1} to declare a break in mean coverage.
        gap_threshold_std:  minimum ratio c_i/c_{i+1} to declare a break in std coverage.
        min_length: contigs shorter than this are candidates for removal.
        k_max: only examine the top k_max values for break detection.

    Returns:
        dict with keys:
            n_removed, n_clipped, n_std_only_protected,
            old_mean_val, new_mean_val, old_var_val, new_var_val,
            mean_clip_thresholds, std_clip_thresholds
    """
    contigname2seq = readPickle(contigname2seq_path)
    contigname2bp = readPickle(contigname2bp_nparray_list_path)

    # Detect number of BAM files
    num_bam = None
    for bp_array in contigname2bp.values():
        num_bam = bp_array.shape[0]
        break
    if num_bam is None:
        logger.warning("--> [clip] No coverage data found, skipping.")
        return {}

    # ── 1. Compute per-contig mean and std for each BAM ──
    mean_cov = {}   # mean_cov[name][j]
    std_cov = {}    # std_cov[name][j]
    for name, bp_array in contigname2bp.items():
        seq_len = bp_array.shape[1]
        m_vals, s_vals = [], []
        for j in range(num_bam):
            trimmed = bp_array[j, 75:-75].astype(np.float32) if seq_len > 150 else bp_array[j].astype(np.float32)
            m_vals.append(float(np.mean(trimmed)))
            s_vals.append(float(np.std(trimmed)))
        mean_cov[name] = m_vals
        std_cov[name] = s_vals

    # ── 2. Per-BAM gap-based detection ──
    mean_outlier_names = [set() for _ in range(num_bam)]
    std_outlier_names = [set() for _ in range(num_bam)]
    mean_clip_thresholds = [None] * num_bam
    std_clip_thresholds = [None] * num_bam

    for j in range(num_bam):
        # --- mean ---
        sorted_mean = sorted(mean_cov.keys(), key=lambda n: mean_cov[n][j], reverse=True)
        vals_mean = [mean_cov[n][j] for n in sorted_mean]
        top_vals = vals_mean[:k_max]
        break_idx = None
        for idx in range(len(top_vals) - 1):
            if top_vals[idx + 1] <= 0:
                break
            g = top_vals[idx] / top_vals[idx + 1]
            if g >= gap_threshold_mean:
                break_idx = idx
                break
        if break_idx is not None:
            # 回溯找到异常簇起点（不碰 break_idx，保留其"断崖位置"语义）
            start_idx = break_idx
            while start_idx > 0:
                prev_g = top_vals[start_idx - 1] / top_vals[start_idx] if top_vals[start_idx] > 0 else 999
                if prev_g >= gap_threshold_mean:
                    break
                start_idx -= 1
            # 异常簇: sorted_mean[start_idx .. break_idx]
            for k in range(start_idx, break_idx + 1):
                mean_outlier_names[j].add(sorted_mean[k])
            # 截断阈值: 断崖之后第一个非 outlier 值
            if break_idx + 1 < len(sorted_mean):
                mean_clip_thresholds[j] = mean_cov[sorted_mean[break_idx + 1]][j]

        # --- std ---
        sorted_std = sorted(std_cov.keys(), key=lambda n: std_cov[n][j], reverse=True)
        vals_std = [std_cov[n][j] for n in sorted_std]
        top_vals_s = vals_std[:k_max]
        break_idx_s = None
        for idx in range(len(top_vals_s) - 1):
            if top_vals_s[idx + 1] <= 0:
                break
            g = top_vals_s[idx] / top_vals_s[idx + 1]
            if g >= gap_threshold_std:
                break_idx_s = idx
                break
        if break_idx_s is not None:
            start_idx_s = break_idx_s
            while start_idx_s > 0:
                prev_g = top_vals_s[start_idx_s - 1] / top_vals_s[start_idx_s] if top_vals_s[start_idx_s] > 0 else 999
                if prev_g >= gap_threshold_std:
                    break
                start_idx_s -= 1
            for k in range(start_idx_s, break_idx_s + 1):
                std_outlier_names[j].add(sorted_std[k])

    # ── 3. Union across BAMs ──
    all_mean_outliers = set().union(*mean_outlier_names)
    all_std_outliers = set().union(*std_outlier_names)
    all_outliers = all_mean_outliers | all_std_outliers

    n_removed = 0
    n_clipped = 0

    # ── 4. Process each outlier ──
    names_to_remove = set()
    names_to_clip = set()       # all long outliers -> per-base clip

    for name in all_outliers:
        seq_len = len(contigname2seq[name])

        if seq_len < min_length:
            names_to_remove.add(name)
            n_removed += 1
        else:
            # 任何类型 outlier, len >= min_length: 截断 per-base 覆盖度
            # 无论被 mean 还是 std 检测到, 上冲的根因都是 per-base 中有位点覆盖度过高
            # np.clip 切掉高值同时压低 mean 和 std
            names_to_clip.add(name)
            n_clipped += 1

    # Log details
    if all_outliers:
        logger.info(f"--> [clip] Total outlier candidates: {len(all_outliers)} "
                    f"(mean={len(all_mean_outliers)}, std={len(all_std_outliers)})")
        logger.info(f"--> [clip] Removing {n_removed} short contigs (len < {min_length})")
        logger.info(f"--> [clip] Clipping {n_clipped} long contigs (per-base coverage)")
        if names_to_remove:
            logger.info("--> [clip] Removed contigs:")
            for name in sorted(names_to_remove, key=lambda n: len(contigname2seq[n]), reverse=True):
                cov_str = ", ".join([f"BAM{j}={mean_cov[name][j]:.1f}" for j in range(num_bam)])
                std_str = ", ".join([f"BAM{j}={std_cov[name][j]:.1f}" for j in range(num_bam)])
                logger.info(f"    {name}  len={len(contigname2seq[name])}  mean=[{cov_str}]  std=[{std_str}]")

    # ── 5. Remove short outliers ──
    for name in names_to_remove:
        del contigname2seq[name]
        del contigname2bp[name]

    # ── 6. Clip per-base coverage for long outliers ──
    for name in names_to_clip:
        bp_array = contigname2bp[name]
        for j in range(num_bam):
            if name in mean_outlier_names[j] and mean_clip_thresholds[j] is not None:
                bp_array[j] = np.clip(bp_array[j], 0, mean_clip_thresholds[j])
        contigname2bp[name] = bp_array

    # ── 7. Recompute max_cov_mean / max_cov_var ──
    cov_vals = [[] for _ in range(num_bam)]
    var_vals = [[] for _ in range(num_bam)]
    for name, bp_array in contigname2bp.items():
        seq_len = bp_array.shape[1]
        for j in range(num_bam):
            trimmed = bp_array[j, 75:-75].astype(np.float32) if seq_len > 150 else bp_array[j].astype(np.float32)
            cov_vals[j].append(float(np.mean(trimmed)))
            var_vals[j].append(float(np.std(trimmed)))

    new_mean_val = np.max(np.array(cov_vals, dtype=np.float32), axis=1, keepdims=False)
    new_var_val = np.max(np.array(var_vals, dtype=np.float32), axis=1, keepdims=False)

    old_mean_val, old_var_val = readPickle(mean_var_path)
    logger.info(f"--> [clip] Old max_cov_mean: {old_mean_val}  ->  New: {new_mean_val}")
    logger.info(f"--> [clip] Old max_cov_std:  {old_var_val}  ->  New: {new_var_val}")
    if mean_clip_thresholds:
        logger.info(f"--> [clip] Mean clip thresholds per BAM: {mean_clip_thresholds}")
    if std_clip_thresholds:
        logger.info(f"--> [clip] Std clip thresholds per BAM: {std_clip_thresholds}")

    # ── 8. Rewrite pkl files ──
    writePickle(contigname2seq_path, contigname2seq)
    writePickle(contigname2bp_nparray_list_path, contigname2bp)
    writePickle(mean_var_path, (new_mean_val, new_var_val))

    logger.info(f"--> [clip] Done. Removed={n_removed}, Clipped={n_clipped}")
    return dict(
        n_removed=n_removed, n_clipped=n_clipped, n_std_only_protected=0,
        old_mean_val=old_mean_val, new_mean_val=new_mean_val,
        old_var_val=old_var_val, new_var_val=new_var_val,
        mean_clip_thresholds=mean_clip_thresholds,
        std_clip_thresholds=std_clip_thresholds,
    )

