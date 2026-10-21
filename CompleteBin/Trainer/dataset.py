
import os
import random
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from CompleteBin.DataProcess.data_utils import get_features_of_one_seq
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_utils import (generate_feature_mapping_reverse, random_generate_view, sampleSeqFromFasta)
from CompleteBin.DataProcess.data_utils import split_seq_randomly


logger = get_logger()

# 连续 N（gap）——用来把草图组装的记录切成可用的无 N 片段。
_N_RUN = re.compile(r"N+")


def _overlaps(intervals, start, end):
    """intervals 是按起点排好序的 (start, end) 列表；判断 [start, end) 是否与它们相交。"""
    for other_start, other_end in intervals:
        if start < other_end and other_start < end:
            return True
    return False


def plan_disjoint_test_windows(sequences,
                               length_bins,
                               per_bin,
                               max_seq_length,
                               rng,
                               budget_fraction=0.6,
                               attempts_per_window=400):
    """为**一个基因组**规划 per_bin × len(length_bins) 条**互不重叠**的测试窗口。

    为什么必须不重叠：一个基因组要抽几百条片段，如果允许重叠，抽出来的窗口会大面积
    互相覆盖，最近邻基本就是"同一个位置附近的那条"。这样算出来的 R@1 / Pair AUPRC
    是被人为抬高的（实测第一版 500 条/基因组时，同基因组片段的 4-mer 余弦中位数
    接近 1，指标直接饱和到 0.99）。

    做法：
      1. 按长度档从短到长各放 per_bin 条（短窗口便宜，先放能保证总数凑得够）；
      2. 放不满就在最短档（768–1500 bp）继续补，直到凑够总数；
      3. 全程维护每条原始记录上已占用的区间，新窗口与它们相交就重抽；
      4. 总占用不超过该基因组可用序列的 budget_fraction（默认 60%），
         避免在剩余空隙里做无用的重试。

    返回 (plan, stats)：
      plan   —— [(bin_index, sequence_index, start, end), ...]，长度 = 目标总数
      stats  —— {"placed_per_bin": [...], "used_bases": int, "genome_bases": int}
    放不下时抛 ValueError（调用方据此换一个基因组）。
    """
    bins = len(length_bins)
    total_target = per_bin * bins
    lengths = [len(seq) for seq in sequences]
    genome_bases = sum(lengths)
    if not sequences or genome_bases == 0:
        raise ValueError("genome has no sequence")
    intervals = [[] for _ in sequences]
    used_bases = 0
    budget = int(budget_fraction * genome_bases)
    plan = []
    placed_per_bin = [0] * bins

    def try_place(bin_index):
        nonlocal used_bases
        lower, upper = length_bins[bin_index]
        upper = min(upper, max_seq_length)
        for _ in range(attempts_per_window):
            index = rng.choices(range(len(sequences)), weights=lengths, k=1)[0]
            sequence_length = lengths[index]
            if sequence_length < lower:
                continue
            length = int(round(math.exp(rng.uniform(
                math.log(lower), math.log(min(upper, sequence_length))))))
            if used_bases + length > budget:
                return None
            start = rng.randint(0, sequence_length - length)
            end = start + length
            if _overlaps(intervals[index], start, end):
                continue
            return index, start, end, length
        return None

    def commit(bin_index, placement):
        nonlocal used_bases
        index, start, end, length = placement
        intervals[index].append((start, end))
        intervals[index].sort()
        used_bases += length
        plan.append((bin_index, index, start, end))
        placed_per_bin[bin_index] += 1

    # 第一轮：短档 → 长档，各放 per_bin 条
    for bin_index in range(bins):
        for _ in range(per_bin):
            placement = try_place(bin_index)
            if placement is None:
                break
            commit(bin_index, placement)

    # 第二轮：凑不够就用最短档补（最短窗口最容易塞进空隙）
    while len(plan) < total_target:
        placement = try_place(0)
        if placement is None:
            break
        commit(0, placement)

    if len(plan) < total_target:
        raise ValueError(
            f"只能在 {genome_bases} bp 上放下 {len(plan)} 条互不重叠的测试片段，"
            f"少于要求的 {total_target} 条（per_bin={per_bin}，{bins} 个长度档）")
    return plan, {"placed_per_bin": placed_per_bin,
                  "used_bases": int(used_bases),
                  "genome_bases": int(genome_bases)}


class GenomeRoundBatchSampler(Sampler):
    """Batch genomes without replacement within each independent sampling round."""

    def __init__(self, num_genomes, rounds, batch_size, seed=2048):
        if num_genomes < 1 or rounds < 1 or batch_size < 1:
            raise ValueError("num_genomes, rounds and batch_size must be positive")
        self.num_genomes = num_genomes
        self.rounds = rounds
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        for round_index in range(self.rounds):
            genome_indices = list(range(self.num_genomes))
            random.Random(
                self.seed + self.epoch * 1000003 + round_index * 104729
            ).shuffle(genome_indices)
            offset = round_index * self.num_genomes
            for start in range(0, self.num_genomes, self.batch_size):
                yield [
                    offset + genome_index
                    for genome_index in genome_indices[start:start + self.batch_size]
                ]

    def __len__(self):
        return self.rounds * math.ceil(self.num_genomes / self.batch_size)


def _fasta_has_eligible_sequence(path, min_length=768, max_n_fraction=0.05):
    """Return whether a FASTA contains a sequence passing source filtering."""
    length = 0
    n_count = 0

    def eligible():
        return length >= min_length and n_count / float(length) <= max_n_fraction

    # 快路径：逐条记录流式判断，命中就立刻返回（15329 个文件要靠它把预检压到几分钟）。
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if length and eligible():
                    return True
                length = 0
                n_count = 0
            else:
                sequence_line = line.upper()
                length += len(sequence_line)
                n_count += sequence_line.count("N")
    if length and eligible():
        return True

    # 慢路径（兜底）：整条记录 N 太多时按连续 N 切开，看能不能切出够长的无 N 片段。
    # 库里有一批草图组装把 gap 写成一长串 N，整个文件就一条 4.5–7 Mb 的记录、
    # N 比例 5.5%–37.7%，按"整条记录 N<=5%"会被直接丢掉——但它内部其实有大段
    # 可用序列。切开后它们仍能作为训练类别，而不是被整条排除。
    with open(path, "r", encoding="utf-8") as handle:
        sequence = []
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(">"):
                continue
            sequence.append(line.upper())
    return any(len(piece) >= min_length
               for piece in _N_RUN.split("".join(sequence)))


class TrainingDataset(Dataset):

    def __init__(self,
                 data,
                 data_name,
                 n_views: int,
                 min_contig_len: int,
                 count_kmer: int,
                 split_parts_list,
                 N50,
                 batch_size,
                 train_valid_test: str = None,
                 dropout_p=0.2) -> None:
        super().__init__()
        self.n_view = n_views
        assert n_views >= 3, ValueError(f"The number of n_views must bigger or equal with 3.")
        self.N50 = N50
        self.min_contig_len = min_contig_len
        if train_valid_test.lower() == "train":
            logger.info(f"--> The min contig length for training is {self.min_contig_len}.")
        self.train_valid_test = train_valid_test
        self.count_kmer_dict_rev, self.count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)
        self.count_kmer = count_kmer
        self.split_parts_list = split_parts_list
        self.data = data
        self.data_name = data_name
        self.dropout_p = dropout_p  # 0.15

        # ensure
        if train_valid_test.lower() == "train" and len(data) % batch_size != 0:
            gaps = len(data) // batch_size * batch_size
            assert gaps != 0, ValueError(
                f"The number of contigs is {len(data)}, but the batch size is {batch_size}. It can not consist a batch for training.")
            new_data = []
            for i, item in enumerate(self.data):
                # progressBar(i, N)
                (ori_seq, cov_bp_array_list, cur_seq_rad_tokens,
                 cur_mean_tokens, cur_std_tokens,
                 cur_whole_bp_cov_tnf_array) = item
                new_data.append((ori_seq, cov_bp_array_list, cur_seq_rad_tokens,
                                 cur_mean_tokens, cur_std_tokens,
                                 cur_whole_bp_cov_tnf_array,
                                 self.data_name[i], len(ori_seq)))
            new_data = list(sorted(new_data, key=lambda x: x[-1], reverse=True))
            gap_data = []
            gap_data_name = []
            # new_data 每一行是 8 元组：
            #   (ori_seq, cov_bp_array_list, seq_rad_tokens, mean_tokens,
            #    std_tokens, whole_bp_cov_tnf_array, contig_name, contig_length)
            # 前 6 项正好是 TrainingDataset.__getitem__ 要解的 6 元组，第 7 项是
            # 名字、第 8 项（-1）是长度。（白化删除后特征少了一项，这里跟着改。）
            for i in range(gaps):
                if i < 3:
                    # new_data 每行是 8 元组：
                    #   [0] ori_seq              [1] cov_bp_array_list
                    #   [2] seq_rad_tokens       [3] mean_tokens
                    #   [4] std_tokens           [5] whole_bp_cov_tnf_array
                    #   [6] contig_name          [7] contig_length
                    # 只打印形状和取值范围；把整条数组 dump 出来会刷屏（
                    # whole_bp_cov_tnf_array 的 136 维尤其长）。
                    cur_mean_tokens = new_data[i][3]
                    cur_std_tokens = new_data[i][4]
                    logger.info(
                        f"--> The top {i + 1} contig length for training is {new_data[i][-1]}. "
                        f"mean_tokens{tuple(np.shape(cur_mean_tokens))} "
                        f"range=[{float(np.min(cur_mean_tokens)):.3f}, {float(np.max(cur_mean_tokens)):.3f}], "
                        f"std_tokens{tuple(np.shape(cur_std_tokens))} "
                        f"range=[{float(np.min(cur_std_tokens)):.3f}, {float(np.max(cur_std_tokens)):.3f}], "
                        f"whole_bp_cov_tnf_array{tuple(np.shape(new_data[i][5]))}")
                gap_data.append(tuple(new_data[i][0: 6]))
                gap_data_name.append(new_data[i][6])
            logger.info(f"--> There are {len(gap_data)} contigs for training.")
            self.data = gap_data
            self.data_name = gap_data_name

    def __len__(self):
        return len(self.data)

    def generate_view(self, seq, cov_bp_array_list, seed=None):
        cur_view_seq, start_i, end_i = random_generate_view(seq, min_contig_len=self.min_contig_len, seed=seed)
        # if seed is None and random.random() <= 0.5:
        #     cur_view_seq = sequence_data_augmentation(cur_view_seq)
        # bp array
        cur_bp_array_list = cov_bp_array_list[:, start_i: end_i]
        return get_features_of_one_seq(cur_view_seq,
                                       cur_bp_array_list,
                                       self.count_kmer,
                                       self.count_kmer_dict_rev,
                                       self.count_nr_feature_rev,
                                       self.split_parts_list,
                                       self.min_contig_len)

    def get_view(self, sub_seq, sub_cov_bp_array_list):
        return get_features_of_one_seq(sub_seq,
                                       sub_cov_bp_array_list,
                                       self.count_kmer,
                                       self.count_kmer_dict_rev,
                                       self.count_nr_feature_rev,
                                       self.split_parts_list,
                                       self.min_contig_len)
    # 每条 view 是一个 4 元组：
    #   seq_rad_tokens          [6, 136]  4-mer 组成向量（白化已删除）
    #   mean_tokens             [6, B]
    #   std_tokens              [6, B]
    #   whole_bp_cov_tnf_array  [B, 136]
    # whole bp_ cov_ tnf_array shape: (2, 136)

    def __getitem__(self, index):
        # [(deepurify_nparray, seq_kmer_freq, cov_kmer_freq, mean_val, std_val), ...,
        # (deepurify_nparray, seq_kmer_freq, cov_kmer_freq, mean_val, std_val)]
        (ori_seq, cov_bp_array_list, seq_rad_tokens,
         mean_tokens, std_tokens, whole_bp_cov_tnf_array) = self.data[index]
        ori_view_tuple = (seq_rad_tokens,
                          mean_tokens, std_tokens, whole_bp_cov_tnf_array)
        l, _ = seq_rad_tokens.shape
        # whole_bp_cov_tnf_array_shape = whole_bp_cov_tnf_array.shape
        # for differnet cases
        if self.train_valid_test.lower() == "train":
            # generate n_views of original sequence for contrastiv learning
            n_view_list = [ori_view_tuple]
            n_views_info = split_seq_randomly(ori_seq, self.min_contig_len, n_views=self.n_view - 1)
            # n_views_info.append(generate_view_with_fixed_len(ori_seq, self.min_contig_len + 20))
            for sub_seq, start_i, end_i in n_views_info:
                (cur_seq_rad_tokens, cur_mean_tokens,
                 cur_std_tokens, cur_whole_bp_cov_tnf_array) = \
                    self.get_view(sub_seq, cov_bp_array_list[:, start_i: end_i])
                if random.random() < 0.25:
                    mask = np.array(np.random.random(size=[l, 1]) > self.dropout_p, dtype=np.float32)
                    cur_seq_rad_tokens = cur_seq_rad_tokens * mask
                    cur_mean_tokens = cur_mean_tokens * mask
                    cur_std_tokens = cur_std_tokens * mask
                n_view_list.append((cur_seq_rad_tokens,
                                    cur_mean_tokens, cur_std_tokens,
                                    cur_whole_bp_cov_tnf_array))
            return n_view_list
        elif self.train_valid_test.lower() == "valid":
            return [ori_view_tuple] + [self.generate_view(ori_seq, cov_bp_array_list, index),
                                       self.generate_view(ori_seq, cov_bp_array_list, index + 1)], self.data_name[index]
        else:
            return [ori_view_tuple], self.data_name[index]


class PretrainDataset(Dataset):

    # Train-ref（train_eval 模式）的评测口径：每个训练基因组、每个长度档抽几条
    # 查询片段。5 档 × 2 条 = 每个基因组 10 条查询，各长度档等权。
    TRAIN_EVAL_PER_BIN = 2

    def __init__(self,
                 genome_records,
                 count_kmer: int,
                 split_parts_list: List,
                 mode: str,
                 min_contig_len: int,
                 max_seq_length: int = 50000,
                 expand_ratio: int = 40,
                 fixed_seed: int = 2048,
                 max_sampling_retries: int = 20,
                 n_views: int = 6,
                 test_contigs_per_bin: int = 1):
        if mode not in {"train", "test", "train_eval"}:
            raise ValueError(f"unsupported pretrain dataset mode: {mode}")
        if int(test_contigs_per_bin) < 1:
            raise ValueError(
                f"test_contigs_per_bin must be >= 1, got {test_contigs_per_bin}")
        if n_views % 2 != 0 or n_views < 2:
            raise ValueError(f"n_views must be an even number >= 2, got {n_views}")
        self.genome_data = list(genome_records)
        if not self.genome_data:
            raise ValueError(f"no genome records supplied for mode={mode}")
        self.mode = mode
        self.min_contig_len = min_contig_len
        self.max_seq_length = max_seq_length
        self.split_parts_list = split_parts_list
        self.count_kmer = count_kmer
        self.expand_ratio = expand_ratio if mode == "train" else 1
        self.fixed_seed = fixed_seed
        self.max_sampling_retries = max_sampling_retries
        self.n_views = n_views
        # test 模式下每个基因组、每个长度档抽几条片段。测试集基因组少的时候
        # （例如只留 10 个），把这里调大才能把"同基因组配对"的正例数量堆起来，
        # 否则 AUPRC / R@1 这些指标会被几个正例的噪声主导。这些片段**互不重叠**，
        # 规划一次后缓存（见 plan_disjoint_test_windows）。
        self.test_contigs_per_bin = int(test_contigs_per_bin)
        self._test_plan_cache = {}
        self.count_kmer_dict_rev, self.count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)

        # 长度档：训练时按"每档等概率"抽（每档各 1/5），档内在该档区间上抽长度；
        # 同时用于给片段打长度标签（日志/统计）和给 test 模式指定档位。
        self.length_bins = (
            (768, 1500),
            (1500, 2500),
            (2500, 5000),
            (5000, 10000),
            (10000, max_seq_length),
        )

    @staticmethod
    def _read_fasta_sequences(path):
        sequences = []
        current = []
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current:
                        sequences.append("".join(current).upper())
                    current = []
                else:
                    current.append(line)
        if current:
            sequences.append("".join(current).upper())
        eligible = [
            seq for seq in sequences
            if len(seq) >= 768 and seq.count("N") / float(len(seq)) <= 0.05
        ]
        if not eligible:
            # 兜底，和 _fasta_has_eligible_sequence 的慢路径一致：整条记录 N 太多时
            # 按连续 N 切成无 N 片段，避免把这类草图组装整个丢掉（采样本来就不会
            # 跨记录边界，切开和原来的约束是一致的）。
            for seq in sequences:
                eligible.extend(piece for piece in _N_RUN.split(seq) if len(piece) >= 768)
        # Keep all qualified reference scaffolds. Sampling chooses among them
        # by length and never crosses an original FASTA-record boundary.
        return eligible

    @staticmethod
    def _log_uniform_integer(rng, lower, upper):
        if upper <= lower:
            return int(lower)
        return int(round(math.exp(rng.uniform(math.log(lower), math.log(upper)))))

    def _sample_contig(self, sequences, bin_index, rng, avoid=None):
        lower, upper = self.length_bins[bin_index]
        upper = min(upper, self.max_seq_length)
        candidates = [i for i, seq in enumerate(sequences) if len(seq) >= lower]

        # A requested long bin may be unavailable for a fragmented genome.
        # Fall back through shorter bins, as specified in the coding plan.
        fallback_bin = bin_index
        while not candidates and fallback_bin > 0:
            fallback_bin -= 1
            lower, upper = self.length_bins[fallback_bin]
            upper = min(upper, self.max_seq_length)
            candidates = [i for i, seq in enumerate(sequences) if len(seq) >= lower]
        if not candidates:
            raise ValueError("genome has no sequence of at least 768 bp")

        last_sample = None
        for _ in range(self.max_sampling_retries):
            sequence_index = rng.choices(
                candidates,
                weights=[len(sequences[i]) for i in candidates],
                k=1,
            )[0]
            sequence = sequences[sequence_index]
            effective_upper = min(upper, len(sequence))
            length = self._log_uniform_integer(rng, lower, effective_upper)
            start = rng.randint(0, len(sequence) - length)
            end = start + length
            sample = sequence[start:end]
            last_sample = (sample, fallback_bin, (sequence_index, start, end))

            if sample.count("N") / float(len(sample)) > 0.05:
                continue
            if avoid is not None and sequence_index == avoid[0]:
                overlap = max(0, min(end, avoid[2]) - max(start, avoid[1]))
                if overlap > 0:
                    continue
            return last_sample

        if last_sample is not None and last_sample[0].count("N") / float(len(last_sample[0])) <= 0.05:
            # Non-overlap is preferred, not mandatory when the genome cannot
            # provide two disjoint fragments of the requested lengths.
            return last_sample

        raise ValueError(
            f"failed to sample a sequence with <=5% N after "
            f"{self.max_sampling_retries} attempts"
        )

    def _sample_contig_any_length(self, sequences, rng, avoid=None, length_sampling="uniform"):
        """Sample one fragment over the whole 768–max_seq_length range.

        Mirrors :meth:`_sample_contig`, but the length bounds come from the
        configured minimum and maximum instead of a length bin. The
        train-reference metric uses this.

        length_sampling="uniform"     长度在 [768, 上限] 上均匀抽（默认）：
                                      各长度区间的抽中概率与区间宽度成正比，长片段占多数。
        length_sampling="log_uniform" 旧口径，按对数均匀抽：每个数量级等概率，
                                      短片段占比高得多（768–1500 bp 约 16%）。
        length_sampling="upstream"    上游 CompleteBin 预训练脚本的口径：
                                       50% 在 [768, 上限] 均匀、40% 在 [2000, 上限] 均匀、
                                       10% 在 [768, 2000] 均匀（长片段占 ~74%）。

        avoid: 同一来源序列上已经抽过的区间 (序列号, 起点, 终点)，给了就尽量不再重叠。
        """
        lower = self.length_bins[0][0]
        upper = self.max_seq_length
        candidates = [i for i, seq in enumerate(sequences) if len(seq) >= lower]
        if not candidates:
            raise ValueError("genome has no sequence of at least 768 bp")

        last_sample = None
        for _ in range(self.max_sampling_retries):
            sequence_index = rng.choices(
                candidates,
                weights=[len(sequences[i]) for i in candidates],
                k=1,
            )[0]
            sequence = sequences[sequence_index]
            effective_upper = min(upper, len(sequence))
            if length_sampling == "uniform":
                length = rng.randint(lower, effective_upper)
            elif length_sampling == "upstream":
                roll = rng.random()
                if roll <= 0.5:
                    length = rng.randint(lower, effective_upper)
                elif roll <= 0.6:
                    length = rng.randint(lower, min(2000, effective_upper))
                else:
                    length = rng.randint(min(2000, effective_upper), effective_upper)
            else:
                length = self._log_uniform_integer(rng, lower, effective_upper)
            start = rng.randint(0, len(sequence) - length)
            end = start + length
            sample = sequence[start:end]
            actual_bin = next(
                (i for i, (_, bin_upper) in enumerate(self.length_bins) if length < bin_upper),
                len(self.length_bins) - 1,
            )
            last_sample = (sample, actual_bin, (sequence_index, start, end))

            if sample.count("N") / float(len(sample)) > 0.05:
                continue
            if avoid is not None and sequence_index == avoid[0]:
                overlap = max(0, min(end, avoid[2]) - max(start, avoid[1]))
                if overlap > 0:
                    continue
            return last_sample

        if last_sample is not None and last_sample[0].count("N") / float(len(last_sample[0])) <= 0.05:
            return last_sample

        raise ValueError(
            f"failed to sample a sequence with <=5% N after "
            f"{self.max_sampling_retries} attempts"
        )

    def _pair_type(self, length_1: int, length_2: int) -> str:
        """按两条片段的实际长度给这一对起个名字，只用于采样统计（日志用）。

        训练不再按固定比例构造这三种对，名字是从抽出来的长度反推的：
        短 = 落在 768–2500 bp（前两个长度档），长 = 2500 bp 以上。
        """
        short_upper = self.length_bins[1][1]
        short_1 = length_1 < short_upper
        short_2 = length_2 < short_upper
        if short_1 and short_2:
            return "short-short"
        if short_1 or short_2:
            return "short-anchor"
        return "long-long"

    def _tokens(self, sequence):
        # 预训练只用原始 136 维那一流，且强制不白化：
        # 白化器是在别处（binning 的样本）拟合的，基不同，预训练不能用。
        # 白化删除后 get_features_of_one_seq 返回 4 元组：
        # (seq_rad_tokens, mean_tokens, std_tokens, whole_bp_cov_tnf_array)
        # 预训练只用第一条（seq_rad_tokens）。
        tokens, _, _, _ = get_features_of_one_seq(
            sequence,
            None,
            self.count_kmer,
            self.count_kmer_dict_rev,
            self.count_nr_feature_rev,
            self.split_parts_list,
            self.min_contig_len,
        )
        return tokens.astype(np.float32, copy=False)

    def _test_sequences(self, genome_index):
        """test 模式下该基因组切好的可用序列（顺带触发规划缓存）。"""
        self._get_test_plan(genome_index)
        return self._test_plan_cache[genome_index][0]

    def _get_test_plan(self, genome_index):
        """返回 [(bin, sequence_index, start, end), ...]；第一次调用时规划并缓存。

        顺带把"每个 item 重读一遍 FASTA"也修掉了：规划时读一次，之后 200 条片段
        都从缓存里切。
        """
        cached = self._test_plan_cache.get(genome_index)
        if cached is not None:
            return cached[1]
        record = self.genome_data[genome_index]
        sequences = self._read_fasta_sequences(record["path"])
        if not sequences:
            raise ValueError(
                f"no >=768 bp sequence with <=5% N in {record['path']}")
        rng = random.Random(self.fixed_seed * 1000003 + genome_index * 104729)
        plan, stats = plan_disjoint_test_windows(
            sequences, self.length_bins, self.test_contigs_per_bin,
            self.max_seq_length, rng)
        logger.info(
            f"--> [test-plan] {os.path.basename(record['path'])}: "
            f"{len(plan)} 条互不重叠测试片段，各档 {stats['placed_per_bin']}，"
            f"占用 {stats['used_bases'] / 1e6:.2f} / "
            f"{stats['genome_bases'] / 1e6:.2f} Mb")
        self._test_plan_cache[genome_index] = (sequences, plan)
        return plan

    def __len__(self):
        if self.mode == "test":
            return (len(self.genome_data) * len(self.length_bins)
                    * self.test_contigs_per_bin)
        if self.mode == "train_eval":
            return len(self.genome_data) * self.train_eval_queries_per_genome
        return len(self.genome_data) * self.expand_ratio

    @property
    def train_eval_queries_per_genome(self) -> int:
        return len(self.length_bins) * self.TRAIN_EVAL_PER_BIN

    def __getitem__(self, index):
        if self.mode == "test":
            per_genome = len(self.length_bins) * self.test_contigs_per_bin
            genome_index = index // per_genome
            slot = index % per_genome
            plan = self._get_test_plan(genome_index)
            actual_bin, sequence_index, start, end = plan[slot]
            sequence = self._test_sequences(genome_index)[sequence_index][start:end]
            return self._tokens(sequence), genome_index, actual_bin
        elif self.mode == "train_eval":
            genome_index = index // self.train_eval_queries_per_genome
            requested_bin = ((index % self.train_eval_queries_per_genome)
                             // self.TRAIN_EVAL_PER_BIN)
        else:
            genome_index = index % len(self.genome_data)
            requested_bin = None

        record = self.genome_data[genome_index]
        sequences = self._read_fasta_sequences(record["path"])
        if not sequences:
            raise ValueError(
                f"no >=768 bp sequence with <=5% N in {record['path']}"
            )

        if self.mode == "train":
            rng = random
            view_tokens = []
            view_bins = []
            pair_types = []
            # 每个基因组抽 n_views 条片段，按 n_views/2 对组织；同一对的两条尽量
            # 不重叠。长度档**每档等概率**（5 档各 1/5），档内再抽具体长度。
            # 旧做法是 40% 双短 / 50% 一短一长 / 10% 双长，再按档内权重（35:30 /
            # 15:10:10）选档，短片段被过采样到 65%；那套固定比例已经删除。
            for _ in range(self.n_views // 2):
                bin_1 = rng.randrange(len(self.length_bins))
                bin_2 = rng.randrange(len(self.length_bins))
                try:
                    seq_1, actual_bin_1, location_1 = self._sample_contig(
                        sequences, bin_1, rng)
                    seq_2, actual_bin_2, _ = self._sample_contig(
                        sequences, bin_2, rng, avoid=location_1)
                except ValueError as error:
                    raise ValueError(f"{record['path']}: {error}") from error
                view_tokens.append(self._tokens(seq_1))
                view_tokens.append(self._tokens(seq_2))
                view_bins.extend([actual_bin_1, actual_bin_2])
                pair_types.append(self._pair_type(len(seq_1), len(seq_2)))
            return (*view_tokens, int(record["class_id"]), *view_bins, pair_types)

        rng = random.Random(self.fixed_seed + index * 104729)
        # Train-ref 分类评测：每个训练基因组在每个长度档抽 TRAIN_EVAL_PER_BIN 条
        # 查询片段（默认 5 档 × 2 条 = 10 条/基因组）。查询种子由 index 决定，
        # 所以每个 epoch 考的是同一批片段，曲线可比。
        # 注意：如果某个基因组没有足够长的序列，_sample_contig 会向较短的档回退，
        # 返回的 actual_bin 就是实际落到的档（可能和 requested_bin 不同）。
        try:
            sequence, actual_bin, _ = self._sample_contig(sequences, requested_bin, rng)
        except ValueError as error:
            raise ValueError(f"{record['path']}: {error}") from error
        return self._tokens(sequence), int(record["class_id"]), actual_bin


def collate_pretrain_train(batch):
    """Collate train-mode PretrainDataset samples.

    Each sample is (n_views token arrays [6, kmer_width], label int,
    n_views bin ints, list of n_views/2 pair-type strings).  Returns a tuple
    (n_views stacked tensors [B, 6, kmer_width], labels [B], n_views bin
    tensors [B], flat pair-type list).
    """
    n_views = (len(batch[0]) - 2) // 2
    tokens = [torch.stack([torch.as_tensor(item[i]) for item in batch]) for i in range(n_views)]
    labels = torch.tensor([item[n_views] for item in batch], dtype=torch.long)
    bins = [torch.tensor([item[n_views + 1 + i] for item in batch], dtype=torch.long)
            for i in range(n_views)]
    pair_types = [pair_type for item in batch for pair_type in item[-1]]
    return (*tokens, labels, *bins, pair_types)


def build_or_load_pretrain_manifest(genome_db_folder_path,
                                    manifest_path,
                                    train_fraction=0.90,
                                    split_seed=2048,
                                    train_count=None,
                                    test_count=None,
                                    test_selection="random"):
    """Create a stable genome-level split and global training class mapping.

    train_count: 直接指定训练用的参考基因组数量，其余全部作 test。
                 给 None 时退回按 train_fraction 切分。
    test_count:  直接指定 **test** 用的基因组数量。给了它就忽略 train_count /
                 train_fraction：留出的这 test_count 个当测试集，其余全部训练。
    test_selection: `random`（随机抽）或 `far`（用 4-mer 组成向量挑"离其他基因组
                 最远"的那几个，见 far_test_selection.py）。只有给 test_count 时
                 才有意义。
    """
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("version", 1) >= 3:
            expected_database = os.path.abspath(genome_db_folder_path)
            if manifest.get("genome_db_folder_path") != expected_database:
                raise ValueError(
                    f"manifest database mismatch: {manifest.get('genome_db_folder_path')} "
                    f"!= {expected_database}"
                )
            if test_count is not None and len(manifest.get("test", [])) != int(test_count):
                raise ValueError(
                    f"manifest {manifest_path} holds out {len(manifest.get('test', []))} "
                    f"test genomes but test_count={int(test_count)} was requested; delete "
                    "that manifest (or point the output directory elsewhere) to rebuild."
                )
            if (test_count is None and train_count is not None
                    and len(manifest.get("train", [])) != int(train_count)):
                raise ValueError(
                    f"manifest {manifest_path} has {len(manifest.get('train', []))} training "
                    f"genomes but train_count={int(train_count)} was requested; delete that "
                    f"manifest (or point the output directory elsewhere) to rebuild the split."
                )
            # 只要 manifest 记录了挑选方式，就必须和这次请求的一致（不只在传了
            # test_count 的时候才查）——否则 --test-selection 换了会被静默忽略。
            recorded_selection = manifest.get("test_selection")
            if recorded_selection is not None and recorded_selection != test_selection:
                raise ValueError(
                    f"manifest {manifest_path} was built with test_selection="
                    f"{recorded_selection!r} but {test_selection!r} was requested; delete that "
                    "manifest (or point the output directory elsewhere) to rebuild."
                )
            class_ids = [record.get("class_id") for record in manifest.get("train", [])]
            if class_ids != list(range(len(class_ids))):
                raise ValueError("manifest training class IDs are not contiguous and ordered")
            missing_paths = [
                record["path"] for split in ("train", "test")
                for record in manifest.get(split, []) if not os.path.isfile(record["path"])
            ]
            if missing_paths:
                raise FileNotFoundError(f"manifest genome path is missing: {missing_paths[0]}")
            return manifest
        logger.info(
            f"--> Rebuilding version-{manifest.get('version', 1)} manifest "
            "(needs version 3: test_count / test_selection support)."
        )

    suffixes = (".fa", ".fna", ".fasta", ".fas")
    file_names = sorted(
        name for name in os.listdir(genome_db_folder_path)
        if name.lower().endswith(suffixes)
        and os.path.isfile(os.path.join(genome_db_folder_path, name))
    )
    if len(file_names) < 2:
        raise ValueError("at least two reference genome FASTA files are required")

    paths = [os.path.abspath(os.path.join(genome_db_folder_path, name)) for name in file_names]
    logger.info(f"--> Prechecking {len(paths)} FASTA files for eligible reference sequences.")
    with ThreadPoolExecutor(max_workers=16) as executor:
        eligibility = list(executor.map(_fasta_has_eligible_sequence, paths))
    excluded_names = [name for name, keep in zip(file_names, eligibility) if not keep]
    file_names = [name for name, keep in zip(file_names, eligibility) if keep]
    if len(file_names) < 2:
        raise ValueError("fewer than two genomes remain after 5% N source filtering")
    logger.info(
        f"--> Source filtering retained {len(file_names)} genomes and excluded "
        f"{len(excluded_names)} genomes."
    )

    far_report = None
    if test_count is not None:
        holdout = min(len(file_names) - 1, max(1, int(test_count)))
        if test_selection == "far":
            from CompleteBin.Trainer.far_test_selection import select_far_test_genomes
            held_paths, far_report = select_far_test_genomes(
                [os.path.join(genome_db_folder_path, name) for name in file_names],
                holdout, logger=logger)
            test_names = {os.path.basename(path) for path in held_paths}
            logger.info(
                f"--> Genome split: test={len(test_names)} 个「远离其他基因组」的基因组, "
                f"train={len(file_names) - len(test_names)}."
            )
        elif test_selection == "random":
            shuffled = list(file_names)
            random.Random(split_seed).shuffle(shuffled)
            test_names = set(shuffled[:holdout])
            logger.info(
                f"--> Genome split: test={len(test_names)} (random, split_seed={split_seed}), "
                f"train={len(file_names) - len(test_names)}."
            )
        else:
            raise ValueError(
                f"unknown test_selection={test_selection!r}; use 'random' or 'far'")
        train_size = len(file_names) - len(test_names)
    else:
        test_names = None
        far_report = None
        shuffled = list(file_names)
        random.Random(split_seed).shuffle(shuffled)
        if train_count is not None:
            train_size = min(len(shuffled) - 1, max(1, int(train_count)))
        else:
            train_size = min(len(shuffled) - 1, max(1, int(len(shuffled) * train_fraction)))
        train_names = set(shuffled[:train_size])
        logger.info(
            f"--> Genome split: train={train_size}, test={len(file_names) - train_size} "
            f"(split_seed={split_seed}, train_count={train_count})."
        )
    if test_names is not None:
        train_names = set(file_names) - test_names

    train_records = []
    test_records = []
    for name in file_names:
        record = {
            "genome_name": name,
            "path": os.path.abspath(os.path.join(genome_db_folder_path, name)),
            "split": "train" if name in train_names else "test",
        }
        if name in train_names:
            record["class_id"] = len(train_records)
            train_records.append(record)
        else:
            record["class_id"] = None
            test_records.append(record)

    manifest = {
        "version": 3,
        "genome_db_folder_path": os.path.abspath(genome_db_folder_path),
        "split_seed": split_seed,
        # 记实际切出来的比例，避免 train_count 指定时这里的 0.9 和真实划分对不上
        "train_fraction": train_size / float(len(file_names)),
        "train_fraction_arg": train_fraction,
        "test_selection": test_selection if test_count is not None else None,
        "test_selection_report": far_report,
        "train_genomes": train_size,
        "test_genomes": len(file_names) - train_size,
        "train": train_records,
        "test": test_records,
        "excluded": [
            {"genome_name": name, "reason": "no sequence >=768 bp with N <=5%"}
            for name in excluded_names
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


# def mask_view(self, view_tuple, l, cov_shape):
#     if random.random() < 0.5:
#         mask_dim = 1
#     else:
#         mask_dim = self.count_nr_feature_rev
#     mask = np.array(np.random.random(size=[l, mask_dim]) > self.dropout_p, dtype=np.float32)
#     bp_shape = view_tuple[3].shape
#     if random.random() < 0.5:
#         bp_aug = view_tuple[3] * np.array(np.random.random(size=bp_shape) > self.dropout_p, dtype=np.float32)
#     else:
#         bp_aug = np.clip(np.random.randn(*bp_shape) * 0.001 + view_tuple[3], a_min=0, a_max=None)
#     # print(view_tuple[0], mask, view_tuple[0] * mask)
#     mask_view = (view_tuple[0] * mask,
#                 np.clip(np.random.randn(*cov_shape) * 0.01 + view_tuple[1], a_min=0, a_max=None),
#                 np.clip(np.random.randn(*cov_shape) * 0.01 + view_tuple[2], a_min=0, a_max=None),
#                 bp_aug)
#     return mask_view


# def generate_view(self, seq, cov_bp_array_list, seed=None):
# cur_view_seq, start_i, end_i = random_generate_view(seq, min_contig_len=self.min_contig_len, seed=seed)
# cur_bp_array_list = cov_bp_array_list[:, start_i: end_i]
# cur_seq_3mer_tokens, cur_seq_rad_tokens, cur_mean_tokens, cur_std_tokens, cur_whole_bp_cov_tnf_array = get_features_of_one_seq(
#     cur_view_seq, cur_bp_array_list,
#     self.count_3mer_nr_features, self.count_3mer_dict,
#     self.count_kmer, self.count_kmer_dict_rev, self.count_nr_feature_rev,
#     self.split_parts_list,
#     self.min_contig_len)
# return cur_seq_3mer_tokens, cur_seq_rad_tokens, cur_mean_tokens, cur_std_tokens, cur_whole_bp_cov_tnf_array
