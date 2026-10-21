#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方案 H1 的实现 —— 对应 `scripts/grid_reform_run_20261001/IMPROVEMENT.md`。

与现有管线（`ensemble_clustering=True`）的区别：

    现有管线：
        第一轮 Leiden 网格(144) → 第二轮 SCG 重聚类 → Galah 共识 → contig 去重 → 第三轮 k-means

    本模块（`ensemble_clustering=False`）：
        第一轮 Leiden 网格(144) → **从 144 个里挑出唯一 1 个** → 粗筛 + 精筛 → k-means 重聚

两大部分（与 IMPROVEMENT.md 的章节一一对应）：

    §0 第一步（`select_best_partition`）
        打分 S = rank01(SCG分) + 0.5·rank01(n_pure_small) + 0.5·rank01(n_pure_all)
        其中
            SCG分        = 0.6·n5010 + 1.0·n7010 + 2.0·n9010
                         + 0.8·n505  + 1.5·n705  + 3.0·n905
                           （6 个 SCG 档位的箱数加权和，只在**箱长 ≥100 kb** 的箱上统计）
            n_pure_small = 长度∈[20kb,100kb) 且「嵌入紧致」的箱数
            n_pure_all   = 长度≥5kb        且「嵌入紧致」的箱数
            rank01(·)    = 在全部候选划分之间做秩归一化到 [0,1]
        取 S 最大的那个划分。

    §1 第二步（`filter_bins_by_purity`）
        粗筛：(箱长 ≥20 kb)  或  (contig 数 ≥3)          ← 并集，两条路径坏处互补
        精筛：(emb_spread ≤0.15) 或 (箱长 ≥200 kb) 或 (只有 1 条 contig)
        最后：把丢掉的 contig 交给 `cluster_kmeans_for_low_v2` 重聚（k 由 SCG 自适应）

参数取值与理由见 IMPROVEMENT.md §3.1；实测结果见 §1.2。

注意：本模块**不修改**任何既有文件，只被 `Binning_steps.binning_with_all_steps`
在 `ensemble_clustering=False` 时调用。
"""
import multiprocessing as mp
import os
from shutil import copy
from typing import Dict, List, Tuple

import numpy as np

from CompleteBin.Cluster.first_cluster import first_cluster
from CompleteBin.Cluster.k_means_v2 import cluster_kmeans_for_low_v2
from CompleteBin.Cluster.sec_cluster_utils import determine_domain
from CompleteBin.IO import readMarkerSets, readPickle, writeFasta
from CompleteBin.logger import get_logger

logger = get_logger()

# ─────────────────────────── 参数（IMPROVEMENT.md §3.1 的表） ───────────────────────────
EMB_SPREAD_MAX = 0.15          # 嵌入紧致度阈值（精筛）
MIN_LENGTH_COARSE = 20_000     # 粗筛的长度条件
MIN_N_CONTIGS = 3              # 粗筛的 contig 数条件
MIN_LENGTH_KEEP_BIG = 200_000  # 大箱无条件保留（防 emb_spread 的尺寸偏差误杀）

# §0 打分公式用到的三个统计口径
SCG_SCORE_FLOOR = 100_000      # SCG 的 6 个档位只在 ≥100 kb 的箱上统计
PURE_SMALL_LO = 20_000         # n_pure_small 的长度下界
PURE_SMALL_HI = 100_000        # n_pure_small 的长度上界
PURE_ALL_FLOOR = 5_000         # n_pure_all 的长度下界

# SCG 6 个档位的权重，顺序 = (n5010, n7010, n9010, n505, n705, n905)
SCG_WEIGHTS = (0.6, 1.0, 2.0, 0.8, 1.5, 3.0)
PURE_SMALL_WEIGHT = 0.5        # n_pure_small 的权重
PURE_ALL_WEIGHT = 0.5          # n_pure_all 的权重

# 重聚（交给管线自己的第三轮实现）
MIN_K_RATIO = 0.1
PERCENTILE = 10

# ── 输出精简（IMPROVEMENT.md §8.2 的「保守版」，只用嵌入紧致度）──────────────
# 之所以**不用 contig 数**做判据：contig 数是"组装产物"的属性，随组装软件/参数变化；
# 本数据集的金标准里质粒被切成 10~33 条 contig，换个组装软件拼成 1~2 条长 contig 时，
# 「contig 数 ≥5」这条规则就会把完整质粒误判成垃圾。
# 而嵌入紧致度是"基因组"的属性（同源序列的向量必然聚在一起），所以可移植。
MAG_TIGHT_BAND_LO = 20_000     # 窄带下界：<20 kb 的箱用较松的阈值
MAG_WIDE_BAND_LO = 100_000     # 宽带上界：≥100 kb 的箱不做紧致度筛选
MAG_TIGHT_EMB = 0.12           # <20 kb 段的 emb_spread 上限（该段有用箱最大到 0.118）
MAG_WIDE_EMB = 0.08            # 20~100 kb 段的 emb_spread 上限


# ─────────────────────────────────── 基础度量 ───────────────────────────────────
def embedding_spread(contigs, contigname2emb) -> float:
    """箱内各 contig 的归一化嵌入到**箱质心**的平均余弦距离（越小越纯）。

    `emb_spread(b) = mean over c in b of ( 1 - cos(emb_c, centroid_b) )`
    `centroid_b` = 归一化后的箱内嵌入均值。

    只有 1 条 contig（或全都拿不到嵌入）时返回 0.0。
    ⚠️ 这个 0.0 是**退化的**、不代表纯度高 —— 所以粗筛必须先排掉单 contig 箱，
    否则它们会全部通过阈值。这正是 `MIN_N_CONTIGS` 存在的理由（IMPROVEMENT.md §1.4.1）。
    """
    idx = [c for c in contigs if c in contigname2emb]
    if len(idx) <= 1:
        return 0.0
    X = np.stack([np.asarray(contigname2emb[c], dtype=np.float64) for c in idx])
    centroid = X.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm <= 0:
        return 0.0
    centroid /= norm
    return float(np.mean(1.0 - X @ centroid))


def scg_cells(per_bin_comp_cont) -> Tuple[int, int, int, int, int, int]:
    """把「每个箱的 (comp, cont)」汇总成 SCG 的 6 个档位计数。

    与管线 `sec_cluster.py:stat_quality` 的判据**完全一致**（边界用严格不等号，
    只有 n905 用 >= / <=）。
    """
    n5010 = n7010 = n9010 = n505 = n705 = n905 = 0
    for comp, cont in per_bin_comp_cont:
        if comp > 50 and cont < 10:
            n5010 += 1
        if comp > 70 and cont < 10:
            n7010 += 1
        if comp > 90 and cont < 10:
            n9010 += 1
        if comp > 50 and cont < 5:
            n505 += 1
        if comp > 70 and cont < 5:
            n705 += 1
        if comp >= 90 and cont <= 5:
            n905 += 1
    return n5010, n7010, n9010, n505, n705, n905


def is_pure(length: int, n_contigs: int, emb_spread: float) -> bool:
    """「嵌入紧致」的判定：`emb_spread ≤0.15`，或大箱兜底，或只有 1 条 contig。

    这个判据在两个地方用，含义不同：
      · §0 的 `n_pure_small` / `n_pure_all`（**打分**用，不删箱）
      · §1 的精筛（**筛选**用，会删箱）
    """
    return (emb_spread <= EMB_SPREAD_MAX
            or length >= MIN_LENGTH_KEEP_BIG
            or n_contigs == 1)


# ───────────────────────── §1 第二步：粗筛 + 精筛 ─────────────────────────
def filter_bins_by_purity(
    clu2contigs: Dict[str, List[str]],
    contigname2emb,
    contigname2seq,
    emb_spread_max: float = EMB_SPREAD_MAX,
    min_length_coarse: int = MIN_LENGTH_COARSE,
    min_n_contigs: int = MIN_N_CONTIGS,
    min_length_keep_big: int = MIN_LENGTH_KEEP_BIG,
) -> Tuple[Dict[str, List[str]], List[str], Dict[str, float]]:
    """按「粗筛 + 精筛」保留箱（IMPROVEMENT.md §1.1）。

    Returns
    -------
    kept    : {bin_name: [contig, ...]}   保留下来的箱
    dropped : [contig, ...]               被丢掉的 contig（交给 k-means 重聚）
    spreads : {bin_name: emb_spread}      每个箱的紧致度（便于打日志）
    """
    kept, dropped, spreads = {}, [], {}
    for name, contigs in clu2contigs.items():
        contigs = list(contigs)
        length = sum(len(contigname2seq[c]) for c in contigs if c in contigname2seq)
        n = len(contigs)
        spread = embedding_spread(contigs, contigname2emb)
        spreads[name] = spread
        # 粗筛：长度 与 contig 数**取并集**（IMPROVEMENT.md §1.4.4）
        if not (length >= min_length_coarse or n >= min_n_contigs):
            dropped.extend(contigs)
            continue
        # 精筛
        if spread <= emb_spread_max or length >= min_length_keep_big or n == 1:
            kept[name] = contigs
        else:
            dropped.extend(contigs)
    return kept, dropped, spreads


# ─────────────────── §8.2 输出精简：只用嵌入紧致度（保守版） ───────────────────
def filter_mags_for_output(
    kept: Dict[str, List[str]],
    spreads: Dict[str, float],
    contigname2seq: dict,
    tight_band_lo: int = MAG_TIGHT_BAND_LO,
    wide_band_lo: int = MAG_WIDE_BAND_LO,
    tight_emb: float = MAG_TIGHT_EMB,
    wide_emb: float = MAG_WIDE_EMB,
) -> Tuple[Dict[str, List[str]], List[str]]:
    """对 H1 保留下来的箱再做一次「输出精简」（IMPROVEMENT.md §8.2）。

    规则：
        长度 < tight_band_lo            -> 要求 emb_spread <= tight_emb
        tight_band_lo <= 长度 < wide_band_lo -> 要求 emb_spread <= wide_emb
        长度 >= wide_band_lo            -> **不筛**（任何规则都会漏掉真实染色体，见 §7.3）

    ⚠️ 只用嵌入紧致度，**刻意不用 contig 数**（理由见文件顶部常量处的注释）。

    Returns
    -------
    kept_final : {bin_name: [contig, ...]}   精简后要输出的箱
    discarded  : [contig, ...]               被精简掉的箱里的 contig（交给 k-means 重聚）
    """
    kept_final, discarded = {}, []
    for name, contigs in kept.items():
        length = sum(len(contigname2seq[c]) for c in contigs if c in contigname2seq)
        emb = spreads.get(name, 0.0)
        if length >= wide_band_lo:
            keep = True
        elif length >= tight_band_lo:
            keep = emb <= wide_emb
        else:
            keep = emb <= tight_emb
        if keep:
            kept_final[name] = contigs
        else:
            discarded.extend(contigs)
    return kept_final, discarded


# ───────────────────────── §0 第一步：从 144 个里挑 1 个 ─────────────────────────
# 子进程里缓存的大对象（避免每个任务重复加载）
_WORKER: Dict[str, object] = {}


def _score_worker_init(temp_file_folder_path, markerset_path, length_only):
    """子进程初始化：加载打分所需的只读数据。

    刻意**不加载** `contigname2seq`（实测 456 MB），只加载长度字典，
    以免 N 个子进程把内存放大 N 倍。
    """
    _WORKER["markerset"] = readMarkerSets(markerset_path)
    bac_g2c, bac_c2g = readPickle(os.path.join(temp_file_folder_path, "bac_gene_info.pkl"))
    arc_g2c, arc_c2g = readPickle(os.path.join(temp_file_folder_path, "arc_gene_info.pkl"))
    _WORKER["bac_c2g"] = bac_c2g
    _WORKER["arc_c2g"] = arc_c2g
    _WORKER["length"] = length_only
    _WORKER["emb"] = readPickle(os.path.join(
        temp_file_folder_path, "SimCLR_contigname2emb_norm_ndarray.pkl"))


def _score_one_partition(tsv_path: str) -> Dict[str, object]:
    """给一个 Leiden 划分算 §0 需要的三个信号。"""
    ms = _WORKER["markerset"]
    bac_c2g = _WORKER["bac_c2g"]
    arc_c2g = _WORKER["arc_c2g"]
    length_of = _WORKER["length"]
    emb = _WORKER["emb"]

    clu2contigs: Dict[str, List[str]] = {}
    with open(tsv_path, "r", encoding="utf-8") as rh:
        for line in rh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            clu2contigs.setdefault(parts[1], []).append(parts[0])

    n_pure_small = 0
    n_pure_all = 0
    scg_pairs = []
    for contigs in clu2contigs.values():
        length = 0
        for c in contigs:
            length += length_of.get(c, 0)
        spread = embedding_spread(contigs, emb)
        n = len(contigs)
        pure = (spread <= EMB_SPREAD_MAX
                or length >= MIN_LENGTH_KEEP_BIG or n == 1)
        if pure:
            if PURE_ALL_FLOOR <= length:
                n_pure_all += 1
            if PURE_SMALL_LO <= length < PURE_SMALL_HI:
                n_pure_small += 1
        # SCG 档位只在 >=100 kb 的箱上统计
        if length >= SCG_SCORE_FLOOR:
            _, _, _, comp, cont = determine_domain(ms, contigs, bac_c2g, arc_c2g)
            scg_pairs.append((comp, cont))

    cells = scg_cells(scg_pairs)
    scg_score = float(sum(w * v for w, v in zip(SCG_WEIGHTS, cells)))
    return {
        "partition": os.path.basename(tsv_path),
        "tsv_path": tsv_path,
        "n_bins": len(clu2contigs),
        "scg_score": scg_score,
        "n5010": cells[0], "n7010": cells[1], "n9010": cells[2],
        "n505": cells[3], "n705": cells[4], "n905": cells[5],
        "n_pure_small": n_pure_small,
        "n_pure_all": n_pure_all,
    }


def _rank01(values: np.ndarray) -> np.ndarray:
    """秩归一化到 [0,1]（同分取平均秩）。"""
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    # 处理并列：用平均秩
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    if len(values) <= 1:
        return np.zeros(len(values))
    return ranks / (len(values) - 1)


def _make_pool_with_init(ctx, num_workers, initializer, initargs,
                         attempts: int = 5, delay: float = 5.0):
    """建一个带 `initializer` 的 spawn 进程池，遇到信号量类错误时重试。

    与 `CompleteBin/Utils/mp_utils.py:make_spawn_pool` 同样的重试策略，
    但那个函数不支持 `initializer`，而本模块需要用它把只读数据装进子进程
    （避免把大字典随每个任务反复 pickle）。
    """
    import time
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return ctx.Pool(num_workers, initializer=initializer, initargs=initargs)
        except (OSError, BlockingIOError) as exc:      # 多为信号量不足
            last_exc = exc
            logger.warning(f"--> [H1] Pool({num_workers}) 创建失败（第 {attempt} 次）：{exc}")
            if attempt < attempts:
                time.sleep(delay)
    raise last_exc


def select_best_partition(
    cluster_folder: str,
    temp_file_folder_path: str,
    contigname2seq: dict,
    markerset_path: str,
    num_workers: int,
    pure_small_weight: float = PURE_SMALL_WEIGHT,
    pure_all_weight: float = PURE_ALL_WEIGHT,
) -> Tuple[str, "object"]:
    """§0：从 `cluster_folder` 里的全部 Leiden 划分中挑出 S 最大的那一个。

    Returns
    -------
    best_tsv : 选中划分的 tsv 路径
    table    : pandas.DataFrame，每个划分一行（含三个信号与最终 S），便于复查
    """
    import pandas as pd

    tsv_list = sorted(
        os.path.join(cluster_folder, f)
        for f in os.listdir(cluster_folder) if f.endswith(".tsv")
    )
    if not tsv_list:
        raise FileNotFoundError(f"{cluster_folder} 里没有任何 .tsv 划分")
    logger.info(f"--> [H1] 共 {len(tsv_list)} 个 Leiden 划分待打分。")

    length_only = {name: len(seq) for name, seq in contigname2seq.items()}

    # 用 initializer 把只读数据装进每个子进程，避免把大字典随每个任务反复 pickle。
    ctx = mp.get_context("spawn")
    pool = _make_pool_with_init(
        ctx, num_workers, _score_worker_init,
        (temp_file_folder_path, markerset_path, length_only))
    try:
        rows = pool.map(_score_one_partition, tsv_list, chunksize=1)
    finally:
        pool.close()
        pool.join()

    df = pd.DataFrame(rows)
    df["S"] = (_rank01(df["scg_score"].values)
               + pure_small_weight * _rank01(df["n_pure_small"].values)
               + pure_all_weight * _rank01(df["n_pure_all"].values))
    df = df.sort_values("S", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    logger.info(
        f"--> [H1] 选中划分 {best['partition']}："
        f"SCG分={best['scg_score']:.1f}, n_pure_small={int(best['n_pure_small'])}, "
        f"n_pure_all={int(best['n_pure_all'])}, S={best['S']:.4f}"
    )
    return best["tsv_path"], df


# ─────────────────────────────────── 主流程 ───────────────────────────────────
def run_improved_clustering(
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
    leiden_iter_mode: str,
    coverage_profile_path=None,
    leiden_seed: int = 42,
    bac_gene2contigNames: dict = None,
    residual_min_k_metainfo_path: str = None,
    apply_output_reduction: bool = True,
    apply_final_reclustering: bool = True,
) -> None:
    """`ensemble_clustering=False` 时的聚类入口（IMPROVEMENT.md 的 H1 方案）。

    `bac_gene2contigNames` 建议由调用方传入（管线里本来就有），
    这样 `cluster_kmeans_for_low_v2` 拿到的基因→contig 映射与其它环节完全一致；
    为 None 时才从 `bac_contigName2_gene2num` 反推。

    `residual_min_k_metainfo_path` 指定**残余 k-means 的 k 下界**要从哪个
    `MetaInfo.tsv` 推（`cluster_kmeans_for_low_v2` 内部用它算
    `min_k = max(3, round((HQ+MQ) × min_k_ratio))`）。为 None 时用本方案自己刚写出的
    `MetaInfo.tsv`（自洽、不依赖其它产物）。
    ⚠️ 该值只影响残余 k-means 的簇数，**不影响 NC / NC+MQ**
    （IMPROVEMENT.md §3.2 实测：两种做法下 NC 与 NC+MQ 完全相同）。
    IMPROVEMENT.md §1.2 的实测数字是在「第一轮集成输出的 MetaInfo」上跑的；
    要逐位复现那组数字，把该参数指到对应的 `first_clustering_bins/MetaInfo.tsv`。

    `apply_output_reduction=True`（默认）时，在 H1 保留下来的箱上再套一层
    **输出精简**（IMPROVEMENT.md §8.2 的保守版：只用嵌入紧致度、刻意不用 contig 数），
    把 3000+ 个 MAG 降到一千多个，而 **NC / MQ 一个都不掉**；
    被精简掉的箱里的 contig 会与 H1 原本丢掉的 contig 合并，一起交给
    `cluster_kmeans_for_low_v2` 重聚（IMPROVEMENT.md §9）。
    设为 False 则跳过这一步，输出与 §1.2 的实测完全一致。

    apply_final_reclustering=True adds a final pass AFTER existing residual
    clustering: keep bins with >=20,000 bp OR >=5 contigs unchanged, and
    recluster only the remaining sequences with cluster_kmeans_for_low_v2.
    This empirical rule was checked on Marine samples 0-9; it is not a
    universal quality guarantee. False restores the previous output path.

    步骤与 IMPROVEMENT.md 的章节对应：
        1. 跑第一轮 Leiden 网格，拿到全部候选划分            （与现有管线相同）
        2. §0 用 S 从全部划分里挑出唯一一个                  （本模块 select_best_partition）
        3. §1 对该划分做粗筛 + 精筛                          （本模块 filter_bins_by_purity）
        4. 保留下来的箱直接写出；丢掉的 contig 交给
           `cluster_kmeans_for_low_v2` 重聚（k 由 SCG 自适应）（IMPROVEMENT.md §3.2）
    """
    clustering_all_folder = os.path.join(temp_file_folder_path, clustering_folder_name)
    if not os.path.exists(clustering_all_folder):
        os.mkdir(clustering_all_folder)
    if not os.path.exists(bin_output_folder_path):
        os.mkdir(bin_output_folder_path)

    # ── 步骤 1：第一轮 Leiden 网格（与现有管线用同一个函数、同一个输出目录）──
    simclr_emb_list, length_list, sub_contigname_list = [], [], []
    for contigname, seq in contigname2seq.items():
        length = len(seq)
        if length < min_contig_length:
            continue
        sub_contigname_list.append(contigname)
        simclr_emb_list.append(simclr_contigname2emb_norm_array[contigname])
        length_list.append(length)
    logger.info(f"--> [H1] 有 {len(sub_contigname_list)} 条 contig 参与聚类"
                f"（min_contig_length={min_contig_length}）。")
    cluster_folder = os.path.join(clustering_all_folder, "leiden_cluster_results")
    if not os.path.exists(cluster_folder):
        os.mkdir(cluster_folder)
    first_cluster(
        sub_contigname_list,
        np.stack(simclr_emb_list, axis=0),
        length_list,
        cluster_folder,
        min_contig_length,
        cpu_workers,
        seed_path,
        1,                       # clustering_stages=1：只要第一轮的 144 个划分
        leiden_iter_mode,
        leiden_seed,
    )

    # ── 步骤 2：§0 挑出唯一一个划分 ──
    best_tsv, score_table = select_best_partition(
        cluster_folder, temp_file_folder_path, contigname2seq,
        markerset_path, cpu_workers,
    )
    score_table.to_csv(
        os.path.join(clustering_all_folder, "h1_partition_scores.tsv"),
        sep="\t", index=False,
    )

    # ── 步骤 3：§1 粗筛 + 精筛 ──
    clu2contigs: Dict[str, List[str]] = {}
    with open(best_tsv, "r", encoding="utf-8") as rh:
        for line in rh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                clu2contigs.setdefault(parts[1], []).append(parts[0])
    kept, dropped, spreads = filter_bins_by_purity(
        clu2contigs, simclr_contigname2emb_norm_array, contigname2seq)
    logger.info(
        f"--> [H1] 划分 {os.path.basename(best_tsv)}："
        f"{len(clu2contigs)} 个箱 → 保留 {len(kept)} 个，丢掉 {len(set(dropped))} 条 contig。"
    )

    # ── 步骤 3b：§8.2 输出精简（只用嵌入紧致度，不用 contig 数）──
    # 被精简掉的箱里的 contig **不丢弃**，与 H1 丢掉的 contig 合并后一起重聚（§9）。
    discarded_by_reduction: List[str] = []
    if apply_output_reduction:
        n_before = len(kept)
        kept, discarded_by_reduction = filter_mags_for_output(
            kept, spreads, contigname2seq)
        logger.info(
            f"--> [H1] 输出精简（§8.2 保守版）：{n_before} 个箱 → {len(kept)} 个；"
            f"被精简的箱贡献 {len(set(discarded_by_reduction))} 条 contig，将与"
            f"H1 丢掉的 contig 合并后重聚。"
        )

    # ── 步骤 4a：保留下来的箱直接写出 ──
    markerset = readMarkerSets(markerset_path)
    index = 0
    wh = open(os.path.join(bin_output_folder_path, "MetaInfo.tsv"), "w")
    for name, contigs in kept.items():
        # ⚠️ 这里**故意不做**任何长度过滤。
        # IMPROVEMENT.md §1.3 明确写了：H1 相对现有管线的最大优势就是
        # **保留了 <100 kb 的质粒**（现有管线的 mag_length_threshold=100000
        # 会把它们全丢掉）。因此 mag_length_threshold 在本方案里不用于筛箱，
        # 只作为 API 兼容参数保留。
        _, _, _, comp, cont = determine_domain(
            markerset, contigs, bac_contigName2_gene2num, arc_contigName2_gene2num)
        state = ("HighQuality" if comp >= 90 and cont <= 5
                 else "MediumQuality" if comp >= 50 and cont <= 10
                 else "LowQuality")
        out_name = f"CompleteBin_{index}.fasta"
        _write_meta(wh, out_name, comp, cont, state)
        writeFasta({c: contigname2seq[c] for c in contigs if c in contigname2seq},
                   os.path.join(bin_output_folder_path, out_name))
        index += 1

    # ── 步骤 4b：两路丢掉的 contig 合并，交给管线自己的第三轮 k-means（k 由 SCG 自适应）──
    # · dropped                —— H1 粗筛/精筛就没让进箱的 contig
    # · discarded_by_reduction —— 进了箱、但那个箱被 §8.2 精简掉的 contig
    all_discarded = set(dropped) | set(discarded_by_reduction)
    sub_dropped = {c: contigname2seq[c] for c in all_discarded if c in contigname2seq}
    logger.info(f"--> [H1] 待重聚的 contig 共 {len(sub_dropped)} 条。")
    if sub_dropped:
        km_dir = os.path.join(clustering_all_folder, "h1_residual_kmeans")
        if bac_gene2contigNames is None:
            bac_gene2contigNames = _bac_gene2contig(bac_contigName2_gene2num)
        cluster_kmeans_for_low_v2(
            sub_dropped,
            simclr_contigname2emb_norm_array,
            bac_gene2contigNames,
            bac_contigName2_gene2num,
            0,                                  # 长度门槛 0：H1 不做长度过滤
            km_dir,
            first_metainfo_path=(residual_min_k_metainfo_path
                                 if residual_min_k_metainfo_path is not None
                                 else os.path.join(bin_output_folder_path, "MetaInfo.tsv")),
            second_metainfo_path=None,
            min_k_ratio=MIN_K_RATIO,
            percentile=PERCENTILE,
        )
        for filename in sorted(os.listdir(km_dir)):
            if not filename.endswith(".fasta"):
                continue
            src = os.path.join(km_dir, filename)
            if os.path.getsize(src) == 0:
                continue
            out_name = f"CompleteBin_{index}.fasta"
            _write_meta(wh, out_name, "0.", "0.", "LowQuality")
            copy(src, os.path.join(bin_output_folder_path, out_name))
            index += 1
    wh.close()
    if apply_final_reclustering:
        index = recluster_small_output_bins(
            bin_output_folder_path, clustering_all_folder,
            contigname2seq, simclr_contigname2emb_norm_array,
            bac_gene2contigNames if bac_gene2contigNames is not None
            else _bac_gene2contig(bac_contigName2_gene2num),
            bac_contigName2_gene2num,
        )
    logger.info(f"--> [H1] 完成，共写出 {index} 个 MAG 到 {bin_output_folder_path}。")


def _write_meta(wh, out_name, comp, cont, state):
    """与 `Binning_steps.writeMetaInfo` 写出的列格式保持一致。"""
    wh.write(out_name + "\tSCG_EVAL(Comp,Cont,Quality)\t"
             + str(comp) + "\t" + str(cont) + "\t" + state + "\n")


def _bac_gene2contig(bac_contigName2_gene2num) -> Dict[str, List[str]]:
    """从 contig→{gene:num} 反推 gene→[contig]，供 `cluster_kmeans_for_low_v2` 使用。

    与管线里 `bac_gene2contigNames`（由 `readPickle` 直接读出）内容一致；
    这里由调用方已在内存中的 `bac_contigName2_gene2num` 现算，
    避免再读一次 pkl。
    """
    out: Dict[str, List[str]] = {}
    for contig, gene2num in bac_contigName2_gene2num.items():
        for gene in gene2num:
            out.setdefault(gene, []).append(contig)
    return out


def recluster_small_output_bins(
    output_dir, work_dir, sequences, embeddings, gene2contigs, contig2genes,
    min_length=20_000, min_contigs=5,
):
    """Keep accepted final bins unchanged; regroup rejected bins without loss.

    The original rejected bins and metadata remain in a unique work directory.
    No reference labels are used. New clusters are NOT recursively filtered.
    Validate complete, disjoint sequence coverage before changing final output.
    """
    import json
    import tempfile
    from pathlib import Path

    output = Path(output_dir)
    metadata = output / "MetaInfo.tsv"
    lines = metadata.read_text().splitlines(keepends=True)
    accepted, rejected, all_names = [], [], set()
    residual = {}
    for line in lines:
        filename = line.split("\t", 1)[0]
        names = []
        with (output / filename).open() as fh:
            for row in fh:
                if row.startswith(">"):
                    name = row.rstrip("\r\n")
                    if name not in sequences or name not in embeddings:
                        raise ValueError(f"Missing sequence or embedding: {name}")
                    if name in all_names:
                        raise ValueError(f"Overlapping default output sequence: {name}")
                    all_names.add(name)
                    names.append(name)
        if not names:
            raise ValueError(f"Empty bin: {filename}")
        if sum(len(sequences[n]) for n in names) >= min_length or len(names) >= min_contigs:
            accepted.append(line)
        else:
            rejected.append(filename)
            residual.update((n, sequences[n]) for n in names)
    if not rejected:
        return len(accepted)

    stage = Path(tempfile.mkdtemp(prefix="final_small_bins_", dir=work_dir))
    clusters = stage / "reclustered"
    clusters.mkdir()
    previous_meta = stage / "accepted.MetaInfo.tsv"
    previous_meta.write_text("".join(accepted))
    copy(metadata, stage / "original.MetaInfo.tsv")
    if len(residual) <= 3:
        # k_means_v2 intentionally returns without writing for <=3 sequences.
        writeFasta(residual, str(clusters / "0.fasta"))
    else:
        cluster_kmeans_for_low_v2(
            residual, embeddings, gene2contigs, contig2genes, 0, str(clusters),
            first_metainfo_path=str(previous_meta), second_metainfo_path=None,
            min_k_ratio=MIN_K_RATIO, percentile=PERCENTILE,
            max_clusters=len(rejected),
        )
    files = sorted(clusters.glob("*.fasta"))
    recovered = set()
    for path in files:
        with path.open() as fh:
            for row in fh:
                if row.startswith(">"):
                    name = row.rstrip("\r\n")
                    if name not in residual or name in recovered:
                        raise ValueError(f"Invalid reclustered sequence: {name}")
                    recovered.add(name)
    if recovered != set(residual):
        raise ValueError("Residual clustering lost sequences; final output unchanged")

    # Stage metadata and new files before retiring the original small bins.
    index = max(int(line.split("\t")[0].removeprefix("CompleteBin_").removesuffix(".fasta"))
                for line in lines) + 1
    new_lines = list(accepted)
    for path in files:
        name = f"CompleteBin_{index}.fasta"
        copy(path, output / name)
        new_lines.append(name + "\tSCG_EVAL(Comp,Cont,Quality)\t0.\t0.\tLowQuality\n")
        index += 1
    original = stage / "original_small_bins"
    original.mkdir()
    for name in rejected:
        os.replace(output / name, original / name)
    staged_meta = output / "MetaInfo.tsv.tmp"
    staged_meta.write_text("".join(new_lines))
    os.replace(staged_meta, metadata)
    report = dict(input_bins=len(lines), retained_bins=len(accepted),
                  regrouped_input_bins=len(rejected), residual_sequences=len(residual),
                  recovered_sequences=len(recovered), new_bins=len(files),
                  output_bins=len(new_lines), min_length=min_length,
                  min_contigs=min_contigs, archive=str(stage),
                  new_bin_quality="not evaluated; 0/0 LowQuality follows residual convention")
    (output / "output_reduction.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info(f"--> Final output reduction: {report}")
    return len(new_lines)
