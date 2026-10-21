"""Leiden clustering with independent ``accurate`` and ``fast`` modes.

``accurate`` preserves the original full-convergence behaviour.
``fast`` runs one Leiden iteration at a time and stops only after a sustained
plateau, or when a conservative iteration/time safety limit is reached.

随机性：leidenalg 的 ``Optimiser`` 有自己的随机数发生器，``seed_everything``
管不到它。这里所有 ``Optimiser`` 都通过 ``set_rng_seed`` 显式固定，
种子来自 ``leiden_seed`` 参数（与训练用的全局 ``seed`` 是两个独立参数）。
"""

import os
import time
from typing import Dict, List, Optional, Set, Union

import leidenalg
import numpy as np
from igraph import Graph

from CompleteBin.Cluster.first_cluster_utils import (
    build_undirected_weighted_edges,
    canonicalize_undirected_edges,
)
from CompleteBin.logger import get_logger

logger = get_logger()

# fast 模式默认每跑一轮 Leiden 都打一行日志；105 个网格点 × 约 20 轮 ≈ 2000+ 行，
# 属于噪声。设 COMPLETEBIN_LEIDEN_VERBOSE=1 可以重新打开（降级/恢复只影响日志级别，
# 不影响任何计算结果）。
_LEIDEN_VERBOSE = os.environ.get(
    "COMPLETEBIN_LEIDEN_VERBOSE", ""
).strip().lower() not in ("", "0", "false", "no")


def _iter_log(message: str) -> None:
    """逐轮迭代日志：默认只进 debug，设了环境变量则进 info。"""
    if _LEIDEN_VERBOSE:
        logger.info(message)
    else:
        logger.debug(message)


def _make_optimiser(rng_seed: int) -> "leidenalg.Optimiser":
    """创建并固定随机种子的 Optimiser。

    leidenalg 的随机性体现在节点访问顺序等内部决策上，
    不调用 set_rng_seed 时同一份输入每次都会给出不同的分区。
    """
    optimiser = leidenalg.Optimiser()
    optimiser.set_rng_seed(int(rng_seed))
    return optimiser


# ============================================================
# 优化核心
# ============================================================

def _optimise_partition_accurate(partition, is_membership_fixed, n_iterations,
                                 rng_seed: int = 2048):
    """
    精确模式：完全等同于原版 optimise_partition 行为。

    当 n_iterations=-1 时跑满直到 leidenalg 内部判定收敛；
    当 n_iterations>0 时跑固定轮数。
    """
    optimiser = _make_optimiser(rng_seed)
    t0 = time.time()
    improvement = optimiser.optimise_partition(
        partition,
        n_iterations=n_iterations,
        is_membership_fixed=is_membership_fixed,
    )
    elapsed = time.time() - t0
    logger.info(
        f"--> [accurate] Leiden done: improvement={improvement:.4f}, "
        f"time={elapsed:.1f}s, n_iterations={n_iterations}, rng_seed={rng_seed}"
    )
    return improvement


def _optimise_partition_fast(
    partition,
    is_membership_fixed,
    early_stop_threshold,
    hard_cap,
    min_iterations,
    patience,
    window_size,
    quality_threshold,
    max_seconds,
    rng_seed: int = 2048,
):
    """Optimise one iteration at a time until a sustained plateau.

    ``partition.quality()`` is evaluated once before optimisation. Subsequent
    quality values are derived from the improvements returned by leidenalg,
    avoiding an additional full graph scan after every iteration.

    The first iteration is treated as warm-up and is excluded from the
    ``recent_rel`` ratio (both its sliding window and its reference total).
    Otherwise the very first iteration, which typically contributes well over
    99% of the total improvement, would sit in ``recent_rel``'s denominator
    forever and make the ratio collapse as soon as the warm-up iteration
    leaves the sliding window. ``total_improvement`` (and therefore the
    quality estimate) still includes the warm-up iteration.
    """
    if min_iterations < 1:
        raise ValueError("min_iterations must be >= 1")
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if hard_cap < min_iterations:
        raise ValueError("hard_cap must be >= min_iterations")
    if early_stop_threshold < 0 or quality_threshold < 0:
        raise ValueError("fast-mode convergence thresholds must be >= 0")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0 or None")

    optimiser = _make_optimiser(rng_seed)
    initial_quality = float(partition.quality())
    total_improvement = 0.0
    # 第 1 轮（warm-up）之后的累计提升，只用于 recent_rel 的分母
    post_warmup_improvement = 0.0
    improvement_history = []
    n_iters_run = 0
    plateau_count = 0
    stop_reason = "hard_cap"
    eps = np.finfo(float).eps

    t0 = time.time()

    for i in range(hard_cap):
        iteration_start = time.time()
        imp = optimiser.optimise_partition(
            partition,
            n_iterations=1,
            is_membership_fixed=is_membership_fixed,
        )
        imp = float(imp)
        n_iters_run = i + 1

        if not np.isfinite(imp) or imp < 0:
            stop_reason = "invalid_improvement"
            logger.warning(
                f"--> [fast] Invalid improvement at iter {n_iters_run}: {imp}. Stopping."
            )
            break

        total_improvement += imp
        if n_iters_run > 1:
            # 第 1 轮作为 warm-up 不计入平台期判据
            post_warmup_improvement += imp
            improvement_history.append(imp)
        current_quality = initial_quality + total_improvement
        elapsed = time.time() - t0
        iteration_seconds = time.time() - iteration_start

        recent_relative_gain = None
        relative_quality_gain = None
        has_full_window = len(improvement_history) >= window_size
        if has_full_window:
            recent_gain = sum(improvement_history[-window_size:])
            previous_total = total_improvement - recent_gain
            previous_quality = initial_quality + previous_total
            recent_relative_gain = recent_gain / max(abs(post_warmup_improvement), eps)
            relative_quality_gain = recent_gain / max(abs(previous_quality), eps)

        can_test_plateau = (
            imp != 0.0
            and n_iters_run >= min_iterations
            and has_full_window
            and (max_seconds is None or elapsed < max_seconds)
        )
        if can_test_plateau:
            is_plateau = (
                recent_relative_gain <= early_stop_threshold
                and relative_quality_gain <= quality_threshold
            )
            plateau_count = plateau_count + 1 if is_plateau else 0

        _iter_log(
            f"--> [fast] iter={n_iters_run}, improvement={imp:.6f}, "
            f"total_improvement={total_improvement:.6f}, quality={current_quality:.6f}, "
            f"recent_rel={recent_relative_gain if recent_relative_gain is not None else 'NA'}, "
            f"quality_rel={relative_quality_gain if relative_quality_gain is not None else 'NA'}, "
            f"patience={plateau_count}/{patience}, iter_time={iteration_seconds:.1f}s, "
            f"elapsed={elapsed:.1f}s"
        )

        if imp == 0.0:
            stop_reason = "exact_zero"
            break

        if max_seconds is not None and elapsed >= max_seconds:
            stop_reason = "time_budget"
            break

        if plateau_count >= patience:
            stop_reason = "plateau"
            break

    elapsed = time.time() - t0
    logger.info(
        f"--> [fast] Leiden done: {n_iters_run} iters, "
        f"total_improvement={total_improvement:.4f}, final_quality="
        f"{initial_quality + total_improvement:.4f}, time={elapsed:.1f}s, "
        f"stop_reason={stop_reason}, rng_seed={rng_seed}"
    )
    return total_improvement


def _default_fast_hard_cap(vcount: int) -> int:
    """Return a conservative graph-size-aware fast-mode safety cap."""
    if vcount < 100_000:
        return 60
    if vcount < 250_000:
        return 55
    if vcount < 500_000:
        return 50
    return 45


# ============================================================
# 图构建（提取自原 run_leiden，逻辑完全一致）
# ============================================================

def _build_leiden_graph(
    norm_embeddings: np.ndarray,
    ann_neighbor_indices: np.ndarray,
    ann_distances: np.ndarray,
    max_edges: int,
    partgraph_ratio: int,
    bandwidth: float,
    lmode: str,
    length_weight: List[float],
    initial_list: Optional[List[Union[int, None]]],
    resolution: float,
    vcount: int,
    alpha: float = 1.0,
    min_degree: int = 0,
):
    """
    构建 igraph Graph 和 RBERVertexPartition。

    建图（KNN → 无向加权边）**与第二轮 ``first_cluster_utils.run_leiden`` 共用**
    ``build_undirected_weighted_edges``，两条路径口径不会再各自漂移：

      · 全局距离百分位筛边（``partgraph_ratio``，100 = 不筛）；
      · 距离→相似度 ``exp(-d/bandwidth)``；
      · 归并成无序对（同端点保留相似度最大的一条），不再按 contig 名任意丢一半；
      · 单向边权重乘 ``alpha``（``alpha=1`` 即不降权）；
      · ``min_degree`` 保底边（消除孤立点）。

    相对旧的 `sources > targets` 写法这是**行为变更**（边集与权重都会变），
    依据与实测见 ``PLAN_GRID_REFORM.md``。
    """
    # 建图统一走共用函数（各步骤说明见其 docstring）
    sources, targets, wei = build_undirected_weighted_edges(
        ann_neighbor_indices, ann_distances, vcount, max_edges,
        partgraph_ratio, bandwidth, lmode,
        alpha=alpha, min_degree=min_degree)

    edgelist = list(zip(sources.tolist(), targets.tolist()))

    # ---- 构建 igraph 图和 Leiden 分区 ----
    graph = Graph(vcount, edgelist, directed=False)

    assert len(wei) == len(edgelist), ValueError(
        f"wei len is {len(wei)}, edgelist len is {len(edgelist)}."
    )

    partition = leidenalg.RBERVertexPartition(
        graph,
        initial_membership=initial_list,
        weights=wei.tolist(),
        node_sizes=length_weight,
        resolution_parameter=resolution,
    )

    return graph, partition, len(wei)


# ============================================================
# 主函数：run_leiden_adaptive
# ============================================================

def run_leiden_adaptive(
    # ---- 原 run_leiden 参数（完全兼容）----
    cur_i: int,
    total_n: int,
    output_file: str,
    contig_name_list: List[str],
    ann_neighbor_indices: np.ndarray,
    ann_distances: np.ndarray,
    length_weight: List[float],
    max_edges: int,
    norm_embeddings: np.ndarray,
    bandwidth: float = 0.1,
    lmode: str = 'l2',
    initial_list: Optional[List[Union[int, None]]] = None,
    partgraph_ratio: int = 50,
    resolution: float = 1.0,
    is_membership_fixed: List[bool] = None,
    n_iterations: int = -1,
    write_file: bool = True,
    # ---- 新增参数 ----
    mode: str = 'accurate',
    early_stop_threshold: float = 5e-3,
    hard_cap: Optional[int] = None,
    min_iterations: int = 8,
    patience: int = 4,
    window_size: int = 4,
    quality_threshold: float = 3e-6,
    max_seconds: Optional[float] = 7200.0,
    rng_seed: int = 2048,
    # ---- 建图规则（与第二轮 run_leiden 同口径；默认值 = 历史行为）----
    alpha: float = 1.0,
    min_degree: int = 0,
) -> Dict[str, Set[str]]:
    """
    运行 Leiden 聚类（支持快速/精确两种模式）。

    参数
    ----
    mode : str
        'fast'      — 自适应早停，速度快
        'accurate'  — 完全收敛，精度最高 （默认）
    early_stop_threshold : float
        fast 模式最近窗口 improvement 占 warm-up 之后累计 improvement 的阈值。
        默认 5e-3（0.5%）。第 1 轮作为 warm-up 不计入该比例。
        历史沿革（2026-09）：旧 run 日志实测为 0.01，后被改为 1e-4（改动未留记录），
        2026-09-29 先后试过 0.01 → 1e-3，最终定为 5e-3。
        改动此值会改变聚类结果，必须同步更新基线。
    hard_cap : int, optional
        fast 模式最大迭代轮数。默认按节点数选择 40--60 轮；
        250,000--500,000 节点使用 45 轮。
    min_iterations : int
        允许平台期早停前至少运行的轮数，默认 8。
    patience : int
        连续多少轮满足平台期条件后停止，默认 4。
    window_size : int
        计算近期累计 improvement 的滑动窗口大小，默认 4。
    quality_threshold : float
        最近窗口 improvement 相对于窗口前 partition quality 的阈值，
        默认 3e-6。该条件比 early_stop_threshold 更严，实际决定停止时机。
    max_seconds : float, optional
        fast 模式单个参数组合的时间上限，默认 7200 秒；None 表示不限时。
    n_iterations : int
        accurate 模式下的迭代数（-1 为跑满直到收敛）。
        fast 模式下此参数被忽略，由 early_stop_threshold + hard_cap 控制。
    rng_seed : int
        leidenalg ``Optimiser`` 的随机种子，默认 2048。与训练用的全局 ``seed``
        是两个独立参数；同一个 rng_seed + 同一份输入 → 同一个分区。

    其余参数与原 run_leiden 完全相同。

    返回
    ----
    clu2contigs : Dict[str, Set[str]]
        {group_name: {contig_name, ...}, ...}
    """
    if mode not in ('fast', 'accurate'):
        raise ValueError(f"mode must be 'fast' or 'accurate', got '{mode}'")

    vcount = len(norm_embeddings)
    resolved_hard_cap = (
        _default_fast_hard_cap(vcount) if hard_cap is None else hard_cap
    )

    # ---- 图构建 ----
    _, partition, n_edges = _build_leiden_graph(
        norm_embeddings=norm_embeddings,
        ann_neighbor_indices=ann_neighbor_indices,
        ann_distances=ann_distances,
        max_edges=max_edges,
        partgraph_ratio=partgraph_ratio,
        bandwidth=bandwidth,
        lmode=lmode,
        length_weight=length_weight,
        initial_list=initial_list,
        resolution=resolution,
        vcount=vcount,
        alpha=alpha,
        min_degree=min_degree,
    )

    # ---- 日志 ----
    if write_file:
        method = os.path.split(output_file)[-1]
    else:
        method = output_file

    if mode == 'accurate':
        mode_info = f"n_iter={n_iterations}"
    else:
        mode_info = (
            f"min_iter={min_iterations}, hard_cap={resolved_hard_cap}, "
            f"window={window_size}, patience={patience}, "
            f"gain_threshold={early_stop_threshold}, "
            f"quality_threshold={quality_threshold}, max_seconds={max_seconds}"
        )
    logger.info(
        f"--> Start Leiden ({mode} mode, {mode_info}, rng_seed={rng_seed}): "
        f"{vcount} nodes, {n_edges} edges, "
        f"max_edges={max_edges}, partgraph_ratio={partgraph_ratio}, "
        f"method={method}. {cur_i} / {total_n}"
    )

    # ---- 优化 ----
    if mode == 'accurate':
        _optimise_partition_accurate(
            partition=partition,
            is_membership_fixed=is_membership_fixed,
            n_iterations=n_iterations,
            rng_seed=rng_seed,
        )
    else:  # mode == 'fast'
        _optimise_partition_fast(
            partition=partition,
            is_membership_fixed=is_membership_fixed,
            early_stop_threshold=early_stop_threshold,
            hard_cap=resolved_hard_cap,
            min_iterations=min_iterations,
            patience=patience,
            window_size=window_size,
            quality_threshold=quality_threshold,
            max_seconds=max_seconds,
            rng_seed=rng_seed,
        )

    # ---- 结果收集（与原 run_leiden 完全一致）----
    part = list(partition)
    contig_labels_dict: Dict[str, str] = {}

    for ci in range(len(part)):
        for node_id in part[ci]:
            contig_labels_dict[contig_name_list[node_id]] = f'group_{ci}'

    # 写文件
    if write_file:
        logger.info(
            f"--> End Clustering with output path: {output_file}. {cur_i} / {total_n}"
        )
        with open(output_file, 'w') as f:
            for contig_idx in range(len(contig_labels_dict)):
                contig_name = contig_name_list[contig_idx]
                label = contig_labels_dict[contig_name]
                f.write(f"{contig_name}\t{label}\n")

    # 构建返回值
    clu2contigs: Dict[str, Set[str]] = {}
    for contig_idx in range(len(contig_labels_dict)):
        cur_contig_name = contig_name_list[contig_idx]
        cur_contig_group = contig_labels_dict[cur_contig_name]
        if cur_contig_group not in clu2contigs:
            clu2contigs[cur_contig_group] = {cur_contig_name}
        else:
            clu2contigs[cur_contig_group].add(cur_contig_name)

    return clu2contigs
