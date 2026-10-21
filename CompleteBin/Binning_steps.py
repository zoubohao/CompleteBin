

import warnings

warnings.filterwarnings("ignore")

import ast
import math
import multiprocessing as mp
import os
import random
import sys
import time
from shutil import copy, rmtree
from typing import List

import numpy as np
import psutil
import torch

from CompleteBin.CallGenes.gene_utils import callMarkerGenesByHMM
# 改进方案 H1（ensemble_clustering=False 时使用），见 IMPROVEMENT.md
from CompleteBin.Cluster.improved_clustering import run_improved_clustering
from CompleteBin.Cluster.k_means_v2 import cluster_kmeans_for_low_v2
from CompleteBin.Cluster.polish_bin_improved import \
    ensure_compact_coverage_path
from CompleteBin.Cluster.split_utils import kmeans_split
from CompleteBin.DataProcess.data_utils_fixed import \
    build_training_seq_data_numpy_save_fixed as \
    build_training_seq_data_numpy_save
from CompleteBin.IO import (readFasta, readMarkerSets, readPickle,
                            readSeedFile, writeFasta)
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.generate_seed import gen_seed
from CompleteBin.Seqs.seq_info import (calculateN50, clip_coverage_outliers,
                                       prepare_sequences_coverage,
                                       resolve_coverage_stats_path)
from CompleteBin.Trainer.sampler import DeeperBinSampler
from CompleteBin.Trainer.ssmt_v2 import SelfSupervisedMethodsTrainer
from CompleteBin.Utils.pretrain_decision import (apply_epoch_safety_valve,
                                                 auto_min_contig_length,
                                                 compute_O, compute_S,
                                                 decide_pretrained)
from CompleteBin.version import bin_v

from .combine_cluster_derep import (clustering_and_dereplication,
                                    selected_recluster_contigs)

logger = get_logger()


def get_time(f):

    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        logger.info(f"--> Time: {e_time - s_time:.2f} seconds")
        return res
    return inner


def _write_time_tsv(temp_file_folder_path: str, entries: dict) -> None:
    """Incrementally record per-step timings into ``time.tsv``.

    ``binning_with_all_steps`` is resumable and, in split SLURM execution
    (``step_num=1/2/3``), each step runs in its own process.  Every step writes
    only its own measurements through this helper; rows recorded by an earlier
    step are preserved and ``SummedTime(s)`` is recomputed from all recorded
    rows.
    """
    path = os.path.join(temp_file_folder_path, "time.tsv")
    times = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                label, _, value = line.rpartition("\t")
                try:
                    times[label] = float(value)
                except ValueError:
                    continue
    times.update(entries)
    times["SummedTime(s)"] = sum(v for k, v in times.items() if k != "SummedTime(s)")
    with open(path, "w") as wh:
        for label, value in times.items():
            wh.write(f"{label}\t{value}\n")


def writeMetaInfo(wh, outName, comp, cont, state):
    wh.write(outName
             + "\t"
             + "SCG_EVAL(Comp,Cont,Quality)"
             + "\t"
             + str(comp)
             + "\t"
             + str(cont)
             + "\t"
             + state
             + "\n")


def _ast_calls(func):
    """返回 func 源码里所有 Call 节点（解析为 AST，**不受注释/字符串干扰**）。"""
    import ast
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(func))
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]


def _has_attr_call(calls, attr):
    """是否存在 ``<something>.attr(...)`` 形式的真实调用。"""
    for c in calls:
        f = c.func
        if isinstance(f, ast.Attribute) and f.attr == attr:
            return True
    return False


def _has_named_call(calls, name, kwarg=None):
    """是否存在名为 name 的调用；给了 kwarg 则要求该关键字参数存在。"""
    for c in calls:
        f = c.func
        if isinstance(f, ast.Name):
            fname = f.id
        elif isinstance(f, ast.Attribute):
            fname = f.attr
        else:
            continue
        if fname != name:
            continue
        if kwarg is None:
            return True
        if any(k.arg == kwarg for k in c.keywords):
            return True
    return False


def _grid_axes_by_stage():
    """读 `Cluster/first_cluster.py`，返回 ``{1: {轴名: 字面量值}, 2: {...}}``。

    只认 ``clustering_stages == 1`` 这个 If 节点的**两个分支体内部**、
    形如 ``name = <字面量>`` 的直接赋值（用 `ast.literal_eval` 取值，绝不执行代码）。
    非常量赋值（如 ``min_degree = MIN_DEGREE``）记成 ``("name", "MIN_DEGREE")`` ——
    本检查只关心"有没有在分支内部赋值"，不解析它的值。

    这个函数存在的唯一目的是查「某个网格轴是不是被写到了 if/else 外面」：
    写入分支内部 → 该分支的 key 出现；被提到外面 → 两个分支都缺这个 key。
    """
    import ast

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Cluster", "first_cluster.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "clustering_stages"
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == 1):
            continue
        # 同一个 If 节点：body 是第一轮（==1），orelse 是第二轮（else）
        for stage, branch in ((1, node.body), (2, node.orelse)):
            for stmt in branch:
                if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)):
                    continue
                name = stmt.targets[0].id
                try:
                    out.setdefault(stage, {})[name] = ast.literal_eval(stmt.value)
                except ValueError:
                    # 非常量（如 `min_degree = MIN_DEGREE`）。这里只关心"有没有在
                    # 分支内部赋值"，所以记下名字本身即可，不解析它的值。
                    if isinstance(stmt.value, ast.Name):
                        out.setdefault(stage, {})[name] = ("name", stmt.value.id)
    return out


def _first_cluster_run_leiden_kwargs():
    """返回 `first_cluster.py` 里 ``run_leiden(...)`` 调用用到**关键字参数名**的集合。

    注意：`first_cluster.py:17` 把它 import 成 ``run_leiden_adaptive as run_leiden``，
    调用点写的是 ``run_leiden``，所以这里按名字 `run_leiden` 找。

    存在的理由：网格只把 `alpha` / `min_degree` 放在 `kwds=` 里传。如果这两个 key
    被删掉，函数会用默认值 `alpha=1.0, min_degree=0` 跑——**不报错**，但文件名里
    仍然写着 `_alpha_0.1`，于是"看起来扫了 α，其实一个都没生效"。这是静默失效，
    和当初 `set_rng_seed` 被删是同一类错误，所以必须显式检查调用点。
    """
    import ast

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Cluster", "first_cluster.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # (a) 直接调用：run_leiden(..., alpha=..., min_degree=...)
        if isinstance(node.func, ast.Name) and node.func.id == "run_leiden":
            out |= {k.arg for k in node.keywords if k.arg}
        # (b) 经进程池转交：apply_async(run_leiden, args=(...), kwds={...})
        #     —— first_cluster.py 用的是这一种，真正的参数在 kwds 字典的 key 上。
        if isinstance(node.func, ast.Attribute) and node.func.attr == "apply_async":
            if not (node.args and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "run_leiden"):
                continue
            for kw in node.keywords:
                if kw.arg == "kwds" and isinstance(kw.value, ast.Dict):
                    out |= {k.value for k in kw.value.keys
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str)}
    return out


def assert_leiden_seed_plumbing() -> None:
    """启动自检：Leiden 的随机种子透传链路必须完整，否则立刻报错。

    背景（真实事故）：这条链路曾在**运行期间**被改回旧版——`run_leiden` 丢掉
    `rng_seed` 参数与 `set_rng_seed` 调用。后果有两种，都很糟：
      · 如果某个 bin 触发 `cluster_split`（SCG 污染 > 12% 时），会在跑了几小时后
        抛 `TypeError: run_leiden() got an unexpected keyword argument 'rng_seed'`；
      · 更糟的是如果只丢了 `set_rng_seed` 而参数还在，程序**不会报错**，
        只会静默地产生不可复现的结果——这种错误没有任何运行期症状。

    所以在开跑前检查一遍签名与关键调用，把上面两种情况都变成"开跑即失败"。

    实现注意：检查"有没有调用 set_rng_seed"必须走 **AST**，不能用
    ``"set_rng_seed" in source`` 这种文本匹配——注释或 docstring 里提到这个词
    就会骗过文本检查（这一点是被 `test_startup_guard.py` 测出来的）。
    """
    import ast  # noqa: F401  (供 _ast_calls 使用，同时保留公开可见性)
    import inspect

    problems = []

    try:
        import leidenalg
        if not hasattr(leidenalg.Optimiser(), "set_rng_seed"):
            problems.append("leidenalg.Optimiser 没有 set_rng_seed 方法")
    except Exception as exc:  # pragma: no cover
        problems.append(f"leidenalg 不可用: {exc!r}")

    from CompleteBin.Cluster import first_cluster_utils as _fcu
    from CompleteBin.Cluster import leiden_adaptive as _la
    from CompleteBin.Cluster import sec_cluster_utils as _scu

    def _calls_of(fn):
        try:
            return _ast_calls(fn)
        except Exception as exc:  # pragma: no cover
            problems.append(f"无法解析 {fn} 的源码: {exc!r}")
            return []

    # 1) 第二轮的 run_leiden：参数要有，且必须真的调用 set_rng_seed
    if "rng_seed" not in inspect.signature(_fcu.run_leiden).parameters:
        problems.append("first_cluster_utils.run_leiden 缺 rng_seed 参数")
    if not _has_attr_call(_calls_of(_fcu.run_leiden), "set_rng_seed"):
        problems.append("first_cluster_utils.run_leiden 没有真正调用 set_rng_seed")

    # 2) 第一轮/第二轮的 leiden_adaptive
    if not hasattr(_la, "_make_optimiser"):
        problems.append("leiden_adaptive 缺 _make_optimiser 工厂函数")
    elif not _has_attr_call(_calls_of(_la._make_optimiser), "set_rng_seed"):
        problems.append("leiden_adaptive._make_optimiser 没有真正调用 set_rng_seed")

    for fn_name in ("_optimise_partition_fast", "_optimise_partition_accurate"):
        fn = getattr(_la, fn_name, None)
        if fn is None:
            problems.append(f"leiden_adaptive 缺 {fn_name}")
            continue
        if "rng_seed" not in inspect.signature(fn).parameters:
            problems.append(f"leiden_adaptive.{fn_name} 缺 rng_seed 参数")
        if not _has_named_call(_calls_of(fn), "_make_optimiser"):
            problems.append(f"leiden_adaptive.{fn_name} 没有用 _make_optimiser")

    if "rng_seed" not in inspect.signature(_la.run_leiden_adaptive).parameters:
        problems.append("leiden_adaptive.run_leiden_adaptive 缺 rng_seed 参数")
    # run_leiden_adaptive 必须把 rng_seed 传给优化函数
    if not (_has_named_call(_calls_of(_la.run_leiden_adaptive),
                            "_optimise_partition_fast", kwarg="rng_seed")
            or _has_named_call(_calls_of(_la.run_leiden_adaptive),
                               "_optimise_partition_accurate", kwarg="rng_seed")):
        problems.append("leiden_adaptive.run_leiden_adaptive 没有把 rng_seed 传下去")

    # 3) cluster_split 是曾出错的那一处：必须把种子传给 run_leiden
    if "leiden_seed" not in inspect.signature(_scu.cluster_split).parameters:
        problems.append("sec_cluster_utils.cluster_split 缺 leiden_seed 参数")
    if not _has_named_call(_calls_of(_scu.cluster_split), "run_leiden",
                           kwarg="rng_seed"):
        problems.append("sec_cluster_utils.cluster_split 没有把 rng_seed 传给 run_leiden")

    # 4) KNN 后端常量（faiss 分流依赖它们）
    for const in ("FAISS_MAX_NODES", "HNSW_M"):
        if not hasattr(_fcu, const):
            problems.append(f"first_cluster_utils 缺常量 {const}")

    # 4b) leiden_seed 的默认值必须处处一致（2026-10-01：用户要求统一为 42）。
    #     这个值散在 6 个函数的形参默认上，任何一个漏改都会在"忘记往下传"时
    #     静默换一个种子。这里逐一断言，避免只改一处。
    try:
        from CompleteBin import combine_cluster_derep as _ccd
        from CompleteBin.Cluster import cluster as _cl
        from CompleteBin.Cluster import first_cluster as _fc
        from CompleteBin.Cluster import sec_cluster as _sec
        _seed_sites = (
            (_cl, "combine_two_cluster_steps"),
            (_fc, "first_cluster"),
            (_sec, "second_cluster"),
            (_scu, "cluster_split"),
            (_scu, "re_cluster_procedure_for_one_method"),
            (_ccd, "clustering_and_dereplication"),
        )
        _want = 42
        for _mod, _name in _seed_sites:
            _fn = getattr(_mod, _name, None)
            if _fn is None:
                problems.append(f"找不到 {_name}（无法检查 leiden_seed 默认值）")
                continue
            _d = inspect.signature(_fn).parameters["leiden_seed"].default
            if _d != _want:
                problems.append(f"{_name} 的 leiden_seed 默认是 {_d}，应为 {_want}")
    except Exception as exc:  # pragma: no cover
        problems.append(f"无法检查 leiden_seed 默认值: {exc!r}")

    # 5) 建图规则（k / alpha / min_degree）—— 2026-10 重整，同样是「被回退过」的高危链路。
    #    这里只检查**结构与透传**，不检查具体取值（取值见 PLAN_GRID_REFORM.md）。
    if not hasattr(_fcu, "build_undirected_weighted_edges"):
        problems.append("first_cluster_utils 缺 build_undirected_weighted_edges（建图共用函数）")
    else:
        # 两条 Leiden 路径必须**都**走这个共用函数，否则口径会各自漂移
        if not _has_named_call(_calls_of(_fcu.run_leiden),
                               "build_undirected_weighted_edges"):
            problems.append("first_cluster_utils.run_leiden 没有用 build_undirected_weighted_edges 建图")
        if not _has_named_call(_calls_of(_la._build_leiden_graph),
                               "build_undirected_weighted_edges"):
            problems.append("leiden_adaptive._build_leiden_graph 没有用 build_undirected_weighted_edges 建图")

    if not hasattr(_fcu, "canonicalize_undirected_edges"):
        problems.append("first_cluster_utils 缺 canonicalize_undirected_edges")
    else:
        if "return_mutual" not in inspect.signature(
                _fcu.canonicalize_undirected_edges).parameters:
            problems.append("canonicalize_undirected_edges 缺 return_mutual 参数（alpha 折扣要用）")
        # PLAN §5.1/§5.4 明确要求「建图段真的调了 canonicalize_undirected_edges」。
        # 只查两条路径调了共用函数还不够 —— 共用函数内部也可能被改成自己实现一套
        # 归并逻辑（那样 mutual 判定就与 canonicalize 分叉了），所以这里直接查它内部。
        if hasattr(_fcu, "build_undirected_weighted_edges") and not _has_named_call(
                _calls_of(_fcu.build_undirected_weighted_edges),
                "canonicalize_undirected_edges"):
            problems.append(
                "build_undirected_weighted_edges 内部没有调用 canonicalize_undirected_edges"
                "（归并/互近邻判定会与 canonicalize 分叉）")

    if not hasattr(_fcu, "MIN_DEGREE"):
        problems.append("first_cluster_utils 缺常量 MIN_DEGREE")

    # 第二轮：run_leiden 必须接受 alpha / min_degree
    for kw in ("alpha", "min_degree"):
        if kw not in inspect.signature(_fcu.run_leiden).parameters:
            problems.append(f"first_cluster_utils.run_leiden 缺 {kw} 参数")

    # 第一轮：run_leiden_adaptive 必须接受并透传 alpha / min_degree
    for kw in ("alpha", "min_degree"):
        if kw not in inspect.signature(_la.run_leiden_adaptive).parameters:
            problems.append(f"leiden_adaptive.run_leiden_adaptive 缺 {kw} 参数")
    if not _has_named_call(_calls_of(_la.run_leiden_adaptive),
                           "_build_leiden_graph", kwarg="alpha"):
        problems.append("leiden_adaptive.run_leiden_adaptive 没有把 alpha 传给 _build_leiden_graph")
    if not _has_named_call(_calls_of(_la.run_leiden_adaptive),
                           "_build_leiden_graph", kwarg="min_degree"):
        problems.append("leiden_adaptive.run_leiden_adaptive 没有把 min_degree 传给 _build_leiden_graph")

    # 第二轮 cluster_split：必须把 alpha / min_degree 传给 run_leiden
    # （第二轮按实测保持历史取值，但仍要显式传递，避免参数漂移）
    for kw in ("alpha", "min_degree"):
        if not _has_named_call(_calls_of(_scu.cluster_split), "run_leiden", kwarg=kw):
            problems.append(f"sec_cluster_utils.cluster_split 没有把 {kw} 传给 run_leiden")

    # 6) 网格分支的**结构**自检（同样是被回退过的高危点）。
    #    真实事故：`max_edges_list` 曾被写成 if/else **外面的共用行**，于是第二轮
    #    （clustering_stages=2）也跟着扫第一轮的三档 k——把第二轮的网格点数从 36
    #    悄悄变成 108。这种"某个轴被提到分支外面"的错误，看值看不出来（`[10,50,100]`
    #    本身是合法取值），只有看**结构**才能发现。这里只查结构，不查具体取值
    #    （取值与依据见 scripts/graph_direction_analysis_20260930/PLAN_GRID_REFORM.md）。
    _GRID_AXES = ("parameter_list", "bandwidth_list", "partgraph_ratio_list",
                  "max_edges_list", "alpha_list", "min_degree")
    try:
        _grid_axes = _grid_axes_by_stage()
        for _stage in (1, 2):
            _assigned = _grid_axes.get(_stage, {})
            for _ax in _GRID_AXES:
                if _ax not in _assigned:
                    problems.append(
                        f"first_cluster.py 的 clustering_stages=={_stage} 分支没有在"
                        f"分支内部给 {_ax} 赋值（可能被提到了共有位置）")
    except Exception as exc:  # pragma: no cover
        problems.append(f"无法解析 first_cluster.py 的网格分支: {exc!r}")

    # 6b) 网格的调用点必须真的把新轴传下去（否则静默失效：文件名写着 alpha，
    #     实际用默认 alpha=1.0 在跑）。只查"传没传"，不查取值。
    try:
        _kw = _first_cluster_run_leiden_kwargs()
        for _k in ("alpha", "min_degree"):
            if _k not in _kw:
                problems.append(
                    f"first_cluster.py 的 run_leiden 调用没有传 {_k} —— "
                    f"网格上的该轴会静默失效（默认值会顶替，且不报错）")
    except Exception as exc:  # pragma: no cover
        problems.append(f"无法解析 first_cluster.py 的 run_leiden 调用: {exc!r}")

    if problems:
        detail = "\n".join(f"  - {p}" for p in problems)
        raise RuntimeError(
            "Leiden 种子透传自检失败，拒绝开始运行（否则会产生不可复现的结果）：\n"
            f"{detail}\n"
            "请检查 CompleteBin/Cluster/ 下的改动是否被回退。"
        )
    logger.info("--> Leiden 种子透传自检通过（run_leiden / leiden_adaptive / cluster_split 均已固定种子）。")


def seed_everything(seed=2048):
    random.seed(seed)
    # 注意：字符串哈希随机化的种子在**解释器启动时**就已固定，这里在运行时写
    # os.environ 对**当前进程无效**（见 ensure_hash_seed 的说明），只对之后由本进程
    # 新起的子进程（multiprocessing 的 'spawn' worker）生效。
    # 这里用 setdefault 给"没能重启解释器"的场景留一道子进程级兜底，
    # 并与 ensure_hash_seed 使用同一个固定值。
    os.environ.setdefault('PYTHONHASHSEED', f'{seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── 字符串哈希随机化的固定 ────────────────────────────────────────────────
# 为什么必须单独处理：Python 的哈希随机化种子在**解释器启动时**就固定了，之后写
# os.environ 对**当前进程无效**。而本管线的可复现性依赖它——聚类的重聚类阶段用
# `set` 迭代顺序（Cluster/sec_cluster_utils.py）决定嵌入矩阵的行序，行序又决定
# 节点编号 → KNN 邻居表 → Leiden 划分。实测（2026-09-29）：
#   · 同一份数据、同一个 rng_seed，仅因进程哈希种子不同，d0≈931 contig 时 3 次运行
#     有 2 次得到不同划分；
#   · 在启动环境固定 PYTHONHASHSEED=0 后，3 次全部相同。
_HASHSEED_VALUE = "2048"
_HASHSEED_SENTINEL = "_COMPLETEBIN_HASHSEED_REEXEC"


def ensure_hash_seed(reexec: bool = True) -> bool:
    """把 PYTHONHASHSEED 固定下来，使 ``set``/``dict`` 的迭代顺序跨运行一致。

    参数
    ----
    reexec : bool
        True（默认）时，若可以直接重启解释器，就重启一次以使**当前进程与它派生的
        所有子进程**都被固定；False 时只设置环境变量（仅固定之后 spawn 的子进程）。

    行为
    ----
    1. 已经设置过 PYTHONHASHSEED → 直接返回（尊重调用方/启动脚本的设定）。
    2. 可以安全重启（``sys.argv[0]`` 是一个真实存在的文件）且 ``reexec=True`` →
       设置环境变量后 ``os.execve`` 重启自身。新进程会发现变量已设，不会再次重启
       （另设哨兵环境变量双重保险），因此**不存在无限重启**。
    3. 无法安全重启（Jupyter / 交互式 / ``argv[0]`` 不是文件等）→ 只设置环境变量，
       当前进程仍保持随机，并给出明确的告警，提示改用
       ``export PYTHONHASHSEED=0``。

    返回
    ----
    bool : True 表示"当前进程的哈希种子已是确定的"，False 表示只固定到了子进程。

    ⚠️ 重要副作用
        选择重启（情况 2）时，**调用方脚本会被重新执行一次**（是进程替换，不是并行
        跑两遍；管线的实际工作只会做一次）。因此调用方在进入本函数**之前**的副作用
        （写文件、训练等）会重复发生一次。若调用方不能承受这种重复，请：
          · 用 ``ensure_hash_seed(reexec=False)`` 并自行在脚本最开头
            ``os.environ.setdefault("PYTHONHASHSEED", "0")``；或
          · 在启动命令前 ``export PYTHONHASHSEED=0``。
    """
    if os.environ.get("PYTHONHASHSEED"):
        return True
    os.environ["PYTHONHASHSEED"] = _HASHSEED_VALUE
    if os.environ.get(_HASHSEED_SENTINEL):
        return True  # 已经重启过一次，不再重启
    argv0 = sys.argv[0] if sys.argv else ""
    can_reexec = bool(argv0) and os.path.isfile(argv0)
    if reexec and can_reexec and getattr(sys, "executable", None):
        os.environ[_HASHSEED_SENTINEL] = "1"
        logger.info(
            f"--> 为保证 set/dict 迭代顺序可复现，将以 PYTHONHASHSEED={_HASHSEED_VALUE} "
            f"重新启动解释器一次（进程替换，管线只会实际执行一次）。"
        )
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
        except OSError as exc:  # pragma: no cover - 仅在无法 exec 时触发
            logger.warning(
                f"--> 无法重启解释器（{exc}）：当前进程的 set/dict 迭代顺序仍会逐次随机，"
                f"仅之后 spawn 的子进程被固定。如需完全可复现，请在启动前 "
                f"'export PYTHONHASHSEED={_HASHSEED_VALUE}'。"
            )
            return False
    logger.warning(
        f"--> 未设置 PYTHONHASHSEED 且无法安全重启解释器（argv[0]={argv0!r}）："
        f"当前进程的 set/dict 迭代顺序仍会逐次随机，仅之后 spawn 的子进程被固定。"
        f"如需完全可复现，请在启动前 'export PYTHONHASHSEED={_HASHSEED_VALUE}'，"
        f"或改用脚本方式运行（使 argv[0] 指向 .py 文件）。"
    )
    return False


def filterSpaceInFastaFile(input_fasta, output_fasta):
    with open(input_fasta, "r") as rh, open(output_fasta, "w") as wh:
        for line in rh:
            oneline = line.strip("\n")
            if ">" in oneline and " " in oneline:
                oneline = oneline.split()[0]
            wh.write(oneline + "\n")


def temp_decision(
    contigname2seq_ori: dict,
    min_contig_length: int,
    N50: int,
    large_data_size_thre: int
):
    # Weak Augmentations: Views are more similar; use higher temperatures to smooth similarity distribution. --> N50 samaller, temp higher
    # Data Noise: Higher noise levels benefit from a higher temperature to smoothen similarity scores. --> N50 samaller, temp higher
    # Dataset Size: For large datasets, lower temperatures often work better as they increase the discriminative power of the embeddings.
    # --> larger size, lower temp
    count_contigs = 0
    for _, seq in contigname2seq_ori.items():
        if len(seq) >= min_contig_length:
            count_contigs += 1
    temp = None
    if N50 >= 15000:
        if count_contigs >= large_data_size_thre:
            temp = 0.115
        else:
            temp = 0.125
    elif N50 <= 5000:
        if count_contigs >= large_data_size_thre:
            temp = 0.135
        else:
            temp = 0.145
    else:
        temp = -0.000002 * N50 + 0.155
        temp = float("%.3f" % temp)
        if count_contigs >= large_data_size_thre:
            temp -= 0.01
    return temp


def decision_lower_bound(
    contigname2seq: dict,
    min_contig_length: int,
    batch_size: int,
    extra_batches: int = 3,
):
    """为了凑够若干个完整 batch，返回 min_contig_length 需要**下调**的幅度。

    返回值满足 0 <= gap <= min_contig_length：调用方做 `min_contig_length -= gap`
    之后阈值只会下降或不变；原来的实现会返回负值，导致阈值被**抬高**（见下）。

    extra_batches: 除了"至少覆盖 cover_count 条"之外，再往缺口里多吃几个 batch
                   （默认 3；调大 = 阈值降得更多、训练用到的短 contig 更多）。

    修的三处（都是原实现的实际 bug）：
      1. 目标条数原来用 `cover_count // batch_size` 向下取整，可能小于 cover_count，
         于是取的"分界长度"落回"本来就合格"的区间里，返回负 gap；
         现在改成向上取整，保证至少覆盖 cover_count 条。
      2. 排序后取的是第 `cover_count` 长（0 基），比"覆盖 cover_count 条"多一条，
         有并列长度时会被放大；现在按目标条数取第 `target` 长的长度，语义明确。
      3. 不再用 assert + ValueError(...) 的写法（-O 下会被去掉），改成直接 raise。
    """
    lengths = sorted((len(seq) for seq in contigname2seq.values()), reverse=True)
    N = len(lengths)
    if N <= batch_size:
        raise ValueError(
            f"There are only {N} contigs in this dataset. But we need at least "
            f"{batch_size} (one batch size) contigs for training."
        )
    cover_count = sum(1 for length in lengths if length >= min_contig_length)
    logger.info(f"--> The number of {cover_count} contigs are longer than {min_contig_length}.")
    gap_num = N - cover_count
    # 先向上对齐到 batch 的整数倍（至少覆盖 cover_count 条），
    # 再额外吃掉最多 extra_batches 个 batch 的缺口（可调，默认 3），最后不超过总数。
    target = max(batch_size, math.ceil(cover_count / batch_size) * batch_size)
    target += min(gap_num // batch_size, max(0, int(extra_batches))) * batch_size
    target = min(target, N)
    logger.info(f"--> We cover {target} contigs for training.")
    new_min_length = lengths[target - 1]
    # 只允许往下调；不允许抬高阈值（原实现会返回负值）
    return max(0, min_contig_length - new_min_length)


def binning_with_all_steps(
    contig_file_path: str,
    sorted_bam_file_list: List[str],
    temp_file_folder_path: str,
    bin_output_folder_path: str,
    n_views=6,
    feature_dim=100,
    auto_feature_dim=False,
    db_folder_path: str = None,
    count_kmer: int = 4,
    min_contig_length=850,
    leiden_iter_mode="fast",  # "accurate" or "fast"
    leiden_seed=42,
    drop_p=0.15,
    lr=1e-5,
    lr_multiple=10,
    lr_warmup_epoch=2,
    weight_deay=1e-3,
    batch_size=544,
    base_epoch=35,
    log_every_n_steps=10,
    training_device="cpu",
    cpu_workers=None,
    gpu_dataloader_workers=32,
    min_training_step=36,
    step_num=None,
    remove_temp_files=True,
    auto_disable_pretrain=True,
    pretrain_model_weight_path=None,
    low_gap_extra_batches=3,
    reexec_for_hash_seed=True,
    ensemble_clustering=True,
):
    """Run the complete, resumable CompleteBin binning workflow.

    The workflow has three stages: (1) contig filtering, coverage/k-mer
    feature preparation and marker-gene calling; (2) contrastive-model
    training and embedding generation; and (3) two Leiden clustering rounds,
    SCG-aware dereplication, and K-means recovery of residual low-quality
    contigs. Cached products in ``temp_file_folder_path`` are reused, making
    it possible to run the three stages on different CPU/GPU nodes.

    Args:
        contig_file_path (str): Input contig FASTA. Contig identifiers are
            normalised internally to remove whitespace.
        sorted_bam_file_list (List[str]): Coordinate-sorted BAM files used to
            construct contig coverage profiles.
        temp_file_folder_path (str): Persistent work directory for cached
            features, training artefacts, marker-gene calls and clustering data.
        bin_output_folder_path (str): Destination directory for final MAG FASTAs
            and ``MetaInfo.tsv``.
        n_views (int): Number of augmented views used for contrastive training.
        feature_dim (int): Requested feature dimension before any automatic
            feature-dimension selection.
        auto_feature_dim (bool): Whether to select the feature dimension from
            the input data automatically.
        db_folder_path (str, optional): CompleteBin database directory. If None,
            the ``CompleteBin_DB`` environment variable is used.
        count_kmer (int): k-mer size used during feature construction.
        min_contig_length (int): Initial minimum contig length in bp. The
            effective threshold may be adjusted from input statistics.
        leiden_iter_mode (str): Leiden optimisation mode: ``"accurate"`` for
            convergence or ``"fast"`` for adaptive early stopping.
        leiden_seed (int): Random seed for the leidenalg ``Optimiser``. This is
            a separate parameter from the global training ``seed`` (which stays
            hardcoded at 2048): it only determines which local optimum Leiden
            lands on for a given graph. Default 42 — the only seed that scored
            the highest deduplicated NC (HQ genome count) on both marine-7 and
            marine-9 in the seed sweep. It also has the lowest MQ, since NC and
            MQ are mutually exclusive in the deduplicated accounting.
        drop_p (float): Training dropout probability.
        lr (float): Base learning rate.
        lr_multiple (float): Multiplier applied to the learning rate schedule.
        lr_warmup_epoch (int): Number of learning-rate warm-up epochs.
        weight_deay (float): L2 regularisation weight. The spelling is retained
            for API compatibility.
        batch_size (int): Training batch size.
        base_epoch (int): Baseline number of training epochs.
        log_every_n_steps (int): Logging interval during training.
        training_device (str): PyTorch device for training, e.g. ``"cpu"`` or
            ``"cuda:0"``.
        cpu_workers (int, optional): CPU workers for preprocessing, marker-gene
            calling and clustering. If None, it is auto-detected.
        gpu_dataloader_workers (int): Data-loader workers used during training.
        min_training_step (int): Minimum training steps per epoch.
        step_num (int, optional): Split-workflow stop point: ``1`` prepares
            data then returns; ``2`` trains using prepared data then returns;
            ``3`` runs binning/clustering using cached stages 1 and 2; None
            runs the complete workflow in one invocation.
        remove_temp_files (bool): Remove first/second clustering temporary
            directories after their corresponding outputs are finalised.
        auto_disable_pretrain (bool): Automatically skip pretrained weights when
            the input statistics indicate that pretraining is unsuitable.
    """
    seed = 2048
    cov_time_s = time.time()
    # 可复现性：在任何子进程被创建之前，先把字符串哈希随机化固定下来。
    # 这一步可能以相同参数重启解释器一次（见 ensure_hash_seed 的说明），
    # 因此必须放在"开始干活/写文件"之前。
    ensure_hash_seed(reexec=reexec_for_hash_seed)
    logger.info(f"--> CompleteBin version: *** {bin_v} ***. The random seed is {seed}.")
    logger.info(f"--> Leiden rng_seed is {leiden_seed} (a separate parameter from the global seed above).")
    # 开跑前自检：种子透传链路必须完整（曾被回退过，见函数 docstring）
    assert_leiden_seed_plumbing()
    mp.set_start_method("fork", force=True)
    seed_everything(seed)

    if cpu_workers is None:
        cpu_workers = psutil.cpu_count() // 3 + 1
    logger.info(f"--> Total CPUs: {psutil.cpu_count()}. Number of {cpu_workers} CPUs are applied.")

    if os.path.exists(temp_file_folder_path) is False:
        os.mkdir(temp_file_folder_path)

    #############################################
    # Remove the space in the name of contigs.
    logger.info("--> Start to filter the space in contig name. Make sure the first string of contig name is unique in fasta file.")
    output_fasta_path = os.path.join(temp_file_folder_path, "filtered_space_in_name.contigs.fasta")
    if os.path.exists(output_fasta_path) is False:
        filterSpaceInFastaFile(contig_file_path, output_fasta_path)
    contig_file_path = output_fasta_path

    ####################################
    # prepare the files in databases
    if db_folder_path is None:
        db_folder_path = os.environ["CompleteBin_DB"]
    markerset_path = os.path.join(db_folder_path, "data", "markersets.ms")
    pfma_file_path = os.path.join(db_folder_path, "data", "pfam_file.dat")
    layers = 4
    if pretrain_model_weight_path is None:
        pretrain_model_weight_path = os.path.join(
            db_folder_path, "CheckPoint", "pretrain_weight_ep127.pth",
        )
    split_parts_list = [1, 11]
    logger.info("--> Start to read contigs.")

    training_data_path = os.path.join(temp_file_folder_path, "training_data.npy")
    contigname2seq_path = os.path.join(temp_file_folder_path, "contigname2seq_str.pkl")
    contigname2bp_nparray_list_path = os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl")
    coverage_profile_path = None
    # 覆盖度统计文件：新名字 coverage_stats.pkl，旧目录里可能还是 mean_var.pkl。
    coverage_stats_path = resolve_coverage_stats_path(temp_file_folder_path)

    contigname2seq_ori = readFasta(contig_file_path)

    min_contig_length = auto_min_contig_length(contigname2seq_ori, current_min_len=min_contig_length)
    N50 = calculateN50(contigname2seq_ori)
    if min_contig_length < 768:
        min_contig_length = 768
        logger.info(f"--> Set min contig length as 768 since the constraints of pretrain model.")
    large_data_size_thre = 115000
    temp = temp_decision(contigname2seq_ori, min_contig_length, N50, large_data_size_thre)
    low_gap = decision_lower_bound(
        contigname2seq_ori,
        min_contig_length,
        batch_size,
        extra_batches=low_gap_extra_batches,
    )
    # 阈值不允许降到 768 bp 以下：预训练编码器是按"片段 ≥768 bp、切 5 个子片段"
    # 训练的，再短就超出它的设计范围；同时保持
    # "聚类阈值 = 训练阈值 + low_gap = 原始阈值" 这个不变式（step3 会 +low_gap 还原）。
    max_gap = max(min_contig_length - 768, 0)
    if low_gap > max_gap:
        logger.info(
            f"--> Lowered threshold limited to 768 bp: gap {low_gap} -> {max_gap} "
            f"(would otherwise train on fragments shorter than the pretrain minimum)."
        )
        low_gap = max_gap
    min_contig_length -= low_gap

    # Step 1 is resumable.  Keep track of whether this invocation actually
    # performed work so a cache-hit invocation cannot overwrite a previous
    # real timing with 0 seconds.
    step1_preparation_ran = os.path.exists(training_data_path) is False
    cal_coverage_time = 0.0
    if step1_preparation_ran:
        ########################################################
        # STEP1: Get the coverage information of contigs.
        logger.info(f"--> Model Weight Path: {pretrain_model_weight_path}")
        logger.info(f"--> N50: {N50}, training temperature: {temp}, cluster mode: leiden + leiden.")
        logger.info(
            f"--> The min contigs length for training is: {min_contig_length}, for clustering is {min_contig_length + low_gap}.")
        if os.path.exists(contigname2seq_path) is False:
            prepare_sequences_coverage(
                contig_file_path,
                sorted_bam_file_list,
                temp_file_folder_path,
                min_contig_length,
                os.path.join(db_folder_path, "HMM", "40_marker.hmm"),
                os.path.join(db_folder_path, "HMM", "get_40marker.pl"),
                cpu_workers
            )

        cov_time_e = time.time()
        cal_coverage_time = cov_time_e - cov_time_s

        #########################################################
        # the following four files would be generated by function "prepare_sequences_coverage"
        mean_val, std_val = readPickle(coverage_stats_path)
        if auto_feature_dim:
            feature_dim = int(math.log(len(readPickle(contigname2seq_path))) * 8.33)

        # ── Gap-based coverage outlier 检测与截断 ──
        clip_result = clip_coverage_outliers(
            contigname2seq_path,
            contigname2bp_nparray_list_path,
            coverage_stats_path,
            gap_threshold_mean=1.15,
            gap_threshold_std=1.265,
            min_length=2500,
        )
        logger.info(f"--> Coverage outlier processing: "
                    f"removed={clip_result.get('n_removed', 0)}, "
                    f"clipped={clip_result.get('n_clipped', 0)}, "
                    f"std_protected={clip_result.get('n_std_only_protected', 0)}")

        # 重新读取更新后的 mean_var
        mean_val, std_val = readPickle(coverage_stats_path)

        # 用过滤后的 contig 重新生成 40-marker 种子
        filtered_fasta = os.path.join(temp_file_folder_path, "filtered_contigs.fasta")
        filtered_contigname2seq = readPickle(contigname2seq_path)
        writeFasta(filtered_contigname2seq, filtered_fasta)
        seed_folder = os.path.join(temp_file_folder_path, "seed_folder")
        if os.path.exists(seed_folder) is False:
            os.mkdir(seed_folder)
        gen_seed(
            filtered_fasta,
            cpu_workers,
            seed_folder,
            os.path.join(db_folder_path, "HMM", "40_marker.hmm"),
            os.path.join(db_folder_path, "HMM", "get_40marker.pl"),
            min_contig_length,
        )
        if os.path.exists(filtered_fasta):
            os.remove(filtered_fasta)
        # ── Outlier 处理结束 ──

    logger.info(
        f"--> Dropout probability: {drop_p}, n-views: {n_views}, base epoch is {base_epoch}, batch size is {batch_size}, feature_dim: {feature_dim}.")
    #########################################################
    # build training data
    # this function would generate 'training_data.npy' file in 'temp_file_folder_path'
    cov_time_s = time.time()
    process_data_ran = os.path.exists(training_data_path) is False and \
        os.path.exists(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl")) is False
    if process_data_ran:
        build_training_seq_data_numpy_save(
            contigname2seq_path,
            contigname2bp_nparray_list_path,
            temp_file_folder_path,
            count_kmer,
            min_contig_length,
            cpu_workers,
        )
    logger.info(f"--> Step 1 is over.")
    cov_time_e = time.time()
    cal_data_time = cov_time_e - cov_time_s
    if step_num is None or step_num == 1:
        step1_times = {}
        if step1_preparation_ran:
            step1_times["CalculateCoverageTime(s)"] = cal_coverage_time
        if process_data_ran:
            step1_times["ProcessDataTime(s)"] = cal_data_time
        if step1_times:
            _write_time_tsv(temp_file_folder_path, step1_times)
    if step_num is not None and step_num == 1:
        return 0
    ##########################################################
    # STEP2:  Strat to training
    cov_time_s = time.time()
    if os.path.exists(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl")) is False:
        # model training
        contigname2seq = readPickle(contigname2seq_path)
        mean_val, std_val = readPickle(coverage_stats_path)
        # Compute k-mer sufficiency S from training contig lengths
        if auto_disable_pretrain:
            train_lengths = [len(seq) for seq in contigname2seq.values()]
            train_avg_len = sum(train_lengths) / max(len(train_lengths), 1)
            S = compute_S(train_avg_len, min_contig_length)
            logger.info(
                f"--> Training contigs: n={len(train_lengths)}, "
                f"avg_len={train_avg_len:.0f}bp, S={S:.3f}"
            )
        else:
            S = 0.0
        model_save_folder = os.path.join(temp_file_folder_path, "model_save")
        if os.path.exists(model_save_folder) is False:
            os.mkdir(model_save_folder)
        # epoch setting, base epoch
        if N50 >= 1536:
            base_epoch += 4
        sampler = DeeperBinSampler(len(contigname2seq) // batch_size * batch_size, batch_size, min_training_step=min_training_step, seed=seed)
        num_contigs = len(sampler.final_sample)
        if num_contigs >= large_data_size_thre:
            train_epoch = base_epoch
        else:
            train_epoch = large_data_size_thre * base_epoch // num_contigs
        if train_epoch > 200:
            train_epoch = 200

        # ---- Automatic pretrained model decision ----
        if auto_disable_pretrain:
            O = compute_O(train_epoch, num_contigs)
            use_pretrained, decision_reason = decide_pretrained(S, O)
            logger.info(f"--> Pretrain decision: {decision_reason}")
            # Safety valve: reduce epoch cap only under extreme overfitting
            train_epoch, valve_reason = apply_epoch_safety_valve(train_epoch, S, O)
            if valve_reason:
                logger.info(f"--> Epoch safety valve: {valve_reason}")
        else:
            use_pretrained = True
            logger.info("--> Auto pretrain disabled: always using pretrained model.")

        trainer_obj = SelfSupervisedMethodsTrainer(
            feature_dim,
            n_views,
            drop_p,
            training_device,
            temperature_simclr=temp,
            min_contig_len=min_contig_length,
            batch_size=batch_size,
            lr=lr,
            lr_multiple=lr_multiple,
            lr_warmup_epoch=lr_warmup_epoch,
            sampler=sampler,
            train_epoch=train_epoch,
            weight_decay=weight_deay,
            training_data_path=training_data_path,
            model_save_folder=model_save_folder,
            emb_output_folder=temp_file_folder_path,
            count_kmer=count_kmer,
            split_parts_list=split_parts_list,
            N50=N50,
            num_bam_files=len(sorted_bam_file_list),
            mean_std_val=(mean_val, std_val),
            pretrain_model_weight_path=pretrain_model_weight_path,
            log_every_n_steps=log_every_n_steps,
            dataloader_workers=gpu_dataloader_workers,
            layers=layers,
            seed=seed,
            use_pretrained=use_pretrained,
        )

        logger.info(f"--> Start to train model. The temperature is {temp}.")
        trainer_obj.train(load_epoch_set=None)
        logger.info(f"--> Start to inference contig embeddings with model.")
        min_epoch_set = trainer_obj.inference(min_epoch_set=None)
        copy(os.path.join(model_save_folder, f'checkpoint_{min_epoch_set}.pth'), temp_file_folder_path)
        if os.path.exists(model_save_folder):
            rmtree(model_save_folder, ignore_errors=True)

    logger.info(f"--> Step 2 is over.")
    cov_time_e = time.time()
    train_time = cov_time_e - cov_time_s
    if step_num is None or step_num == 2:
        _write_time_tsv(temp_file_folder_path, {"TrainingTime(s)": train_time})
    if step_num is not None and step_num == 2:
        return 0
    #############################################################
    # STEP3: Start Clustring
    cov_time_s = time.time()
    min_contig_length += low_gap
    contigname2seq = readPickle(contigname2seq_path)
    simclr_contigname2emb_norm_array = readPickle(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl"))
    simclr_emb_list = []
    length_list = []
    sub_contigname_list = []
    for contigname, seq in contigname2seq.items():
        length = len(seq)
        if length < min_contig_length:
            continue
        sub_contigname_list.append(contigname)
        simclr_emb_list.append(simclr_contigname2emb_norm_array[contigname])
        length_list.append(length)
    logger.info(f"--> There are {len(contigname2seq)} contigs for training and {len(simclr_emb_list)} for clustering.")

    initial_fasta_path = os.path.join(temp_file_folder_path, "split_contigs_initial_kmeans")
    if os.path.exists(initial_fasta_path) is False:
        os.mkdir(initial_fasta_path)

    seed_path = os.path.join(temp_file_folder_path, "seed_folder", "bacar_marker.2quarter.seed")
    seed_set = readSeedFile(seed_path)
    bin_number = int(len(seed_set) * 1.5) + 1
    if len(os.listdir(initial_fasta_path)) != bin_number:
        kmeans_split(
            logger,
            initial_fasta_path,
            contigname2seq,
            sub_contigname_list,
            np.stack(simclr_emb_list, axis=0),
            np.array(length_list),
            bin_number,
            seed_set,
            min_contig_length
        )
    call_genes_folder = os.path.join(temp_file_folder_path, "call_genes")
    if os.path.exists(os.path.join(temp_file_folder_path, "bac_gene_info.pkl")) is False or \
            os.path.exists(os.path.join(temp_file_folder_path, "arc_gene_info.pkl")) is False:
        bac_acc_set = set()
        arc_acc_set = set()
        marset = readMarkerSets(markerset_path)
        for cur_gene_set in marset["d__Bacteria"]:
            for cur_gene in cur_gene_set:
                bac_acc_set.add(cur_gene)
        for cur_gene_set in marset["d__Archaea"]:
            for cur_gene in cur_gene_set:
                arc_acc_set.add(cur_gene)
        callMarkerGenesByHMM(
            initial_fasta_path,
            call_genes_folder,
            cpu_workers,
            hmm_model_path=os.path.join(db_folder_path, "HMM", "bac_arc_hmm_model_extended.hmm"),
            bin_suffix="fasta",
            bac_acc_set=bac_acc_set,
            arc_acc_set=arc_acc_set,
            pfma_file_path=pfma_file_path,
            output_folder=temp_file_folder_path
        )
    bac_gene2contigNames, bac_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "bac_gene_info.pkl"))
    arc_gene2contigNames, arc_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "arc_gene_info.pkl"))
    if os.path.exists(call_genes_folder):
        rmtree(call_genes_folder, ignore_errors=True)

    coverage_profile_path = ensure_compact_coverage_path(contigname2bp_nparray_list_path)
    logger.info(f"--> Improved polish coverage profile ready: {coverage_profile_path}")

    # ── 聚类方案分支 ────────────────────────────────────────────────────────
    # ensemble_clustering=False 时走改进方案 H1（IMPROVEMENT.md），
    # 它自己完成「选划分 → 粗筛+精筛 → k-means 重聚 → 写 MAG」，
    # 因此这里直接返回，下面原有的三段式流程完全不执行、也一行未改。
    if not ensemble_clustering:
        logger.info("--> ensemble_clustering=False: 使用 IMPROVEMENT.md 的 H1 方案"
                    "（先选划分 → 粗筛+精筛 → k-means 重聚）。")
        run_improved_clustering(
            temp_file_folder_path=temp_file_folder_path,
            clustering_folder_name="first_clustering_temp",
            contigname2seq=contigname2seq,
            simclr_contigname2emb_norm_array=simclr_contigname2emb_norm_array,
            markerset_path=markerset_path,
            min_contig_length=min_contig_length,
            cpu_workers=cpu_workers,
            seed_path=seed_path,
            bac_contigName2_gene2num=bac_contigName2_gene2num,
            arc_contigName2_gene2num=arc_contigName2_gene2num,
            mag_length_threshold=100000,
            bin_output_folder_path=bin_output_folder_path,
            leiden_iter_mode=leiden_iter_mode,
            coverage_profile_path=coverage_profile_path,
            leiden_seed=leiden_seed,
            bac_gene2contigNames=bac_gene2contigNames,
        )
        cluster_time = time.time() - cov_time_s
        _write_time_tsv(temp_file_folder_path, {"ClusteringTime(s)": cluster_time})
        return 0

    # Decide once from the full input, not each round's remaining contigs.
    from CompleteBin.Dereplication.tied_representative import (
        PLANT_MAX_CONTIGS_GE_768, TIED_POLICY_MIN_CONTIG_LENGTH,
        should_prefer_longer_representative,
    )
    num_contigs_ge_768 = sum(
        len(seq) >= TIED_POLICY_MIN_CONTIG_LENGTH
        for seq in contigname2seq_ori.values()
    )
    prefer_longer_representative = should_prefer_longer_representative(
        len(sorted_bam_file_list), num_contigs_ge_768,
    )
    logger.info(
        "--> Longer tied representative replacement: %s; BAM files=%d; "
        "full-input contigs >=768 bp=%d; small-input cutoff: <%d. "
        "Same policy applies to both clustering stages.",
        "enabled" if prefer_longer_representative else "disabled",
        len(sorted_bam_file_list), num_contigs_ge_768, PLANT_MAX_CONTIGS_GE_768,
    )

    # first clustering
    first_clustering_bins = os.path.join(temp_file_folder_path, "first_clustering_bins")
    if os.path.exists(os.path.join(first_clustering_bins, "MetaInfo.tsv")) is False:
        clustering_and_dereplication(
            temp_file_folder_path,
            "first_clustering_temp",
            contigname2seq,
            simclr_contigname2emb_norm_array,
            markerset_path,
            min_contig_length,
            cpu_workers,
            seed_path,
            bac_contigName2_gene2num,
            arc_contigName2_gene2num,
            100000,
            first_clustering_bins,
            leiden_iter_mode,
            clustering_stages=1,
            coverage_profile_path=coverage_profile_path,
            leiden_seed=leiden_seed,
            prefer_longer=prefer_longer_representative,
        )
    if remove_temp_files and os.path.exists(os.path.join(temp_file_folder_path, "first_clustering_temp")):
        rmtree(os.path.join(temp_file_folder_path, "first_clustering_temp"))
    # selected contigs for next clustering
    sec_seleceted_contigs = selected_recluster_contigs(contigname2seq, first_clustering_bins, ["HighQuality", "MediumQuality"], comp_thre=50.)
    sec_contigname2seq = {}
    for name in sec_seleceted_contigs:
        sec_contigname2seq[name] = contigname2seq[name]
    # ## second clustering
    second_clustering_bins = os.path.join(temp_file_folder_path, "second_clustering_bins")
    if os.path.exists(os.path.join(second_clustering_bins, "MetaInfo.tsv")) is False:
        clustering_and_dereplication(
            temp_file_folder_path,
            "second_clustering_temp",
            sec_contigname2seq,
            simclr_contigname2emb_norm_array,
            markerset_path,
            min_contig_length,
            cpu_workers,
            seed_path,
            bac_contigName2_gene2num,
            arc_contigName2_gene2num,
            100000,
            second_clustering_bins,
            leiden_iter_mode,
            clustering_stages=2,
            coverage_profile_path=coverage_profile_path,
            leiden_seed=leiden_seed,
            prefer_longer=prefer_longer_representative,
        )
    if remove_temp_files and os.path.exists(os.path.join(temp_file_folder_path, "second_clustering_temp")):
        rmtree(os.path.join(temp_file_folder_path, "second_clustering_temp"))
    # third clustering with K-Mearns
    thi_seleceted_contigs = selected_recluster_contigs(sec_contigname2seq, second_clustering_bins, ["HighQuality", "MediumQuality"], comp_thre=50.)
    thi_contigname2seq = {}
    for name in thi_seleceted_contigs:
        thi_contigname2seq[name] = contigname2seq[name]
    third_clustering_bins = os.path.join(temp_file_folder_path, "thi_clustering_bins")
    cluster_kmeans_for_low_v2(
        thi_contigname2seq,
        simclr_contigname2emb_norm_array,
        bac_gene2contigNames,
        bac_contigName2_gene2num,
        0,
        third_clustering_bins,
        first_metainfo_path=os.path.join(first_clustering_bins, "MetaInfo.tsv"),
        second_metainfo_path=os.path.join(second_clustering_bins, "MetaInfo.tsv"),
        min_k_ratio=0.1,
        percentile=10
    )
    # select final bins from first and second results.
    first_selected_list = readPickle(os.path.join(first_clustering_bins, "bin_copy_list.pkl"))
    second_selected_list = readPickle(os.path.join(second_clustering_bins, "bin_copy_list.pkl"))
    index = 0
    if os.path.exists(bin_output_folder_path) is False:
        os.mkdir(bin_output_folder_path)
    wh = open(os.path.join(bin_output_folder_path, "MetaInfo.tsv"), "w")
    for item_tuple in first_selected_list + second_selected_list:
        ori_path, comp, cont, state = item_tuple
        outName = f"CompleteBin_{index}.fasta"
        writeMetaInfo(wh, outName, comp, cont, state)
        copy(ori_path, os.path.join(bin_output_folder_path, outName))
        index += 1
    for filename in os.listdir(third_clustering_bins):
        comp = "0."
        cont = "0."
        state = "LowQuality"
        outName = f"CompleteBin_{index}.fasta"
        writeMetaInfo(wh, outName, comp, cont, state)
        copy(os.path.join(third_clustering_bins, filename), os.path.join(bin_output_folder_path, outName))
        index += 1
    wh.close()

    cov_time_e = time.time()
    cluster_time = cov_time_e - cov_time_s
    _write_time_tsv(temp_file_folder_path, {"ClusteringTime(s)": cluster_time})
    return 0
