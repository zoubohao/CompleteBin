"""挑"离其他基因组最远"的基因组当测试集。

为什么要专门挑：如果测试集里的基因组在训练集里有近缘（同一物种/属的近identical
菌株），那"留出基因组上的检索"就变得过分容易——模型只要把训练里见过的近缘基因组
的判别能力搬过来就行，测出来的数字不能代表它对**没见过的远缘类群**的能力。

判据分两级：

  1. **主判据：skani 的 ANI**（平均核苷酸一致性，Average Nucleotide Identity，
     两条基因组能比对上的区域里碱基一致的比例，单位 %）。用 `skani triangle`
     全库两两算一遍，对每条基因组取"它和最近缘那条的 ANI"，记作 max_ani。
     **没有任何一条与它比对覆盖 >= min_af 的基因组，max_ani 记 0**，是最孤立的一档。
  2. **同分次序与互斥：4-mer 组成向量**。当一批基因组的 max_ani 同为 0（都很孤立）
     时，用"该基因组与全库最像的那条的余弦相似度"来排序（组成向量先按全库均值
     中心化，避免被 GC 含量这种共同成分主导），并保证选中的彼此之间不撞在一起。

（早期版本只用 4-mer 组成余弦，实测区分度不够：14734 条里只有 3 条的最近邻余弦
低于 0.5。skani 的 ANI 才是"是不是同一个种/属"的直接度量。）
"""

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from CompleteBin.DataProcess.data_utils import get_normlized_count_vec_of_seq_fast
from CompleteBin.Seqs.seq_utils import generate_feature_mapping_reverse

# 和 PretrainDataset 保持一致的长度档与上限（测试片段必须先满足这个设计）。
DEFAULT_LENGTH_BINS = (
    (768, 1500),
    (1500, 2500),
    (2500, 5000),
    (5000, 10000),
    (10000, 50000),
)
DEFAULT_MAX_SEQ_LENGTH = 50000
# 每个测试基因组至少要凑出的片段条数（用户要求）。
MIN_TEST_CONTIGS_PER_GENOME = 200


def verify_test_genome(path,
                       contigs_per_bin,
                       length_bins=DEFAULT_LENGTH_BINS,
                       max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                       seed=2048):
    """检查一条基因组能不能真的抽出 contigs_per_bin × 档数 条**互不重叠**片段。

    返回 (ok, message)。这一步是必须的：片段互不重叠之后，一个基因组要放下 200 条
    窗口就需要约 1.55 Mb 的空闲序列（分档抽长度时的期望值），小基因组或碎片化
    组装放不下。放不下的基因组不能当测试集，否则要么抽不满、要么又退回互相重叠。
    """
    from CompleteBin.Trainer.dataset import (  # 局部导入，避免模块级循环依赖
        PretrainDataset,
        plan_disjoint_test_windows,
    )
    import random as _random

    sequences = PretrainDataset._read_fasta_sequences(path)
    if not sequences:
        return False, "没有 >=768 bp 且 N<=5% 的序列"
    try:
        plan, stats = plan_disjoint_test_windows(
            sequences, length_bins, contigs_per_bin, max_seq_length,
            _random.Random(seed))
    except ValueError as error:
        return False, str(error)
    return True, (f"{len(plan)} 条互不重叠，各档 {stats['placed_per_bin']}，"
                  f"占用 {stats['used_bases'] / 1e6:.2f}/"
                  f"{stats['genome_bases'] / 1e6:.2f} Mb")


def read_fasta_prefix(path, prefix_bp):
    """读 FASTA 开头最多 prefix_bp 个碱基（跨记录累加，遇到 header 只跳过不截断）。

    注意不能只看第一条记录：库里有很多碎片化组装，第一条 contig 可能只有几百 bp，
    只看它会有约 15% 的基因组算不出组成向量。
    """
    chunks = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            chunks.append(line)
            total += len(line)
            if total >= prefix_bp:
                break
    return "".join(chunks)[:prefix_bp]


def genome_composition_vector(path, count_kmer=4, prefix_bp=200_000):
    """一条基因组的 136 维 4-mer 组成向量（已 L2 归一化）；算不出来返回 None。"""
    sequence = read_fasta_prefix(path, prefix_bp).upper().replace("N", "A")
    if len(sequence) < 4000:
        return None
    kmer_dict, nr_features = generate_feature_mapping_reverse(count_kmer)
    vector, _ = get_normlized_count_vec_of_seq_fast(
        sequence, kmer_dict, nr_features, count_kmer, None, False)
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        return None
    return vector / norm


def skani_max_ani(paths, threads=64, min_af=15.0, work_dir=None, skani_bin="skani",
                  logger=None):
    """用 skani 算每条基因组到"最近缘"的 ANI。

    返回 (max_ani, best_partner)：两个 dict，键是 FASTA 路径。
    max_ani 为 0 表示全库两两比对里没有任何一条与它有 >= min_af 的比对覆盖，
    也就是这一档里最孤立的。
    """
    def log(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message, flush=True)

    own_tmp = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(
        prefix="skani_far_", dir=os.environ.get("TMPDIR") or None)
    os.makedirs(work_dir, exist_ok=True)
    list_path = os.path.join(work_dir, "fasta_list.txt")
    edges_path = os.path.join(work_dir, "skani_edges.tsv")
    with open(list_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(paths) + "\n")

    command = [skani_bin, "triangle", "-t", str(int(threads)),
               "--sparse", "--min-af", str(min_af),
               "-l", list_path, "-o", edges_path]
    log(f"--> [far-test] skani: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        tail = result.stdout.decode("utf-8", "ignore").strip().splitlines()[-2:]
        for line in tail:
            log(f"--> [far-test] skani: {line.strip()}")
    except (OSError, subprocess.CalledProcessError) as error:
        if own_tmp:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"skani failed: {error}") from error

    max_ani = {path: 0.0 for path in paths}
    best_partner = {path: None for path in paths}
    n_edges = 0
    with open(edges_path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        position = {name: index for index, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < len(header):
                continue
            left = fields[position["Ref_file"]]
            right = fields[position["Query_file"]]
            try:
                ani = float(fields[position["ANI"]])
            except ValueError:
                continue
            n_edges += 1
            for this, other in ((left, right), (right, left)):
                if this not in max_ani:
                    continue
                if ani > max_ani[this]:
                    max_ani[this] = ani
                    best_partner[this] = os.path.basename(other)

    n_isolated = sum(1 for value in max_ani.values() if value == 0.0)
    log(f"--> [far-test] skani 完成：{n_edges} 条配对达到覆盖阈值，"
        f"{n_isolated}/{len(paths)} 条基因组的 max_ani = 0（全库找不到近缘）")
    if own_tmp:
        shutil.rmtree(work_dir, ignore_errors=True)
    return max_ani, best_partner


def select_far_test_genomes(paths,
                            n_test,
                            prefix_bp=200_000,
                            min_file_bytes=3_000_000,
                            workers=32,
                            similarity_cap=0.5,
                            chunk=512,
                            center=True,
                            skani_threads=64,
                            skani_min_af=15.0,
                            skani_bin="skani",
                            skani_work_dir=None,
                            required_contigs_per_genome=MIN_TEST_CONTIGS_PER_GENOME,
                            logger=None):
    """返回 (selected_paths, report)。"""
    def log(message):
        if logger is not None:
            logger.info(message)
        else:
            print(message, flush=True)

    # 所有合格基因组都参与排名；`min_file_bytes` 只用来决定"能不能被选中"——
    # 它保证被选中的那 10 条有足够序列去抽 5 档 × test_contigs_per_bin 条测试片段。
    # （早先的版本把这条当成候选池的入场券，会无谓地把 570 条小基因组整个踢出排名。）
    candidates = list(paths)
    is_large = {path: os.path.getsize(path) >= min_file_bytes for path in candidates}
    n_large = sum(1 for value in is_large.values() if value)
    log(f"--> [far-test] 参与排名 {len(candidates)} 条基因组；其中 {n_large} 条 >= "
        f"{min_file_bytes / 1e6:.1f} MB 可被选中，其余 {len(candidates) - n_large} 条"
        f"只参与排名（序列太短，抽不出足够的测试片段）")
    if n_large <= n_test:
        raise ValueError(
            f"only {n_large} genomes reach {min_file_bytes} bytes, cannot hold out {n_test}")

    # ── 主判据：skani ANI（全库两两，取每条的"最近缘 ANI"）──────────────
    max_ani, best_partner = skani_max_ani(
        candidates, threads=skani_threads, min_af=skani_min_af,
        work_dir=skani_work_dir, skani_bin=skani_bin, logger=logger)

    # ── 次序/互斥判据：中心化 4-mer 组成余弦 ──────────────────────────
    def job(path):
        try:
            return genome_composition_vector(path, prefix_bp=prefix_bp)
        except Exception:  # noqa: BLE001 - 单条文件坏掉不该拖垮整个划分
            return None

    log(f"--> [far-test] 读取每条的 FASTA 前缀（<= {prefix_bp} bp）并算 4-mer 组成向量")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        vectors = list(executor.map(job, candidates))
    usable = [(path, vec) for path, vec in zip(candidates, vectors) if vec is not None]
    if len(usable) <= n_test:
        raise ValueError("too few genomes produced a usable composition vector")
    paths_used = [path for path, _ in usable]
    matrix = np.stack([vec for _, vec in usable]).astype(np.float64)
    if center:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        log("--> [far-test] 组成向量已按全库均值中心化（去掉共同成分）")

    max_similarity = np.full(len(paths_used), -1.0, dtype=np.float32)
    for start in range(0, len(paths_used), chunk):
        end = min(start + chunk, len(paths_used))
        block = matrix[start:end] @ matrix.T
        block[np.arange(end - start), np.arange(start, end)] = -1.0
        max_similarity[start:end] = block.max(axis=1)

    # 排序键：(最近缘 ANI 升序, 组成余弦升序)。max_ani 为 0 的一批全在最前面，
    # 它们之间再按"组成有多不寻常"排。
    key = [(float(max_ani.get(path, 0.0)), float(max_similarity[index]))
           for index, path in enumerate(paths_used)]
    order = sorted(range(len(paths_used)), key=lambda index: key[index])

    selected = []
    rejection_log = []

    def can_host_test_contigs(path):
        """这一条能不能放下 required_contigs_per_genome 条互不重叠片段。"""
        per_bin = max(1, int(round(required_contigs_per_genome
                                   / len(DEFAULT_LENGTH_BINS))))
        ok, message = verify_test_genome(path, per_bin)
        if not ok:
            rejection_log.append((os.path.basename(path), message))
        return ok

    for index in order:
        if len(selected) >= n_test:
            break
        if not is_large[paths_used[index]]:
            continue
        if selected:
            conflicts = matrix[[index]] @ matrix[np.array(selected)].T
            if float(conflicts.max()) >= similarity_cap:
                continue
        if not can_host_test_contigs(paths_used[index]):
            continue
        selected.append(index)
    # 第一遍按 similarity_cap 挑互相不撞的；挑不满（例如整个库里所有基因组的组成
    # 向量都很接近）就按排序顺次补齐，不要让整个划分因此报错退出。补进来的记在
    # relaxed 里，事后可查。
    relaxed = []
    if len(selected) < n_test:
        logger_warn = (logger.warning if logger is not None else print)
        logger_warn(
            f"--> [far-test] 只有 {len(selected)} 个基因组满足 similarity_cap="
            f"{similarity_cap}，剩下的按孤立度顺序补齐（会放宽这个上限）。")
        for index in order:
            if len(selected) >= n_test:
                break
            if index in selected:
                continue
            if not is_large[paths_used[index]]:
                continue
            if not can_host_test_contigs(paths_used[index]):
                continue
            selected.append(index)
            relaxed.append(index)

    selected_paths = [paths_used[index] for index in selected]
    report = {
        "candidates": len(candidates),
        "candidates_large_enough_to_be_picked": n_large,
        "scored": len(paths_used),
        "prefix_bp": prefix_bp,
        "min_file_bytes": min_file_bytes,
        "similarity_cap": similarity_cap,
        "relaxed_picks": len(relaxed),
        "required_contigs_per_genome": int(required_contigs_per_genome),
        "rejected_cannot_host_test_contigs": [
            {"name": name, "reason": reason} for name, reason in rejection_log
        ],
        "skani_min_af": skani_min_af,
        "skani_isolated_count": sum(1 for value in max_ani.values() if value == 0.0),
        "selected": [
            {"name": os.path.basename(path),
             "max_ani_to_closest_genome": float(max_ani.get(path, 0.0)),
             "closest_genome": best_partner.get(path),
             "max_similarity_to_any_other": float(max_similarity[index]),
             "relaxed": index in relaxed}
            for path, index in zip(selected_paths, selected)
        ],
        "max_similarity_quantiles": {
            "min": float(max_similarity.min()),
            "p01": float(np.percentile(max_similarity, 1)),
            "median": float(np.median(max_similarity)),
            "max": float(max_similarity.max()),
        },
    }
    for item in report["selected"]:
        partner = item["closest_genome"] or "（全库没有近缘）"
        log(f"--> [far-test] 选中 {item['name']}  "
            f"最近缘 ANI={item['max_ani_to_closest_genome']:.2f} ({partner}) "
            f"组成余弦={item['max_similarity_to_any_other']:.4f}")
    if rejection_log:
        log(f"--> [far-test] 有 {len(rejection_log)} 个候选因为放不下 "
            f"{required_contigs_per_genome} 条互不重叠测试片段被跳过（最多显示 3 条）:")
        for name, reason in rejection_log[:3]:
            log(f"    · {name}: {reason}")
    log("--> [far-test] 全库组成余弦分布: "
        f"min={report['max_similarity_quantiles']['min']:.4f} "
        f"p01={report['max_similarity_quantiles']['p01']:.4f} "
        f"中位={report['max_similarity_quantiles']['median']:.4f}")
    return selected_paths, report
