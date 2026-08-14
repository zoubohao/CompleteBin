import functools
import os
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

from CompleteBin.IO import writeFasta, readMetaInfo
from CompleteBin.Cluster.split_utils import partial_seed_init
from CompleteBin.logger import get_logger


logger = get_logger()


def summedLengthCal(name2seq: Dict[str, str]) -> int:
    return sum(len(seq) for seq in name2seq.values())


def get_min_k_from_previous_rounds(
    first_metainfo_path: str,
    second_metainfo_path: str,
    min_k_ratio: float = 0.15
) -> int:
    """
    Calculate the minimum k for k-means clustering based on HQ + MQ MAG counts
    from the first two clustering rounds.

    Args:
        first_metainfo_path: Path to MetaInfo.tsv from the first clustering round.
        second_metainfo_path: Path to MetaInfo.tsv from the second clustering round.
        min_k_ratio: The ratio of (HQ + MQ) count to use as minimum k.
                     Defaults to 0.15 (15%).

    Returns:
        int: The minimum k value (always >= 3).
    """
    hq_mq_total = 0
    for metainfo_path in [first_metainfo_path, second_metainfo_path]:
        if metainfo_path is not None and os.path.exists(metainfo_path):
            _, h, m, _ = readMetaInfo(metainfo_path, comp_i=2, cont_i=3)
            hq_mq_total += (h + m)
            logger.info(
                f"--> Read MetaInfo: {metainfo_path} -> "
                f"HQ={h}, MQ={m}"
            )

    min_k = max(3, round(hq_mq_total * min_k_ratio))
    logger.info(
        f"--> Total HQ + MQ from previous rounds: {hq_mq_total}, "
        f"min_k_ratio: {min_k_ratio}, computed min_k: {min_k}"
    )
    return min_k


def cluster_kmeans_for_low_v2(
    sub_contigName2seq: Dict[str, str],
    contigName2RepNormV: Dict,
    gene2contigNames: Dict[str, List[str]],
    contigName2_gene2num: Dict[str, Dict[str, int]],
    mag_length_threshold: int,
    bin_output_folder_path: str,
    first_metainfo_path: str = None,
    second_metainfo_path: str = None,
    min_k_ratio: float = 0.15,
    percentile: float = 10,
) -> None:
    """
    K-means clustering for contigs from low-quality MAGs (third round).

    Improvements over the original:
      1. Multi-SCG k estimation — uses a configurable percentile (default 10th)
         of all SCG contig counts instead of a single SCG at the ~8th percentile.
      2. Multi-SCG seeds — seeds are drawn across all SCGs, prioritising
         contigs that carry more SCGs (better marker contigs).
      3. partial_seed_init — always seeds the first centres from SCG contigs
         and fills the rest via k-means++ (imported from split_utils).
      4. Log-transformed length weights — compresses the dominance of very
         long contigs so shorter contigs also influence the clustering.
      5. Multiple runs (n_init=10) — runs KMeans multiple times with
         different random completions of the remaining centres and keeps
         the best (lowest inertia) result.

    Args:
        sub_contigName2seq: Dict mapping contig names to sequences for the
                            remaining (low-quality) contigs.
        contigName2RepNormV: Dict mapping contig names to their normalized
                             representation vectors (embeddings).
        gene2contigNames: Dict mapping gene names to lists of contig names
                          that contain that gene.
        contigName2_gene2num: Dict mapping contig names to dicts of
                              {gene_name: copy_number}.
        mag_length_threshold: Minimum total length for a MAG to be written.
        bin_output_folder_path: Path to output folder for the new bins.
        first_metainfo_path: Path to MetaInfo.tsv from the first clustering
                             round. Used to count HQ + MQ MAGs.
        second_metainfo_path: Path to MetaInfo.tsv from the second clustering
                              round. Used to count HQ + MQ MAGs.
        min_k_ratio: Fraction of (HQ + MQ) count to set as the minimum k.
                     Defaults to 0.15 (15%).
        percentile: Percentile of the SCG contig-count distribution used to
                    estimate the number of clusters (k). Defaults to 10,
                    giving a conservative k that is lower than the median.
    """
    os.makedirs(bin_output_folder_path, exist_ok=True)

    contigSeqPair = [
        (contigName, len(seq)) for contigName, seq in sub_contigName2seq.items()
    ]
    if len(contigSeqPair) <= 3:
        return

    exist_contigs = [
        contig for contig, _ in sorted(contigSeqPair, key=lambda x: x[1], reverse=True)
    ]

    # ---- Build subset gene info for the existing contigs ----
    existGene2contigNames = {}
    existContig2RepNormV = {}
    for contig in exist_contigs:
        if contigName2RepNormV is not None:
            existContig2RepNormV[contig] = contigName2RepNormV[contig]
        if contig in contigName2_gene2num:
            curExistGenes2num = contigName2_gene2num[contig]
            for gene, _ in curExistGenes2num.items():
                if gene not in existGene2contigNames:
                    cur_set = set()
                else:
                    cur_set = existGene2contigNames[gene]
                for cur_contigName in gene2contigNames[gene]:
                    if cur_contigName in sub_contigName2seq:
                        cur_set.add(cur_contigName)
                existGene2contigNames[gene] = cur_set

    # ============================================================
    # Improvement 1: Multi-SCG estimation of cluster number k.
    # Use the median of contig counts across all SCGs instead of
    # a single SCG at the ~8th percentile.  The given percentile is robust
    # against outlier SCGs with unusually high / low copy numbers.
    # ============================================================
    scg_counts = [len(contigs) for contigs in existGene2contigNames.values()]
    if len(scg_counts) == 0:
        logger.warning("--> No SCG information available, fallback to k=3.")
        scg_based_cluster_num = 3
    else:
        scg_based_cluster_num = max(3, int(round(np.percentile(scg_counts, percentile))))
        logger.info(
            f"--> SCG count distribution across {len(scg_counts)} SCGs: "
            f"min={min(scg_counts)}, p{percentile}={np.percentile(scg_counts, percentile):.1f}, "
            f"median={np.median(scg_counts):.1f}, max={max(scg_counts)} "
            f"-> SCG-based k={scg_based_cluster_num}"
        )

    # ---- Determine minimum k from previous rounds' MetaInfo ----
    min_k = get_min_k_from_previous_rounds(
        first_metainfo_path,
        second_metainfo_path,
        min_k_ratio
    )

    # ---- Decide final k: apply only the lower bound (min_k). ----
    # No upper bound — the SCG-median estimate can grow freely;
    # only constrained by the number of available contigs.
    final_k = max(scg_based_cluster_num, min_k)

    n_samples = len(existContig2RepNormV)
    if final_k > n_samples:
        logger.warning(
            f"--> final_k ({final_k}) > n_contigs ({n_samples}), "
            f"clamping to {n_samples}."
        )
        final_k = n_samples

    logger.info(
        f"--> Final k = max(SCG_median={scg_based_cluster_num}, "
        f"min_k={min_k}) = {final_k}"
    )

    # ---- Build feature matrix X ----
    length_weights_raw = []
    X = []
    index2contigName = {}
    for j, (contigName, repNormVec) in enumerate(existContig2RepNormV.items()):
        X.append(repNormVec)
        index2contigName[j] = contigName
        length_weights_raw.append(len(sub_contigName2seq[contigName]))

    X = np.array(X, dtype=np.float64)

    # ============================================================
    # Improvement 2: Multi-SCG seed selection.
    # Collect all SCG-carrying contigs, rank them by (a) how many
    # different SCGs they carry (more = better marker), then
    # (b) contig length.  Take up to final_k seeds.  This draws
    # seeds from across ALL SCG families, covering species that
    # may be missing any single SCG.
    # ============================================================
    contig_2_scg_count = {}  # contig -> how many SCGs it carries
    for gene_name, contig_set in existGene2contigNames.items():
        for c in contig_set:
            if c in existContig2RepNormV:
                contig_2_scg_count[c] = contig_2_scg_count.get(c, 0) + 1

    seed_candidates = sorted(
        contig_2_scg_count.keys(),
        key=lambda c: (
            contig_2_scg_count[c],              # more SCGs first
            len(sub_contigName2seq.get(c, ""))  # then longer contig
        ),
        reverse=True
    )
    seed_contigs_set = set(seed_candidates[:final_k])

    # Build seed_index from the ordered positions in X
    seed_index = []
    for j in range(len(X)):
        if index2contigName[j] in seed_contigs_set:
            seed_index.append(j)

    logger.info(
        f"--> Collected {len(seed_index)} seed contigs from "
        f"{len(contig_2_scg_count)} SCG-carrying contigs "
        f"(across {len(existGene2contigNames)} SCGs)."
    )

    # ============================================================
    # Improvement 4: Sqrt-transformed length weights.
    # Raw lengths span orders of magnitude (1 kb - 1 Mb).  Without
    # transformation a 500 kb contig has 500× the weight of a 1 kb
    # contig and completely dominates the cluster centres.  Sqrt
    # compression keeps the ordering while preserving meaningful
    # length differentiation (~22× for 500 kb vs 1 kb).
    # ============================================================
    min_contig_len = min(length_weights_raw) if length_weights_raw else 1
    div = np.sqrt(min_contig_len)
    length_weights = np.array(
        [np.sqrt(w) / div for w in length_weights_raw]
    )

    # ============================================================
    # Improvement 3 & 5: partial_seed_init + multiple runs.
    # - partial_seed_init (from split_utils) places the first
    #   len(seed_index) centres at the SCG-seeded contig embeddings
    #   and fills the remaining (final_k - len(seed_index)) centres
    #   via k-means++ sampling.  This never wastes SCG information,
    #   even when seeds < final_k.
    # - n_init=10 runs KMeans 10 times.  Because partial_seed_init
    #   uses random sampling for the non-seed centres, each run
    #   explores a different region of the solution space; we keep
    #   the run with the lowest inertia.
    # ============================================================
    if len(seed_index) > 0:
        init_func = functools.partial(partial_seed_init, seed_idx=seed_index)
        n_init = 10
        logger.info(
            f"--> Start K-Means: k={final_k}, seeds={len(seed_index)}, "
            f"n_init={n_init} (partial_seed_init)."
        )
        kmeans = KMeans(
            n_clusters=final_k,
            init=init_func,
            n_init=n_init,
            random_state=3407
        )
    else:
        n_init = 10
        logger.info(
            f"--> Start K-Means: k={final_k}, no seeds available, "
            f"n_init={n_init} (k-means++)."
        )
        kmeans = KMeans(
            n_clusters=final_k,
            init='k-means++',
            n_init=n_init,
            random_state=3407
        )

    kmeans.fit(X, sample_weight=length_weights)

    logger.info(
        f"--> K-Means converged in {kmeans.n_iter_} iterations, "
        f"inertia={kmeans.inertia_:.2f}."
    )

    # ---- Group contigs by cluster label ----
    cluster_out = {}
    for i, label in enumerate(kmeans.labels_):
        contigName = index2contigName[i]
        if label not in cluster_out:
            cur_name2seq = {}
            cur_name2seq[contigName] = sub_contigName2seq[contigName]
            cluster_out[label] = cur_name2seq
        else:
            cur_name2seq = cluster_out[label]
            cur_name2seq[contigName] = sub_contigName2seq[contigName]

    # ---- Write results ----
    index = 0
    for _, name2seq in cluster_out.items():
        if summedLengthCal(name2seq) >= mag_length_threshold:
            writeFasta(
                name2seq,
                os.path.join(bin_output_folder_path, f"{index}.fasta")
            )
            index += 1

    logger.info(
        f"--> K-Means v2 finished: {len(cluster_out)} clusters, "
        f"{index} bins written (length >= {mag_length_threshold})."
    )
