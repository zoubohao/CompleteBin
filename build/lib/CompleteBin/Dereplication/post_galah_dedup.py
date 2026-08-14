"""
Post-Galah de-duplication for CompleteBin output MAGs.

Removes redundant low-completeness bins (comp < 70%, >= 100 kbp) that are
> 97.2% contained within another such bin, keeping the best bin per
containment cluster.

Uses the same ``getScore`` with piecewise contamination penalty as galah_utils.

Usage (after ``collect_galah_result_with_SCGs``)::

    from CompleteBin.Dereplication.post_galah_dedup import post_galah_dedup
    post_galah_dedup(output_folder=outputBinFolder)
"""


import os
import subprocess
from collections import defaultdict

import numpy as np

from CompleteBin.IO import readFasta, readMetaInfo
from CompleteBin.logger import get_logger
from CompleteBin.Dereplication.galah_utils import getScore

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summed_length(name2seq):
    return sum(len(seq) for seq in name2seq.values())


def _run_skani_dist(bin_paths, tmp_dir):
    """Run ``skani dist`` on a set of bin FASTA files.

    Returns a square matrix of aligned fractions (float, 0-1) where
    M[i][j] = aligned fraction of query j vs reference i.
    """
    n = len(bin_paths)
    if n <= 1:
        return np.eye(n)

    out_file = os.path.join(tmp_dir, "skani_output.tsv")
    # skani dist expects FASTA files, not a list file.  Pass each path
    # as a separate -q/-r argument for all-vs-all comparison.
    cmd = ["skani", "dist", "-t", "8", "-o", out_file]
    for p in bin_paths:
        cmd.extend(["-q", p])
    for p in bin_paths:
        cmd.extend(["-r", p])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.warning("skani dist failed: %s", e.stderr)
        return np.eye(n)

    # Parse skani output: columns: Ref, Query, ANI, AlignedRef, AlignedQuery, ...
    matrix = np.eye(n)
    try:
        with open(out_file, "r") as fh:
            for line in fh:
                if line.startswith("Ref") or not line.strip():
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                ref_path, query_path = parts[0], parts[1]
                # Find indices
                ref_idx = None
                query_idx = None
                for idx, p in enumerate(bin_paths):
                    if p == ref_path:
                        ref_idx = idx
                    if p == query_path:
                        query_idx = idx
                if ref_idx is None or query_idx is None:
                    continue
                # AlignedRef = fraction of reference covered (col 3)
                # AlignedQuery = fraction of query covered (col 4)
                try:
                    aligned_ref = float(parts[3])
                    aligned_query = float(parts[4])
                    matrix[ref_idx][query_idx] = aligned_query / 100.0  # query in ref
                    matrix[query_idx][ref_idx] = aligned_ref / 100.0   # ref in query
                except (ValueError, IndexError):
                    pass
    except FileNotFoundError:
        pass

    return matrix


def _connected_components(n, edges):
    """Union-Find connected components."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in edges:
        union(a, b)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return list(comps.values())


# ---------------------------------------------------------------------------
# Low-completeness bin dedup by containment
# ---------------------------------------------------------------------------

def dedup_lowcomp_bins(bin_paths, quality_map, tmp_dir):
    """De-duplicate low-completeness bins by skani containment.

    Clusters bins that are > 97.2% mutually contained and keeps the
    highest-scoring bin per cluster.

    Parameters
    ----------
    bin_paths : list of str
        Paths to bin FASTA files.
    quality_map : dict
        {bin_path: (comp, contam, quality_label)}
    tmp_dir : str
        Temporary directory for skani output.

    Returns
    -------
    list of str
        De-duplicated bin paths (one per containment super-cluster).
    """
    n = len(bin_paths)
    if n <= 1:
        return list(bin_paths)

    logger.info("--> Post-Galah dedup: %d bins, computing containment ...", n)

    aligned = _run_skani_dist(bin_paths, tmp_dir)

    # containment[i][j] = max(aligned_i_j, aligned_j_i)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            cont = max(aligned[i][j], aligned[j][i])
            if cont >= 0.978:
                edges.append((i, j))
                logger.debug(
                    "  low-comp containment: %s <-> %s = %.3f",
                    os.path.basename(bin_paths[i])[:30],
                    os.path.basename(bin_paths[j])[:30],
                    cont,
                )

    components = _connected_components(n, edges)

    kept = []
    n_removed = 0
    for comp in components:
        if len(comp) == 1:
            kept.append(bin_paths[comp[0]])
        else:
            # SELECT best by getScore (with N50)
            best_idx = None
            best_score = -float("inf")
            for idx in comp:
                path = bin_paths[idx]
                qv = quality_map[path]
                s = getScore(qv)
                if s > best_score:
                    best_score = s
                    best_idx = idx
            kept.append(bin_paths[best_idx])
            n_removed += len(comp) - 1

    logger.info(
        "  bin dedup: %d -> %d bins (%d redundant removed)",
        n, len(kept), n_removed,
    )
    return kept


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def post_galah_dedup(
    output_folder,
    tmp_dir=None,
    mag_length_threshold=100000,
    completeness=90.
):
    """Run post-Galah dedup on output MAGs.

    Removes redundant low-completeness bins (comp < 90%) that are
    > 97.2% contained within another such bin (measured by skani aligned fraction).
    Keeps the best bin per containment cluster by getScore.

    Parameters
    ----------
    output_folder : str
        Folder containing the output bin FASTA files and MetaInfo.tsv.
    tmp_dir : str, optional
        Temporary directory for skani.  Defaults to ``output_folder/post_galah_tmp``.
    mag_length_threshold : int
        Minimum total bp for a MAG to be considered.

    Returns
    -------
    None (modifies output_folder in-place).
    """
    if tmp_dir is None:
        tmp_dir = os.path.join(output_folder, "post_galah_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # --- Read SCG quality ---
    scg_meta, _, _, _ = readMetaInfo(os.path.join(output_folder, "MetaInfo.tsv"), 2, 3)

    # --- Scan output folder for bin FASTA files ---
    bin_paths = []
    for fname in sorted(os.listdir(output_folder)):
        if fname.endswith(".fasta") or fname.endswith(".fa"):
            bin_paths.append(os.path.join(output_folder, fname))

    if not bin_paths:
        logger.warning("No bin FASTA files found in %s", output_folder)
        return

    logger.info("--> Post-Galah dedup: %d bins in output folder", len(bin_paths))

    # --- Select low-completeness bins ---
    low_comp_paths = []
    quality_map = {}  # {path: (comp, contam, quality_label)}

    for bp in bin_paths:
        fname = os.path.basename(bp)
        if fname in scg_meta:
            qv = scg_meta[fname]
            quality_map[bp] = qv
            comp, contam = qv[0], qv[1]
            name2seq = readFasta(bp)
            size = _summed_length(name2seq) if name2seq else 0
            if size < mag_length_threshold:
                continue
            if comp < completeness: #############
                low_comp_paths.append(bp)
        else:
            logger.debug("  Bin %s not found in SCG report, skipping", fname)

    logger.info(
        "  Selected: %d low-completeness bins (comp < %.0f%%, >= %d bp)",
        len(low_comp_paths), completeness, mag_length_threshold,
    )

    # --- Low-comp dedup ---
    kept = low_comp_paths
    if len(low_comp_paths) > 1:
        kept = dedup_lowcomp_bins(low_comp_paths, quality_map, tmp_dir)
        # Remove discarded bins
        keep_set = set(kept)
        for bp in low_comp_paths:
            if bp not in keep_set:
                try:
                    os.remove(bp)
                except OSError:
                    pass

    # --- Cleanup ---
    try:
        import shutil
        shutil.rmtree(tmp_dir)
    except OSError:
        pass

    final_count = len([f for f in os.listdir(output_folder)
                       if f.endswith(".fasta") or f.endswith(".fa")])
    logger.info("--> Post-Galah dedup complete: %d final bins", final_count)
