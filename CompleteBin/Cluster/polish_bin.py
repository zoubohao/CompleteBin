"""
Drop-in replacement for the ``elif first_cont > 5:`` block in
``sec_cluster_utils.re_cluster_procedure_for_one_method`` (line 438).

Instead of SCG-only greedy allocation (``cluster_split_no_learning``), this
module uses embedding distances to identify and remove contaminant contigs one
at a time, guided by SCG quality feedback.

Algorithm (two-phase removal):
  1. Compute initial SCG (comp_init, cont_init).
  2. Find core contigs (unique SCG markers within the bin).
  3. Compute core centroid embedding.
  4. Compute cosine distance from each contig to the centroid.
  5. Split non-core into SCG-bearing (can be verified) and non-SCG (SCG-blind).
  Phase 1 -- SCG-guided greedy removal:
    Sort SCG-bearing non-core by distance descending.
    Remove one at a time, re-evaluate SCG after each.
    Accept if cont drops AND comp does not drop too much.
    Stop if cont <= trigger.
  Phase 2 -- Distance-threshold removal (non-SCG contigs):
    Compute outlier threshold from core distance distribution.
    Remove all non-SCG contigs whose distance exceeds the threshold.
  7. Write the cleaned bin as a fasta candidate.

Usage (replace the elif block at sec_cluster_utils.py line 438):
    from CompleteBin.Cluster.embedding_contam_removal import (
        clean_bin_by_embedding,
    )

    index, quality_record = clean_bin_by_embedding(
        first_cluster_contigName2seq,
        contigname2repNormVector,
        first_cluster_gene2contig_list,
        first_cluster_contigName2_gene2num,
        first_dom,
        tname2markerset,
        bac_contigName2_gene2num,
        arc_contigName2_gene2num,
        output_folder, index, quality_record, mag_length_threshold,
    )
"""

import os
import numpy as np

from CompleteBin.IO import writeFasta
from CompleteBin.Seqs.seq_info import calculateN50
from CompleteBin.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summed_length(name2seq):
    return sum(len(seq) for seq in name2seq.values())


def _find_core_contigs(contig_names, c2g, g2c):
    """Contigs whose every SCG marker appears exactly once in the bin.

    Args:
        contig_names: list of contig names.
        c2g: {contig: {gene: count}} for the bin's domain.
        g2c: {gene: set(contig_names)} for the bin's domain.

    Returns:
        set of core contig names.  Falls back to all contigs if < 3 found.
    """
    core = set()
    for c in contig_names:
        gene_counts = c2g.get(c, {})
        if not gene_counts:
            continue
        if all(len(g2c.get(g, set())) == 1 for g in gene_counts):
            core.add(c)
    if len(core) < 3:
        core = set(contig_names)
    return core


def _centroid(core_names, contigname2emb):
    """Mean embedding of core contigs.  Returns None if no embeddings found."""
    embs = []
    for c in core_names:
        if c in contigname2emb:
            embs.append(contigname2emb[c])
    if not embs:
        return None
    return np.mean(np.stack(embs, axis=0), axis=0)


def _cosine_dists(contig_names, contigname2emb, centroid):
    """{contig: 1 - cos_sim(emb, centroid)}.  Values in [0, 2]."""
    if centroid is None:
        return {}
    c_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    dists = {}
    for c in contig_names:
        if c not in contigname2emb:
            continue
        e = contigname2emb[c]
        cos_sim = np.dot(e / (np.linalg.norm(e) + 1e-12), c_norm)
        dists[c] = 1.0 - float(cos_sim)
    return dists


def _scg_eval(contig_names, tname2markerset,
              bac_c2g, arc_c2g, dom):
    """Compute (comp, cont) for a set of contigs."""
    if len(contig_names) == 0:
        return 0.0, 0.0
    from CompleteBin.Cluster.sec_cluster_utils import determine_domain
    _, _, _, comp, cont = determine_domain(
        tname2markerset, list(contig_names), bac_c2g, arc_c2g, dom,
    )
    return comp, cont


# ---------------------------------------------------------------------------
# Public API  --  drop-in replacement for the elif block
# ---------------------------------------------------------------------------

def clean_bin_by_embedding(
    first_cluster_contigName2seq,
    contigname2repNormVector,
    first_cluster_gene2contig_list,
    first_cluster_contigName2_gene2num,
    first_dom,
    tname2markerset,
    bac_contigName2_gene2num,
    arc_contigName2_gene2num,
    output_folder,
    index,
    quality_record,
    mag_length_threshold,
    contamination_trigger=5.0,
    completeness_drop_tolerance=10.0,
):
    """Greedy embedding-guided contaminant removal for one bin.

    Replaces the ``elif first_cont > 5:`` block.  Writes the cleaned bin as a
    fasta candidate (in addition to the original bin written by the caller).

    Args:
        first_cluster_contigName2seq: {contig: seq} for this bin.
        contigname2repNormVector: {contig: np.ndarray} global embeddings.
        first_cluster_gene2contig_list: {gene: set(contigs)} for this bin.
        first_cluster_contigName2_gene2num: {contig: {gene: count}}.
        first_dom: "bac" or "arc".
        tname2markerset: SCG marker sets.
        bac_contigName2_gene2num: global bacteria SCG.
        arc_contigName2_gene2num: global archaea SCG.
        output_folder: directory for fasta output.
        index: current candidate index (int, returned incremented).
        quality_record: dict updated in-place.
        mag_length_threshold: minimum total bp for a bin.
        contamination_trigger: only process if cont > this value.
        completeness_drop_tolerance: max allowed completeness loss.

    Returns:
        (index, quality_record) -- index may have been incremented.
    """
    contig_names = list(first_cluster_contigName2seq.keys())
    n_total = len(contig_names)
    if n_total < 5:
        return index, quality_record

    # ---- Step 1: initial SCG quality ----
    comp_init, cont_init = _scg_eval(
        contig_names, tname2markerset,
        bac_contigName2_gene2num, arc_contigName2_gene2num, first_dom,
    )
    if cont_init <= contamination_trigger:
        return index, quality_record

    # ---- Step 2: find core contigs ----
    core = _find_core_contigs(
        contig_names,
        first_cluster_contigName2_gene2num,
        first_cluster_gene2contig_list,
    )
    noncore = [c for c in contig_names if c not in core]
    if len(noncore) == 0:
        return index, quality_record

    # ---- Step 3: core centroid ----
    centroid = _centroid(core, contigname2repNormVector)
    if centroid is None:
        return index, quality_record

    # ---- Step 4: cosine distances ----
    dists = _cosine_dists(contig_names, contigname2repNormVector, centroid)

    # ---- Step 5: split non-core by SCG status ----
    # SCG-bearing: carry at least one SCG marker; SCG eval changes on removal.
    # Non-SCG:     carry no SCG markers; SCG eval is blind to these.
    scg_noncore = [c for c in noncore
                   if c in first_cluster_contigName2_gene2num
                   and len(first_cluster_contigName2_gene2num[c]) > 0]
    nonscg_noncore = [c for c in noncore if c not in scg_noncore]

    kept = set(contig_names)
    removed = set()
    current_comp = comp_init
    current_cont = cont_init

    # ---- Phase 1: SCG-guided greedy removal (for SCG-bearing contigs) ----
    if scg_noncore:
        sorted_scg = sorted(scg_noncore, key=lambda c: dists.get(c, 0.0),
                            reverse=True)
        for c in sorted_scg:
            trial_kept = kept - {c}
            trial_comp, trial_cont = _scg_eval(
                list(trial_kept), tname2markerset,
                bac_contigName2_gene2num, arc_contigName2_gene2num, first_dom,
            )
            if (trial_cont < current_cont and
                    trial_comp >= current_comp - completeness_drop_tolerance):
                kept = trial_kept
                removed.add(c)
                current_comp = trial_comp
                current_cont = trial_cont
            if current_cont <= contamination_trigger:
                break

    # ---- Phase 2: distance-threshold removal (for non-SCG contigs) ----
    if nonscg_noncore:
        core_dists = [dists[c] for c in core if c in dists]
        arr = np.array(core_dists, dtype=np.float32)
        mean_d = float(np.mean(arr))
        if len(core_dists) >= 3:
            std_d = float(np.std(arr))
            # Adaptive k: fewer SCG-bearing contigs → more conservative threshold.
            # Bins with low SCG coverage (e.g. LjRoot84 at 5.9%) have most contigs
            # invisible to SCG validation — Phase 2 must not blindly remove them.
            scg_fraction = len(first_cluster_contigName2_gene2num) / max(len(contig_names), 1)
            k_adaptive = 3.0 + 2.0 * (1.0 - scg_fraction)
            threshold = mean_d + k_adaptive * std_d
        else:
            threshold = mean_d * 5.0

        for c in nonscg_noncore:
            if dists.get(c, 0.0) > threshold:
                kept.discard(c)
                removed.add(c)

    # ---- Step 7: write cleaned bin if anything was removed ----
    if len(removed) == 0:
        return index, quality_record

    cleaned_name2seq = {c: first_cluster_contigName2seq[c] for c in kept}
    size = _summed_length(cleaned_name2seq)
    if size >= mag_length_threshold:
        fname = f"CompleteBin_cand_{index}.fasta"
        writeFasta(cleaned_name2seq, os.path.join(output_folder, fname))
        n50 = np.log(max(1, calculateN50(cleaned_name2seq)))
        quality_record[fname] = (current_comp, current_cont, n50, size)
        index += 1

    return index, quality_record


# ---------------------------------------------------------------------------
# SCG + embedding dual-signal polish trigger
# ---------------------------------------------------------------------------

def compute_cohens_d(
    contig_names,
    contigname2emb,
    contigName2_gene2num,
    gene2contig_list,
):
    """Compute Cohen's d effect size between core and non-core contig distances.

    Core contigs = those whose every SCG marker appears exactly once in the bin
    (via :func:`_find_core_contigs`).  Their centroid models the "true" bin
    identity.  Non-core contigs that are systematically farther from this
    centroid suggest embedding-detectable contamination that SCG may have missed
    (e.g. closely related species sharing the same SCG markers).

    Returns:
        d: Cohen's d (positive = non-core farther from centroid).
        core_dists: array of core contig distances (None if insufficient data).
        noncore_dists: array of non-core contig distances (None if insufficient).
        centroid: core centroid vector (None if insufficient data).
    """
    if len(contig_names) < 5:
        return 0.0, None, None, None

    core = _find_core_contigs(
        list(contig_names), contigName2_gene2num, gene2contig_list,
    )
    noncore = [c for c in contig_names if c not in core]

    if len(noncore) == 0:
        return 0.0, None, None, None

    centroid = _centroid(core, contigname2emb)
    if centroid is None:
        return 0.0, None, None, None

    dists = _cosine_dists(contig_names, contigname2emb, centroid)

    core_dists = [dists[c] for c in core if c in dists]
    noncore_dists = [dists[c] for c in noncore if c in dists]

    if len(core_dists) < 3 or len(noncore_dists) < 3:
        return 0.0, None, None, None

    core_arr = np.array(core_dists, dtype=np.float32)
    noncore_arr = np.array(noncore_dists, dtype=np.float32)

    mean_c = float(np.mean(core_arr))
    mean_nc = float(np.mean(noncore_arr))
    var_c = float(np.var(core_arr, ddof=1))
    var_nc = float(np.var(noncore_arr, ddof=1))
    pooled_std = np.sqrt(max((var_c + var_nc) / 2.0, 1e-12))

    d = (mean_nc - mean_c) / pooled_std
    return d, core_arr, noncore_arr, centroid


def should_trigger_polish(
    scg_cont: float,
    d: float,
    gray_zone_lower: float = 2.0,
    scg_artifact_ceiling: float = 3.45,
    gray_embed_threshold: float = 1.5,
    light_embed_threshold: float = 1.0,
):
    """Three-level polish trigger combining SCG contamination and embedding effect size.

    Levels:
        cont <= 2.0%                          -> no trigger (clean)
        2.0% < cont <= 3.45%  AND d > 1.5     -> trigger, zone="gray"
        3.45% < cont <= 5.0%  AND d > 1.0     -> trigger, zone="light"
        cont > 5.0%                            -> trigger, zone="clear"

    The SCG artifact ceiling (3.25%) is calibrated on 439 bacterial/archaeal
    reference genomes -- see ``getScore`` in ``galah_utils.py``.

    Returns:
        (should_trigger: bool, zone: str or None)
    """
    if scg_cont <= gray_zone_lower:
        return False, None
    if scg_cont <= scg_artifact_ceiling:
        if d > gray_embed_threshold:
            return True, "gray"
        return False, None
    if scg_cont <= 5.0:
        if d > light_embed_threshold:
            return True, "light"
        return False, None
    return True, "clear"
