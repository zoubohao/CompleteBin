"""
Contig-level deduplication for CompleteBin output MAGs.

Ensures each contig appears in at most one output bin using an iterative
greedy bin-level tournament (adapted from ``scg_disjoint_utils.py``).

Algorithm
---------
Each round:
  1. Score every active bin from its **current** contig set (recompute SCG
     comp/cont via ``determine_domain``, score via ``getScore``).
  2. The highest-scoring bin "claims" all its currently held contigs.
  3. Those contigs are removed from every other active bin.
  4. Empty bins are dropped.  The winner is removed from the active set.
  5. Repeat until no active bins remain.

This directly optimises for final bin quality: the best bins get to keep
their contigs first, and lower-quality bins receive the leftovers.  Because
SCG is recomputed from scratch every round there are no stale-data issues.

Usage (after ``binning_with_all_steps`` assembles final bins)::

    from CompleteBin.Dereplication.contig_dedup import contig_level_dedup
    contig_level_dedup(
        bin_output_folder=bin_output_folder_path,
        contigname2seq=contigname2seq,
        simclr_embeddings=simclr_contigname2emb_norm_array,
        bac_contigName2_gene2num=bac_contigName2_gene2num,
        arc_contigName2_gene2num=arc_contigName2_gene2num,
        bac_gene2contigNames=bac_gene2contigNames,
        arc_gene2contigNames=arc_gene2contigNames,
        markerset=markerset,
    )
"""

import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from CompleteBin.Cluster.sec_cluster_utils import determine_domain
from CompleteBin.Dereplication.galah_utils import getScore
from CompleteBin.IO import readFasta, writeFasta
from CompleteBin.logger import get_logger

logger = get_logger()


# ===================================================================
# Phase 1: Scan bins and detect duplicates.
# ===================================================================

def _scan_bins(bin_output_folder: str) -> Dict[str, Dict[str, str]]:
    """Read all ``CompleteBin_*.fasta`` files from *bin_output_folder*."""
    bin_data = {}
    for fname in sorted(os.listdir(bin_output_folder)):
        if fname.startswith("CompleteBin_") and fname.endswith(".fasta"):
            fpath = os.path.join(bin_output_folder, fname)
            name2seq = readFasta(fpath)
            if name2seq:
                bin_data[fname] = name2seq
    logger.info(
        "--> Contig dedup: scanned %d bin FASTA files.", len(bin_data)
    )
    return bin_data


def _find_duplicates(
    bin_data: Dict[str, Dict[str, str]],
) -> Dict[str, List[str]]:
    """Find contigs that appear in two or more bins.

    Returns {contig_name: [bin_filename, ...]} for contigs with ≥ 2 bins.
    """
    contig_to_bins = defaultdict(list)
    for bin_fname, name2seq in bin_data.items():
        for contig_name in name2seq:
            contig_to_bins[contig_name].append(bin_fname)
    duplicates = {
        c: bins for c, bins in contig_to_bins.items() if len(bins) >= 2
    }
    logger.info(
        "--> Contig dedup: %d duplicate contigs found across bins.",
        len(duplicates),
    )
    return duplicates


# ===================================================================
# Phase 2: Iterative greedy bin-level tournament.
# ===================================================================

def _classify_state(comp, cont):
    """Map (comp, cont) to quality tier string (matches ``IO.readMetaInfo``)."""
    if comp >= 90.0 and cont <= 5.0:
        return "HighQuality"
    if comp >= 50.0 and cont <= 10.0:
        return "MediumQuality"
    return "LowQuality"


def _resolve_duplicates_scg_disjoint(
    bin_data: Dict[str, Dict[str, str]],
    bac_contigName2_gene2num: Dict[str, Dict[str, int]],
    arc_contigName2_gene2num: Dict[str, Dict[str, int]],
    bac_marker_set: List[Set[str]],
    arc_marker_set: List[Set[str]],
) -> int:
    """Iterative greedy bin-level tournament — scg_disjoint algorithm.

    Each round:
      1. Recompute SCG comp/cont for every active bin from its CURRENT
         contig set (calling ``determine_domain``).
      2. Score each bin via ``getScore``.
      3. The highest-scoring bin claims **all** its contigs.
      4. Those contigs are removed from every other active bin.
      5. Empty bins are dropped; the winner leaves the active set.

    This handles SCG-bearing and non-SCG duplicate contigs uniformly:
    the tournament order ensures the best bins keep their contigs first.

    Returns the number of contig-in-bin removal events.
    """
    markerset = {"d__Bacteria": bac_marker_set, "d__Archaea": arc_marker_set}

    # active: bin_name → set(contig_names still in this bin)
    active = {bf: set(n2s.keys()) for bf, n2s in bin_data.items()}

    round_no = 0
    n_removals = 0

    while active:
        round_no += 1

        # ---- Score every active bin ----
        best_bin = None
        best_score = -float("inf")
        best_contigs = None

        for bf, live in active.items():
            if not live:
                continue
            _, _, _, comp, cont = determine_domain(
                markerset, list(live),
                bac_contigName2_gene2num, arc_contigName2_gene2num,
            )
            state = _classify_state(comp, cont)
            score = getScore((comp, cont, state))
            if score > best_score:
                best_score = score
                best_bin = bf
                best_contigs = live

        if best_bin is None:
            break

        # ---- Remove winner's contigs from all other active bins ----
        for other_bf in list(active.keys()):
            if other_bf == best_bin:
                continue

            to_remove = best_contigs & active[other_bf]
            if not to_remove:
                continue

            active[other_bf] -= to_remove
            n_removals += len(to_remove)

            # Keep bin_data in sync (used for final FASTA rewrite)
            for c in to_remove:
                if other_bf in bin_data:
                    bin_data[other_bf].pop(c, None)

            if not active[other_bf]:
                del active[other_bf]

        # Winner is settled — its contigs are finalised
        del active[best_bin]

    logger.info(
        "--> Contig dedup tournament: %d rounds, %d contig-in-bin removals.",
        round_no, n_removals,
    )
    return n_removals


# ===================================================================
# Phase 3: Re-evaluate and rewrite output.
# ===================================================================

def _reevaluate_and_rewrite(
    bin_output_folder: str,
    updated_bins: Dict[str, Dict[str, str]],
    bac_contigName2_gene2num: Dict[str, Dict[str, int]],
    arc_contigName2_gene2num: Dict[str, Dict[str, int]],
    bac_marker_set: List[Set[str]],
    arc_marker_set: List[Set[str]],
    mag_length_threshold: int,
) -> int:
    """Re-evaluate SCG quality, delete small bins, rewrite FASTA + MetaInfo.tsv.

    Returns number of bins removed due to size threshold.
    """
    markerset = {"d__Bacteria": bac_marker_set, "d__Archaea": arc_marker_set}

    final_bins = []
    n_removed_small = 0

    for bin_fname, name2seq in sorted(updated_bins.items()):
        total_len = sum(len(seq) for seq in name2seq.values())
        if total_len < mag_length_threshold:
            fpath = os.path.join(bin_output_folder, bin_fname)
            if os.path.exists(fpath):
                os.remove(fpath)
            n_removed_small += 1
            logger.debug(
                "  Removing bin %s: total length %d < %d.",
                bin_fname, total_len, mag_length_threshold,
            )
            continue

        contig_names = list(name2seq.keys())
        _, _, _, comp, cont = determine_domain(
            markerset, contig_names,
            bac_contigName2_gene2num, arc_contigName2_gene2num,
        )
        state = _classify_state(comp, cont)
        final_bins.append((bin_fname, name2seq, comp, cont, state))

    if n_removed_small:
        logger.info(
            "--> Contig dedup: %d bins removed (below %d bp threshold).",
            n_removed_small, mag_length_threshold,
        )

    for bin_fname, name2seq, _, _, _ in final_bins:
        fpath = os.path.join(bin_output_folder, bin_fname)
        writeFasta(name2seq, fpath)

    metainfo_path = os.path.join(bin_output_folder, "MetaInfo.tsv")
    with open(metainfo_path, "w") as wh:
        for bin_fname, _, comp, cont, state in final_bins:
            wh.write(
                f"{bin_fname}\tSCGs_EVAL(Comp,Cont,Quality)\t"
                f"{comp}\t{cont}\t{state}\n"
            )

    logger.info(
        "--> Contig dedup: %d final bins written to %s.",
        len(final_bins), metainfo_path,
    )
    return n_removed_small


# ===================================================================
# Public entry point
# ===================================================================

def contig_level_dedup(
    bin_output_folder: str,
    bac_contigName2_gene2num: Dict[str, Dict[str, int]],
    arc_contigName2_gene2num: Dict[str, Dict[str, int]],
    markerset: Dict[str, List[Set[str]]],
    mag_length_threshold: int = 100000,
) -> Tuple[int, int]:
    """Ensure each contig appears in at most one output bin.

    Uses an iterative greedy bin-level tournament adapted from
    ``scg_disjoint_utils.py``: in each round the highest-scoring active
    bin claims all its contigs, which are then removed from every other
    bin.  SCG quality is recomputed from scratch each round so there
    are no stale-data issues.

    Parameters
    ----------
    bin_output_folder : str
        Folder containing ``CompleteBin_*.fasta`` and ``MetaInfo.tsv``.
    bac_contigName2_gene2num : dict
        Bacterial SCG gene counts per contig.
    arc_contigName2_gene2num : dict
        Archaeal SCG gene counts per contig.
    markerset : dict
        SCG marker sets with keys ``"d__Bacteria"`` and ``"d__Archaea"``.
    mag_length_threshold : int
        Minimum total bp for a bin to be retained (default 100000).

    Returns
    -------
    (n_duplicates_found, n_bins_removed) : Tuple[int, int]
    """
    bac_marker_set = markerset["d__Bacteria"]
    arc_marker_set = markerset["d__Archaea"]

    # ---- Scan and index ----
    bin_data = _scan_bins(bin_output_folder)
    if len(bin_data) <= 1:
        logger.info("--> Contig dedup: ≤1 bin, nothing to deduplicate.")
        return 0, 0

    duplicates = _find_duplicates(bin_data)
    if not duplicates:
        logger.info("--> Contig dedup: no duplicate contigs found. Done.")
        return 0, 0

    # ---- Iterative greedy bin-level tournament ----
    _resolve_duplicates_scg_disjoint(
        bin_data,
        bac_contigName2_gene2num, arc_contigName2_gene2num,
        bac_marker_set, arc_marker_set,
    )

    # ---- Cleanup: Remove empty bins ----
    empty_bins = [bf for bf, n2s in bin_data.items() if len(n2s) == 0]
    for bf in empty_bins:
        del bin_data[bf]
        fpath = os.path.join(bin_output_folder, bf)
        if os.path.exists(fpath):
            os.remove(fpath)
    if empty_bins:
        logger.info(
            "--> Contig dedup: %d empty bins removed.", len(empty_bins),
        )

    # ---- Re-evaluate and rewrite ----
    n_removed_small = _reevaluate_and_rewrite(
        bin_output_folder, bin_data,
        bac_contigName2_gene2num, arc_contigName2_gene2num,
        bac_marker_set, arc_marker_set, mag_length_threshold,
    )

    n_removed = len(empty_bins) + n_removed_small
    logger.info(
        "--> Contig dedup complete: %d duplicate contigs processed, "
        "%d bins removed.", len(duplicates), n_removed,
    )
    return len(duplicates), n_removed
