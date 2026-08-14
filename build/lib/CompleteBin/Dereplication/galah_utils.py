import math
import os
import re
from shutil import copy
from typing import Dict, List, Tuple


from CompleteBin.IO import readFasta, readGalahClusterTSV, readMarkerSets, readMetaInfo
from CompleteBin.logger import get_logger


logger = get_logger()


# ---------------------------------------------------------------------------
# Galah input order optimisation
# ---------------------------------------------------------------------------

def _parse_method_name(method_name: str) -> Tuple[int, float, float]:
    """Extract (partgraphRatio, resolution, bandwidth) from a Leiden method name."""
    m = re.match(
        r'Leiden_embMat0_maxedges_\d+_partgraphRatio_(\d+)_resolution_([0-9.]+)_bandwidth_([0-9.]+)',
        method_name,
    )
    if not m:
        return (0, 0.0, 0.0)
    return (int(m.group(1)), float(m.group(2)), float(m.group(3)))


def _diversity_score(
    pg: int,
    res: float,
    bw: float,
    num_905: int = 0,
    anchor_pg: int = 100,
    anchor_res: float = 8,
    anchor_bw: float = 0.15,
) -> float:
    """Quantify how much a (pg, res, bw) combo differs from the anchor.

    Higher scores mean the combo provides a *different perspective* from the
    anchor and should be placed earlier in Galah input.

    Parameters
    ----------
    num_905 : int
        Number of near-complete bins (comp > 90, contam < 5) from this combo.
        Used as a micro-adjustment tiebreaker (capped at 0.02) because SCGs
        are ~60 % reliable.  Only NC differences >= 14 (>1.5 sigma) can flip
        a P10 diversity gap; P25+ gaps are never affected.
    """
    log_range = math.log10(50 / 0.25)  # approx 2.301
    res_dist = abs(math.log10(res) - math.log10(anchor_res)) / log_range
    pg_dist = abs(pg - anchor_pg) / 50.0
    bw_dist = abs(bw - anchor_bw) / 0.25

    diversity = pg_dist * 0.35 + res_dist * 0.45 + bw_dist * 0.2

    # Coarse high-coverage bonus: preserve large complete genomes
    if res <= 1 and pg >= 80:
        diversity += 0.25

    # NC tiebreaker: SCG quality micro-adjustment (max +0.02).
    # Linear in [0, 50]; NC diff of 14 produces 0.0056 shift.
    diversity += min(num_905 / 50.0, 1.0) * 0.02

    if res >= 30 and bw >= 0.15:
        diversity -= 3.0
    elif res >= 30 and bw >= 0.1:
        diversity -= 1.5

    if res >= 50 and bw <= 0.05:
        diversity += 0.15

    return diversity


def _select_anchor_params(
    ensemble_list: List[Tuple],
    target_pg: int = 2,
    target_res: int = 3,
    target_bw: int = 2,
) -> Tuple[tuple, tuple, tuple, float, float]:
    """Select optimal anchor parameters from ensemble quality data.

    Two-phase selection balancing quality (NC + all_summed) and diversity
    (coverage of parameter space).

    Phase A — resolution-first: pick the highest-Q representative from
    each distinct resolution until *target_res* resolutions are covered.
    Phase B — pg / bw supplement: add the highest-Q entry with a new pg
    or bw until *target_pg* and *target_bw* are satisfied.

    Returns
    -------
    chosen_pg : tuple
        Selected partgraph_ratio values (length ≤ *target_pg*).
    chosen_res : tuple
        Selected resolution values (length ≤ *target_res*).
    chosen_bw : tuple
        Selected bandwidth values (length ≤ *target_bw*).
    best_res : float
        Resolution of the highest-quality combination (used in anchor_score).
    best_bw : float
        Bandwidth of the highest-quality combination (used in anchor_score).
    """
    # ── Parse and compute quality ──────────────────────────────────────
    candidates = []  # [pg, res, bw, nc, as_, Q]
    for item in ensemble_list:
        method_name = item[0]
        pg, res, bw = _parse_method_name(method_name)
        if pg == 0 and res == 0.0 and bw == 0.0:
            continue
        adj_nc = item[4] if len(item) > 4 else (item[2] if len(item) > 2 else 0)
        adj_as = item[5] if len(item) > 5 else (item[3] if len(item) > 3 else 0)
        candidates.append([pg, res, bw, adj_nc, adj_as, 0.0])

    if not candidates:
        logger.warning("No valid candidates; using default anchor params.")
        return ((100,), (5, 8, 10), (0.1, 0.15), 8.0, 0.15)

    nc_vals = [c[3] for c in candidates]
    as_vals = [c[4] for c in candidates]
    nc_min, nc_max = min(nc_vals), max(nc_vals)
    as_min, as_max = min(as_vals), max(as_vals)
    nc_range = max(nc_max - nc_min, 1)
    as_range = max(as_max - as_min, 1)

    for c in candidates:
        nc_norm = (c[3] - nc_min) / nc_range
        as_norm = (c[4] - as_min) / as_range
        c[5] = 0.9 * nc_norm + 0.1 * as_norm  # Q

    # ── Phase A: resolution-first selection ─────────────────────────────
    candidates.sort(key=lambda c: c[5], reverse=True)
    selected = []
    used_res = set()

    for c in candidates:
        if c[1] not in used_res:
            selected.append(c)
            used_res.add(c[1])
        if len(used_res) >= target_res:
            break

    # ── Phase B: pg / bw supplement ────────────────────────────────────
    used_pg = set(c[0] for c in selected)
    used_bw = set(c[2] for c in selected)

    for c in candidates:
        if c in selected:
            continue
        if c[0] not in used_pg and len(used_pg) < target_pg:
            selected.append(c)
            used_pg.add(c[0])
        if c[2] not in used_bw and len(used_bw) < target_bw:
            if c not in selected:
                selected.append(c)
                used_bw.add(c[2])
        if len(used_pg) >= target_pg and len(used_bw) >= target_bw:
            break

    # ── Extract chosen sets ─────────────────────────────────────────────
    chosen_pg_list = []
    chosen_res_list = []
    chosen_bw_list = []
    seen_pg, seen_res, seen_bw = set(), set(), set()
    for c in selected:
        if c[0] not in seen_pg and len(chosen_pg_list) < target_pg:
            chosen_pg_list.append(c[0])
            seen_pg.add(c[0])
        if c[1] not in seen_res and len(chosen_res_list) < target_res:
            chosen_res_list.append(c[1])
            seen_res.add(c[1])
        if c[2] not in seen_bw and len(chosen_bw_list) < target_bw:
            chosen_bw_list.append(c[2])
            seen_bw.add(c[2])

    chosen_pg = tuple(sorted(chosen_pg_list))
    chosen_res = tuple(sorted(chosen_res_list))
    chosen_bw = tuple(sorted(chosen_bw_list))

    # Anchor = highest-Q combination
    anchor_combo = selected[0]
    best_res = anchor_combo[1]
    best_bw = anchor_combo[2]

    logger.info(
        "--> Auto-selected anchor params: pg=%s res=%s bw=%s | "
        "best_res=%.2f best_bw=%.2f (Q=%.3f adj_NC=%.2f adj_AS=%.2f)",
        chosen_pg, chosen_res, chosen_bw,
        best_res, best_bw, anchor_combo[5], anchor_combo[3], anchor_combo[4],
    )

    return (chosen_pg, chosen_res, chosen_bw, best_res, best_bw)


def _galah_sort_key(item,
                    anchor_pg=100, anchor_res=8.0, anchor_bw=0.15,
                    chosen_pg=(100,), chosen_res=(5, 8, 10), chosen_bw=(0.1, 0.15)) -> Tuple[int, float]:
    """
    Compute a sort key for a single *ensemble_methods* entry.

    Returns ``(tier, sub_key)``.  Lower values appear earlier in Galah input.

    Tier assignment is driven by :func:`_diversity_score` ranges rather
    than hardcoded parameter thresholds, so the strategy adapts to any
    (pg, res, bw) combination automatically.

    Tiers
    -----
    0 — **Anchor family.**  ``pg=100``, ``res`` ∈ {5, 8, 10}, ``bw`` ∈ {0.1, 0.15}.
        Sorted by closeness to the optimal ``(100, 8, 0.15)``.
    1 — **Diversity injectors** (diversity >= 0.15).  Combos that differ
        substantially from the anchor and are not fragmented.
    2 — **Gap fillers** (0 <= diversity < 0.15).  Combos similar to the
        anchor — different resolution/bandwidth within the same pg=100
        family, or moderate changes in pg.
    3 — **Mildly penalised** (−1.5 <= diversity < 0).  Transition-band
        high-resolution combos (e.g. ``res >= 30, bw = 0.1``).
    4 — **Heavily penalised** (diversity < −1.5).  Dense high-resolution
        (``res >= 30, bw >= 0.15``) — extreme fragmentation.  **Last.**
    """
    method_name = item[0]
    pg, res, bw = _parse_method_name(method_name)

    # ── Tier 0: anchor family ──────────────────────────────────────────
    if pg == 100 and res in (5, 8, 10) and bw in (0.1, 0.15):
        # Closeness to the optimal (100, 8, 0.15)
        anchor_score = abs(res - 8) * 1.0 + abs(bw - 0.15) * 10.0
        return (0, anchor_score)

    # Completeness anchor: coarse full-graph — preserves large complete
    # genomes that the medium-resolution anchor may split.  Placed right
    # after the main anchor family, before diversity injectors.
    if pg == anchor_pg and res == 1 and bw in (0.05, 0.1, 0.15):
        return (0, 100 + abs(bw - 0.1) * 10.0)

    nc = item[2] if len(item) > 2 else 0
    div = _diversity_score(pg, res, bw, num_905=nc)

    # ── Tier 4: heavily penalised (dense high-res, LAST) ────────────────
    if div < -1.5:
        return (4, -div)

    # ── Tier 3: mildly penalised (transition high-res) ──────────────────
    if div < 0:
        return (3, -div)

    # ── Tier 2: gap fillers (low diversity, not fragmented) ─────────────
    if div < 0.15:
        return (2, -div)

    # ── Tier 1: diversity injectors (high diversity, not fragmented) ────
    return (1, -div)


def reorder_ensemble_for_galah(
    ensemble_list: List[Tuple],
    nc_first: bool = True,
) -> List[Tuple]:
    """
    Reorder *ensemble_list* to maximise Galah cluster count and NCMAGs.

    .. note::

        Galah processes genomes in the order they appear in
        ``files_path.txt`` (which is built by iterating *ensemble_list*
        sequentially — see :func:`run_galah`).  Because Galah uses a greedy
        algorithm, **the first parameter combination contributes ~80 % of
        the final representatives**.  Putting the right combo first is the
        single most impactful change you can make.

    The Tier-0 anchor family parameters (pg, res, bw) are now selected
    **automatically** from the ensemble quality data via
    :func:`_select_anchor_params`, replacing the previous hardcoded
    ``pg=100, res∈{5,8,10}, bw∈{0.1,0.15}``.

    Parameters
    ----------
    ensemble_list : list of tuple
        The ensemble quality list.
    nc_first : bool, optional
        If True, within each resolution the combo with the **highest NC**
        (near-complete count) is chosen as the ``is_first`` representative,
        instead of the first occurrence in *ensemble_list*.  Default False.
    """
    # ── Auto-select anchor parameters from ensemble data ──────────────
    try:
        chosen_pg, chosen_res, chosen_bw, best_res, best_bw = \
            _select_anchor_params(ensemble_list)
    except Exception:
        logger.warning("Anchor param selection failed; using defaults.")
        chosen_pg = (100,)
        chosen_res = (5, 8, 10)
        chosen_bw = (0.1, 0.15)
        best_res = 8.0
        best_bw = 0.15

    # Pre-compute the best combo per resolution (for Pass 1 ordering)
    res_best_combo = {}  # res -> (pg, bw)
    for item in ensemble_list:
        pg, res, bw = _parse_method_name(item[0])
        if pg in chosen_pg and res in chosen_res and bw in chosen_bw:
            score = abs(res - best_res) * 1.0 + abs(bw - best_bw) * 10.0
            if nc_first:
                nc_val = item[2] if len(item) > 2 else 0
                if res not in res_best_combo or nc_val > res_best_combo[res][2]:
                    res_best_combo[res] = ((pg, bw), score, nc_val)
            else:
                if res not in res_best_combo or score < res_best_combo[res][1]:
                    res_best_combo[res] = ((pg, bw), score)

    # Dynamic coarse supplement: only when chosen_res lacks res=1
    _has_coarse = 1.0 in chosen_res

    def _sort_key(item):
        method_name = item[0]
        pg, res, bw = _parse_method_name(method_name)

        # ── Tier 0: data-driven anchor family ──────────────────────────
        if pg in chosen_pg and res in chosen_res and bw in chosen_bw:
            anchor_score = abs(res - best_res) * 1.0 + abs(bw - best_bw) * 10.0
            best_combo = res_best_combo.get(res, ((None, None), 999))[0]
            is_first = (pg, bw) == best_combo
            # Pass1 (is_first): (0, 0, anchor_score) — one per res at front
            #   Different resolutions naturally interleave because
            #   anchor_score = |res-best_res|*1 + |bw-best_bw|*10 is
            #   dominated by the resolution gap term.
            # Pass2: (0, 100, anchor_score) — normal sort, no grouping
            return (0, 0 if is_first else 100, anchor_score)

        # ── Coarse supplement (dynamic) ────────────────────────────────
        if not _has_coarse:
            if pg == (chosen_pg[0] if chosen_pg else 100) and res == 1 and bw in (0.05, 0.1, 0.15):
                return (0, 200, 0, abs(bw - 0.1) * 10.0)

        # ── Tiers 1-4: unchanged (same as original _galah_sort_key) ────
        nc = item[2] if len(item) > 2 else 0
        div = _diversity_score(pg, res, bw, num_905=nc)
        if div < -1.5:
            return (4, -div)
        if div < 0:
            return (3, -div)
        if div < 0.15:
            return (2, -div)
        return (1, -div)

    reordered = sorted(ensemble_list, key=_sort_key)

    # Log the reordered list
    _log_reorder_result(reordered, sort_key_fn=_sort_key,
                        anchor_pg=chosen_pg[0] if chosen_pg else 100,
                        anchor_res=best_res, anchor_bw=best_bw)

    return reordered


def _log_reorder_result(reordered: List[Tuple], sort_key_fn=None,
                        anchor_pg=100, anchor_res=8.0, anchor_bw=0.15) -> None:
    """Print the reordered ensemble list with tier labels and diversity scores."""
    if sort_key_fn is None:
        sort_key_fn = _galah_sort_key
    tier_names = {
        0: "ANCHOR",
        1: "DIVERSITY",
        2: "GAP FILLER",
        3: "TRANSITION",
        4: "REDUNDANCY",
    }
    logger.info("--> Galah input order (after reorder_ensemble_for_galah):")
    current_tier = None
    for i, item in enumerate(reordered):
        tier = sort_key_fn(item)[0]
        pg, res, bw = _parse_method_name(item[0])
        nc = item[2] if len(item) > 2 else "?"
        div = _diversity_score(pg, res, bw, num_905=(nc if isinstance(nc, (int, float)) else 0))
        bins = len(item[1])

        if tier != current_tier:
            current_tier = tier
            label = tier_names.get(tier, f"TIER {tier}")
            logger.info(f"  --- {label} (Tier {tier}) ---")

        # Only log first 3 and last 2 of each tier, plus every entry in Tier 0
        count_in_tier = sum(1 for it in reordered if sort_key_fn(it)[0] == tier)
        pos_in_tier = sum(1 for j in range(i) if sort_key_fn(reordered[j])[0] == tier)
        show = (
            tier == 0
            or pos_in_tier < 3
            or pos_in_tier >= count_in_tier - 2
        )
        if show:
            logger.info(
                f"  [{i:3d}] pg={pg:>3d} res={res:>5.2f} bw={bw:.2f}  "
                f"div={div:+.3f}  bins={bins:>4d}  NC={str(nc):>3s}"
            )
        elif pos_in_tier == 3:
            logger.info(f"  ...  ({count_in_tier - 6} more in this tier)")


def summedLengthCal(name2seq: Dict[str, str]) -> int:
    return sum(len(seq) for seq in name2seq.values())


def getScore(
    qualityValues,
    apply_quality=True
) -> float:
    comp = qualityValues[0]
    cont = qualityValues[1]
    # Piecewise contamination penalty calibrated on 439 bacterial/archaeal
    # reference genomes (CheckM1-aligned SCG evaluation).
    # Contamination ≤ 3.42% (p95) is treated as inherent SCG marker
    # cross-detection, not real pollution.
    # → cont ≤ 3.42%: light penalty (k=1, SCG artifact regime)
    # → cont > 3.42%: heavy penalty (k=4, real contamination regime)
    cont_threshold = 3.42
    if cont <= cont_threshold:
        penalty = 1.0 * cont
    else:
        penalty = cont_threshold + 4.0 * (cont - cont_threshold)
    if apply_quality:
        if qualityValues[-1] == "HighQuality":
            score = comp - penalty + 100.
        elif qualityValues[-1] == "MediumQuality":
            score = comp - penalty + 50.
        else:
            score = comp - penalty
    else:
        score = comp - penalty
    return score


def run_galah(
    galah_out_folder: str,
    temp_flspp_bin_output: str,
    ensemble_list: list,
    cpu_num: int,
    bin_suffix: str
):
    if not os.path.exists(galah_out_folder):
        os.mkdir(galah_out_folder)
    cur_out_files_txt = os.path.join(galah_out_folder, "files_path.txt")
    with open(cur_out_files_txt, "a") as wh:
        for i, item in enumerate(ensemble_list):
            cur_method_name = item[0]
            cur_method_bin_folder = os.path.join(temp_flspp_bin_output, cur_method_name)
            for j, file_name, in enumerate(os.listdir(cur_method_bin_folder)):
                _, suffix = os.path.splitext(file_name)
                if suffix[1:] != bin_suffix:
                    continue
                wh.write(os.path.join(cur_method_bin_folder, file_name) + "\n")
    cmd = f"galah cluster --ani 95 --precluster-ani 90 --precluster-method skani " + \
        f" --genome-fasta-list {cur_out_files_txt} --output-cluster-definition {os.path.join(galah_out_folder, 'clusters.tsv')} " + \
        f" -t {cpu_num}"
    os.system(cmd)


def collect_galah_result_with_SCGs(
    scg_quality_report_path,
    galah_tsv_path: str,
    output_folder: str,
    mag_length_threshold=150000
):
    collect = {}
    clu_res_info = readGalahClusterTSV(galah_tsv_path)
    scg_meta_info = readMetaInfo(scg_quality_report_path)[0]
    wh = open(os.path.join(output_folder, "MetaInfo.tsv"), "w")
    # n: name, q: quality, v: path of file
    for c, vals in clu_res_info.items():
        for v in vals:
            n = os.path.split(v)[-1]
            q_scg = scg_meta_info[n]
            if c not in collect:
                collect[c] = [(n, q_scg, v, getScore(q_scg), q_scg)]
            else:
                collect[c].append((n, q_scg, v, getScore(q_scg), q_scg))
    res = []
    for _, q_l in collect.items():
        res.append(list(sorted(q_l, key=lambda x: x[3], reverse=True))[0])

    # print(f"The number of clusters is {len(res)}")
    for i, r in enumerate(res):
        outName = f"CompleteBin_{i}.fasta"
        size = summedLengthCal(readFasta(r[2]))
        if size > mag_length_threshold:
            wh.write(outName
                     + "\t"
                     + "SCGs_EVAL(Comp,Cont,Quality)"
                     + "\t"
                     + str(r[1][0])
                     + "\t"
                     + str(r[1][1])
                     + "\t"
                     + str(r[1][-1])
                     + "\n")
            copy(r[2], os.path.join(output_folder, outName))
    wh.close()


def process_galah_with_SCGs(
    clustering_all_folder,
    temp_flspp_bin_output: str,
    ensemble_list: list,
    outputBinFolder,
    scg_quality_report_path,
    gmm_flspp,
    mag_length_threshold,
    markerset_path: str,
    bac_contigName2_gene2num: dict,
    arc_contigName2_gene2num: dict,
    reuse_contig=True,
    cpus=64,
):
    logger.info("--> Start to Use Galah to Ensemble the Results.")
    ########
    galah_out = os.path.join(clustering_all_folder, f"galah_out_info_{gmm_flspp}")
    if os.path.exists(galah_out) is False:
        os.mkdir(galah_out)
    # # Drep gather and filter results
    galah_tsv = os.path.join(galah_out, "clusters.tsv")
    if os.path.exists(galah_tsv) is False:
        run_galah(galah_out, temp_flspp_bin_output, ensemble_list, cpus, "fasta")
    if os.path.exists(outputBinFolder) is False:
        os.makedirs(outputBinFolder)
    logger.info("--> Start to Process Galah Results.")
    collect_galah_result_with_SCGs(
        scg_quality_report_path,
        galah_tsv,
        outputBinFolder,
        mag_length_threshold=mag_length_threshold
    )
    # ---- Contig-level deduplication ----
    logger.info("--> Start contig-level deduplication across final bins.")

    if not reuse_contig:
        from CompleteBin.Dereplication.contig_dedup import contig_level_dedup
        dedup_markerset = readMarkerSets(markerset_path)
        n_resolved, n_removed = contig_level_dedup(
            bin_output_folder=outputBinFolder,
            bac_contigName2_gene2num=bac_contigName2_gene2num,
            arc_contigName2_gene2num=arc_contigName2_gene2num,
            markerset=dedup_markerset,
            mag_length_threshold=0,
        )
        logger.info(
            "--> Contig dedup: %d duplicate contigs resolved, %d bins removed.",
            n_resolved, n_removed,
        )
    else:
        from CompleteBin.Dereplication.post_galah_dedup import post_galah_dedup
        post_galah_dedup(output_folder=outputBinFolder)
