"""Experimental Phase-2 polishing implementation.

This module deliberately does not modify :mod:`polish_bin`.  Its public
``clean_bin_by_embedding`` entry point always uses the hybrid-reference
Phase-2 implementation described in ``PHASE2_POLISH_IMPROVEMENT_PLAN.md``.

Coverage is optional.  It can be supplied as ``contigname2coverage`` (a
contig -> vector mapping) or as ``coverage_profile_path`` pointing to the
existing coverage pickle.  If coverage is unavailable, the 20% bp cap is
used; the 25% extension requires a significant length-stratified
permutation pseudo-F test.
"""

import hashlib
import os
import pickle

import numpy as np

from CompleteBin.IO import writeFasta


DEFAULT_PERMUTATIONS = 199
DEFAULT_MIN_REFERENCE = 3
STANDARD_REFERENCE = 8
DEFAULT_BP_CAP = 0.20
ABSOLUTE_BP_CAP = 0.25
PERMUTATION_ALPHA = 0.05
MIN_COVERAGE_EFFECT = 0.5
COMPACT_COVERAGE_FILENAME = "contigname2coverage_profile.pkl"
_COVERAGE_CACHE = {}


def _length(name2seq, name):
    return max(1, len(name2seq[name]))


def _sum_length(name2seq):
    return sum(len(seq) for seq in name2seq.values())


def _log_n50(name2seq):
    # Keep the coverage/statistics helpers importable in lightweight
    # environments; seq_info has optional runtime dependencies.
    from CompleteBin.Seqs.seq_info import calculateN50
    return np.log(max(1, calculateN50(name2seq)))


def _normalise_profile(value):
    """Convert a scalar/per-BAM/per-base coverage object to one vector."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=np.float64)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] > 150:
        arr = arr[:, 75:-75]
    return np.nanmean(arr, axis=1).astype(np.float64, copy=False)


def ensure_compact_coverage_path(raw_path):
    """Create a small contig-by-BAM coverage profile for worker processes."""
    if not raw_path or not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Coverage profile does not exist: {raw_path}"
        )
    if os.path.basename(raw_path) == COMPACT_COVERAGE_FILENAME:
        return raw_path
    compact_path = os.path.join(
        os.path.dirname(raw_path), COMPACT_COVERAGE_FILENAME,
    )
    if (os.path.exists(compact_path)
            and os.path.getmtime(compact_path) >= os.path.getmtime(raw_path)):
        return compact_path

    with open(raw_path, "rb") as handle:
        raw = pickle.load(handle)
    if (isinstance(raw, dict)
            and raw.get("__format__") == "completebin_coverage_profile_v1"):
        profiles = raw.get("profiles", {})
    else:
        profiles = raw
    compact = {}
    for name, value in profiles.items():
        profile = _normalise_profile(value)
        if profile.size and np.all(np.isfinite(profile)):
            compact[name] = profile.astype(np.float32, copy=False)
    payload = {
        "__format__": "completebin_coverage_profile_v1",
        "profiles": compact,
    }
    temporary_path = f"{compact_path}.{os.getpid()}.tmp"
    try:
        with open(temporary_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, compact_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return compact_path


def _load_coverage(contigname2coverage=None, coverage_profile_path=None):
    if contigname2coverage is not None:
        raw = contigname2coverage
    elif coverage_profile_path:
        cache_key = os.path.realpath(coverage_profile_path)
        if cache_key in _COVERAGE_CACHE:
            return _COVERAGE_CACHE[cache_key]
        with open(coverage_profile_path, "rb") as handle:
            raw = pickle.load(handle)
        if (isinstance(raw, dict)
                and raw.get("__format__") == "completebin_coverage_profile_v1"):
            raw = raw.get("profiles", {})
    else:
        return {}

    result = {}
    for name, value in raw.items():
        name = name[1:] if isinstance(name, str) and name.startswith(">") else name
        profile = _normalise_profile(value)
        if profile.size and np.all(np.isfinite(profile)):
            result[name] = profile
    if contigname2coverage is None and coverage_profile_path:
        _COVERAGE_CACHE[os.path.realpath(coverage_profile_path)] = result
    return result


def _cosine_distance(embedding, centroid):
    e = np.asarray(embedding, dtype=np.float64)
    c = np.asarray(centroid, dtype=np.float64)
    e_norm = np.linalg.norm(e)
    c_norm = np.linalg.norm(c)
    if e_norm == 0.0 or c_norm == 0.0:
        return None
    return 1.0 - float(np.dot(e / e_norm, c / c_norm))


def _weighted_centroid(names, name2seq, contigname2emb):
    vectors = []
    weights = []
    lengths = np.asarray([_length(name2seq, n) for n in names], dtype=np.float64)
    p90 = float(np.quantile(lengths, 0.90)) if len(lengths) else 1.0
    for name, length in zip(names, lengths):
        if name not in contigname2emb:
            continue
        vector = np.asarray(contigname2emb[name], dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm == 0.0 or not np.all(np.isfinite(vector)):
            continue
        vectors.append(vector / norm)
        weights.append(np.sqrt(min(float(length), p90)))
    if not vectors:
        return None
    return np.average(np.stack(vectors, axis=0), axis=0,
                      weights=np.asarray(weights, dtype=np.float64))


def _distance_map(names, contigname2emb, centroid):
    if centroid is None:
        return {}
    result = {}
    for name in names:
        if name in contigname2emb:
            distance = _cosine_distance(contigname2emb[name], centroid)
            if distance is not None and np.isfinite(distance):
                result[name] = distance
    return result


def _unique_scg_anchors(names, contig2genes, gene2contigs):
    anchors = set()
    for name in names:
        genes = contig2genes.get(name, {})
        if genes and all(len(gene2contigs.get(gene, set())) == 1
                         for gene in genes):
            anchors.add(name)
    return anchors


def _hybrid_reference(names, name2seq, contigname2emb,
                      contig2genes, gene2contigs):
    """Build the hybrid reference set and return (reference, all_distances)."""
    temporary_centroid = _weighted_centroid(names, name2seq, contigname2emb)
    all_distances = _distance_map(names, contigname2emb, temporary_centroid)
    if not all_distances:
        return set(), {}

    central_cutoff = float(np.quantile(list(all_distances.values()), 0.80))
    anchors = _unique_scg_anchors(names, contig2genes, gene2contigs)
    reference = {
        name for name in anchors
        if name in all_distances and all_distances[name] <= central_cutoff
    }

    # Fill sparse SCG anchors with the nearest contigs to the temporary
    # centre.  These are reference contigs, not deletion candidates.
    ranked = sorted(all_distances, key=all_distances.get)
    for name in ranked:
        if len(reference) >= STANDARD_REFERENCE:
            break
        reference.add(name)
    return reference, all_distances


def _length_strata(lengths, number=5):
    if len(lengths) < 2:
        return np.zeros(len(lengths), dtype=np.int64)
    quantiles = np.quantile(lengths, np.linspace(0.0, 1.0, number + 1))
    edges = np.unique(quantiles[1:-1])
    return np.digitize(lengths, edges, right=True)


def _pseudo_f(matrix, labels):
    """Two-group pseudo-F based on Euclidean coverage-vector distances."""
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.unique(labels)
    if len(groups) != 2:
        return 0.0
    grand = np.mean(matrix, axis=0)
    between = 0.0
    within = 0.0
    for group in groups:
        rows = matrix[labels == group]
        if len(rows) == 0:
            return 0.0
        centre = np.mean(rows, axis=0)
        between += len(rows) * float(np.sum((centre - grand) ** 2))
        within += float(np.sum((rows - centre) ** 2))
    between_ms = between / (len(groups) - 1)
    within_df = len(matrix) - len(groups)
    if within_df <= 0 or within <= 1e-15:
        return float("inf") if between > 0 else 0.0
    return between_ms / (within / within_df)


def _permutation_seed(names, random_seed):
    digest = hashlib.sha256(
        (str(random_seed) + "||" + "||".join(sorted(names))).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def coverage_vector_permutation_score(names, retained, removed,
                                      contigname2coverage,
                                      name2seq,
                                      permutations=DEFAULT_PERMUTATIONS,
                                      random_seed=2048):
    """Return coverage evidence for ``retained`` versus ``removed``.

    The coverage of each contig remains a vector over BAMs.  Labels are
    permuted within length strata, preserving both group sizes and coarse
    length composition.  The return value is a dictionary suitable for
    logging and for selecting a threshold candidate.
    """
    valid = [name for name in names if name in contigname2coverage]
    if len(valid) < 6:
        return {"supported": False, "reason": "insufficient_coverage"}
    vectors = [contigname2coverage[name] for name in valid]
    widths = {len(vector) for vector in vectors}
    if len(widths) != 1:
        return {"supported": False, "reason": "inconsistent_coverage_width"}
    matrix = np.log1p(np.maximum(np.stack(vectors, axis=0), 0.0))
    labels = np.asarray([1 if name in retained else 0 for name in valid],
                        dtype=np.int8)
    if np.count_nonzero(labels == 0) < 3 or np.count_nonzero(labels == 1) < 3:
        return {"supported": False, "reason": "group_too_small"}

    lengths = np.asarray([_length(name2seq, name) for name in valid],
                         dtype=np.float64)
    strata = _length_strata(lengths)
    observed = _pseudo_f(matrix, labels)
    rng = np.random.default_rng(_permutation_seed(valid, random_seed))
    greater_or_equal = 0
    for _ in range(int(permutations)):
        permuted = labels.copy()
        for stratum in np.unique(strata):
            indices = np.flatnonzero(strata == stratum)
            if len(indices) > 1:
                permuted[indices] = labels[rng.permutation(indices)]
        if _pseudo_f(matrix, permuted) >= observed:
            greater_or_equal += 1
    p_value = (1.0 + greater_or_equal) / (float(permutations) + 1.0)

    retained_matrix = matrix[labels == 1]
    removed_matrix = matrix[labels == 0]
    centre_gap = float(np.linalg.norm(
        np.mean(retained_matrix, axis=0) - np.mean(removed_matrix, axis=0)
    ))
    retained_residual = retained_matrix - np.mean(retained_matrix, axis=0)
    removed_residual = removed_matrix - np.mean(removed_matrix, axis=0)
    within_rms = float(np.sqrt(
        (np.sum(retained_residual ** 2) + np.sum(removed_residual ** 2))
        / len(matrix)
    ))
    effect = centre_gap / max(within_rms, 1e-12)
    supported = bool(
        p_value <= PERMUTATION_ALPHA and effect >= MIN_COVERAGE_EFFECT
    )
    return {
        "supported": supported,
        "reason": "evaluated",
        "pseudo_f": float(observed),
        "permutation_p": float(p_value),
        "coverage_effect": float(effect),
        "covered_contigs": len(valid),
        "removed_covered_contigs": int(np.count_nonzero(labels == 0)),
    }


def _scg_eval(contig_names, tname2markerset,
              bac_c2g, arc_c2g, dom):
    if not contig_names:
        return 0.0, 0.0
    from CompleteBin.Cluster.sec_cluster_utils import determine_domain
    _, _, _, comp, cont = determine_domain(
        tname2markerset, list(contig_names), bac_c2g, arc_c2g, dom,
    )
    return comp, cont


def _fallback_to_original(
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
    contamination_trigger,
    completeness_drop_tolerance,
):
    """Use the original cleaner when improved reference evidence is weak."""
    from CompleteBin.Cluster.polish_bin import clean_bin_by_embedding as original
    return original(
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
        contamination_trigger,
        completeness_drop_tolerance,
    )


def _bp_limited_delete(candidates, name2seq, original_bp, cap):
    """Keep the farthest candidates until the bp cap is reached."""
    removed = set()
    removed_bp = 0
    for name in sorted(candidates, key=candidates.get, reverse=True):
        length = len(name2seq[name])
        if removed_bp + length > original_bp * cap:
            continue
        removed.add(name)
        removed_bp += length
    return removed


def _improved_clean(
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
    contigname2coverage=None,
    coverage_profile_path=None,
    permutations=DEFAULT_PERMUTATIONS,
    random_seed=2048,
):
    names = list(first_cluster_contigName2seq)
    if len(names) < 5:
        return index, quality_record
    comp_init, cont_init = _scg_eval(
        names, tname2markerset, bac_contigName2_gene2num,
        arc_contigName2_gene2num, first_dom,
    )
    if cont_init <= contamination_trigger:
        return index, quality_record

    reference, temporary_distances = _hybrid_reference(
        names, first_cluster_contigName2seq, contigname2repNormVector,
        first_cluster_contigName2_gene2num, first_cluster_gene2contig_list,
    )
    if len(reference) < DEFAULT_MIN_REFERENCE:
        return _fallback_to_original(
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
            contamination_trigger,
            completeness_drop_tolerance,
        )
    centroid = _weighted_centroid(
        list(reference), first_cluster_contigName2seq,
        contigname2repNormVector,
    )
    distances = _distance_map(names, contigname2repNormVector, centroid)
    reference_distances = [distances[name] for name in reference
                           if name in distances]
    if len(reference_distances) < DEFAULT_MIN_REFERENCE:
        return _fallback_to_original(
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
            contamination_trigger,
            completeness_drop_tolerance,
        )

    # Preserve the original SCG-guided Phase 1.  The improved module only
    # changes the subsequent marker-free Phase 2.
    kept = set(names)
    removed_total = set()
    current_comp = comp_init
    current_cont = cont_init
    original_core = _unique_scg_anchors(
        names, first_cluster_contigName2_gene2num,
        first_cluster_gene2contig_list,
    )
    if len(original_core) < 3:
        original_core = set(names)
    scg_noncore = [name for name in names
                   if name not in original_core
                   and first_cluster_contigName2_gene2num.get(name, {})
                   and name in distances]
    for name in sorted(scg_noncore, key=distances.get, reverse=True):
        trial_kept = kept - {name}
        trial_comp, trial_cont = _scg_eval(
            list(trial_kept), tname2markerset,
            bac_contigName2_gene2num, arc_contigName2_gene2num, first_dom,
        )
        if (trial_cont < current_cont
                and trial_comp >= current_comp - completeness_drop_tolerance):
            kept = trial_kept
            removed_total.add(name)
            current_comp = trial_comp
            current_cont = trial_cont
        if current_cont <= contamination_trigger:
            break

    all_central = [name for name, distance in temporary_distances.items()
                   if distance <= np.quantile(
                       list(temporary_distances.values()), 0.80)]
    central_distances = [distances[name] for name in all_central
                         if name in distances]
    if len(central_distances) < DEFAULT_MIN_REFERENCE:
        central_distances = reference_distances

    scg_fraction = len(first_cluster_contigName2_gene2num) / max(len(names), 1)
    k_adaptive = float(np.clip(3.0 + 2.0 * (1.0 - scg_fraction), 3.0, 5.0))
    if len(reference) < STANDARD_REFERENCE:
        k_sparse = 5.0 - min(
            1.0,
            max(0.0, cont_init - contamination_trigger)
            / max(5.0, contamination_trigger),
        )
        k_adaptive = max(k_adaptive, k_sparse)

    reference_mean = float(np.mean(reference_distances))
    reference_std = float(np.std(reference_distances))
    central_mean = float(np.mean(central_distances))
    central_std = float(np.std(central_distances))
    # Reference members are always retained.  Only marker-free contigs
    # outside the reference are eligible for this Phase-2 operation.
    nonscg = [name for name in kept
              if name not in reference
              and not first_cluster_contigName2_gene2num.get(name, {})
              and name in distances]
    if not nonscg:
        if not removed_total:
            return index, quality_record
        cleaned = {name: first_cluster_contigName2seq[name] for name in kept}
        size = _sum_length(cleaned)
        if size < mag_length_threshold:
            return index, quality_record
        filename = f"CompleteBin_cand_{index}.fasta"
        writeFasta(cleaned, os.path.join(output_folder, filename))
        n50 = _log_n50(cleaned)
        quality_record[filename] = (current_comp, current_cont, n50, size)
        return index + 1, quality_record

    coverage = _load_coverage(contigname2coverage, coverage_profile_path)
    original_bp = _sum_length(first_cluster_contigName2seq)
    candidate_results = []
    length_values = np.asarray([len(first_cluster_contigName2seq[name])
                                for name in names], dtype=np.float64)
    short_cutoff = float(np.quantile(length_values, 1.0 / 3.0))
    for candidate_k in sorted({k_adaptive,
                               min(k_adaptive + 0.5, 5.0), 5.0}):
        candidate_distances = {}
        for name in nonscg:
            local_k = candidate_k + (0.5 if
                                     len(first_cluster_contigName2seq[name])
                                     <= short_cutoff else 0.0)
            threshold = max(
                reference_mean + local_k * reference_std,
                central_mean + local_k * central_std,
            )
            if distances[name] > threshold:
                candidate_distances[name] = distances[name]
        if not candidate_distances:
            continue

        removed = _bp_limited_delete(
            candidate_distances, first_cluster_contigName2seq,
            original_bp, DEFAULT_BP_CAP,
        )
        if not removed:
            continue
        coverage_score = coverage_vector_permutation_score(
            list(kept), set(kept) - removed, removed, coverage,
            first_cluster_contigName2seq, permutations, random_seed,
        )
        if coverage_score.get("supported", False):
            expanded_removed = _bp_limited_delete(
                candidate_distances, first_cluster_contigName2seq,
                original_bp, ABSOLUTE_BP_CAP,
            )
            expanded_score = coverage_vector_permutation_score(
                list(kept), set(kept) - expanded_removed,
                expanded_removed, coverage, first_cluster_contigName2seq,
                permutations, random_seed,
            )
            if expanded_score.get("supported", False):
                removed = expanded_removed
                coverage_score = expanded_score
        if not removed:
            continue
        candidate_results.append((candidate_k, removed, coverage_score))

    if not candidate_results:
        # Phase 1 may have produced a valid cleaned bin even when Phase 2
        # has no marker-free contig eligible for removal.  Do not discard
        # that Phase-1-only result.
        if not removed_total:
            return index, quality_record
        cleaned = {
            name: first_cluster_contigName2seq[name]
            for name in kept
        }
        size = _sum_length(cleaned)
        if size < mag_length_threshold:
            return index, quality_record
        filename = f"CompleteBin_cand_{index}.fasta"
        writeFasta(cleaned, os.path.join(output_folder, filename))
        n50 = _log_n50(cleaned)
        quality_record[filename] = (current_comp, current_cont, n50, size)
        return index + 1, quality_record

    # Coverage changes the selection objective.  If coverage supports a
    # candidate, use the largest supported removal (the strongest supported
    # cleanup).  If no candidate has coverage support, remain conservative
    # and use the candidate with the largest k / smallest removal.
    supported_results = [item for item in candidate_results
                         if item[2].get("supported", False)]
    if supported_results:
        _, removed, _ = max(
            supported_results,
            key=lambda item: (
                sum(len(first_cluster_contigName2seq[name])
                    for name in item[1]),
                -item[0],
            ),
        )
    else:
        _, removed, _ = max(
            candidate_results,
            key=lambda item: (
                item[0],
                -sum(len(first_cluster_contigName2seq[name])
                     for name in item[1]),
            ),
        )
    kept -= removed
    removed_total.update(removed)
    cleaned = {name: first_cluster_contigName2seq[name] for name in kept}
    size = _sum_length(cleaned)
    if size < mag_length_threshold:
        return index, quality_record
    filename = f"CompleteBin_cand_{index}.fasta"
    writeFasta(cleaned, os.path.join(output_folder, filename))
    n50 = _log_n50(cleaned)
    quality_record[filename] = (current_comp, current_cont, n50, size)
    return index + 1, quality_record


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
    contigname2coverage=None,
    coverage_profile_path=None,
    permutations=DEFAULT_PERMUTATIONS,
    random_seed=2048,
):
    """Clean one bin using the mandatory improved Phase-2 polish."""
    if contigname2coverage is None:
        if not coverage_profile_path:
            raise FileNotFoundError(
                "Improved polish requires coverage_profile_path when "
                "contigname2coverage is not provided."
            )
        if not os.path.isfile(coverage_profile_path):
            raise FileNotFoundError(
                f"Coverage profile does not exist: {coverage_profile_path}"
            )
    return _improved_clean(
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
        contamination_trigger,
        completeness_drop_tolerance,
        contigname2coverage,
        coverage_profile_path,
        permutations,
        random_seed,
    )
