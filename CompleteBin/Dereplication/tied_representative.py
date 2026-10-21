"""Choose longer equally scored representatives only with supporting evidence.

No reference labels are used. Thresholds are conservative starting settings,
not a guarantee of biological purity. Coverage is required; embeddings are
diagnostic only and never veto selection.
"""
import numpy as np
from CompleteBin.IO import readFasta


def _lookup(mapping, name):
    value = mapping.get(name)
    return mapping.get(name.lstrip('>')) if value is None else value


def _quantile(values, weights, q):
    order = np.argsort(values)
    return float(np.asarray(values)[order][np.searchsorted(
        np.cumsum(np.asarray(weights)[order]), q * sum(weights), side='left')])


def _coverage_vector_support(coverage, core_weights):
    """Check one distance for the full log-coverage vector, not each axis.

    RMS distance is Euclidean distance divided by sqrt(number of samples).
    The floor log(2) and length-weighted 95th percentile are unchanged in
    meaning for a single sample; multiple samples share one distance limit.
    """
    values = np.asarray(coverage, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n_core = len(core_weights)
    if (values.ndim != 2 or values.shape[1] == 0 or n_core == 0
            or n_core > len(values) or not np.all(np.isfinite(values))
            or np.any(values < 0)):
        raise ValueError('invalid coverage vectors')
    log_values = np.log1p(values)
    center = np.array([_quantile(v, core_weights, .5)
                       for v in log_values[:n_core].T])
    distance = np.sqrt(np.mean((log_values - center) ** 2, axis=1))
    limit = max(float(np.log(2)), _quantile(distance[:n_core], core_weights, .95))
    return distance[n_core:] <= limit, {
        'coverage_check': 'log1p_vector_rms',
        'coverage_dimensions': int(values.shape[1]),
        'coverage_distance_limit': limit,
        'coverage_log_center': center.tolist(),
        'added_coverage_distance_max': float(distance[n_core:].max())
            if len(distance) > n_core else None,
    }


PLANT_MAX_CONTIGS_GE_768 = 80317
TIED_POLICY_MIN_CONTIG_LENGTH = 768


def should_prefer_longer_representative(num_bam_files, num_contigs_ge_768):
    """Enable for multiple BAMs OR inputs strictly smaller than Plant's max.

    The count is taken once from the full input, before clustering/training
    length filtering, and reused for both clustering stages.
    """
    if num_bam_files < 0 or num_contigs_ge_768 < 0:
        raise ValueError('BAM and contig counts must be nonnegative')
    return num_bam_files > 1 or num_contigs_ge_768 <= PLANT_MAX_CONTIGS_GE_768


def select_tied_representative(items, embeddings=None, coverage=None, prefer_longer=True):
    """Items retain galah_utils' (name, quality, path, score, quality) shape."""
    anchor = max(items, key=lambda x: (x[3], x[2]))
    top = [x for x in items if x[3] == anchor[3]]
    record = dict(original=anchor[2], selected=anchor[2], score=float(anchor[3]),
                  tied=len(top), attempts=[])
    if not prefer_longer:
        record['replacement_disabled'] = True
        return anchor, record
    if len(top) < 2:
        return anchor, record
    sequences = {x[2]: readFasta(x[2]) for x in top}
    lengths = {p: sum(map(len, seq.values())) for p, seq in sequences.items()}
    original = sequences[anchor[2]]
    anchor_length = lengths[anchor[2]]
    if not anchor_length:
        return anchor, record
    for candidate in sorted(top, key=lambda x: (lengths[x[2]], x[2]), reverse=True):
        if lengths[candidate[2]] <= anchor_length:
            break
        seq = sequences[candidate[2]]
        result = dict(path=candidate[2], length=lengths[candidate[2]])
        record['attempts'].append(result)
        common = set(original) & set(seq)
        containment = sum(len(original[c]) for c in common) / anchor_length
        result['retained_fraction'] = containment
        if containment < .95:
            result['reason'] = 'less_than_95_percent_original_length'; continue
        if candidate[1][0] < anchor[1][0] or candidate[1][1] > anchor[1][1]:
            result['reason'] = 'marker_quality_worsened'; continue
        if not coverage:
            result['reason'] = 'missing_evidence'; continue
        added = sorted(set(seq) - set(original))
        core = sorted(original)
        if not added:
            result['reason'] = 'no_added_sequences'; continue
        names = core + added
        cv = [_lookup(coverage, c) for c in names]
        if any(v is None for v in cv):
            result['reason'] = 'missing_coverage_features'; continue
        weights = np.array([len(original[n]) for n in core], dtype=float)
        aw = np.array([len(seq[n]) for n in added], dtype=float)
        try:
            coverage_ok, coverage_details = _coverage_vector_support(cv, weights)
            support = float(aw[coverage_ok].sum() / aw.sum())
        except (ValueError, TypeError, IndexError):
            result['reason'] = 'invalid_coverage_features'; continue
        result.update(supported_added_fraction=support,
                      coverage_supported_added_fraction=support,
                      support_basis='coverage_only', **coverage_details)
        # Embedding evidence is recorded separately; it cannot reject a candidate.
        result['embedding_diagnostic'] = 'unavailable'
        result['embedding_supported_added_fraction'] = None
        if embeddings:
            ev = [_lookup(embeddings, c) for c in names]
            if all(v is not None for v in ev):
                try:
                    e = np.asarray(ev, dtype=float)
                    norms = np.linalg.norm(e, axis=1)
                    if not np.all(np.isfinite(e)) or np.any(norms == 0):
                        raise ValueError('invalid embeddings')
                    e = e / norms[:, None]
                    centroid = np.average(e[:len(core)], axis=0, weights=weights)
                    norm = np.linalg.norm(centroid)
                    if norm == 0: raise ValueError('zero centroid')
                    dist = np.clip(1 - e @ (centroid / norm), 0, 2)
                    emb_limit = max(.15, _quantile(dist[:len(core)], weights, .95))
                    result.update(embedding_diagnostic='recorded',
                        embedding_limit=float(emb_limit),
                        embedding_supported_added_fraction=float(
                            aw[dist[len(core):] <= emb_limit].sum() / aw.sum()))
                except (ValueError, TypeError, IndexError):
                    result['embedding_diagnostic'] = 'invalid'
        if support < .90:
            result['reason'] = 'less_than_90_percent_coverage_supported_added_length'; continue
        result['reason'] = 'accepted'
        record['selected'] = candidate[2]
        return candidate, record
    return anchor, record
