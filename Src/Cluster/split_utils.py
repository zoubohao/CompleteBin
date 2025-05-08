
import functools
import os
from typing import List

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import (KMeans, check_random_state,
                                     euclidean_distances, row_norms,
                                     stable_cumsum)

from Src.CallGenes.gene_utils import splitListEqually
from Src.DataProcess.data_utils import split_seq_equally
from Src.IO import writeFasta


def gen_seed_idx(seed_list, contig_id_list: List[str]) -> List[int]:
    """
    Generate a list of indices corresponding to seed contig IDs from a given URL.

    :param seedURL: The URL or path to the file containing seed contig names.
    :param contig_id_list: List of all contig IDs to match with the seed contig names.
    :return: List[int]
    """
    name_map = dict(zip(contig_id_list, range(len(contig_id_list))))
    seed_idx = [name_map[seed_name] for seed_name in seed_list]
    return seed_idx


# change from sklearn.cluster.kmeans
def partial_seed_init(X, n_clusters: int, random_state, seed_idx, n_local_trials=None) -> np.ndarray:
    """
    Partial initialization of KMeans centers with seeds from seed_idx.

    Parameters:
    :param X: Features.
    :param n_clusters: The number of clusters.
    :param random_state: Determines random number generation for centroid initialization. Use an int for reproducibility.
    :param seed_idx: Indices of seed points for initialization.
    :param n_local_trials: The number of local seeding trials. Default is None.

    Returns:
    :return centers (ndarray): The initialized cluster centers.

    This function initializes a KMeans clustering by partially seeding the centers with provided seeds.
    It is a modification of the KMeans initialization algorithm.
    """
    random_state = check_random_state(random_state)
    x_squared_norms = row_norms(X, squared=True)

    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly

    center_id = seed_idx[0]

    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)

    for c, center_id in enumerate(seed_idx[1:], 1):
        if sp.issparse(X):
            centers[c] = X[center_id].toarray()
        else:
            centers[c] = X[center_id]
        closest_dist_sq = np.minimum(closest_dist_sq,
                                     euclidean_distances(
                                         centers[c, np.newaxis], X, Y_norm_squared=x_squared_norms,
                                         squared=True))
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(len(seed_idx), n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def kmeans_split(logger,
                out_fasta_path: str,
                contigname2seq: dict,
                namelist: List[str], 
                X_mat: np.ndarray, 
                length_weight: np.ndarray,
                bin_number,
                contignames_40mar: set,
                min_contig_len):
    logger.info(f"--> Running kmeans to get initial split. Bin number: {bin_number}")
    seed_idx = gen_seed_idx(contignames_40mar, namelist)
    div = np.log(min_contig_len + 0.)
    len_vals_list = []
    for len_val in length_weight:
        len_vals_list.append(np.log(len_val) / div)
    len_vals_list = np.array(len_vals_list)
    km = KMeans(n_clusters=bin_number, random_state=3407,
                init=functools.partial(partial_seed_init, seed_idx=seed_idx),
                n_init=1)
    km.fit(X_mat, sample_weight=len_vals_list)
    idx = km.labels_
    res = {}
    for clu, name in zip(idx, namelist):
        if clu not in res:
            res[clu] = [name]
        else:
            res[clu].append(name)
    index = 0
    for clu, names in res.items():
        cur_clu_res = {}
        cur_summed_bps = 0
        for name in names:
            cur_clu_res[name] = contigname2seq[name]
            cur_summed_bps += len(contigname2seq[name])
        if cur_summed_bps >= 32000000:
            cur_name_list = list(cur_clu_res.keys())
            split_names_list = splitListEqually(cur_name_list, cur_summed_bps // 32000000 + 2)
            for one_split_names in split_names_list:
                cur_split_clu_res = {}
                for split_name in one_split_names:
                    cur_split_clu_res[split_name] = contigname2seq[split_name]
                writeFasta(cur_split_clu_res, os.path.join(out_fasta_path, f"Init_Bins_{index}.fasta"))
                index += 1
        else:
            writeFasta(cur_clu_res, os.path.join(out_fasta_path, f"Init_Bins_{index}.fasta"))
            index += 1
    logger.info("--> End of Running kmeans.")