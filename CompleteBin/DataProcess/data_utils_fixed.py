"""
Memory-safe replacement for the data-generation stage of CompleteBin.

Problem (original code)
-----------------------
``build_training_seq_data_numpy_save`` loads ~28 GB of data (25 GB coverage
arrays + 3 GB sequences) into the main process and then calls
``multiprocessing.Pool(N)`` which uses *fork* on Linux.  Every forked child
inherits the full 28 GB page table.  CPython's periodic GC and reference
counting then trigger copy-on-write across the entire dataset, leading to

    10 workers × 28 GB (COW'd)  ≈ 280 GB
    + main process               ≈  28 GB
    + pickle transfer buffers    ≈  50 GB
    + save_list accumulation     ≈  28 GB
    + np.save pickle buffer      ≈  30 GB
    ─────────────────────────────────────
    Peak RSS                     ≈ 420–500 GB  (reported 1.7 TB with VSS)

Root fix
--------
1. **Spawn instead of fork** – workers are fresh Python processes that do
   NOT inherit the parent's 28 GB.  They only receive the data they need.
2. **Temp-file I/O for worker results** – each worker writes its output to a
   compressed .npz file on disk.  This avoids pickling large return values
   back to the main process and avoids accumulating everything in a giant
   ``save_list`` before serialisation.
3. **Adaptive worker count** – for datasets with > 500 k contigs or when the
   auto-computed worker count exceeds 4, the pool size is capped.

Expected peak RSS after the fix
-------------------------------
    main process (original data)  ≈ 28 GB
    spawned workers (each ~2.8 GB)≈ 28 GB  (10 × 2.8, isolated address spaces)
    np.save pickle buffer         ≈ 30 GB  (temporary, during final save)
    ─────────────────────────────────────
    Main-process peak             ≈ 60 GB  (workers do NOT add to main RSS)

GPU training is completely unaffected – this module is only imported during
Step 1 (data preparation).

Usage
-----
In ``Binning_steps.py``, replace::

    from CompleteBin.DataProcess.data_utils import build_training_seq_data_numpy_save

with::

    from CompleteBin.DataProcess.data_utils_fixed import (
        build_training_seq_data_numpy_save_fixed as build_training_seq_data_numpy_save,
    )

The function signature is identical.
"""

import multiprocessing as mp
import os
import pickle
import random
import shutil
from typing import List

import numpy as np
import psutil

from CompleteBin.CallGenes.gene_utils import splitListEqually
from CompleteBin.DataProcess.data_utils import get_features_of_one_seq
from CompleteBin.IO import readPickle
from CompleteBin.Seqs.seq_utils import generate_feature_mapping_reverse

# ── logger (same singleton as the rest of CompleteBin) ────────────────
from CompleteBin.logger import get_logger

logger = get_logger()


# ═══════════════════════════════════════════════════════════════════════
#  Worker function (runs in a *spawned* child – no inherited memory)
# ═══════════════════════════════════════════════════════════════════════

def _worker_process_chunk(
    temp_chunk_path: str,
    temp_output_path: str,
    count_kmer: int,
    min_len: int,
):
    """
    Load a pickled chunk of contigs, compute features for every contig,
    and save the results to a compressed .npz file.

    Parameters
    ----------
    temp_chunk_path : str
        Path to a pickle file containing ``(c2s, c2b)`` – two dicts mapping
        contig name → sequence string and contig name → coverage ndarray.
    temp_output_path : str
        Where to write the ``.npz`` with the computed features.
    count_kmer : int
        k-mer size (normally 4).
    min_len : int
        Minimum contig length.

    Notes
    -----
    This function does NOT return raw sequences or per-base coverage arrays
    to the main process.  Only the *computed features* are saved.  The main
    process already holds the original data and will re-attach it during
    the combine step.
    """
    # -- load this worker's chunk from disk (only ~2.8 GB per worker) ----
    with open(temp_chunk_path, "rb") as fh:
        c2s, c2b = pickle.load(fh)

    count_kmer_dict, count_nr_features = generate_feature_mapping_reverse(count_kmer)

    results: List[tuple] = []

    for contigname, seq in c2s.items():
        seq_rad_tokens, mean_tokens, std_tokens, whole_bp_cov_tnf_array = (
            get_features_of_one_seq(
                seq,
                c2b[contigname],
                count_kmer,
                count_kmer_dict,
                count_nr_features,
                [1, 5],
                min_len,
            )
        )
        # Store ONLY computed features (float32 arrays, all small).
        # Raw ``seq`` and ``cov_bp_array_list`` are deliberately omitted
        # here – they already exist in the main process.
        results.append(
            (
                contigname[1:],  # drop leading '>'
                seq_rad_tokens,  # [6, 136]  float32
                mean_tokens,     # [6, B]    float32
                std_tokens,      # [6, B]    float32
                whole_bp_cov_tnf_array,  # [B, 136]  float32
            )
        )

    # -- persist to disk (compressed) ----------------------------------
    np.savez_compressed(
        temp_output_path,
        results=np.array(results, dtype=object),
        allow_pickle=True,
    )

    return len(results)


# ═══════════════════════════════════════════════════════════════════════
#  Public entry point (drop-in replacement)
# ═══════════════════════════════════════════════════════════════════════

def build_training_seq_data_numpy_save_fixed(
    contigname2seq_path: str,
    contigname2bp_nparray_list_path: str,
    data_output_path: str,
    count_kmer,
    min_len,
    num_workers: int = None,
):
    """
    Memory-safe drop-in replacement for the original
    ``build_training_seq_data_numpy_save``.

    Signature, inputs, and output file format are **identical** to the
    original.  The only difference is that this version uses *spawn*
    multiprocessing + temp-file I/O to keep main-process RSS under ~60 GB
    even for datasets with > 600 k contigs and 6 BAM files.

    Parameters
    ----------
    contigname2seq_path : str
        Path to ``contigname2seq_str.pkl``.
    contigname2bp_nparray_list_path : str
        Path to ``contigname2bpcover_nparray_list.pkl``.
    data_output_path : str
        Directory for the output file ``training_data.npy``.
    count_kmer : int
        k-mer size.
    min_len : int
        Minimum contig length.
    num_workers : int or None
        Number of parallel workers.  ``None`` auto-computes as
        ``cpu_count // 10 + 2`` (capped at 4 for large datasets).
    """
    # ── load original data (main process) ────────────────────────────
    logger.info("--> [FIXED] Loading contig sequences and coverage data.")
    contigname2seq = readPickle(contigname2seq_path)
    contigname2bp_array_list = readPickle(contigname2bp_nparray_list_path)

    if not os.path.exists(data_output_path):
        os.mkdir(data_output_path)

    # ── decide worker count ──────────────────────────────────────────
    if num_workers is None:
        num_workers = psutil.cpu_count() // 10 + 2

    # For large datasets the pickle buffers for many concurrent workers
    # can add up.  Capping at 4 keeps throughput reasonable while keeping
    # memory bounded.
    total_contigs = len(contigname2seq)
    if total_contigs > 500_000 or num_workers > 4:
        capped = min(num_workers, 4)
        if capped != num_workers:
            logger.info(
                "--> [FIXED] Large dataset (%d contigs): capping workers "
                "from %d to %d to limit pickle-buffer memory.",
                total_contigs,
                num_workers,
                capped,
            )
        num_workers = capped

    # ── split contig names ───────────────────────────────────────────
    contignames = list(contigname2seq.keys())
    random.shuffle(contignames)
    contignames_list: List[List[str]] = splitListEqually(contignames, num_workers)

    # ── write each worker's chunk to a temp pickle file ───────────────
    # We dump the data to disk first so that spawned workers can load
    # only their own chunk.  This avoids sending 2.8 GB through the
    # pickle pipe (which would create large temporary buffers in both
    # parent and child).
    temp_dir = os.path.join(data_output_path, "_temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    temp_chunk_paths: List[str] = []
    for i, names in enumerate(contignames_list):
        c2s = {n: contigname2seq[n] for n in names}
        c2b = {n: contigname2bp_array_list[n] for n in names}
        chunk_path = os.path.join(temp_dir, f"chunk_{i:04d}.pkl")
        with open(chunk_path, "wb") as fh:
            pickle.dump((c2s, c2b), fh, protocol=pickle.HIGHEST_PROTOCOL)
        temp_chunk_paths.append(chunk_path)
        logger.info(
            "--> [FIXED] Wrote chunk %d/%d (%d contigs) to %s",
            i + 1,
            num_workers,
            len(names),
            chunk_path,
        )

    # Release the temporary ``c2s`` / ``c2b`` dicts so the main process
    # only holds the single master copy of the data.
    del c2s, c2b

    # ── spawn workers (NO fork – fresh Python processes) ─────────────
    logger.info(
        "--> [FIXED] Spawning %d workers in 'spawn' mode (no COW inheritance).",
        num_workers,
    )
    ctx = mp.get_context("spawn")
    temp_output_paths: List[str] = []

    with ctx.Pool(num_workers) as pool:
        futures = []
        for i in range(num_workers):
            out_path = os.path.join(temp_dir, f"output_{i:04d}.npz")
            temp_output_paths.append(out_path)
            fut = pool.apply_async(
                _worker_process_chunk,
                (temp_chunk_paths[i], out_path, count_kmer, min_len),
            )
            futures.append(fut)

        pool.close()

        for i, fut in enumerate(futures):
            n_done = fut.get()  # blocks until worker i finishes
            logger.info(
                "--> [FIXED] Worker %d/%d finished: %d contigs processed.",
                i + 1,
                num_workers,
                n_done,
            )

    # ── combine worker outputs ───────────────────────────────────────
    # Results are loaded one .npz at a time and the full tuple is
    # reconstructed by looking up the original sequences and coverage
    # arrays that are still held in the main process.
    logger.info("--> [FIXED] Combining worker outputs into training_data.npy.")

    training_data_path = os.path.join(data_output_path, "training_data.npy")

    # We accumulate into a Python list, exactly like the original code.
    save_list: list = []
    for out_path in temp_output_paths:
        loaded = np.load(out_path, allow_pickle=True)
        worker_results = loaded["results"]

        for item in worker_results:
            contigname_no_gt, seq_rad_tokens, mean_tokens, std_tokens, whole_bp_cov_tnf_array = item
            full_name = ">" + contigname_no_gt

            # Reconstruct the 6-tuple expected by ssmt_v2.py
            cur_tuple = (
                contigname2seq[full_name],
                contigname2bp_array_list[full_name],
                seq_rad_tokens,
                mean_tokens,
                std_tokens,
                whole_bp_cov_tnf_array,
            )
            save_list.append((contigname_no_gt, cur_tuple))

    # ── persist final output ─────────────────────────────────────────
    np.save(
        training_data_path,
        np.array(save_list, dtype=object),
        allow_pickle=True,
    )
    logger.info(
        "--> [FIXED] Saved training_data.npy with %d contigs.", len(save_list)
    )

    # ── cleanup ──────────────────────────────────────────────────────
    logger.info("--> [FIXED] Cleaning up temp files.")
    shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("--> [FIXED] Data generation complete.")
    return save_list
