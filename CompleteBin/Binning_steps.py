

from CompleteBin.Utils.pretrain_decision import (
    compute_S, compute_O, decide_pretrained, apply_epoch_safety_valve, auto_min_contig_length
)
from CompleteBin.Cluster.polish_bin_improved import ensure_compact_coverage_path
from CompleteBin.Cluster.k_means_v2 import cluster_kmeans_for_low_v2
from .combine_cluster_derep import clustering_and_dereplication, selected_recluster_contigs
from CompleteBin.version import bin_v
from CompleteBin.Trainer.ssmt_v2 import SelfSupervisedMethodsTrainer
from CompleteBin.Seqs.seq_info import calculateN50, clip_coverage_outliers, prepare_sequences_coverage
from CompleteBin.Seqs.generate_seed import gen_seed
from CompleteBin.logger import get_logger
from CompleteBin.Trainer.sampler import DeeperBinSampler
from CompleteBin.IO import readFasta, readMarkerSets, readPickle, readSeedFile, writeFasta
from CompleteBin.DataProcess.data_utils_fixed import (
    build_training_seq_data_numpy_save_fixed as build_training_seq_data_numpy_save,
)
from CompleteBin.Cluster.split_utils import kmeans_split
from CompleteBin.CallGenes.gene_utils import callMarkerGenesByHMM
import torch
import psutil
import numpy as np
from typing import List
from shutil import copy, rmtree
import random
import os
import multiprocessing as mp
import math
import time
import warnings
warnings.filterwarnings("ignore")

logger = get_logger()


def get_time(f):

    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        logger.info(f"--> Time: {e_time - s_time:.2f} seconds")
        return res
    return inner


def _write_time_tsv(temp_file_folder_path: str, entries: dict) -> None:
    """Incrementally record per-step timings into ``time.tsv``.

    ``binning_with_all_steps`` is resumable and, in split SLURM execution
    (``step_num=1/2/3``), each step runs in its own process.  Every step writes
    only its own measurements through this helper; rows recorded by an earlier
    step are preserved and ``SummedTime(s)`` is recomputed from all recorded
    rows.
    """
    path = os.path.join(temp_file_folder_path, "time.tsv")
    times = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                label, _, value = line.rpartition("\t")
                try:
                    times[label] = float(value)
                except ValueError:
                    continue
    times.update(entries)
    times["SummedTime(s)"] = sum(v for k, v in times.items() if k != "SummedTime(s)")
    with open(path, "w") as wh:
        for label, value in times.items():
            wh.write(f"{label}\t{value}\n")


def writeMetaInfo(wh, outName, comp, cont, state):
    wh.write(outName
             + "\t"
             + "SCG_EVAL(Comp,Cont,Quality)"
             + "\t"
             + str(comp)
             + "\t"
             + str(cont)
             + "\t"
             + state
             + "\n")


def seed_everything(seed=2048):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def filterSpaceInFastaFile(input_fasta, output_fasta):
    with open(input_fasta, "r") as rh, open(output_fasta, "w") as wh:
        for line in rh:
            oneline = line.strip("\n")
            if ">" in oneline and " " in oneline:
                oneline = oneline.split()[0]
            wh.write(oneline + "\n")


def temp_decision(
    contigname2seq_ori: dict,
    min_contig_length: int,
    N50: int,
    large_data_size_thre: int
):
    # Weak Augmentations: Views are more similar; use higher temperatures to smooth similarity distribution. --> N50 samaller, temp higher
    # Data Noise: Higher noise levels benefit from a higher temperature to smoothen similarity scores. --> N50 samaller, temp higher
    # Dataset Size: For large datasets, lower temperatures often work better as they increase the discriminative power of the embeddings.
    # --> larger size, lower temp
    count_contigs = 0
    for _, seq in contigname2seq_ori.items():
        if len(seq) >= min_contig_length:
            count_contigs += 1
    temp = None
    if N50 >= 15000:
        if count_contigs >= large_data_size_thre:
            temp = 0.115
        else:
            temp = 0.125
    elif N50 <= 5000:
        if count_contigs >= large_data_size_thre:
            temp = 0.135
        else:
            temp = 0.145
    else:
        temp = -0.000002 * N50 + 0.155
        temp = float("%.3f" % temp)
        if count_contigs >= large_data_size_thre:
            temp -= 0.01
    return temp


def decision_lower_bound(
    contigname2seq: dict,
    min_contig_length: int,
    batch_size: int,
):
    cover_count = 0
    tmp_store = []
    for _, seq in contigname2seq.items():
        tmp_store.append(len(seq))
        if len(seq) < min_contig_length:
            continue
        cover_count += 1
    tmp_store = list(sorted(tmp_store, reverse=True))
    N = len(contigname2seq)
    assert N > batch_size, ValueError(
        f"There are only {N} contigs in this dataset. But we need at least {batch_size} (one batch size) contigs for training.")
    logger.info(f"--> The number of {cover_count} contigs are longer than {min_contig_length}.")
    if cover_count <= batch_size:
        cover_count = batch_size
    else:
        gap_num = N - cover_count
        multi = gap_num // batch_size
        if multi > 2:
            multi = 2
        cover_count = (cover_count // batch_size + multi) * batch_size
    logger.info(f"--> We cover {cover_count} contigs for training.")
    min_length_decision = tmp_store[min(cover_count, len(tmp_store) - 1)]
    return min_contig_length - min_length_decision


def binning_with_all_steps(
    contig_file_path: str,
    sorted_bam_file_list: List[str],
    temp_file_folder_path: str,
    bin_output_folder_path: str,
    n_views=6,
    feature_dim=100,
    auto_feature_dim=False,
    db_folder_path: str = None,
    count_kmer: int = 4,
    min_contig_length=850,
    leiden_iter_mode="accurate",  # "accurate" or "fast"
    # model training config
    drop_p=0.15,
    lr=1e-5,
    lr_multiple=10,
    lr_warmup_epoch=2,
    weight_deay=1e-3,
    batch_size=544,
    base_epoch=35,
    log_every_n_steps=10,
    training_device="cpu",
    cpu_workers=None,
    gpu_dataloader_workers=32,
    min_training_step=36,
    step_num=None,
    remove_temp_files=True,
    auto_disable_pretrain=True
):
    """Run the complete, resumable CompleteBin binning workflow.

    The workflow has three stages: (1) contig filtering, coverage/k-mer
    feature preparation and marker-gene calling; (2) contrastive-model
    training and embedding generation; and (3) two Leiden clustering rounds,
    SCG-aware dereplication, and K-means recovery of residual low-quality
    contigs. Cached products in ``temp_file_folder_path`` are reused, making
    it possible to run the three stages on different CPU/GPU nodes.

    Args:
        contig_file_path (str): Input contig FASTA. Contig identifiers are
            normalised internally to remove whitespace.
        sorted_bam_file_list (List[str]): Coordinate-sorted BAM files used to
            construct contig coverage profiles.
        temp_file_folder_path (str): Persistent work directory for cached
            features, training artefacts, marker-gene calls and clustering data.
        bin_output_folder_path (str): Destination directory for final MAG FASTAs
            and ``MetaInfo.tsv``.
        n_views (int): Number of augmented views used for contrastive training.
        feature_dim (int): Requested feature dimension before any automatic
            feature-dimension selection.
        auto_feature_dim (bool): Whether to select the feature dimension from
            the input data automatically.
        db_folder_path (str, optional): CompleteBin database directory. If None,
            the ``CompleteBin_DB`` environment variable is used.
        count_kmer (int): k-mer size used during feature construction.
        min_contig_length (int): Initial minimum contig length in bp. The
            effective threshold may be adjusted from input statistics.
        leiden_iter_mode (str): Leiden optimisation mode: ``"accurate"`` for
            convergence or ``"fast"`` for adaptive early stopping.
        drop_p (float): Training dropout probability.
        lr (float): Base learning rate.
        lr_multiple (float): Multiplier applied to the learning rate schedule.
        lr_warmup_epoch (int): Number of learning-rate warm-up epochs.
        weight_deay (float): L2 regularisation weight. The spelling is retained
            for API compatibility.
        batch_size (int): Training batch size.
        base_epoch (int): Baseline number of training epochs.
        log_every_n_steps (int): Logging interval during training.
        training_device (str): PyTorch device for training, e.g. ``"cpu"`` or
            ``"cuda:0"``.
        cpu_workers (int, optional): CPU workers for preprocessing, marker-gene
            calling and clustering. If None, it is auto-detected.
        gpu_dataloader_workers (int): Data-loader workers used during training.
        min_training_step (int): Minimum training steps per epoch.
        step_num (int, optional): Split-workflow stop point: ``1`` prepares
            data then returns; ``2`` trains using prepared data then returns;
            ``3`` runs binning/clustering using cached stages 1 and 2; None
            runs the complete workflow in one invocation.
        remove_temp_files (bool): Remove first/second clustering temporary
            directories after their corresponding outputs are finalised.
        auto_disable_pretrain (bool): Automatically skip pretrained weights when
            the input statistics indicate that pretraining is unsuitable.
    """
    seed = 2048
    cov_time_s = time.time()
    logger.info(f"--> CompleteBin version: *** {bin_v} ***. The random seed is {seed}.")
    mp.set_start_method("fork", force=True)
    seed_everything(seed)

    if cpu_workers is None:
        cpu_workers = psutil.cpu_count() // 3 + 1
    logger.info(f"--> Total CPUs: {psutil.cpu_count()}. Number of {cpu_workers} CPUs are applied.")

    if os.path.exists(temp_file_folder_path) is False:
        os.mkdir(temp_file_folder_path)

    #############################################
    # Remove the space in the name of contigs.
    logger.info("--> Start to filter the space in contig name. Make sure the first string of contig name is unique in fasta file.")
    output_fasta_path = os.path.join(temp_file_folder_path, "filtered_space_in_name.contigs.fasta")
    if os.path.exists(output_fasta_path) is False:
        filterSpaceInFastaFile(contig_file_path, output_fasta_path)
    contig_file_path = output_fasta_path

    ####################################
    # prepare the files in databases
    if db_folder_path is None:
        db_folder_path = os.environ["CompleteBin_DB"]
    markerset_path = os.path.join(db_folder_path, "data", "markersets.ms")
    pfma_file_path = os.path.join(db_folder_path, "data", "pfam_file.dat")
    num_classes = 15434  # 0 index pretrain model is the best
    layers = 4
    pretrain_model_weight_path = os.path.join(db_folder_path, "CheckPoint", "pretrain_weight_large_hidden_dim_768_layers_4.pth")
    split_parts_list = [1, 11]
    logger.info("--> Start to read contigs.")

    training_data_path = os.path.join(temp_file_folder_path, "training_data.npy")
    contigname2seq_path = os.path.join(temp_file_folder_path, "contigname2seq_str.pkl")
    contigname2bp_nparray_list_path = os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl")
    coverage_profile_path = None
    mean_var_path = os.path.join(temp_file_folder_path, "mean_var.pkl")

    contigname2seq_ori = readFasta(contig_file_path)
    min_contig_length = auto_min_contig_length(contigname2seq_ori, current_min_len=min_contig_length)
    N50 = calculateN50(contigname2seq_ori)
    if min_contig_length < 768:
        min_contig_length = 768
        logger.info(f"--> Set min contig length as 768 since the constraints of pretrain model.")
    large_data_size_thre = 115000
    temp = temp_decision(contigname2seq_ori, min_contig_length, N50, large_data_size_thre)
    low_gap = decision_lower_bound(
        contigname2seq_ori,
        min_contig_length,
        batch_size
    )
    min_contig_length -= low_gap

    # Step 1 is resumable.  Keep track of whether this invocation actually
    # performed work so a cache-hit invocation cannot overwrite a previous
    # real timing with 0 seconds.
    step1_preparation_ran = os.path.exists(training_data_path) is False
    cal_coverage_time = 0.0
    if step1_preparation_ran:
        ########################################################
        # STEP1: Get the coverage information of contigs.
        logger.info(f"--> Model Weight Path: {pretrain_model_weight_path}")
        logger.info(f"--> N50: {N50}, training temperature: {temp}, cluster mode: leiden + leiden.")
        logger.info(
            f"--> The min contigs length for training is: {min_contig_length}, for clustering is {min_contig_length + low_gap}.")
        if os.path.exists(contigname2seq_path) is False:
            prepare_sequences_coverage(
                contig_file_path,
                sorted_bam_file_list,
                temp_file_folder_path,
                min_contig_length,
                os.path.join(db_folder_path, "HMM", "40_marker.hmm"),
                os.path.join(db_folder_path, "HMM", "get_40marker.pl"),
                cpu_workers
            )

        cov_time_e = time.time()
        cal_coverage_time = cov_time_e - cov_time_s

        #########################################################
        # the following four files would be generated by function "prepare_sequences_coverage"
        mean_val, std_val = readPickle(mean_var_path)
        if auto_feature_dim:
            feature_dim = int(math.log(len(readPickle(contigname2seq_path))) * 8.33)

        # ── Gap-based coverage outlier 检测与截断 ──
        clip_result = clip_coverage_outliers(
            contigname2seq_path,
            contigname2bp_nparray_list_path,
            mean_var_path,
            gap_threshold_mean=1.15,
            gap_threshold_std=1.265,
            min_length=2500,
        )
        logger.info(f"--> Coverage outlier processing: "
                    f"removed={clip_result.get('n_removed', 0)}, "
                    f"clipped={clip_result.get('n_clipped', 0)}, "
                    f"std_protected={clip_result.get('n_std_only_protected', 0)}")

        # 重新读取更新后的 mean_var
        mean_val, std_val = readPickle(mean_var_path)

        # 用过滤后的 contig 重新生成 40-marker 种子
        filtered_fasta = os.path.join(temp_file_folder_path, "filtered_contigs.fasta")
        filtered_contigname2seq = readPickle(contigname2seq_path)
        writeFasta(filtered_contigname2seq, filtered_fasta)
        seed_folder = os.path.join(temp_file_folder_path, "seed_folder")
        if os.path.exists(seed_folder) is False:
            os.mkdir(seed_folder)
        gen_seed(
            filtered_fasta,
            cpu_workers,
            seed_folder,
            os.path.join(db_folder_path, "HMM", "40_marker.hmm"),
            os.path.join(db_folder_path, "HMM", "get_40marker.pl"),
            min_contig_length,
        )
        if os.path.exists(filtered_fasta):
            os.remove(filtered_fasta)
        # ── Outlier 处理结束 ──

    logger.info(
        f"--> Dropout probability: {drop_p}, n-views: {n_views}, base epoch is {base_epoch}, batch size is {batch_size}, feature_dim: {feature_dim}.")
    #########################################################
    # build training data
    # this function would generate 'training_data.npy' file in 'temp_file_folder_path'
    cov_time_s = time.time()
    process_data_ran = os.path.exists(training_data_path) is False and \
        os.path.exists(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl")) is False
    if process_data_ran:
        build_training_seq_data_numpy_save(
            contigname2seq_path,
            contigname2bp_nparray_list_path,
            temp_file_folder_path,
            count_kmer,
            min_contig_length,
            cpu_workers
        )
    logger.info(f"--> Step 1 is over.")
    cov_time_e = time.time()
    cal_data_time = cov_time_e - cov_time_s
    if step_num is None or step_num == 1:
        step1_times = {}
        if step1_preparation_ran:
            step1_times["CalculateCoverageTime(s)"] = cal_coverage_time
        if process_data_ran:
            step1_times["ProcessDataTime(s)"] = cal_data_time
        if step1_times:
            _write_time_tsv(temp_file_folder_path, step1_times)
    if step_num is not None and step_num == 1:
        return 0
    ##########################################################
    # STEP2:  Strat to training
    cov_time_s = time.time()
    if os.path.exists(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl")) is False:
        # model training
        contigname2seq = readPickle(contigname2seq_path)
        mean_val, std_val = readPickle(mean_var_path)
        # Compute k-mer sufficiency S from training contig lengths
        if auto_disable_pretrain:
            train_lengths = [len(seq) for seq in contigname2seq.values()]
            train_avg_len = sum(train_lengths) / max(len(train_lengths), 1)
            S = compute_S(train_avg_len, min_contig_length)
            logger.info(
                f"--> Training contigs: n={len(train_lengths)}, "
                f"avg_len={train_avg_len:.0f}bp, S={S:.3f}"
            )
        else:
            S = 0.0
        model_save_folder = os.path.join(temp_file_folder_path, "model_save")
        if os.path.exists(model_save_folder) is False:
            os.mkdir(model_save_folder)
        # epoch setting, base epoch
        if N50 >= 1536:
            base_epoch += 4
        sampler = DeeperBinSampler(len(contigname2seq) // batch_size * batch_size, batch_size, min_training_step=min_training_step, seed=seed)
        num_contigs = len(sampler.final_sample)
        if num_contigs >= large_data_size_thre:
            train_epoch = base_epoch
        else:
            train_epoch = large_data_size_thre * base_epoch // num_contigs
        if train_epoch > 200:
            train_epoch = 200

        # ---- Automatic pretrained model decision ----
        if auto_disable_pretrain:
            O = compute_O(train_epoch, num_contigs)
            use_pretrained, decision_reason = decide_pretrained(S, O)
            logger.info(f"--> Pretrain decision: {decision_reason}")
            # Safety valve: reduce epoch cap only under extreme overfitting
            train_epoch, valve_reason = apply_epoch_safety_valve(train_epoch, S, O)
            if valve_reason:
                logger.info(f"--> Epoch safety valve: {valve_reason}")
        else:
            use_pretrained = True
            logger.info("--> Auto pretrain disabled: always using pretrained model.")

        trainer_obj = SelfSupervisedMethodsTrainer(
            feature_dim,
            n_views,
            drop_p,
            training_device,
            temperature_simclr=temp,
            min_contig_len=min_contig_length,
            batch_size=batch_size,
            lr=lr,
            lr_multiple=lr_multiple,
            lr_warmup_epoch=lr_warmup_epoch,
            sampler=sampler,
            train_epoch=train_epoch,
            weight_decay=weight_deay,
            training_data_path=training_data_path,
            model_save_folder=model_save_folder,
            emb_output_folder=temp_file_folder_path,
            count_kmer=count_kmer,
            split_parts_list=split_parts_list,
            N50=N50,
            num_bam_files=len(sorted_bam_file_list),
            mean_std_val=(mean_val, std_val),
            pretrain_model_weight_path=pretrain_model_weight_path,
            log_every_n_steps=log_every_n_steps,
            dataloader_workers=gpu_dataloader_workers,
            num_classes=num_classes,
            layers=layers,
            seed=seed,
            use_pretrained=use_pretrained,
        )

        logger.info(f"--> Start to train model. The temperature is {temp}.")
        trainer_obj.train(load_epoch_set=None)
        logger.info(f"--> Start to inference contig embeddings with model.")
        min_epoch_set = trainer_obj.inference(min_epoch_set=None)
        copy(os.path.join(model_save_folder, f'checkpoint_{min_epoch_set}.pth'), temp_file_folder_path)
        if os.path.exists(model_save_folder):
            rmtree(model_save_folder, ignore_errors=True)

    logger.info(f"--> Step 2 is over.")
    cov_time_e = time.time()
    train_time = cov_time_e - cov_time_s
    if step_num is None or step_num == 2:
        _write_time_tsv(temp_file_folder_path, {"TrainingTime(s)": train_time})
    if step_num is not None and step_num == 2:
        return 0
    #############################################################
    # STEP3: Start Clustring
    cov_time_s = time.time()
    min_contig_length += low_gap
    contigname2seq = readPickle(contigname2seq_path)
    simclr_contigname2emb_norm_array = readPickle(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl"))
    simclr_emb_list = []
    length_list = []
    sub_contigname_list = []
    for contigname, seq in contigname2seq.items():
        length = len(seq)
        if length < min_contig_length:
            continue
        sub_contigname_list.append(contigname)
        simclr_emb_list.append(simclr_contigname2emb_norm_array[contigname])
        length_list.append(length)
    logger.info(f"--> There are {len(contigname2seq)} contigs for training and {len(simclr_emb_list)} for clustering.")

    initial_fasta_path = os.path.join(temp_file_folder_path, "split_contigs_initial_kmeans")
    if os.path.exists(initial_fasta_path) is False:
        os.mkdir(initial_fasta_path)

    seed_path = os.path.join(temp_file_folder_path, "seed_folder", "bacar_marker.2quarter.seed")
    seed_set = readSeedFile(seed_path)
    bin_number = int(len(seed_set) * 1.5) + 1
    if len(os.listdir(initial_fasta_path)) != bin_number:
        kmeans_split(
            logger,
            initial_fasta_path,
            contigname2seq,
            sub_contigname_list,
            np.stack(simclr_emb_list, axis=0),
            np.array(length_list),
            bin_number,
            seed_set,
            min_contig_length
        )
    call_genes_folder = os.path.join(temp_file_folder_path, "call_genes")
    if os.path.exists(os.path.join(temp_file_folder_path, "bac_gene_info.pkl")) is False or \
            os.path.exists(os.path.join(temp_file_folder_path, "arc_gene_info.pkl")) is False:
        bac_acc_set = set()
        arc_acc_set = set()
        marset = readMarkerSets(markerset_path)
        for cur_gene_set in marset["d__Bacteria"]:
            for cur_gene in cur_gene_set:
                bac_acc_set.add(cur_gene)
        for cur_gene_set in marset["d__Archaea"]:
            for cur_gene in cur_gene_set:
                arc_acc_set.add(cur_gene)
        callMarkerGenesByHMM(
            initial_fasta_path,
            call_genes_folder,
            cpu_workers,
            hmm_model_path=os.path.join(db_folder_path, "HMM", "bac_arc_hmm_model_extended.hmm"),
            bin_suffix="fasta",
            bac_acc_set=bac_acc_set,
            arc_acc_set=arc_acc_set,
            pfma_file_path=pfma_file_path,
            output_folder=temp_file_folder_path
        )
    bac_gene2contigNames, bac_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "bac_gene_info.pkl"))
    arc_gene2contigNames, arc_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "arc_gene_info.pkl"))
    if os.path.exists(call_genes_folder):
        rmtree(call_genes_folder, ignore_errors=True)

    coverage_profile_path = ensure_compact_coverage_path(contigname2bp_nparray_list_path)
    logger.info(f"--> Improved polish coverage profile ready: {coverage_profile_path}")

    # first clustering
    first_clustering_bins = os.path.join(temp_file_folder_path, "first_clustering_bins")
    if os.path.exists(os.path.join(first_clustering_bins, "MetaInfo.tsv")) is False:
        clustering_and_dereplication(
            temp_file_folder_path,
            "first_clustering_temp",
            contigname2seq,
            simclr_contigname2emb_norm_array,
            markerset_path,
            min_contig_length,
            cpu_workers,
            seed_path,
            bac_contigName2_gene2num,
            arc_contigName2_gene2num,
            100000,
            first_clustering_bins,
            leiden_iter_mode,
            clustering_stages=1,
            coverage_profile_path=coverage_profile_path,
        )
    if remove_temp_files and os.path.exists(os.path.join(temp_file_folder_path, "first_clustering_temp")):
        rmtree(os.path.join(temp_file_folder_path, "first_clustering_temp"))
    # selected contigs for next clustering
    sec_seleceted_contigs = selected_recluster_contigs(contigname2seq, first_clustering_bins, ["HighQuality", "MediumQuality"], comp_thre=50.)
    sec_contigname2seq = {}
    for name in sec_seleceted_contigs:
        sec_contigname2seq[name] = contigname2seq[name]
    # ## second clustering
    second_clustering_bins = os.path.join(temp_file_folder_path, "second_clustering_bins")
    if os.path.exists(os.path.join(second_clustering_bins, "MetaInfo.tsv")) is False:
        clustering_and_dereplication(
            temp_file_folder_path,
            "second_clustering_temp",
            sec_contigname2seq,
            simclr_contigname2emb_norm_array,
            markerset_path,
            min_contig_length,
            cpu_workers,
            seed_path,
            bac_contigName2_gene2num,
            arc_contigName2_gene2num,
            100000,
            second_clustering_bins,
            leiden_iter_mode,
            clustering_stages=2,
            coverage_profile_path=coverage_profile_path,
        )
    if remove_temp_files and os.path.exists(os.path.join(temp_file_folder_path, "second_clustering_temp")):
        rmtree(os.path.join(temp_file_folder_path, "second_clustering_temp"))
    # third clustering with K-Mearns
    thi_seleceted_contigs = selected_recluster_contigs(sec_contigname2seq, second_clustering_bins, ["HighQuality", "MediumQuality"], comp_thre=50.)
    thi_contigname2seq = {}
    for name in thi_seleceted_contigs:
        thi_contigname2seq[name] = contigname2seq[name]
    third_clustering_bins = os.path.join(temp_file_folder_path, "thi_clustering_bins")
    cluster_kmeans_for_low_v2(
        thi_contigname2seq,
        simclr_contigname2emb_norm_array,
        bac_gene2contigNames,
        bac_contigName2_gene2num,
        0,
        third_clustering_bins,
        first_metainfo_path=os.path.join(first_clustering_bins, "MetaInfo.tsv"),
        second_metainfo_path=os.path.join(second_clustering_bins, "MetaInfo.tsv"),
        min_k_ratio=0.1,
        percentile=10
    )
    # select final bins from first and second results.
    first_selected_list = readPickle(os.path.join(first_clustering_bins, "bin_copy_list.pkl"))
    second_selected_list = readPickle(os.path.join(second_clustering_bins, "bin_copy_list.pkl"))
    index = 0
    if os.path.exists(bin_output_folder_path) is False:
        os.mkdir(bin_output_folder_path)
    wh = open(os.path.join(bin_output_folder_path, "MetaInfo.tsv"), "w")
    for item_tuple in first_selected_list + second_selected_list:
        ori_path, comp, cont, state = item_tuple
        outName = f"CompleteBin_{index}.fasta"
        writeMetaInfo(wh, outName, comp, cont, state)
        copy(ori_path, os.path.join(bin_output_folder_path, outName))
        index += 1
    for filename in os.listdir(third_clustering_bins):
        comp = "0."
        cont = "0."
        state = "LowQuality"
        outName = f"CompleteBin_{index}.fasta"
        writeMetaInfo(wh, outName, comp, cont, state)
        copy(os.path.join(third_clustering_bins, filename), os.path.join(bin_output_folder_path, outName))
        index += 1
    wh.close()

    cov_time_e = time.time()
    cluster_time = cov_time_e - cov_time_s
    _write_time_tsv(temp_file_folder_path, {"ClusteringTime(s)": cluster_time})
    return 0
