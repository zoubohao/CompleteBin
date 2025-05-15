
import multiprocessing as mp
import os
from shutil import rmtree
from typing import List, Union

import numpy as np
import psutil

from Src.CallGenes.gene_utils import callMarkerGenesByCheckm
from Src.CallGenes.hmm_utils import processHits
from Src.Cluster.cluster import combine_two_cluster_steps
from Src.Cluster.split_utils import kmeans_split
from Src.DataProcess.data_utils import build_training_seq_data_numpy_save
from Src.Dereplication.galah_utils import process_galah
from Src.IO import readFasta, readPickle
from Src.logger import get_logger
from Src.Seqs.seq_info import calculateN50, prepare_sequences_coverage
from Src.Seqs.seq_utils import getGeneWithLargestCount, callGenesForKmeans
from Src.Trainer.ssmt_v2 import SelfSupervisedMethodsTrainer
from Src.version import bin_v

logger = get_logger()


def filterSpaceInFastaFile(input_fasta, output_fasta):
    with open(input_fasta, "r") as rh, open(output_fasta, "w") as wh:
        for line in rh:
            oneline = line.strip("\n")
            if ">" in oneline and " " in oneline:
                oneline = oneline.split()[0]
            wh.write(oneline + "\n")


def auto_decision(
    contigname2seq_ori: dict,
    short_long_ratio: float,
):
    long_contig_num = 0
    short_contig_num = 0
    collect_list = []
    for name, seq in contigname2seq_ori.items():
        n = len(seq)
        collect_list.append((n, name))
        if 1000 <= n:
            long_contig_num += 1
    sorted_collect_list = list(sorted(collect_list, key=lambda x: x[0], reverse=True))
    estimated_num_short_conitgs = int(long_contig_num * short_long_ratio)
    for n, name in sorted_collect_list:
        if n < 1000:
            if short_contig_num < estimated_num_short_conitgs:
                short_contig_num += 1
            else:
                min_contig_length = n - 1
                break
    ratio = short_contig_num / long_contig_num + 0.
    logger.info(f"--> # of short: {short_contig_num}, long {long_contig_num}, ratio: {ratio}. min length is {min_contig_length}.")
    min_contig_length = min_contig_length // 10 * 10
    if min_contig_length < 768: 
        logger.info(f"--> Set min contig length as 768 since the constrains of pretrain model.")
        min_contig_length = 768
    return min_contig_length


def temp_decision(
    contigname2seq_ori: dict,
    min_contig_length: int,
    N50: int,
    large_data_size_thre: int
):
    ## Weak Augmentations: Views are more similar; use higher temperatures to smooth similarity distribution. --> N50 samaller, temp higher
    ## Data Noise: Higher noise levels benefit from a higher temperature to smoothen similarity scores. --> N50 samaller, temp higher
    ## Dataset Size: For large datasets, lower temperatures often work better as they increase the discriminative power of the embeddings.
    ## --> larger size, lower temp
    count_contigs = 0
    for _, seq in contigname2seq_ori.items():
        if len(seq) >= min_contig_length:
            count_contigs += 1
    temp = None
    if N50 >= 10000:
        if count_contigs >= large_data_size_thre:
            temp = 0.055
        else:
            temp = 0.065
    elif N50 <= 5000:
        if count_contigs >= large_data_size_thre:
            temp = 0.135
        else:
            temp = 0.145
    else:
        temp = -0.000016 * N50 + 0.23
        temp = float("%.3f" % temp)
        if count_contigs >= large_data_size_thre:
            temp -= 0.01
    return temp


def binning_with_all_steps(
    contig_file_path: str,
    sorted_bam_file_list: List[str],
    temp_file_folder_path: str,
    bin_output_folder_path: str,
    db_folder_path: str=None,
    n_views: int=6,
    count_kmer: int=4,
    min_contig_length=900,
    min_contig_length_auto_decision=False,
    short_long_ratio=0.333,
    # model training config
    feature_dim=100,
    drop_p=0.12,
    lr=1e-5,
    lr_multiple=10,
    lr_warmup_epoch=1,
    weight_deay=1e-3,
    batch_size=1024,
    base_epoch=35,
    large_model=False,
    log_every_n_steps=10,
    training_device="cuda:0",
    num_workers: int=None,
    von_flspp_mix = "flspp",
    ensemble_with_SCGs=False,
    multi_seq_contrast=False,
    step_num = None,
    remove_temp_files = False,
    filter_huge_gap = False
):
    """
    The whole binning process of DeeperBin.

    Args:
        contig_file_path (str): The contigs fasta file path.
        sorted_bam_file_list (List[str]): The list of sorted bam files paths.
        temp_file_folder_path (str): The folder path to store temp files.
        bin_output_folder_path (str): The folder path to store final MAGs.
        db_folder_path (str, optional): The path of database folder. Defaults to None. You can ignore it if you set the 'DeeperBin_DB' environmental variable.
        n_views (int, optional): Number of views to generate for each contig during training. Defaults to 6.
        count_kmer (int, optional): The k setting of k-mer. Defaults to 4.
        min_contig_length (int, optional): The minimum length of contigs for binning. Defaults to 900.
        min_contig_length_auto_decision  (bool, optional): Auto determining the length of min contig if it is True. 
        short_long_ratio (float, optional): The min contig length would be shorter if this parameter larger under the auto determing is True.
        feature_dim (int, optional): The feature dim of final embeddings. Defaults to 100.
        drop_p (float, optional): The dropout probability setting. Defaults to 0.12.
        lr (float, optional): The learning rate setting. Defaults to 1e-5.
        lr_multiple (int, optional): The multiple value for learning rate. Defaults to 10.
        lr_warmup_epoch (int, optional): Number of epoches to warm up the learning rate. Defaults to 1.
        weight_deay (float, optional): L2 regularization. Defaults to 1e-3.
        batch_size (int, optional): The batch size. Defaults to 1024.
        base_epoch (int, optional): Number of basic training epoches. Defaults to 35.
        large_model (bool, optional): If use large pretrained model. Defaults to False.
        log_every_n_steps (int, optional): Print log after n training step. Defaults to 10.
        training_device (str, optional): The device for training model. You can set 'cpu' to use CPU. Defaults to "cuda:0".
        num_workers (int, optional): Number of cpus for clustering contigs. Defaults to None. We would set 1 / 3 of total cpus if it is None.
        von_flspp_mix (str, optional): The clustering algorithm for the second stage clustering. You can set  'flspp (FLS++ algorithm)', 
        'von (Estimator for Mixture of von Mises Fisher clustering on the unit sphere)' or 
        'mix (Apply von when number of contigs bigger than 150 and smaller than 1850, otherwise apply flspp)'. 
        'flspp' has the fastest speed. We recommand to use flspp for large datasets and mix for small datasets. Defaults to "flspp". 
        ensemble_with_SCGs (bool, optional): Apply the called SCGs to do quality evaluation and used them in ensembling the results if it is True. 
        multi_seq_contrast (bool, optional): Add sequence embedding for contrastive learning if it is True.
        step_num (int, optional): The whole binning procedure can be divided into 3 steps. 
        The first step (step 1) is to process the training data. Focusing on using CPU.
        The second step (step 2) is training procedure. Focusing on using GPU.
        The third step (step 3) is clustering. Focusing on using CPU.
        This function would combine these 3 steps if this parameter is None.
        remove_temp_files (bool, optional): Remove the temp files if this is true.
        filter_huge_gap (bool, optional): Filter the MAGs if the checkm2's completeness has a huge gap (> 40%) with the SCGs' completeness if it is true. 
        Try to fix the bug of checkm2.
    """
    logger.info(f"CompleteBin version: *** {bin_v} ***")
    mp.set_start_method("fork", force=True) 
    
    if num_workers is None:
        num_workers = psutil.cpu_count() // 3 + 1
    logger.info(f"--> Total CPUs: {psutil.cpu_count()}. Number of {num_workers} CPUs are applied.")
    
    if os.path.exists(temp_file_folder_path) is False:
        os.mkdir(temp_file_folder_path)
    
    #############################################
    ##### Remove the space in the name of contigs.
    logger.info("--> Start to filter the space in contig name. Make sure the first string of contig name is unique in fasta file.")
    output_fasta_path = os.path.join(temp_file_folder_path, "filtered_space_in_name.contigs.fasta")
    if os.path.exists(output_fasta_path) is False:
        filterSpaceInFastaFile(contig_file_path, output_fasta_path)
    contig_file_path = output_fasta_path
    
    ####################################
    ##### prepare the files in databases
    if db_folder_path is None:
        db_folder_path = os.environ["CompleteBin_DB"]
    phy2accs_path = os.path.join(db_folder_path, "HMM", "phy2accs_new.pkl")
    markerset_path = os.path.join(db_folder_path, "markerSets", "markersets.ms")
    if large_model:
        pretrain_model_weight_path = os.path.join(db_folder_path, "CheckPoint", "pretrain_weight_hidden_dim_2048_layers_4.pth")
    else:
        pretrain_model_weight_path = os.path.join(db_folder_path, "CheckPoint", "pretrain_weight_hidden_dim_512_layers_3.pth")
    
    logger.info("--> Start to read contigs.")
    contigname2seq_ori = readFasta(contig_file_path)
    N50 = calculateN50(contigname2seq_ori)
    
    if min_contig_length_auto_decision:
        min_contig_length = auto_decision(contigname2seq_ori, short_long_ratio)
    if min_contig_length < 768:
        min_contig_length = 768
        logger.info(f"--> Set min contig length as 768 since the constrains of pretrain model.")
    large_data_size_thre=153600
    temp = temp_decision(contigname2seq_ori, min_contig_length, N50, large_data_size_thre)
    
    split_parts_list = [1, 16]
    logger.info(f"--> N50 is {N50}, seq split list: {split_parts_list}, temperature is {temp}, cluster mode: leiden + {von_flspp_mix}.")
    logger.info(f"--> Dropout Probability: {drop_p}, n-views: {n_views}, min contigs length: {min_contig_length}.")
    
    ########################################################
    # STEP1: Get the coverage information of contigs.
    contigname2seq_path = os.path.join(temp_file_folder_path, "contigname2seq_str.pkl")
    if os.path.exists(contigname2seq_path) is False:
        prepare_sequences_coverage(
            contigname2seq_ori,
            sorted_bam_file_list,
            temp_file_folder_path,
            min_contig_length,
            os.path.join(db_folder_path, "HMM", "40_marker.hmm"),
            num_workers,
            remove_temp_files
        )
    
    #########################################################
    ## the following four files would be generated by function "prepare_sequences_coverage"
    max_cov_mean = readPickle(os.path.join(temp_file_folder_path, "mean_var.pkl"))
    contigname2seq_path = os.path.join(temp_file_folder_path, "contigname2seq_str.pkl")
    contigname2bp_nparray_list_path = os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl")
    
    #########################################################
    ## build training data
    ## this function would generate 'training_data.npy' file in 'temp_file_folder_path'
    training_data_path = os.path.join(temp_file_folder_path, "training_data.npy")
    if os.path.exists(training_data_path) is False and \
        os.path.exists(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl")) is False:
        build_training_seq_data_numpy_save(
            contigname2seq_path,
            contigname2bp_nparray_list_path,
            temp_file_folder_path,
            count_kmer,
            split_parts_list,
            num_workers
        )  
    
    logger.info(f"--> Step 1 is over.")
    if step_num is not None and step_num == 1:
        return 0
    
    ##########################################################
    ## STEP2:  Strat to training
    if os.path.exists(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl")) is False:
        # model training
        contigname2seq = readPickle(contigname2seq_path)
        model_save_folder = os.path.join(temp_file_folder_path, "model_save")
        if os.path.exists(model_save_folder) is False:
            os.mkdir(model_save_folder)
        ## epoch setting, base epoch
        if N50 >= 1536: 
            base_epoch += 4
        num_contigs = len(contigname2seq)
        if num_contigs >= large_data_size_thre: 
            train_epoch = base_epoch
        else: 
            train_epoch = large_data_size_thre * base_epoch // num_contigs 
        if train_epoch > 200: 
            train_epoch = 200
        
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
            base_epoch=base_epoch,
            train_epoch=train_epoch,
            weight_decay=weight_deay,
            training_data_path=training_data_path,
            model_save_folder=model_save_folder,
            emb_output_folder=temp_file_folder_path,
            count_kmer=count_kmer,
            split_parts_list = split_parts_list,
            N50=N50,
            large_model=large_model,
            num_bam_files=len(sorted_bam_file_list),
            max_cov_mean = max_cov_mean,
            pretrain_model_weight_path=pretrain_model_weight_path,
            log_every_n_steps=log_every_n_steps,
            multi_contrast=multi_seq_contrast
        )
        logger.info(f"--> Start to train model. The tempeature is {temp}.")
        trainer_obj.train(load_epoch_set=None)
        logger.info(f"--> Start to inference contig embeddings with model.")
        trainer_obj.inference(min_epoch_set=None)
        if remove_temp_files:
            rmtree(model_save_folder, ignore_errors=True)
    
    logger.info(f"--> Step 2 is over.")
    if step_num is not None and step_num == 2:
        return 0
    
    #############################################################
    ## STEP3: Start Clustring
    contigname2seq = readPickle(contigname2seq_path)
    simclr_contigname2emb_norm_array = readPickle(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl"))
    simclr_emb_list = []
    length_list = []
    sub_contigname_list = []
    for contigname, seq in contigname2seq.items():
        length = len(seq)
        if length < min_contig_length:
            raise ValueError(f"The length of {contigname} is {length}, which is smaller than {min_contig_length}")
        sub_contigname_list.append(contigname)
        simclr_emb_list.append(simclr_contigname2emb_norm_array[contigname])
        length_list.append(length)
    
    initial_fasta_path = os.path.join(temp_file_folder_path, "split_contigs_initial_kmeans")
    if os.path.exists(initial_fasta_path) is False:
        os.mkdir(initial_fasta_path)
    
    contigname2hits = readPickle(os.path.join(temp_file_folder_path, "contigname2hmmhits_list.pkl"))
    mar40_gene2contigNames, _ = processHits(contigname2hits)
    _, _, contignames_40mar = getGeneWithLargestCount(mar40_gene2contigNames, contigname2seq, None)
    bin_number = int(len(contignames_40mar) * 1.6) + 1
    if len(os.listdir(initial_fasta_path)) != bin_number:
        kmeans_split(
                logger,
                initial_fasta_path,
                contigname2seq,
                sub_contigname_list,
                np.stack(simclr_emb_list, axis=0),
                np.array(length_list),
                bin_number,
                contignames_40mar,
                min_contig_length
        )
        callGenesForKmeans(
            temp_file_folder_path, 
            initial_fasta_path, 
            num_workers, 
            os.path.join(db_folder_path, "HMM", "40_marker.hmm"))
    refined_fasta_path = os.path.join(temp_file_folder_path, "split_contigs_refined_kmeans")
    if os.path.exists(refined_fasta_path) is False:
        os.mkdir(refined_fasta_path)
    contigname2hits = readPickle(os.path.join(temp_file_folder_path, "contigname2hmmhits_list_initial_kmeans.pkl"))
    mar40_gene2contigNames, _ = processHits(contigname2hits)
    _, _, contignames_40mar = getGeneWithLargestCount(mar40_gene2contigNames, contigname2seq, None)
    bin_number = int(len(contignames_40mar) * 1.6) + 1
    if len(os.listdir(refined_fasta_path)) != bin_number:
        kmeans_split(
                logger,
                refined_fasta_path,
                contigname2seq,
                sub_contigname_list,
                np.stack(simclr_emb_list, axis=0),
                np.array(length_list),
                bin_number,
                contignames_40mar,
                min_contig_length
        )
    bac_ms_path = os.path.join(db_folder_path, "checkm", "bacteria.ms")
    arc_ms_path = os.path.join(db_folder_path, "checkm", "archaea.ms")
    call_genes_folder = os.path.join(temp_file_folder_path, "call_genes")
    if os.path.exists(os.path.join(temp_file_folder_path, "bac_gene_info.pkl")) is False or \
        os.path.exists(os.path.join(temp_file_folder_path, "arc_gene_info.pkl")) is False:
        callMarkerGenesByCheckm(temp_file_folder_path,
                                bac_ms_path,
                                arc_ms_path,
                                refined_fasta_path,
                                call_genes_folder,
                                os.path.join(db_folder_path, "checkm", "checkm_db"),
                                num_workers)
    bac_gene2contigNames, bac_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "bac_gene_info.pkl"))
    arc_gene2contigNames, arc_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "arc_gene_info.pkl"))
    if os.path.exists(bin_output_folder_path) is False:
        os.mkdir(bin_output_folder_path)
    clustering_all_folder = os.path.join(temp_file_folder_path, "clustering_res")
    if os.path.exists(os.path.join(clustering_all_folder, f"ensemble_methods_list_{von_flspp_mix}.pkl")) is False:
        phy2accs = readPickle(phy2accs_path)
        ensemble_list = combine_two_cluster_steps(
            contigname2seq,
            simclr_contigname2emb_norm_array,
            markerset_path,
            phy2accs,
            min_contig_length,
            cpu_num=num_workers,
            clustering_all_folder=clustering_all_folder,
            bac_gene2contigNames = bac_gene2contigNames,
            arc_gene2contigNames = arc_gene2contigNames,
            bac_contigName2_gene2num=bac_contigName2_gene2num,
            arc_contigName2_gene2num=arc_contigName2_gene2num,
            gmm_flspp=von_flspp_mix
        )
    # ensemble the grouped results by galah.
    ensemble_list = readPickle(os.path.join(clustering_all_folder, f"ensemble_methods_list_{von_flspp_mix}.pkl"))
    temp_flspp_bin_output = os.path.join(clustering_all_folder, f"temp_binning_results_{von_flspp_mix}")
    scg_quality_report_path = os.path.join(clustering_all_folder, f"quality_record_{von_flspp_mix}.tsv")
    process_galah(
        temp_file_folder_path,
        temp_flspp_bin_output,
        ensemble_list,
        refined_fasta_path,
        db_folder_path,
        bin_output_folder_path,
        ensemble_with_SCGs,
        scg_quality_report_path,
        filter_huge_gap,
        von_flspp_mix,
        cpus=num_workers,
    )
    if remove_temp_files:
        rmtree(temp_file_folder_path, ignore_errors=True)
    return 0
