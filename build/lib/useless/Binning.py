
import multiprocessing as mp
import os
from typing import List, Union

import numpy as np
import psutil

from Src.CallGenes.gene_utils import callMarkerGenesByCheckm
from Src.CallGenes.hmm_utils import processHits
from Src.Cluster.cluster import combine_two_cluster_steps
from Src.Cluster.split_utils import kmeans_split
from Src.DataProcess.data_utils import build_training_seq_data
from Src.Dereplication.galah_utils import process_galah
from Src.IO import readFasta, readPickle
from Src.logger import get_logger
from Src.Seqs.seq_info import calculateN50, prepare_sequences_coverage
from Src.Seqs.seq_utils import getGeneWithLargestCount
from useless.ssmt import SelfSupervisedMethodsTrainer

logger = get_logger()



def filterSpaceInFastaFile(input_fasta, output_fasta):
    with open(input_fasta, "r") as rh, open(output_fasta, "w") as wh:
        for line in rh:
            oneline = line.strip("\n")
            if ">" in oneline and " " in oneline:
                oneline = oneline.split()[0]
            wh.write(oneline + "\n")


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
    ensemble_with_SCGs=False,
    multi_seq_contrast=False
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
        lr_multiple (int, optional): The multiple value for learning rate. Defaults to 8.
        lr_warmup_epoch (int, optional): Number of epoches to warm up the learning rate. Defaults to 1.
        weight_deay (float, optional): L2 regularization. Defaults to 1e-3.
        batch_size (int, optional): The batch size. Defaults to 1024.
        base_epoch (int, optional): Number of basic training epoches. Defaults to 35.
        large_model (bool, optional): If use large pretrained model. Defaults to False.
        log_every_n_steps (int, optional): Print log after n training step. Defaults to 10.
        training_device (str, optional): The device for training model. You can set 'cpu' to use CPU. Defaults to "cuda:0".
        num_workers (int, optional): Number of cpus for clustering contigs. Defaults to None. We would set 1 / 3 of total cpus if it is None.
        ensemble_with_SCGs (bool, optional): Only uses the called SCGs to do quality evaluation and used in ensembling the results if it is True. 
                                                Else, we would apply CheckM2 to ensemble the results.
        multi_seq_contrast (bool, optional): Add sequence embedding for contrastive learning if it is True.
    """
    
    mp.set_start_method("fork", force=True) 
    if num_workers is None:
        num_workers = psutil.cpu_count() // 3 + 1
    logger.info(f"--> Total CPUs: {psutil.cpu_count()}. Number of {num_workers} CPUs are applied.")
    if os.path.exists(temp_file_folder_path) is False:
        os.mkdir(temp_file_folder_path)
    # Remove the space in the name of contigs.
    signal = False
    with open(contig_file_path, "r") as rh:
        for line in rh:
            if " " in line:
                signal = True
            break
    if signal:
        logger.info("--> ======================================================================== <--")
        logger.info("--> !!! WARNING !!! <-- Find space in the contig name. Make sure the first string of contig name is unique in fasta file.")
        output_fasta_path = os.path.join(temp_file_folder_path, "filtered_space_in_name.contigs.fasta")
        if os.path.exists(output_fasta_path) is False:
            filterSpaceInFastaFile(contig_file_path, output_fasta_path)
        contig_file_path = output_fasta_path
    if db_folder_path is None:
        db_folder_path = os.environ["DeeperBin_DB"]
    phy2accs_path = os.path.join(db_folder_path, "HMM", "phy2accs_new.pkl")
    markerset_path = os.path.join(db_folder_path, "markerSets", "markersets.ms")
    if large_model:
        pretrain_model_weight_path = os.path.join(db_folder_path, "CheckPoint", "pretrain_weight_hidden_dim_2048_layers_4.pth")
    else:
        pretrain_model_weight_path = os.path.join(db_folder_path, "CheckPoint", "pretrain_weight_hidden_dim_512_layers_3.pth")
    logger.info("--> Start to read contigs.")
    contigname2seq_ori = readFasta(contig_file_path)
    
    if min_contig_length_auto_decision:
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
        logger.info(f"--> # of short: {short_contig_num}, long {long_contig_num}, ratio: {ratio}. min length is {min_contig_length}")
        min_contig_length = min_contig_length // 10 * 10
        if min_contig_length < 770: 
            logger.info(f"--> Set min contig length as 770 since the constrains of pretrain model.")
            min_contig_length = 770
    
    contig_len = []
    count_contigs = 0
    for _, seq in contigname2seq_ori.items():
        contig_len.append(len(seq))
        if len(seq) >= min_contig_length:
            count_contigs += 1
    N50 = calculateN50(contig_len)
    logger.info(f"--> Dropout Probability: {drop_p}, n-views: {n_views}, min contigs length: {min_contig_length}")
    
    
    ## Weak Augmentations: Views are more similar; use higher temperatures to smooth similarity distribution. --> N50 samaller, temp higher
    ## Data Noise: Higher noise levels benefit from a higher temperature to smoothen similarity scores. --> N50 samaller, temp higher
    ## Dataset Size: For large datasets, lower temperatures often work better as they increase the discriminative power of the embeddings.
    ## --> larger size, lower temp
    large_data_size_thre=153600
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
    
    split_parts_list = [1, 16]
    logger.info(f"--> Original N50 is {N50}, seq split list: {split_parts_list}, temperature is {temp}, cluster mode: leiden + flspp.")
    # Get the coverage information of contigs.
    prepare_sequences_coverage(
        contigname2seq_ori,
        sorted_bam_file_list,
        temp_file_folder_path,
        min_contig_length,
        os.path.join(db_folder_path, "HMM", "40_marker.hmm"),
        num_workers
    )
    phy2accs = readPickle(phy2accs_path)
    split_input_folder = os.path.join(temp_file_folder_path, "split_contigs")
    contigname2seq_path = os.path.join(temp_file_folder_path, "contigname2seq_str.pkl")
    contigname2bp_nparray_list_path = os.path.join(temp_file_folder_path, "contigname2bpcover_nparray_list.pkl")
    contigname2hits = readPickle(os.path.join(temp_file_folder_path, "contigname2hmmhits_list.pkl"))
    mar40_gene2contigNames, _ = processHits(contigname2hits)
    # build training data
    model_training_data_path = os.path.join(temp_file_folder_path, "training_data")
    contigname2seq = readPickle(contigname2seq_path)
    if os.path.exists(os.path.join(temp_file_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl")) is False:
        build_training_seq_data(
            contigname2seq_path,
            contigname2bp_nparray_list_path,
            model_training_data_path,
            count_kmer,
            split_parts_list,
            num_workers
        )
        # model training
        model_save_folder = os.path.join(temp_file_folder_path, "model_save")
        if os.path.exists(model_save_folder) is False:
            os.mkdir(model_save_folder)
        max_cov_mean = readPickle(os.path.join(temp_file_folder_path, "mean_var.pkl"))
        ## epoch setting, base epoch
        if N50 >= 1536:
            base_epoch += 4
        num_contigs = len(contigname2seq)
        if num_contigs >= large_data_size_thre: train_epoch = base_epoch
        else: train_epoch = large_data_size_thre * base_epoch // num_contigs 
        if train_epoch > 150: train_epoch = 150
        # if the training steps less or equal to 20.
        if num_contigs // batch_size <= 20: lr = lr * 1.2
        ## At least 10 steps per epoch
        if num_contigs // batch_size <= 10: 
            batch_size = num_contigs // 10
            logger.info(f"The batch size changes to {batch_size} since the training steps for one epoch smaller than 10.")
        
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
            train_epoch=train_epoch,
            weight_decay=weight_deay,
            training_data_path=model_training_data_path,
            model_save_folder=model_save_folder,
            emb_output_folder=temp_file_folder_path,
            count_kmer=count_kmer,
            contigname2seq=contigname2seq,
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
    ## clustring
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
    initial_fasta_path = os.path.join(temp_file_folder_path, "split_contigs")
    if os.path.exists(initial_fasta_path) is False:
        os.mkdir(initial_fasta_path)
    _, _, contignames_40mar = getGeneWithLargestCount(mar40_gene2contigNames, contigname2seq)
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
    logger.info(f"--> Start to Call SCGs.")
    bac_ms_path = os.path.join(db_folder_path, "checkm", "bacteria.ms")
    arc_ms_path = os.path.join(db_folder_path, "checkm", "archaea.ms")
    call_genes_folder = os.path.join(temp_file_folder_path, "call_genes")
    if os.path.exists(os.path.join(temp_file_folder_path, "bac_gene_info.pkl")) is False or \
        os.path.exists(os.path.join(temp_file_folder_path, "arc_gene_info.pkl")) is False:
        callMarkerGenesByCheckm(temp_file_folder_path,
                                bac_ms_path,
                                arc_ms_path,
                                initial_fasta_path,
                                call_genes_folder,
                                os.path.join(db_folder_path, "checkm", "checkm_db"),
                                num_workers)
    bac_gene2contigNames, bac_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "bac_gene_info.pkl"))
    arc_gene2contigNames, arc_contigName2_gene2num = readPickle(os.path.join(temp_file_folder_path, "arc_gene_info.pkl"))
    if os.path.exists(bin_output_folder_path) is False:
        os.mkdir(bin_output_folder_path)
    clustering_all_folder = os.path.join(temp_file_folder_path, "clustering_res")
    gmm_flspp = "flspp"
    if os.path.exists(os.path.join(clustering_all_folder, "ensemble_methods_list.pkl")) is False:
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
            gmm_flspp=gmm_flspp
        )
    # ensemble the grouped results by galah.
    ensemble_list = readPickle(os.path.join(clustering_all_folder, "ensemble_methods_list.pkl"))
    temp_flspp_bin_output = os.path.join(clustering_all_folder, f"temp_binning_results_{gmm_flspp}")
    quality_report_path = os.path.join(clustering_all_folder, f"quality_record_{gmm_flspp}.tsv")
    process_galah(
        temp_file_folder_path,
        temp_flspp_bin_output,
        ensemble_list,
        split_input_folder,
        db_folder_path,
        bin_output_folder_path,
        ensemble_with_SCGs,
        quality_report_path,
        cpus=num_workers,
    )

