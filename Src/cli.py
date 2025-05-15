
import argparse
import os
import sys

from Src.Binning_steps import binning_with_all_steps

def main():
    
    myparser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]), description=\
            "CompleteBin is a Binner to cluster the contigs with Sequence Patch Embedding, Pretrained Deep Language Model, and Dynamic Contrastive Learning."
    )

    # Add parameters, required settings
    myparser.add_argument(
        "-c",
        "--contig_path",
        type=str,
        required=True,
        help="Contig fasta file path.")
    myparser.add_argument(
        "-b",
        "--sorted_bams_paths",
        required=True,
        type=str,
        nargs="+",
        help="The sorted bam files path. " + \
            " You can set a BAM file for single-sample binning and multiple BAM files for multi-sample binning.")
    myparser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="The folder to store final MAGs.")
    myparser.add_argument(
        "-temp",
        "--temp_file_path",
        required=True,
        help="The folder to store temporary files during binning processing.")
    ## optional settings
    myparser.add_argument(
        "-db",
        "--db_files_path",
        type=str,
        default=None,
        help="The folder contains database files." + \
            " You can ignore it if you set the 'CompleteBin_DB' environmental variable.")
    myparser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="The device uses for training. The default is CPU. " + \
            "We highly recommend using GPU, but not CPU. " + \
            "We need 24GB of GPU memory to run the default settings. You can adjust the 'batch_size' parameter to fit your GPU's memory. " + \
            "You can use CPU if you set this parameter with 'cpu'.")
    myparser.add_argument(
        "--n_views",
        type=int,
        default=6,
        help="Number of views to generate for each contig during training. Defaults to 6.")
    myparser.add_argument(
        "--min_contig_length",
        type=int,
        default=900,
        help="The minimum length of contigs for binning. Defaults to 900.")
    myparser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="The batch size. Defaults to 1024.")
    myparser.add_argument(
        "--base_epoch",
        type=int,
        default=35,
        help="Number of basic training epoches. Defaults to 35.")
    myparser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPUs for clustering contigs. Defaults to None. We would set 1 / 3 of the total CPUs if it is None.")
    myparser.add_argument(
        "--auto_min_length",
        type=bool,
        default=False,
        help="Auto-determining the min length for this sample.")
    myparser.add_argument(
        "--step_num",
        type=int,
        default=None,
        help="The whole binning procedure can be divided into 3 steps. " + \
            "The first step (step 1) is to process the training data. Focusing on using CPU." + \
            "The second step (step 2) is training procedure. Focusing on using GPU." + \
            "The third step (step 3) is clustering. Focusing on using CPU." + \
            "This function would combine these 3 steps if this parameter is None. Defaults to None.")
    myparser.add_argument(
        "--sec_clu_algo",
        type=str,
        default="flspp",
        help="The clustering algorithm for the second stage clustering. You can set  'flspp' (FLS++ algorithm)," + \
        " 'von' (Estimator for Mixture of von Mises Fisher clustering on the unit sphere) or " + \
        " 'mix' (Apply von when number of contigs bigger than 150 and smaller than 1850, otherwise apply flspp). " + \
        " flspp has the fastest speed. We recommand to use flspp for large datasets and mix for small datasets. Defaults to flspp. ")
    myparser.add_argument(
        "--ensemble_with_SCGs",
        type=bool,
        default=False,
        help="Apply the called SCGs to do quality evaluation and used them in ensembling the results if it is True. Defaults to False.")
    
    
    args = myparser.parse_args()
    binning_with_all_steps(
        contig_file_path=args.contig_path,
        sorted_bam_file_list=args.sorted_bams_paths,
        temp_file_folder_path=args.temp_file_path,
        bin_output_folder_path=args.output_path,
        db_folder_path=args.db_files_path,
        n_views=args.n_views,
        min_contig_length=args.min_contig_length,
        min_contig_length_auto_decision=args.auto_min_length,
        batch_size=args.batch_size,
        base_epoch=args.base_epoch,
        num_workers=args.num_workers,
        training_device=args.device,
        step_num=args.step_num,
        von_flspp_mix=args.sec_clu_algo,
        ensemble_with_SCGs=args.ensemble_with_SCGs,
    )


