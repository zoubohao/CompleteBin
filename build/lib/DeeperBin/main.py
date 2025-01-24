
import argparse
import os
import sys

from DeeperBin.Binning import binning_with_all_steps

deeprebin_v = "v1.0.1"

def main():
    print(f"DeeperBin version: *** {deeprebin_v} ***")
    
    myparser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]), description="DeeperBin Is a Binner with Dynamic Contrastive Learning with Pretrained Deep Language Model."
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
            " You can set one bam file for single-sample binning and multiple bam files for multi-sample binning.")
    myparser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="The folder to store final MAGs.")
    myparser.add_argument(
        "-temp",
        "--temp_file_path",
        required=True,
        help="The folder to store temporay files.")
    ## optional settings
    myparser.add_argument(
        "-db",
        "--db_files_path",
        type=str,
        default=None,
        help="The folder contains temporay files." + \
            " You can ignore it if you set the 'DeeperBin_DB' environmental variable.")
    myparser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="The device for training. Default is cuda:0. " + \
            "We highly recommand to use GPU but not CPU. " + \
            "You can adjust 'batch_size' parameter to fit your GPU's memory. We need 24GB GPU memory to run the default settings. " + \
            "You can use CPU if you set this parameter with 'cpu'.")
    myparser.add_argument(
        "--n_views",
        type=int,
        default=6,
        help="Number of views to generate for each contig during training. Defaults to 6.")
    myparser.add_argument(
        "--min_contig_length",
        type=int,
        default=750,
        help="The minimum length of contigs for binning. Defaults to 750.")
    myparser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="The batch size. Defaults to 1024.")
    myparser.add_argument(
        "--epoch_base",
        type=int,
        default=35,
        help="Number of basic training epoches. Defaults to 35.")
    myparser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of cpus for clustering contigs. Defaults to None. We would set 1/3 of total cpus if it is None.")
    
    args = myparser.parse_args()
    binning_with_all_steps(
        contig_file_path=args.contig_path,
        sorted_bam_file_list=args.sorted_bams_paths,
        temp_file_folder_path=args.temp_file_path,
        bin_output_folder_path=args.output_path,
        db_folder_path=args.db_files_path,
        n_views=args.n_views,
        min_contig_length=args.min_contig_length,
        batch_size=args.batch_size,
        epoch_base=args.epoch_base,
        num_workers=args.num_workers,
        training_device=args.device,
    )


