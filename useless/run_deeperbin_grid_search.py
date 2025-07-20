
import argparse
import os
import sys

import numpy as np

from CompleteBin.Binning_steps import binning_with_all_steps

if __name__ == "__main__":
    # home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI2-others-datasets"
    # out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets"
    myparser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]), description="DeeperBin Is a Binner with Dynamic Contrastive Learning with Pretrained Deep Language Model."
    )

    # Add parameters, required settings
    
    myparser.add_argument(
        "-l",
        "--len",
        type=int,
        required=True,
        help="Contig fasta file path.")
    
    myparser.add_argument(
        "-d",
        "--device",
        type=str,
        required=True,
        help="Contig fasta file path.")
    
    args = myparser.parse_args()
    
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-Marine-contigs-bam"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Grid-Search-marine-v1.0.9.7"
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    datanames = set()
    for file in os.listdir(home_path):
        name = file.split(".")[0:-2]
        datanames.add(".".join(name))
    
    drop_p_list = [0.125]
    for drop_p in drop_p_list:
        id_name = f"marine-sample-0"
        # id_name = data
        print(id_name, f"dropout: {drop_p}, min_len: {args.len}")
        contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        bam_path = os.path.join(home_path, f"{id_name}.sorted.bam")
        temp_path = os.path.join(out_folder, f"{id_name}-temp-folder-drop-{drop_p}-min_len-{args.len}")
        print(temp_path)
        output_path = os.path.join(out_folder, f"{id_name}-final-output-bins-drop-{drop_p}-min_len-{args.len}")
        if os.path.exists(os.path.join(output_path, "MetaInfo.tsv")) is False:
            binning_with_all_steps(
                contig_path,
                [bam_path],
                temp_path,
                output_path,
                min_contig_length=args.len,
                drop_p=drop_p,
                db_folder_path = "../DeeperBin-v1.0.8/DeeperBin-DB",
                training_device=args.device,
                remove_temp_files=True
            ) 



