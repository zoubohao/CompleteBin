
import argparse
import os
import sys

import numpy as np

from CompleteBin.Binning_steps import binning_with_all_steps

if __name__ == "__main__":
    # home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI2-others-datasets"
    # out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets"
    
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-Marine-contigs-bam"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin--marine-v1.0.9.6"
    device = "cuda:0"
    
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    # datanames = set()
    # for file in os.listdir(home_path):
    #     name = file.split(".")[0:-2]
    #     datanames.add(".".join(name))
    
    for i in range(10):
        id_name = f"marine-sample-{9}"
        # id_name = data
        print(id_name)
        contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        bam_path = os.path.join(home_path, f"{id_name}.sorted.bam")
        temp_path = os.path.join(out_folder, f"{id_name}-temp-folder")
        print(temp_path)
        output_path = os.path.join(out_folder, f"{id_name}-final-output-bins-100-75")
        if os.path.exists(os.path.join(output_path, "MetaInfo.tsv")) is False:
            binning_with_all_steps(
                contig_path,
                [bam_path],
                temp_path,
                output_path,
                db_folder_path = "../DeeperBin-v1.0.8/DeeperBin-DB",
                training_device=device
            ) 
        break



