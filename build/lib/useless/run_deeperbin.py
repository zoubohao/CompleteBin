
import os

from Src.Binning_steps import binning_with_all_steps

if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/HD-marine-final-results"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Real-data"
    
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    for id_name in ["CS01", "CS03", "CS05", "CS08", "CS10", "CS12"]: # ["CS08", "CS10", "CS12"] "CS01", "CS03", "CS05"
        print(id_name)
        contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        bam_path = os.path.join(home_path, f"{id_name}.sorted.bam")
        temp_path = os.path.join(out_folder, f"{id_name}-temp-folder")
        output_path = os.path.join(out_folder, f"{id_name}-final-output-bins")
        if os.path.exists(os.path.join(output_path, "MetaInfo.tsv")) is False:
            binning_with_all_steps(
                contig_path,
                [bam_path],
                temp_path,
                output_path,
                min_contig_length=770,
                db_folder_path = "./DeeperBin-DB",
                training_device="cuda:3",
            ) 



