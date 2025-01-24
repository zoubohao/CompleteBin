
from DeeperBin.Binning import binning_with_all_steps
import os


if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Plant-contigs-bam"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeepShortBin-plant"
    
    for i in range(10, 21):
        id_name = f"plant-sample-{i}"
        cur_temp_path = os.path.join(out_folder, f"{id_name}-768-v2.1.0")
        cur_bin_output_folder = os.path.join(out_folder, f"{id_name}-750-v2.1.2_final_bin_output")
        meta_path = os.path.join(cur_bin_output_folder, "MetaInfo.tsv")
        cur_contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        bam_list = []
        for j in range(10):
            bam_list.append(os.path.join(home_path, f"plant-sample-c{i}-r{j}.sorted.bam"))
        binning_with_all_steps(
            contig_file_path=cur_contig_path,
            sorted_bam_file_list=bam_list,
            temp_file_folder_path=cur_temp_path,
            bin_output_folder_path=cur_bin_output_folder,
            db_folder_path="./DeepMetaBin-DB",
            training_device="cuda:4",
            )
        # break
        


