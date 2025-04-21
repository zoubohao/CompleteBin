
import os

from Src.Binning_steps import binning_with_all_steps

if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/HD-multi-sample"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-HD-multi-sample"
    
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    ids = ["CS01", "CS03", "CS05", "CS08", "CS10", "CS12"]
    
    for id_name in ["CS01", "CS03", "CS05", "CS08", "CS10", "CS12"]: # "CS01", "CS03", "CS05", "CS08", "CS10", "CS12"
        print(id_name)
        cur_temp_path = os.path.join(out_folder, f"{id_name}-temp-folder")
        cur_bin_output_folder = os.path.join(out_folder, f"{id_name}-final-output-bins")
        meta_path = os.path.join(cur_bin_output_folder, "MestaInfo.tsv")
        cur_contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        if os.path.exists(meta_path):
            continue
        bam_list = []
        for j in ids:
            bam_list.append(os.path.join(home_path, f"HD-c{id_name}-r{j}.sorted.bam"))
        if os.path.exists(os.path.join(cur_bin_output_folder, "MetaInfo.tsv")) is False:
            binning_with_all_steps(
                contig_file_path=cur_contig_path,
                sorted_bam_file_list=bam_list,
                temp_file_folder_path=cur_temp_path,
                bin_output_folder_path=cur_bin_output_folder,
                min_contig_length=769,
                db_folder_path="./DeeperBin-DB",
                training_device="cuda:2",
                remove_temp_files=False,
                step_num=None
                )
        # break
        


