
import os
from CompleteBin.Binning_steps import binning_with_all_steps



if __name__ == "__main__":
    
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-HD-marine-multi-sample"
    out_folder = "/home/comp/21481598/HD-marine-multi-completebin-1.0.9.6"
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    for i in range(5):
        id_name = f"CS01"
        print(id_name)
        contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        bam_path_list = []
        for j in ["01"]: # ["01", "03", "05", "08", "10", "12"]
            bam_path = os.path.join(home_path, f"HD-c{id_name}-rCS{j}.sorted.bam")
            bam_path_list.append(bam_path)
        temp_path = os.path.join(out_folder, f"{id_name}-temp-folder-single-800")
        output_path = os.path.join(out_folder, f"{id_name}-final-output-bins-single-800")
        if os.path.exists(os.path.join(output_path, "MetaInfo.tsv")) is False:
            binning_with_all_steps(
                contig_path,
                bam_path_list,
                temp_path,
                output_path,
                # db_folder_path = "../DeeperBin-v1.0.8/DeeperBin-DB",
                min_contig_length=800,
                training_device="cuda:7",
            ) 
            break














# if __name__ == "__main__":
#     home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI2-others-datasets"
#     out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets"
#     cur_data_names = ["Airways"] # Urogenital_tract Oral Gastrointestinal_tract Skin Airways
    
#     # home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Plant-contigs-bam"
#     # out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-plant-1.0.8"
#     if os.path.exists(out_folder) is False:
#         os.mkdir(out_folder)
    
#     datanames = set()
#     for file in os.listdir(home_path):
#         name = file.split(".")[0:-2]
#         datanames.add(".".join(name))
    
#     # for i in range(21):
#     for id_name in datanames:
#         # i = 2
#         # id_name = f"plant-sample-{i}"
#         signal = True
#         for cur_name in cur_data_names:
#             # print(cur_name, id_name, cur_name not in id_name)
#             if cur_name in id_name:
#                 signal = False
#         if signal: continue
#         print(id_name)
#         contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
#         bam_path = os.path.join(home_path, f"{id_name}.sorted.bam")
#         temp_path = os.path.join(out_folder, f"{id_name}-temp-folder")
#         output_path = os.path.join(out_folder, f"{id_name}-final-output-bins")
#         if os.path.exists(os.path.join(output_path, "MetaInfo.tsv")) is False:
#             binning_with_all_steps(
#                 contig_path,
#                 [bam_path],
#                 temp_path,
#                 output_path,
#                 db_folder_path = "./DeeperBin-DB",
#                 training_device="cuda:2",
#             ) 
#             # break



