


import os
from CompleteBin.Binning_steps import binning_with_all_steps



if __name__ == "__main__":
    # home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-others-datasets"
    # out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets-v1.0.6"
    # cur_data_names = ["Airways"] # Urogenital_tract Oral Gastrointestinal_tract Skin Airways "Airways"
    # device = "cuda:0"
    
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-strain-madness"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-strain-madness"
    cur_data_names = [""] # Urogenital_tract Oral Gastrointestinal_tract Skin Airways "Airways"
    device = "cuda:0"

    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    datanames = set()
    for file in os.listdir(home_path):
        name = file.split(".")[0:-2]
        datanames.add(".".join(name))
    
    for id_name in range(10):
        # signal = True
        # for cur_name in cur_data_names:
        #     # print(cur_name, id_name, cur_name not in id_name)
        #     if cur_name in id_name:
        #         signal = False
        # if signal: continue
        print(id_name)
        id_name = f"sample_{id_name}"
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
                db_folder_path = "../DeeperBin-v1.0.8/DeeperBin-DB",
                training_device=device,
            ) 
        
        
        # if os.path.exists(temp_path) is False:
            # binning_with_all_steps(
            #     contig_path,
            #     [bam_path],
            #     temp_path,
            #     output_path,
            #     db_folder_path = "./DeeperBin-DB",
            #     training_device=device,
            # ) 
            







