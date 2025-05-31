

import os
from CompleteBin.Binning_steps import binning_with_all_steps



if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-others-datasets"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets-v1.0.9.5"
    cur_data_names = [
                      "Skin-2017.12.04_18.56.22_sample_1", 
                      "Gastrointestinal_tract-2017.12.04_18.45.54_sample_3", 
                      "Gastrointestinal_tract-2017.12.04_18.45.54_sample_2",
                      "Airways-2017.12.04_18.56.22_sample_26",
                      "Gastrointestinal_tract-2017.12.04_18.45.54_sample_5",
                      "Skin-2017.12.04_18.56.22_sample_15",
                      "Airways-2017.12.04_18.56.22_sample_12",
                      "Skin-2017.12.04_18.56.22_sample_13",
                      "Urogenital_tract-2017.12.04_18.56.22_sample_6",
                      "Skin-2017.12.04_18.56.22_sample_16",
                      "Skin-2017.12.04_18.56.22_sample_28"
                      ] 
    device = "cuda:1"
    ##  "Oral", "Gastrointestinal_tract", "Skin" "Airways", "Urogenital_tract"
    
    # home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-strain-madness"
    # out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-strain-madness"
    # cur_data_names = [""] # Urogenital_tract Oral Gastrointestinal_tract Skin Airways "Airways"
    # device = "cuda:0"

    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    datanames = set()
    for file in os.listdir(home_path):
        name = file.split(".")[0:-2]
        datanames.add(".".join(name))
    
    for id_name in datanames:
        signal = True
        for cur_name in cur_data_names:
            # print(cur_name, id_name, cur_name not in id_name)
            if cur_name in id_name:
                signal = False
        if signal: continue
        print(id_name)
        # id_name = f"sample_{id_name}"
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
            







