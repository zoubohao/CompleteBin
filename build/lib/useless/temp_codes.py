
import os
from shutil import copytree
import torch

from CompleteBin.IO import readFasta, readPickle
import numpy as np

if __name__ == "__main__":
    
    
    fasta_path = "/home/datasets/ZOUbohao/Proj1-Deepurify/Data_IBS/K0268.contigs.fasta"
    # fasta_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-freshwater-multi-sample/ERR9631077.contigs.fasta"
    output_pat = "./Complex-vs-Simple.tsv"
    contigname2seq = readFasta(fasta_path)
    with open(output_pat, "a") as wh:
        for _, seq in contigname2seq.items():
            wh.write("Simple Metagenomic Sample\t"+str(len(seq)) + "\n")
    
    
    # data_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CompleteBin-v1.0.9.12-tnf-cov-temp/contigname2seq_str.pkl"
    # name2seq = readPickle(data_path)
    # print(len(name2seq))
    
    
    
    # data_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CompleteBin-v1.0.9.12-tnf-cov-temp/contigname2bpcover_nparray_0.pkl"
    # name2bp_array = readPickle(data_path)
    # data = []
    # mean_values = []
    # std_values = []
    # for name, bp_array in name2bp_array.items():
    #     data.append((name, len(bp_array), np.std(bp_array), np.mean(bp_array)))
    #     mean_values.append(np.mean(bp_array))
    #     std_values.append(np.std(bp_array))
    # sorted_data = list(sorted(data, key=lambda x: x[-1]))
    # mean_values = np.array(mean_values)
    # std_values = np.array(std_values)
    # for ele in sorted_data:
    #     print(ele)
    # print(np.percentile(mean_values, 99), np.percentile(mean_values, 99.5), np.percentile(mean_values, 99.99))
    # print(np.percentile(std_values, 99), np.percentile(std_values, 99.5), np.percentile(std_values, 99.99))
    
    
    
    
    # final_output_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets-final"
    # file_list = os.listdir(final_output_folder)
    # wh = open(os.path.join(final_output_folder, "final_meta_info.tsv"), "w")
    # for file in file_list:
    #     with open(os.path.join(final_output_folder, file, "MetaInfo.tsv"), "r") as rh:
    #         for line in rh:
    #             wh.write(line)
    # wh.close()
    
    
    # data_folder1 = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets-v1.0.9.5"
    # data_folder2 = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets-v1.0.6"
    # data_folder3 = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets-v1.0.3"
    # final_output_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets-final"
    
    # if os.path.exists(final_output_folder) is False:
    #     os.mkdir(final_output_folder)

    # print(1)
    # file_list = os.listdir(data_folder1)
    # for file in file_list:
    #     if "final-output-bins" in file and os.path.exists(os.path.join(final_output_folder, file)) is False:
    #         if len(os.listdir(os.path.join(data_folder1, file))) != 0:
    #             copytree(os.path.join(data_folder1, file), os.path.join(final_output_folder, file))
    #         else:
    #             print(file)
    # print(2)
    # file_list = os.listdir(data_folder2)
    # for file in file_list:
    #     if "final-output-bins" in file and os.path.exists(os.path.join(final_output_folder, file)) is False:
    #         if len(os.listdir(os.path.join(data_folder2, file))) != 0:
    #             copytree(os.path.join(data_folder2, file), os.path.join(final_output_folder, file))
    #         else:
    #             print(file)
    # print(3)
    # file_list = os.listdir(data_folder3)
    # for file in file_list:
    #     if "final-output-bins" in file and os.path.exists(os.path.join(final_output_folder, file)) is False:
    #         if len(os.listdir(os.path.join(data_folder3, file))) != 0:
    #             copytree(os.path.join(data_folder3, file), os.path.join(final_output_folder, file))
    #         else:
    #             print(file)