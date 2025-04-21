

import multiprocessing
import os
import random
import sys

import numpy as np

from Src.CallGenes.gene_utils import splitListEqually
from Src.IO import (progressBar, readCheckm2Res, readFasta, readMetaInfo,
                    readPickle)

# import torch
# import os


# import torch
# import torch.nn.functional as F

# # Example embeddings for samples and positive pairs
# embeddings = torch.rand(1000, 128)  # 1000 samples, 128-d embeddings
# positive_pairs = [(torch.rand(128), torch.rand(128)) for _ in range(500)]

# # Normalize embeddings to the unit sphere
# embeddings = F.normalize(embeddings, dim=1)

# # Alignment Calculation
# def calculate_alignment(positive_pairs):
#     alignment = 0.0
#     for x, x_pos in positive_pairs:
#         alignment += torch.norm(x - x_pos, p=2).pow(2).item()
#     return alignment / len(positive_pairs)

# # Uniformity Calculation
# def calculate_uniformity(embeddings):
#     n = embeddings.size(0)
#     pairwise_distances = torch.cdist(embeddings, embeddings, p=2).pow(2)
#     uniformity = torch.log(torch.exp(-2 * pairwise_distances).mean()).item()
#     return uniformity

# alignment = calculate_alignment(positive_pairs)
# uniformity = calculate_uniformity(embeddings)

# print(f"Alignment: {alignment}")
# print(f"Uniformity: {uniformity}")


# def readDiamond(file_path: str, res):
#     with open(file_path, "r") as rh:
#         for line in rh:
#             thisline = line.strip("\n").split("\t")
#             _, contig_name = thisline[0].split("Ω")
#             true_contig_name = ">" + "_".join(contig_name.split("_")[0:-1])
#             if true_contig_name not in res:
#                 res[true_contig_name] = [(contig_name, thisline[1:])]
#             else:
#                 res[true_contig_name].append((contig_name, thisline[1:]))

# import numpy as np
# def schedule_of_temperature(temp: float, epochs: int):
#     ## 0.125 0.15
#     start_temp = temp - 0.025
#     step = 0.025 / epochs
#     res = [0 for _ in range(epochs)]
#     for i, cur_temp in enumerate(np.arange(start_temp, temp, step)):
#         if i <= epochs - 1:
#             res[i] = float("%.3f" % cur_temp)
#     res[-1] = temp
#     return res

def read_list(file_list, training_data_path):
    i = 0
    N = len(file_list)
    data = []
    data_name = []
    for file in file_list:
        progressBar(i, N)
        seq_features_tuple = readPickle(os.path.join(training_data_path, file))
        data.append(seq_features_tuple)
        data_name.append(file)
        i += 1
    return data, data_name

# NODE_1_length_716051_cov_17.026672
if __name__ == "__main__":
    
    from Analysis.F1 import BinningF1
    from Analysis.utils import convert_paf2vamb, get_vamb_cluster_tsv
    from Analysis.vamb_benchmark import Binning, Reference
    
    bin_tool = "DeeperBin"
    data = "plant"
    data_output_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Grid-Search-marine-v1.0.7/"
    data_sample_num = 10
    
    drop_p_list = [0.11, 0.12, 0.13, 0.14]
    # epoch_list = [850, 900]
    # multi_list = [False, True]
    for drop_p in drop_p_list:
        # for min_len in epoch_list:
        #     ### this needs to change !!!!!
        #     for multi in multi_list:
        bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Grid-Search-marine-v1.0.7/"+ \
            f"marine-sample-0-final-output-bins-drop-{drop_p}-min_len-{900}-multi-{False}/"
        
        output_tsv_file = f"{data_output_folder}marine-sample-0-drop-{drop_p}-min_len-{900}-multi-{False}.cluster.tsv"
        print(output_tsv_file)
        get_vamb_cluster_tsv(
            bin_folder,
            output_tsv_file,
            suffix=".fasta" # change this for different binning tools
        )
        # break
    
    p_11 = []
    p_12 = []
    p_13 = []
    p_14 = []
    p_15 = []
    
    l_800 = []
    l_850 = []
    l_900 = []
    
    multi_true = []
    multi_false = []
    
    ## calculate F1
    final_res = dict()
    nc_mag, r90p90_mag, medium_mag = 0., 0., 0.
    for drop_p in drop_p_list:
        # for min_len in epoch_list:
        #     for multi in multi_list:
                # ref_folder = f"{data_output_folder}{data}-sample-{i}.vamb.ref"
                ref_folder = f"{data_output_folder}marine-sample-0.minimap2"
                output_tsv_file = f"{data_output_folder}marine-sample-0-drop-{drop_p}-min_len-{900}-multi-{False}.cluster.tsv"
                output_bin2f1_file = f"{data_output_folder}marine-sample-0-drop-{drop_p}-min_len-{9000}-multi-{False}.bin2f1.tsv"
                
                print(output_tsv_file, os.path.exists(output_tsv_file))
                
                bins = BinningF1(output_tsv_file, ref_folder)
                res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag, cur_bin2f1 = bins.get_f1([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
                print(res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag)
                
                
                if drop_p == 0.11:
                    p_12.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                if drop_p == 0.12:
                    p_12.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                if drop_p == 0.13:
                    p_13.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                if drop_p == 0.14:
                    p_14.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                    
                # if min_len == 850:
                #     l_850.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                # if min_len == 900:
                #     l_900.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                    
                # if multi:
                #     multi_true.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                # else:
                #     multi_false.append(sum([cur_nc_mag * 1.2, cur_r90p90_mag * 0.95, res[0.95], res[0.9]*0.9]))
                
                for k, v in res.items():
                    if k not in final_res:
                        final_res[k] = v
                    else:
                        final_res[k] += v
                r90p90_mag += cur_r90p90_mag
                nc_mag += cur_nc_mag
                medium_mag += cur_medium_mag
                with open(output_bin2f1_file, "w") as wh:
                    for k, v in cur_bin2f1.items():
                        wh.write(k + "\t" + "\t".join(v) + "\n")
        # break
    
    
    
    print(f"0.12: {sum(p_12)}, 0.13: {sum(p_13)}, 0.14: {sum(p_14)}, 0.11: {sum(p_11)}")
    print(f"800: {sum(l_800)}, 850: {sum(l_850)}, 900: {sum(l_900)}")
    print(f"multi-True: {sum(multi_true)}, false: {sum(multi_false)}")
    
    ## grid search results collection
    # ref_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine/marine-sample-0.minimap2"
    # data_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Grid-Search/"
    # files_list = os.listdir(data_folder)
    # collected_list = []
    # for file in files_list:
    #     if "final-output-bins" in file and "cluster.tsv" not in file:
    #         bin_folder = os.path.join(data_folder, file)
    #         output_tsv_file = os.path.join(data_folder, f"{file}.cluster.tsv")
    #         print(output_tsv_file)
    #         # get_vamb_cluster_tsv(
    #         #     bin_folder,
    #         #     output_tsv_file,
    #         #     suffix=".fasta" # change this for different binning tools
    #         # )
    #         bins = BinningF1(output_tsv_file, ref_folder)
    #         res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag, cur_bin2f1 = bins.get_f1([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    #         print(res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag)
    #         collected_list.append((file, (res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag), cur_r90p90_mag))
    
    # sorted_coll = list(sorted(collected_list, key=lambda x: x[-1], reverse=True))
    
    # for ele in sorted_coll:
    #     print(ele)
    
    
    ## Dingyi real data 
    binner = "DeeperBin"
    bin_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Dingyi-data/SRR13060973-final-bins/"
    output_tsv_file = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-Dingyi-data/SRR13060973.cluster.tsv"
    ref_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Dingyi-short-long-reads-data/SRR13060973.minimap2"
    
    # binner = "COMEBin"
    # bin_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin/SRR13060973-COMEBin-bins/"
    # output_tsv_file = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin/SRR13060973.cluster.tsv"
    
    print(output_tsv_file)
    # get_vamb_cluster_tsv(
    #     bin_folder,
    #     output_tsv_file,
    #     suffix=".fa" # change this for different binning tools
    # )
    
    print(f"{binner} F1")
    bins = BinningF1(output_tsv_file, ref_folder)
    res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag, cur_bin2f1 = bins.get_f1([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    print(f"{res[0.5]}, {res[0.6]}, {res[0.7]}, {res[0.8]}, {res[0.9]}, {res[0.95]} MAGs' F1 bigger than 0.5, 0.6, 0.7, 0.8, 0.9, 0.95")
    sys.exit(0)
    
    
    
    # ######construct table
    # bin_tool = "DeeperBin"
    # data = "marine"
    # data_output_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/{data}/"
    # data_sample_num = 10
    
    # for i in range(data_sample_num):
    #     ### this needs to change !!!!!
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Comebin-{data}/comebin-marine-sample-{i}-outputs/comebin_res/comebin_res_bins"
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Semibin2-marine/marine-sample-{i}/output_bins"
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/MetaBAT2-marine/marine-sample-{i}/"
    #     bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine-auto-and-contrast/marine-sample-{i}-final-output-bins/"
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/marine-sample-{i}-final-output-bins-the-last/"
        
        
    #     ### plant
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Comebin-{data}/comebin-plant-sample-{i}-outputs/comebin_res/comebin_res_bins"
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Semibin2-plant/plant-sample-{i}/output_bins"
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-plant/plant-sample-{i}-final-output-bins-with-checkm2/"
    #     # bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/MetaBAT2-plant/plant-sample-{i}/"
    #     output_tsv_file = f"{data_output_folder}{data}-sample-{i}-{bin_tool}.auto.cluster.tsv"
    #     print(output_tsv_file)
    #     get_vamb_cluster_tsv(
    #         bin_folder,
    #         output_tsv_file,
    #         suffix=".fasta" # change this for different binning tools
    #     )
    #     # break
    
    # ## calculate F1
    # final_res = dict()
    # nc_mag, r90p90_mag, medium_mag = 0., 0., 0.
    # for i in range(data_sample_num):
    #     # ref_folder = f"{data_output_folder}{data}-sample-{i}.vamb.ref"
    #     ref_folder = f"{data_output_folder}{data}-sample-{i}.minimap2"
    #     output_tsv_file = f"{data_output_folder}{data}-sample-{i}-{bin_tool}.auto.cluster.tsv"
    #     output_bin2f1_file = f"{data_output_folder}{data}-sample-{i}-{bin_tool}.auto.bin2f1.tsv"
        
    #     print(output_tsv_file, os.path.exists(output_tsv_file))
        
    #     # ref = Reference.from_file(open(ref_folder, "r"))
    #     # bins = Binning.from_file(open(output_tsv_file, "r"), ref)
    #     # cur_nc_mag = bins.print_matrix(rank=0)
    #     # print(f"NC MAGs: {cur_nc_mag}")
        
    #     bins = BinningF1(output_tsv_file, ref_folder)
    #     res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag, cur_bin2f1 = bins.get_f1([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    #     print(res, cur_nc_mag, cur_r90p90_mag, cur_medium_mag)
    #     for k, v in res.items():
    #         if k not in final_res:
    #             final_res[k] = v
    #         else:
    #             final_res[k] += v
    #     r90p90_mag += cur_r90p90_mag
    #     nc_mag += cur_nc_mag
    #     medium_mag += cur_medium_mag
    #     with open(output_bin2f1_file, "w") as wh:
    #         for k, v in cur_bin2f1.items():
    #             wh.write(k + "\t" + "\t".join(v) + "\n")
    #     # break
        
    # print(bin_tool, f"NC MAGs: {nc_mag}, ", f"90 & 90 MAGs: {r90p90_mag}, ", f"Medium MAGs: {medium_mag}", final_res)
    # print(list(final_res.values()))
    
    
    ### convert paf 2 vamb ref format.
    # homepath = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine"
    # for i in range(10):
    #     print(i)
    #     convert_paf2vamb(
    #         os.path.join(homepath, f"marine-sample-{i}.minimap2"),
    #         os.path.join(homepath, f"marines-sample-{i}.vamb.ref")
    #     )
    
    
    # ### align the contig to reference genomes
    # output_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine"
    # home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Marine-contigs-bam"
    # query_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine_ref_genomes.fasta"
    # if os.path.exists(output_folder) is False:
    #     os.mkdir(output_folder)
    # for i in range(10):
    #     id_name = f"marine-sample-{i}"
    #     cur_contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
    #     cur_output_path = os.path.join(output_folder, f"{id_name}.minimap2")
    #     cmd = f"minimap2 {query_path} {cur_contig_path} -o {cur_output_path} -t 64"
    #     print(cmd)
    #     os.system(cmd)





    #### abundance 2 contig length
    # data_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-HD-marine-sample/CS03-temp-folder/training_data"
    # output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/HD-marine-CS03-longVSshort-coverage.tsv"
    # long_threa = 2000
    
    # data = []
    # data_name = []
    # file_list = os.listdir(data_folder)
    # file_list_list = splitListEqually(file_list, 128)
    # pro_list = []
    # res = []
    # with multiprocessing.Pool(len(file_list_list)) as multiprocess:
    #     for i, item in enumerate(file_list_list):
    #         p = multiprocess.apply_async(read_list,
    #                                     (item,
    #                                     data_folder,
    #                                     ))
    #         pro_list.append(p)
    #     multiprocess.close()
    #     for p in pro_list:
    #         res.append(p.get())
    # for cur_data, cur_data_name in res:
    #     data += cur_data
    #     data_name += cur_data_name

    # with open(output_path, "w") as wh:
    #     N = len(data)
    #     for i, cur_data in enumerate(data):
    #         if i % 10 == 0: print(i, N)
    #         ori_seq, cov_bp_array_list, seq_tokens, cov_mean, cov_var_sqrt = cur_data
    #         if len(ori_seq) >= long_threa:
    #             wh.write(f"{data_name[i]}\t{len(ori_seq)}\tLong Contig\t{cov_mean}\n")
    #         else:
    #             wh.write(f"{data_name[i]}\t{len(ori_seq)}\tShort Contig\t{cov_mean}\n")
    
    