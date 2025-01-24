

import os
from DeeperBin.CallGenes.gene_utils import splitListEqually
from DeeperBin.IO import readFasta, readCheckm2Res, readMetaInfo
import random
import numpy as np

from DeeperBin.Seqs.seq_utils import reject_outliers
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


def readDiamond(file_path: str, res):
    with open(file_path, "r") as rh:
        for line in rh:
            thisline = line.strip("\n").split("\t")
            _, contig_name = thisline[0].split("Ω")
            true_contig_name = ">" + "_".join(contig_name.split("_")[0:-1])
            if true_contig_name not in res:
                res[true_contig_name] = [(contig_name, thisline[1:])]
            else:
                res[true_contig_name].append((contig_name, thisline[1:]))

import numpy as np
def schedule_of_temperature(temp: float, epochs: int):
    ## 0.125 0.15
    start_temp = temp - 0.025
    step = 0.025 / epochs
    res = [0 for _ in range(epochs)]
    for i, cur_temp in enumerate(np.arange(start_temp, temp, step)):
        if i <= epochs - 1:
            res[i] = float("%.3f" % cur_temp)
    res[-1] = temp
    return res

# NODE_1_length_716051_cov_17.026672
if __name__ == "__main__":
    
    a = np.random.randn(100)
    
    a = np.array(a, dtype=np.float32)
    print(a)
    print(reject_outliers(a))
    
    
    # output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Result-multi-sample-data/SemiBin2-marine-multi-sample.tsv"
    # input_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Semibin2-marine-multi-sample"
    # num_samples = 10
    # with open(output_path, "w") as wh:
    #     index = 0
    #     for i in range(num_samples):
    #         cur_id = f"marine-sample-{i}-checkm2"
    #         cur_quality_tsv_path = os.path.join(input_folder, cur_id, "quality_report.tsv")
    #         if os.path.exists(cur_quality_tsv_path) is False:
    #             cur_id = f"marine-sample-{i-1}-checkm2"
    #             cur_quality_tsv_path = os.path.join(input_folder, cur_id, "quality_report.tsv")
    #         cur_res, _, _, _ = readCheckm2Res(cur_quality_tsv_path, "fasta")
    #         for _, vals in cur_res.items():
    #             wh.write(f"{index}.fasta\t{vals[0]}\t{vals[1]}\t{vals[2]}\n")
    #             index += 1
    
    
    # output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Result-multi-sample-data/DeepMetaBin-marine-multi-sample.tsv"
    # input_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeepShortBin-marine-multi-sample"
    # num_samples = 10
    # with open(output_path, "w") as wh:
    #     index = 0
    #     for i in range(num_samples):
    #         cur_id = f"marine-sample-{i}-750-v2.1.2_final_bin_output"
    #         cur_quality_tsv_path = os.path.join(input_folder, cur_id, "MetaInfo.tsv")
    #         cur_res, _, _, _ = readMetaInfo(cur_quality_tsv_path)
    #         for _, vals in cur_res.items():
    #             wh.write(f"{index}.fasta\t{vals[0]}\t{vals[1]}\t{vals[2]}\n")
    #             index += 1
    
    
    # contigname2seq = {}
    # for i in range(1000):
    #     contigname2seq[i] = i
    # cur_name_list = list(range(1000))
    # split_names_list = splitListEqually(cur_name_list, 1 + 2)
    # for one_split_names in split_names_list:
    #     cur_split_clu_res = {}
    #     for split_name in one_split_names:
    #         cur_split_clu_res[split_name] = contigname2seq[split_name]
    #     print("##############")
    #     for k, v in cur_split_clu_res.items():
    #         print(k, v)
    
    # for i in range(110):
    #     print()
    #     print(schedule_of_temperature(0.10239232, i + 11))
    
    
    # def random_generate_view(
    #     seq: str,
    #     min_contig_len: int,
    #     seed=None
    # ):
    #     if seed is None:
    #         random.seed()
    #     else:
    #         random.seed(seed)
    #     n = len(seq)
    #     sim_len = random.randint(min_contig_len - 1, n)
    #     start = random.randint(0, n - sim_len)
    #     end = start + sim_len
    #     random.seed()
    #     return seq[start: end], start, end
    
    
    # seq = "ATCGATCGATCGATCG"
    # a = []
    # for i in range(10000):
    #     a.append(random_generate_view(seq, 16))
    # res = {}
    # for item in a:
    #     cur_name = (item[1], item[2])
    #     if cur_name in res:
    #         res[cur_name] += 1
    #     else:
    #         res[cur_name] = 1
    # print(res)
    
    
    
    # diam_file = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/ERR9631077-768-v2.1.0/split_contigs_checkm2_temp/diamond_output"
    # diamond_info = {} # contig name (with ">") with its genes info list
    # for file in os.listdir(diam_file):
    #     print(file)
    #     readDiamond(os.path.join(diam_file, file), diamond_info)
    # print(diamond_info[">NODE_9801_length_10716_cov_27.976175"])