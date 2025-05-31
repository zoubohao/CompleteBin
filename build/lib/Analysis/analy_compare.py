

import os

from CompleteBin.IO import readFasta


def readAllInfo(file_path):
    output_list = []
    with open(file_path, "r") as rh:
        for line in rh:
            line = line.strip("\n").split("\t")
            genome_name = line[-1]
            if line[4] == "HighQuality" or line[4] == "MediumQuality":
                cur_idx, bin_name = line[0].split("-")
                cur_idx = int(cur_idx)
                if cur_idx >= len(output_list):
                    cur_dict = dict()
                    cur_dict[genome_name] = [tuple(line)]
                    output_list.append(cur_dict)
                else:
                    cur_dict = output_list[cur_idx]
                    if genome_name not in cur_dict:
                        cur_dict[genome_name] = [tuple(line)]
                    else:
                        cur_dict[genome_name].append(tuple(line))
    return output_list


def readGenome2ID(file_path):
    res = {}
    with open(file_path, "r") as rh:
        for line in rh:
            line = line.strip("\n").split("\t")
            genome_name = os.path.split(line[-1])[-1]
            genome_name = os.path.splitext(genome_name)[0]
            res[genome_name] = line[0]
    return res


def readAbundance(file_path):
    res = {}
    with open(file_path, "r") as rh:
        for line in rh:
            line = line.strip("\n").split("\t")
            res[line[0]] = float(line[1])
    return res


def calculateN50(seqLens):
    if isinstance(seqLens, dict):
        contig_len = []
        for _, seq in seqLens.items():
            contig_len.append(len(seq))
        seqLens = contig_len
    thresholdN50 = sum(seqLens) / 2.0
    seqLens.sort(reverse=True)
    testSum = 0
    N50 = 0
    for seqLen in seqLens:
        testSum += seqLen
        if testSum >= thresholdN50:
            N50 = seqLen
            break
    return N50


def get_best(info_list):
    new_list = []
    for item in info_list:
        new_list.append((item[0], float(item[1]), item[4], item[5]))
    new_list = list(sorted(new_list, key=lambda x: x[1], reverse=True))
    return new_list[0]


if __name__ == "__main__":
    
    deeper_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-900-all-info.tsv"
    deeperbin_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/" # marine-sample-8-final-output-bins-1000
    # marine-sample-7-final-output-bins-900-no-multi-contrast
    comebin_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-2500-all-info.tsv"
    comebin_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/" #comebin-marine-sample-5-outputs/comebin_res/comebin_res_bins
    
    output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/DeeperBin900-vs-DeeperBin2500.abundance.N50.Unique.Intersect.tsv"
    
    deeperbin_data_list = readAllInfo(deeper_path)
    comebin_data_list = readAllInfo(comebin_path)
    
    sum_de_com = 0
    sum_com_de = 0
    
    genome2id = readGenome2ID("/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/genome_to_id.tsv")
    wh = open(output_path, "w")
    
    for i in range(len(deeperbin_data_list)):
        
        cur_deeperbin_path = os.path.join(deeperbin_folder, f"marine-sample-{i}-final-output-bins-900-no-multi-contrast")
        cur_comebin_path = os.path.join(comebin_folder, f"marine-sample-{i}-final-output-bins-2500")
        
        cur_deeperbin_sample_dict = deeperbin_data_list[i]
        cur_comebin_sample_dict = comebin_data_list[i]
        
        cur_deeperbin_genomes = set(cur_deeperbin_sample_dict.keys())
        cur_comebin_genomes = set(cur_comebin_sample_dict.keys())
        
        de_com = cur_deeperbin_genomes -  cur_comebin_genomes
        com_de = cur_comebin_genomes - cur_deeperbin_genomes
        intersect = cur_comebin_genomes & cur_deeperbin_genomes
        
        print("#" * 10)
        print(f"de_com: {len(de_com)}, com_de: {len(com_de)}, inter: {len(intersect)}")
        
        sum_de_com += len(de_com)
        sum_com_de += len(com_de)
        
        id2abundance = readAbundance(f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/abundance{i}.tsv")
        
        de_com_abu_list = []
        inter_list = []
        com_de_abu_list = []
        for g_name in de_com:
            de_com_abu_list.append(id2abundance[genome2id[g_name]])
            de_ge_name_best_mag = get_best(cur_deeperbin_sample_dict[g_name])
            contigname2seq = readFasta(os.path.join(cur_deeperbin_path, de_ge_name_best_mag[0].split("-")[-1]+".fasta"))
            cur_n50 = calculateN50(contigname2seq)
            wh.write("\t".join([de_ge_name_best_mag[0], "DeeperBin-900", "Unique", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
            
        for g_name in intersect:
            inter_list.append(id2abundance[genome2id[g_name]])
            # deeperbin
            de_ge_name_best_mag = get_best(cur_deeperbin_sample_dict[g_name])
            contigname2seq = readFasta(os.path.join(cur_deeperbin_path, de_ge_name_best_mag[0].split("-")[-1]+".fasta"))
            cur_n50 = calculateN50(contigname2seq)
            wh.write("\t".join([de_ge_name_best_mag[0], "DeeperBin-900", "Intersect", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
            # comebin
            de_ge_name_best_mag = get_best(cur_comebin_sample_dict[g_name])
            contigname2seq = readFasta(os.path.join(cur_comebin_path, de_ge_name_best_mag[0].split("-")[-1]+".fasta"))
            cur_n50 = calculateN50(contigname2seq)
            wh.write("\t".join([de_ge_name_best_mag[0], "DeeperBin-2500", "Intersect", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
            
        for g_name in com_de:
            com_de_abu_list.append(id2abundance[genome2id[g_name]])
            de_ge_name_best_mag = get_best(cur_comebin_sample_dict[g_name])
            contigname2seq = readFasta(os.path.join(cur_comebin_path, de_ge_name_best_mag[0].split("-")[-1]+".fasta"))
            cur_n50 = calculateN50(contigname2seq)
            wh.write("\t".join([de_ge_name_best_mag[0], "DeeperBin-2500", "Unique", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
        
        
        
        print(f"mean de_com: {sum(de_com_abu_list) / (len(de_com_abu_list) + 0.001)}, mean inter: {sum(inter_list) / (len(inter_list) + 0.001)}, mean com_de: {sum(com_de_abu_list) / (len(com_de_abu_list) + 0.001)}")
        
    print(f"sum de_com: {sum_de_com}, sum_com_de: {sum_com_de}")
    wh.close()
    
    
    
    
    
    
    
    # deeper_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-1000-all-info.tsv"
    # deeperbin_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/" # marine-sample-8-final-output-bins-1000
    # # marine-sample-7-final-output-bins-900-no-multi-contrast
    # comebin_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/COMEBin-all-info.tsv"
    # comebin_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin/Comebin-marine" #comebin-marine-sample-5-outputs/comebin_res/comebin_res_bins
    
    # output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/COMEBin-DeeperBin1000.abundance.N50.tsv"
    
    # deeperbin_data_list = readAllInfo(deeper_path)
    # comebin_data_list = readAllInfo(comebin_path)
    
    # sum_de_com = 0
    # sum_com_de = 0
    
    # genome2id = readGenome2ID("/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/genome_to_id.tsv")
    # wh = open(output_path, "w")
    
    # for i in range(len(deeperbin_data_list)):
        
    #     cur_deeperbin_path = os.path.join(deeperbin_folder, f"marine-sample-{i}-final-output-bins-1000")
    #     cur_comebin_path = os.path.join(comebin_folder, f"comebin-marine-sample-{i}-outputs/comebin_res/comebin_res_bins")
        
    #     cur_deeperbin_sample_dict = deeperbin_data_list[i]
    #     cur_comebin_sample_dict = comebin_data_list[i]
        
    #     cur_deeperbin_genomes = set(cur_deeperbin_sample_dict.keys())
    #     cur_comebin_genomes = set(cur_comebin_sample_dict.keys())
        
    #     de_com = cur_deeperbin_genomes -  cur_comebin_genomes
    #     com_de = cur_comebin_genomes - cur_deeperbin_genomes
    #     intersect = cur_comebin_genomes & cur_deeperbin_genomes
        
    #     print("#" * 10)
    #     print(f"de_com: {len(de_com)}, com_de: {len(com_de)}, inter: {len(intersect)}")
        
    #     sum_de_com += len(de_com)
    #     sum_com_de += len(com_de)
        
    #     id2abundance = readAbundance(f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/abundance{i}.tsv")
        
    #     de_com_abu_list = []
    #     inter_list = []
    #     com_de_abu_list = []
    #     for g_name in de_com:
    #         de_com_abu_list.append(id2abundance[genome2id[g_name]])
    #         de_ge_name_best_mag = get_best(cur_deeperbin_sample_dict[g_name])
    #         contigname2seq = readFasta(os.path.join(cur_deeperbin_path, de_ge_name_best_mag[0].split("-")[-1]+".fasta"))
    #         cur_n50 = calculateN50(contigname2seq)
    #         wh.write("\t".join([de_ge_name_best_mag[0], "DeeperBin", "Unique", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
            
    #     for g_name in intersect:
    #         inter_list.append(id2abundance[genome2id[g_name]])
    #         # deeperbin
    #         de_ge_name_best_mag = get_best(cur_deeperbin_sample_dict[g_name])
    #         contigname2seq = readFasta(os.path.join(cur_deeperbin_path, de_ge_name_best_mag[0].split("-")[-1]+".fasta"))
    #         cur_n50 = calculateN50(contigname2seq)
    #         wh.write("\t".join([de_ge_name_best_mag[0], "DeeperBin", "Intersect", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
    #         # comebin
    #         de_ge_name_best_mag = get_best(cur_comebin_sample_dict[g_name])
    #         contigname2seq = readFasta(os.path.join(cur_comebin_path, de_ge_name_best_mag[0].split("-")[-1]+".fa"))
    #         cur_n50 = calculateN50(contigname2seq)
    #         wh.write("\t".join([de_ge_name_best_mag[0], "COMEBin", "Intersect", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
            
    #     for g_name in com_de:
    #         com_de_abu_list.append(id2abundance[genome2id[g_name]])
    #         de_ge_name_best_mag = get_best(cur_comebin_sample_dict[g_name])
    #         contigname2seq = readFasta(os.path.join(cur_comebin_path, de_ge_name_best_mag[0].split("-")[-1]+".fa"))
    #         cur_n50 = calculateN50(contigname2seq)
    #         wh.write("\t".join([de_ge_name_best_mag[0], "COMEBin", "Unique", str(id2abundance[genome2id[g_name]]), str(cur_n50), g_name]) + "\n")
        
        
        
    #     print(f"mean de_com: {sum(de_com_abu_list) / (len(de_com_abu_list) + 0.001)}, mean inter: {sum(inter_list) / (len(inter_list) + 0.001)}, mean com_de: {sum(com_de_abu_list) / (len(com_de_abu_list) + 0.001)}")
        
    # print(f"sum de_com: {sum_de_com}, sum_com_de: {sum_com_de}")
    # wh.close()

