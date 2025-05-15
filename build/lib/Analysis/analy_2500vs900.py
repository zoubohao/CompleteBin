

import os
from shutil import copy

from Src.IO import readFasta


def readAllInfo(file_path):
    output_list = []
    with open(file_path, "r") as rh:
        for line in rh:
            line = line.strip("\n").split("\t")
            genome_name = line[-1]
            if line[4] == "HighQuality" or line[4] == "MediumQuality" or line[4] == "LowQuality":
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
    new_output = []
    for i in range(len(output_list)):
        cur_dict_new = {}
        for genome_name, values_list in output_list[i].items():
            new_list = list(sorted(values_list, key=lambda x: float(x[1]), reverse=True))
            cur_dict_new[genome_name] = new_list[0]
        new_output.append(cur_dict_new)
    return new_output


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


if __name__ == "__main__":
    
    
    # baseline_info_tsv_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/CAMI-Marine-DeeperBin2500-High.abundance.N50.tsv"
    # upgrade_info_tsv_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/CAMI-Marine-DeeperBin2500-DeeperBin900.900upgrade-new.abundance.N50.tsv"
    # output_info_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/2500-High-vs-900-upgrade-vs-900-new.tsv"
    # wh = open(output_info_path, "w")
    
    # baseline_n50 = []
    # baseline_abu = []
    
    # upgrade_n50 = []
    # upgrade_abu = []
    
    # new_n50 = []
    # new_abu = []
    
    # with open(baseline_info_tsv_path, "r") as rh:
    #     for line in rh:
    #         info = line.strip().split("\t")
    #         baseline_abu.append(float(info[8]))
    #         baseline_n50.append(float(info[9]))
    #         wh.write(f"baseline\t{info[8]}\t{info[9]}\n")
    
    # with open(upgrade_info_tsv_path, "r") as rh:
    #     for line in rh:
    #         info = line.strip().split("\t")
    #         if info[6] == "900" and info[7] == "upgrade":
    #             upgrade_n50.append(float(info[8]))
    #             upgrade_abu.append(float(info[9]))
    #             wh.write(f"Upgrade\t{info[9]}\t{info[8]}\n")
    #         if info[6] == "900" and info[7] == "new":
    #             new_n50.append(float(info[8]))
    #             new_abu.append(float(info[9]))
    #             wh.write(f"Unique\t{info[9]}\t{info[8]}\n")
    
    # print(f"baseline mean N50: {sum(baseline_n50) / len(baseline_n50)}, mean abundance: {sum(baseline_abu) / len(baseline_abu)}")
    # print(f"upgrade mean N50: {sum(upgrade_n50) / len(upgrade_n50)}, mean abundance: {sum(upgrade_abu) / len(upgrade_abu)}")
    # print(f"new mean N50: {sum(new_n50) / len(new_n50)}, mean abundance: {sum(new_abu) / len(new_abu)}")
    
    
    
    
    
    
    # ### get 2500 high quality N50, abundance
    # deeperbin_long_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-2500-all-info.tsv"
    # deeperbin_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/" # marine-sample-7-final-output-bins-2500
    # genome2id = readGenome2ID("/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/genome_to_id.tsv")
    
    
    # info_tsv_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/CAMI-Marine-DeeperBin2500-High.abundance.N50.tsv"
    # wh = open(info_tsv_path, 'w')
    # deeperbin_long_data_list = readAllInfo(deeperbin_long_path)
    # for i, cur_dict in enumerate(deeperbin_long_data_list):
    #     cur_deeperbin_folder = os.path.join(deeperbin_folder, f"marine-sample-{i}-final-output-bins-2500")
    #     id2abundance = readAbundance(f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/abundance{i}.tsv")
    #     for g_name, info in cur_dict.items():
    #         if info[4] == "HighQuality":
    #             contigname2seq = readFasta(os.path.join(cur_deeperbin_folder, info[0].split("-")[-1]+".fasta"))
    #             cur_n50 = calculateN50(contigname2seq)
    #             wh.write("\t".join([*list(info), "2500", "baseline", str(id2abundance[genome2id[g_name]]), str(cur_n50), str(len(contigname2seq))]) + "\n")
    
    # wh.close()
    
    
    
    
    
    
    
    # upgrade_list = []
    # new_list = []
    
    # with open(info_tsv_path, "r") as rh:
    #     for line in rh:
    #         info = line.strip("\n").split("\t")
    #         if info[6] == "upgrade":
    #             upgrade_list.append(info)
    #         else:
    #             new_list.append(info)
    
    # for i in range(1, len(upgrade_list), 2):
    #     pair_1 = upgrade_list[i - 1]
    #     pair_2 = upgrade_list[i]
    #     cur_sample_id1, pair1_name = pair_1[0].split("-")
    #     cur_sample_id2, pair2_name = pair_2[0].split("-")
    #     assert cur_sample_id1 == cur_sample_id2
    #     print("1-->", pair_1)
    #     print("2-->", pair_2)
        
    
    # for j in range(len(new_list)):
    #     pair_2 = new_list[j]
    #     cur_sample_id2, pair2_name = pair_2[0].split("-")
        
    
    
    # calculate the number of genes
    input_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-2500-vs-900-upgrade-new-pair-fasta-checkm2/protein_files"
    gene_res = {}
    for file_name in os.listdir(input_path):
        pre_fix, bin_suffix = os.path.splitext(file_name)
        if bin_suffix == ".faa":
            genename2seq = readFasta(os.path.join(input_path, file_name))
            c_gene_num = 0
            for genename, _ in genename2seq.items():
                if "partial=00" in genename:
                    c_gene_num += 1
            pair_index = "_".join(pre_fix.split("_")[0:2])
            pair_cat = pre_fix.split("_")[2]
            print(c_gene_num, pair_index, pair_cat)
            if pair_index not in gene_res:
                gene_res[pair_index] = {"2500": 0, "900": 0}
            if pair_cat == "2500":
                gene_res[pair_index]["2500"] += c_gene_num
            else:
                gene_res[pair_index]["900"] += c_gene_num
    
    gap_ratio = []
    gap_vals = []
    gap_900 = []
    gap_2500 = []
    posi_num = 0
    neg_num = 0
    for pair_index, values in gene_res.items():
        # print(pair_index)
        gene_num_2500 = float(values["2500"])
        gene_num_900 = float(values["900"])
        gap_val = gene_num_900 - gene_num_2500
        if gap_val > 0:
            posi_num += 1
        else:
            neg_num += 1
        if gene_num_2500 != 0:
            gap_ratio.append(gap_val / gene_num_2500 + 0.)
            gap_vals.append(gap_val)
            gap_900.append(gene_num_900)
            gap_2500.append(gene_num_2500)
        # else:
        #     gap_ratio.append(1. + 0.)
    
    # print(gap_ratio)
    print(f"{posi_num} upgrade quality MAGs has higher genes' number.")
    print(f"{neg_num} upgrade quality MAGs has lower genes' number.")
    print(f"The average increased gene percential across all paris is {sum(gap_ratio) / len(gap_ratio)}")
    print(f"The average increased gene number across all paris is {sum(gap_vals) / len(gap_vals)}")
    print(f"The total gene number in 900 is  {sum(gap_900)}. The total gene number in 2500 is {sum(gap_2500)}. The ratio of them is {sum(gap_900)/sum(gap_2500)}")
    
    
    
    # # write pair between 2500 and 900 fasta files
    # input_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-2500-vs-900-all-info-n50-abundance.tsv"
    # mag_folder_path =  "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/"
    # output_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-2500-vs-900-upgrade-new-pair-fasta"
    # if os.path.exists(output_folder) is False:
    #     os.mkdir(output_folder)
    
    # upgrade_list = []
    # new_list = []
    
    # with open(input_path, "r") as rh:
    #     for line in rh:
    #         info = line.strip("\n").split("\t")
    #         if info[6] == "upgrade":
    #             upgrade_list.append(info)
    #         else:
    #             new_list.append(info)
    
    # for i in range(1, len(upgrade_list), 2):
    #     pair_1 = upgrade_list[i - 1]
    #     pair_2 = upgrade_list[i]
    #     cur_sample_id1, pair1_name = pair_1[0].split("-")
    #     cur_sample_id2, pair2_name = pair_2[0].split("-")
    #     assert cur_sample_id1 == cur_sample_id2
    #     print("1-->", pair_1)
    #     print("2-->", pair_2)
    #     pair1_path = os.path.join(mag_folder_path, f"marine-sample-{cur_sample_id1}-final-output-bins-2500", pair1_name + ".fasta")
    #     pair2_path = os.path.join(mag_folder_path, f"marine-sample-{cur_sample_id2}-final-output-bins-900-no-multi-contrast", pair2_name + ".fasta")
        
    #     copy(pair1_path, os.path.join(output_folder, f"pair_{i}_2500.fasta"))
    #     copy(pair2_path, os.path.join(output_folder, f"pair_{i}_900.fasta"))
    
    # for j in range(len(new_list)):
    #     output_pair1_path = os.path.join(output_folder, f"pair_{i + j}_2500.fasta")
    #     with open(output_pair1_path, "w") as wh:
    #         pass
    #     pair_2 = new_list[j]
    #     cur_sample_id2, pair2_name = pair_2[0].split("-")
    #     pair2_path = os.path.join(mag_folder_path, f"marine-sample-{cur_sample_id2}-final-output-bins-900-no-multi-contrast", pair2_name + ".fasta")
    #     copy(pair2_path, os.path.join(output_folder, f"pair_{i + j}_900.fasta"))
    
    
    
    
    
    # #### write key information into the 'marine-2500-vs-900-all-info-n50-abundance.tsv' file
    # deeper_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-900-all-info.tsv"
    # deeperbin_long_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-2500-all-info.tsv"
    # mag_folder_path =  "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/"
    
    
    # output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/CAMI-Marine-DeeperBin2500-DeeperBin900.abundance.N50.tsv"
    
    # deeperbin_data_list = readAllInfo(deeper_path)
    # deeperbin_long_data_list = readAllInfo(deeperbin_long_path)
    # genome2id = readGenome2ID("/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/genome_to_id.tsv")
    
    # long_low_but_short_upgrade = 0
    # short_low_but_long_upgrade = 0
    # long_has_but_short_not = 0
    # short_has_but_long_not = 0
    
    # all_info_gather = []
    
    # for i in range(len(deeperbin_data_list)):
    #     cur_deeperbin_sample_dict = deeperbin_data_list[i]
    #     cur_deeperbin_long_sample_dict = deeperbin_long_data_list[i]
        
    #     cur_deeperbin_genomes = set(cur_deeperbin_sample_dict.keys())
    #     cur_deeperbin_long_genomes = set(cur_deeperbin_long_sample_dict.keys())
        
    #     cur_info_gather = []
    #     id2abundance = readAbundance(f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/abundance{i}.tsv")
        
    #     # long vs short
    #     for genome_name, values in cur_deeperbin_long_sample_dict.items():
    #         if values[4] == "LowQuality":
    #             if genome_name in cur_deeperbin_sample_dict:
    #                 short_info = cur_deeperbin_sample_dict[genome_name]
    #                 if short_info[4] != "LowQuality":
    #                     long_low_but_short_upgrade += 1
                        
    #                     mag_name = values[0].split("-")[-1]
    #                     cur_mag_path = os.path.join(mag_folder_path, f"marine-sample-{i}-final-output-bins-2500", mag_name + ".fasta")
    #                     cur_contigs = readFasta(cur_mag_path)
    #                     cur_n50 = calculateN50(readFasta(cur_mag_path))
    #                     cur_abun = id2abundance[genome2id[values[5]]]
                        
    #                     cur_output = f"\t".join(values) + "\t" + "\t".join(["2500", "upgrade", str(cur_n50), str(cur_abun), str(len(cur_contigs))]) + "\n"
    #                     cur_info_gather.append(cur_output)
                        
    #                     mag_name = short_info[0].split("-")[-1]
    #                     cur_mag_path = os.path.join(mag_folder_path, f"marine-sample-{i}-final-output-bins-900-no-multi-contrast", mag_name + ".fasta")
    #                     cur_contigs = readFasta(cur_mag_path)
    #                     cur_n50 = calculateN50(readFasta(cur_mag_path))
    #                     cur_abun = id2abundance[genome2id[short_info[5]]]
                        
    #                     cur_output = f"\t".join(short_info)+ "\t" + "\t".join(["900", "upgrade", str(cur_n50), str(cur_abun), str(len(cur_contigs))]) + "\n"
    #                     cur_info_gather.append(cur_output)
                        
                        
                        
                        
    #         elif values[4] == "MediumQuality":
    #             if genome_name in cur_deeperbin_sample_dict:
    #                 short_info = cur_deeperbin_sample_dict[genome_name]
    #                 if short_info[4] == "HighQuality":
    #                     long_low_but_short_upgrade += 1
                        
                        
    #                     mag_name = values[0].split("-")[-1]
    #                     cur_mag_path = os.path.join(mag_folder_path, f"marine-sample-{i}-final-output-bins-2500", mag_name + ".fasta")
    #                     cur_contigs = readFasta(cur_mag_path)
    #                     cur_n50 = calculateN50(readFasta(cur_mag_path))
    #                     cur_abun = id2abundance[genome2id[values[5]]]
                        
    #                     cur_output = f"\t".join(values) + "\t" + "\t".join(["2500", "upgrade" ,str(cur_n50), str(cur_abun), str(len(cur_contigs))]) + "\n"
    #                     cur_info_gather.append(cur_output)
                        
    #                     mag_name = short_info[0].split("-")[-1]
    #                     cur_mag_path = os.path.join(mag_folder_path, f"marine-sample-{i}-final-output-bins-900-no-multi-contrast", mag_name + ".fasta")
                        
    #                     cur_n50 = calculateN50(readFasta(cur_mag_path))
    #                     cur_contigs = readFasta(cur_mag_path)
    #                     cur_abun = id2abundance[genome2id[short_info[5]]]
                        
    #                     cur_output = f"\t".join(short_info) + "\t" + "\t".join(["900", "upgrade", str(cur_n50), str(cur_abun), str(len(cur_contigs))]) + "\n"
    #                     cur_info_gather.append(cur_output)
                        
                        
                        
    #             else:
    #                 long_has_but_short_not += 1
    #         else:
    #             if genome_name not in cur_deeperbin_sample_dict:
    #                 long_has_but_short_not += 1
        
    #     # short vs long
    #     for genome_name, values in cur_deeperbin_sample_dict.items():
    #         if values[4] == "LowQuality":
    #             if genome_name in cur_deeperbin_long_sample_dict:
    #                 long_info = cur_deeperbin_long_sample_dict[genome_name]
    #                 if long_info[4] != "LowQuality":
    #                     short_low_but_long_upgrade += 1
    #         elif values[4] == "MediumQuality":
    #             if genome_name in cur_deeperbin_long_sample_dict:
    #                 long_info = cur_deeperbin_long_sample_dict[genome_name]
    #                 if long_info[4] == "HighQuality":
    #                     short_low_but_long_upgrade += 1
    #             else:
    #                 short_has_but_long_not += 1
                    
    #                 mag_name = values[0].split("-")[-1]
    #                 cur_mag_path = os.path.join(mag_folder_path, f"marine-sample-{i}-final-output-bins-900-no-multi-contrast", mag_name + ".fasta")
    #                 cur_contigs = readFasta(cur_mag_path)
    #                 cur_n50 = calculateN50(readFasta(cur_mag_path))
    #                 cur_abun = id2abundance[genome2id[values[5]]]
                    
    #                 cur_output = f"\t".join(values)+ "\t" + "\t".join(["900", "new" ,str(cur_n50), str(cur_abun), str(len(cur_contigs))]) + "\n"
    #                 cur_info_gather.append(cur_output)
                    
    #         else:
    #             if genome_name not in cur_deeperbin_long_sample_dict:
    #                 short_has_but_long_not += 1
                    
    #                 mag_name = values[0].split("-")[-1]
    #                 cur_mag_path = os.path.join(mag_folder_path, f"marine-sample-{i}-final-output-bins-900-no-multi-contrast", mag_name + ".fasta")
    #                 cur_n50 = calculateN50(readFasta(cur_mag_path))
    #                 cur_contigs = readFasta(cur_mag_path)
    #                 cur_abun = id2abundance[genome2id[values[5]]]
                    
    #                 cur_output = f"\t".join(values)+ "\t" + "\t".join(["900", "new" ,str(cur_n50), str(cur_abun), str(len(cur_contigs))]) + "\n"
    #                 cur_info_gather.append(cur_output)
        
    #     all_info_gather.append(cur_info_gather)
    
    # wh = open(output_path, "w")
    # for cur_info in all_info_gather:
    #     for item in cur_info:
    #         wh.write(item)
    # wh.close()
    
    
    
    
    
    
    # # 2500 vs 900 MAG upgrade and can get ref genome that not in 2500
    # deeper_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-900-all-info.tsv"
    # deeperbin_long_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/DeeperBin-2500-all-info.tsv"
    # output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/re_pre.tsv"
    
    # deeperbin_data_list = readAllInfo(deeper_path)
    # deeperbin_long_data_list = readAllInfo(deeperbin_long_path)
    
    # long_low_but_short_upgrade = 0
    # short_low_but_long_upgrade = 0
    # long_has_but_short_not = 0
    # short_has_but_long_not = 0
    
    # for i in range(len(deeperbin_data_list)):
    #     cur_deeperbin_sample_dict = deeperbin_data_list[i]
    #     cur_deeperbin_long_sample_dict = deeperbin_long_data_list[i]
        
    #     cur_deeperbin_genomes = set(cur_deeperbin_sample_dict.keys())
    #     cur_deeperbin_long_genomes = set(cur_deeperbin_long_sample_dict.keys())
        
    #     # long vs short
    #     for genome_name, values in cur_deeperbin_long_sample_dict.items():
    #         if values[4] == "LowQuality":
    #             if genome_name in cur_deeperbin_sample_dict:
    #                 short_info = cur_deeperbin_sample_dict[genome_name]
    #                 if short_info[4] != "LowQuality":
    #                     long_low_but_short_upgrade += 1
    #         elif values[4] == "MediumQuality":
    #             if genome_name in cur_deeperbin_sample_dict:
    #                 short_info = cur_deeperbin_sample_dict[genome_name]
    #                 if short_info[4] == "HighQuality":
    #                     long_low_but_short_upgrade += 1
    #             else:
    #                 long_has_but_short_not += 1
    #         else:
    #             if genome_name not in cur_deeperbin_sample_dict:
    #                 long_has_but_short_not += 1
        
    #     # short vs long
    #     for genome_name, values in cur_deeperbin_sample_dict.items():
    #         if values[4] == "LowQuality":
    #             if genome_name in cur_deeperbin_long_sample_dict:
    #                 long_info = cur_deeperbin_long_sample_dict[genome_name]
    #                 if long_info[4] != "LowQuality":
    #                     short_low_but_long_upgrade += 1
    #         elif values[4] == "MediumQuality":
    #             if genome_name in cur_deeperbin_long_sample_dict:
    #                 long_info = cur_deeperbin_long_sample_dict[genome_name]
    #                 if long_info[4] == "HighQuality":
    #                     short_low_but_long_upgrade += 1
    #             else:
    #                 short_has_but_long_not += 1
    #         else:
    #             if genome_name not in cur_deeperbin_long_sample_dict:
    #                 short_has_but_long_not += 1
    
    # print(f"2500 low quality but 900 upgrade num: {long_low_but_short_upgrade}, 2500 not has the genome but 900 has: {short_has_but_long_not}")
    # print(f"900 low quality but 2500 upgrade num: {short_low_but_long_upgrade}, 900 not has the genome but 2500 has: {long_has_but_short_not}")
    
    
    
    
    
    # ### analysis the number of  short vs long contigs ratio
    # contig_file_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Marine-contigs-bam"
    # output_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/plant-bps-con_num.csv"
    # wh = open(output_path, "a")
    # summed = 0.
    # for i  in range(10):
    #     min_contig_length = 900
    #     cur_contig_file_path = f"{contig_file_path}/marine-sample-{i}.contigs.fasta"
    #     contigname2seq_ori = readFasta(cur_contig_file_path)
        
    #     long_summed = 0.
    #     short_summed = 0.
        
    #     long_contig_num = 0
    #     short_contig_num = 0
        
    #     for _, seq in contigname2seq_ori.items():
    #         n = len(seq)
    #         if  min_contig_length <= n < 1000:
    #             short_summed += n
    #             short_contig_num += 1
    #         elif 1000 <= n:
    #             long_summed += n
    #             long_contig_num += 1

    #     print(f"{short_summed / long_summed + 0.},{short_contig_num / long_contig_num + 0.}")
    #     summed += short_contig_num / long_contig_num + 0.
    #     # wh.write(f"{short_summed / long_summed + 0.},{short_contig_num / long_contig_num + 0.}\n")
        
    # wh.close()
    # print(summed / 10.)
        # print(f"min length: {min_contig_length}, Long summed: {long_summed}, short summed: {short_summed}, ratio: {short_summed / long_summed + 0.}")
        # print(f"short contig num {short_contig_num}, long contig num {long_contig_num}, ratio {short_contig_num / long_contig_num + 0.}")

