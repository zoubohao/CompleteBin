

import os

import numpy as np

from Src.IO import readCheckm2Res

if __name__ == "__main__":
    
    data_name = "marine"
    bin_tool = "DeeperBin"
    samples_num = 10
    data_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine/"
    meta_data_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-marine/"
    info_out_path = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-DeeperBin-vs-COMEBin/{bin_tool}-900-all-info.tsv"
    
    bin2f1 = {}
    bin2quality = {}
    bin2genome = {}
    wh = open(info_out_path, "w")
    
    for i in range(samples_num):
        
        output_bin2f1_file = f"{data_folder}{data_name}-sample-{i}-{bin_tool}.900.bin2f1.tsv"
        metainfo_file = f"{meta_data_folder}{data_name}-sample-{i}-final-output-bins-900-no-multi-contrast/MetaInfo.tsv"
        
        # output_bin2f1_file = f"{data_folder}{data_name}-sample-{i}-{bin_tool}.1000.bin2f1.tsv"
        # metainfo_file = f"{meta_data_folder}{data_name}-sample-{i}-final-output-bins-1000bps/MetaInfo.tsv"
        # metainfo_file = f"{meta_data_folder}comebin-{data_name}-sample-{i}-checkm2/quality_report.tsv"
        
        with open(output_bin2f1_file, "r") as rh:
            for line in rh:
                line = line.strip("\n").split("\t")
                # print(line)
                bin2f1[f"{i}-{line[0]}"] = (line[1], line[2], line[3])
                bin2genome[f"{i}-{line[0]}"] = line[4]
        
        if bin_tool == "DeeperBin":
            with open(metainfo_file, "r") as rh:
                for line in rh:
                    line = line.strip("\n").split("\t")
                    prefix, _ = os.path.splitext(line[0])
                    bin2quality[f"{i}-{prefix}"] = line[-1]
        else:
            cur_checkm2, _, _, _ = readCheckm2Res(metainfo_file, "fasta")
            for k, v in cur_checkm2.items():
                prefix, _ = os.path.splitext(k)
                bin2quality[f"{i}-{prefix}"] = v[-1]
            
    for bin_name, values in bin2f1.items():
        wh.write(bin_name + "\t" + "\t".join(values) + "\t" + bin2quality[bin_name] + "\t" + bin2genome[bin_name] + "\n")
    
    wh.close()
    
    ### try to figure out the relationship between F1 and checkm2 
    # gaps = np.arange(0.5, 1.05, 0.05)
    # gaps_len = len(gaps)
    # res = {}
    # for j in range(gaps_len - 1):
    #     res[gaps[j]] = 0
    # count = 0
    # for binname, f1 in bin2f1.items():
    #     quality = bin2quality[binname]
    #     if quality != "HighQuality":
    #         continue
    #     count += 1
    #     for j in range(gaps_len - 1):
    #         if gaps[j] <= f1 and f1 < gaps[j + 1]:
    #             res[gaps[j]] += 1
    #             break
    
    # print(res, count)


