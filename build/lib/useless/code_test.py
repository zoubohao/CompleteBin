
import torch
import numpy as np

from Src.IO import readPickle

if __name__ == "__main__":
    path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin/Comebin-marine/comebin-marine-sample-0-outputs/data_augmentation/aug0_datacoverage_mean.tsv"
    res = {}
    with open(path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                i += 1
                continue
            info = line.strip("\n").split("\t")
            res[info[0]] = float(info[1])
    print(max(list(res.values())))
    path2 = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/previous-results/DeeperBin-CAMI2-marine-1.0.8/marine-sample-0-temp-folder-2500/contigname2bpcover_nparray_list.pkl"
    res2 = readPickle(path2)
    for key, val in res.items():
        key = ">" + key
        if key in res2:
            print(f"COMEBin: {val}")
            cur_bp_array = res2[key][0]
            print(f"DeeperBin: {sum(cur_bp_array) / len(cur_bp_array)}")
    


