
import os
from Src.IO import readCheckm2Res, readMetaInfo


if __name__ == "__main__":

    home_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin/CAMI2-strain-madness"
    files = os.listdir(home_folder)
    metainfo = False
    special_str = "checkm2"
    sample = ""
    
    summed_h = 0
    summed_med = 0
    for i in files:
        if sample in i and special_str in i:
            cur_checkm_output_folder = os.path.join(home_folder, i)
            if metainfo:
                d, h, m, l = readMetaInfo(os.path.join(cur_checkm_output_folder, "MetaInfo.tsv"), 2, 3)
            else:
                d, h, m, l = readCheckm2Res(os.path.join(cur_checkm_output_folder, "quality_report.tsv"), "fa")
            print(f"{i}-checkm2 --> high: {h}, medium: {m}")
            summed_h += h
            summed_med += m
    print(summed_h, summed_med)
    
    # home_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin/CAMI2-strain-madness"
    # summed_val = 0
    # summed_m = 0
    # for i in range(10):
    #     cur_checkm_output_folder = os.path.join(home_folder, f"{i}-COMEBin-checkm2", "quality_report.tsv")
    #     d, h, m, l = readCheckm2Res(cur_checkm_output_folder, )
    #     print(f"sample-{i}-checkm2 --> high: {h}, medium: {m}")
    #     summed_val += h
    #     summed_m += h + m
    # print(f"COMEBin, high: {summed_val}; high + medium: {summed_m}")
    
    # gunc_res = readGUNC("/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI_medium_1_final_bin_output_28_nc_bin_gunc/GUNC.progenomes_2.1.maxCSS_level.tsv")
    # i = 0
    # score = 0
    # for k, v in d.items():
    #     if  v[-1] == "HighQuality":
    #         # print(k, v)
    #         i += 1
    #     if v[-1] != "LowQuality":
    #         print(k, v)
    #         score += v[0] - v[1] * 5
    #     # if v[0] >= 90:
    #     #     print(k, v)
    # print(f"high: {h}, medium: {m}, low: {l}, Summed Score: {score}")