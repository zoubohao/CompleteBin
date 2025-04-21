
import os

from Src.IO import readCheckm2Res

if __name__ == "__main__":

    home_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Comebin-marine-multi-sample"
    for i in range(10):
        i = 9
        cur_checkm_output_folder = os.path.join(home_folder, f"marine-sample-{i}-checkm2")
        d, h, m, l = readCheckm2Res(os.path.join(cur_checkm_output_folder, "quality_report.tsv"), "fasta")
        print(f"marine-sample-{i}-checkm2 --> high: {h}, medium: {m}")
        break
    
    
    
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