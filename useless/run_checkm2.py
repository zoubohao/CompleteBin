
import os

from CompleteBin.IO import readCheckm2Res, readMetaInfo


def runCheckm2Single(
    input_bin_folder: str,
    output_bin_folder: str,
    bin_suffix: str,
    db_path,
    num_cpu: int):
    if os.path.exists(output_bin_folder) is False:
        os.makedirs(output_bin_folder)
    cmd = f"checkm2 predict -x {bin_suffix} --threads {num_cpu} -i {input_bin_folder} -o {output_bin_folder} --database_path {db_path}"
    os.system(cmd)


if __name__ == "__main__":
    
    # folder_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeeperBin-CAMI2-other-datasets/"
    
    # res = {}
    # for folder in os.listdir(folder_path):
    #     if "-final-output-bins" in folder:
    #         cur_data_set = folder.split("-")[0]
    #         _, h, _, _ = readMetaInfo(os.path.join(folder_path, folder, "MetaInfo.tsv"))
    #         if cur_data_set not in res:
    #             res[cur_data_set] = h
    #         else:
    #             res[cur_data_set] += h
    
    # print(res)
    
    
    for i in ["SRR13060973", "SRR13060977"]:
        bin_folder = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin/{i}-COMEBin-checkm2/quality_report.tsv"
        _, h, m, l = readCheckm2Res(bin_folder, "fa")
        print(f"i: {i}, H: {h}, M: {m}")
    
    
    # home_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/COMEBin"
    # for i in ["SRR13060973", "SRR13060977"]:
    #     cur_input_folder = os.path.join(home_folder, f"{i}-COMEBin-bins")
    #     cur_checkm_output_folder = os.path.join(home_folder, f"{i}-COMEBin-checkm2")
    #     runCheckm2Single(cur_input_folder, cur_checkm_output_folder, "fa", "./DeeperBin-DB/checkm/checkm2_db.dmnd", 128)