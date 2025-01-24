
import os


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
    home_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Comebin-marine-multi-sample"
    for i in range(10):
        cur_input_folder = os.path.join(home_folder, f"marine-sample-{i}-outputs", "comebin_res", "comebin_res_bins")
        cur_checkm_output_folder = os.path.join(home_folder, f"marine-sample-{i}-checkm2")
        runCheckm2Single(cur_input_folder, cur_checkm_output_folder, "fa", "./DeepMetaBin-DB/checkm/checkm2_db.dmnd", 86)