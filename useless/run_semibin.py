

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def runSemibin(input_contigs_path,
            bam_file: str,
            output_folder,
            num_cpu):
    
    cmd = f"SemiBin2 single_easy_bin --threads {num_cpu} -i {input_contigs_path} -o {output_folder} -b {bam_file} " + \
        " --compression none"
    os.system(cmd)


def runSemibin_multi(input_contigs_path,
                    bam_file_list,
                    output_folder,
                    num_cpu):
    multi_str = " "
    for bam_file_path in bam_file_list:
        multi_str += f" {bam_file_path} "
    cmd = f"SemiBin2 single_easy_bin --threads {num_cpu} -i {input_contigs_path} -o {output_folder} -b {multi_str} " + \
        " --compression none"
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Dingyi-short-long-reads-data/"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Semibin2/Dingyi-data/"
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    for id_name in ["SRR13060973", "SRR13060977"]:
        # id_name = f"marine-sample-{i}"
        cur_bin_output_folder = os.path.join(out_folder, id_name)
        cur_contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        runSemibin(cur_contig_path, os.path.join(home_path, f"{id_name}.sorted.bam"), cur_bin_output_folder, 86)



# if __name__ == "__main__":
#     home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Plant-contigs-bam"
    
#     for i in range(10, 21):
#         id_name = f"plant-sample-{i}"
#         outputpath = os.path.join(f"../Semibin2-plant/{id_name}/")
#         if os.path.exists(outputpath) is False:
#             os.mkdir(outputpath)
#         contig_file_path=os.path.join(home_path, f"{id_name}.contigs.fasta")
#         sorted_bam_file_path=os.path.join(home_path, f"{id_name}.sorted.bam")
#         runSemibin(contig_file_path, sorted_bam_file_path,
#                      outputpath,
#                      64)
