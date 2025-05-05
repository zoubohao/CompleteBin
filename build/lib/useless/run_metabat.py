from multiprocessing import Process
import os
from time import sleep


def cal_depth(sorted_bam_path, out_depth_file_path):
    cmd = f"jgi_summarize_bam_contig_depths --outputDepth {out_depth_file_path} {sorted_bam_path[0]}"
    os.system(cmd)


def cal_depth_multi(sorted_bam_path_list, out_depth_file_path):
    multi_str = " "
    for bam_file_path in sorted_bam_path_list:
        multi_str += f" {bam_file_path} "
    cmd = f"jgi_summarize_bam_contig_depths {multi_str} --outputDepth {out_depth_file_path}"
    # print(cmd)
    print(os.system(cmd))


def runMetaBAT2(
    ori_contig_fasta,
    depth_file_path,
    bins_path,
    num_cpu
):
    cmd = f"metabat2  -o {bins_path} -t {num_cpu} --inFile {ori_contig_fasta}  --abdFile {depth_file_path} -m 1500"
    print(cmd)
    os.system(cmd)


def bin_metabat2(
    contigs_path: str,
    sorted_sorted_bam_file,
    res_output_path,
    depth_path,
    num_cpu,
    multi_sample = True
):
    if os.path.exists(res_output_path) is False:
        os.mkdir(res_output_path)
    if depth_path is not None and os.path.exists(depth_path) is False:
        if not multi_sample:
            cal_depth(sorted_sorted_bam_file, depth_path)
        else:
            cal_depth_multi(sorted_sorted_bam_file, depth_path)
    cur_depth_path = os.path.join(res_output_path, "cur_depth.txt")
    bins_path = os.path.join(res_output_path, "output-bins")
    first_line = ""
    ori_depth_info = {}
    with open(depth_path, "r") as rh:
        i = 0
        for line in rh:
            if i == 0:
                first_line = line
            else:
                info = line.strip("\n").split("\t")
                ori_depth_info[info[0]] = info[1:]
            i += 1
    
    with open(contigs_path, "r") as rh, open(cur_depth_path, "w") as wh:
        wh.write(first_line)
        for line in rh:
            if line[0] == ">":
                contig_name = line.strip("\n")[1:].split()[0]
                wh.write("\t".join([contig_name] + ori_depth_info[contig_name]) + "\n")
    p = Process(target=runMetaBAT2, args=(contigs_path, cur_depth_path, bins_path, num_cpu))
    p.start()
    p.join()



if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Dingyi-short-long-reads-data/"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/MetaBAT2/Dingyi-data/"
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    for id_name in ["SRR13060973", "SRR13060977"]:
        # id_name = f"marine-sample-{i}"
        cur_bin_output_folder = os.path.join(out_folder, id_name)
        cur_contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        depth_file_path = os.path.join(cur_bin_output_folder, "depth.file")
        bam_list = []
        bam_list.append(os.path.join(home_path, f"{id_name}.sorted.bam"))
        bin_metabat2(cur_contig_path, bam_list, cur_bin_output_folder, depth_file_path, 86, multi_sample=False)




# if __name__ == "__main__":
#     home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Plant-contigs-bam"
    
#     for i in range(21):
#         id_name = f"plant-sample-{i}"
#         outputpath = os.path.join(f"../MetaBAT2-plant/{id_name}/")
#         if os.path.exists(outputpath) is False:
#             os.mkdir(outputpath)
#         contig_file_path=os.path.join(home_path, f"{id_name}.contigs.fasta")
#         sorted_bam_file_path=os.path.join(home_path, f"{id_name}.sorted.bam")
#         depth_file_path = os.path.join(outputpath, "depth.file")
#         bin_metabat2(contig_file_path, sorted_bam_file_path,
#                      outputpath,
#                      depth_file_path,
#                      86)