
from multiprocessing import Process
import os


def cal_depth(sorted_bam_path, out_depth_file_path):
    cmd = f"jgi_summarize_bam_contig_depths --outputDepth {out_depth_file_path} {sorted_bam_path[0]}"
    os.system(cmd)


def cal_depth_multi(sorted_bam_path_list, out_depth_file_path):
    multi_str = " "
    for bam_file_path in sorted_bam_path_list:
        multi_str += f" {bam_file_path} "
    cmd = f"jgi_summarize_bam_contig_depths {multi_str} --outputDepth {out_depth_file_path}"
    print(cmd)
    os.system(cmd)


def convertMetabat2CONCOCT(depth_file_path, out_depth_path, multi_sample):
    if not multi_sample:
        with open(depth_file_path, "r") as rh, open(out_depth_path, "w") as wh:
            for line in rh:
                info = line.strip("\n").split("\t")
                wh.write("\t".join([info[0], info[3]]) + "\n")
    else:
        with open(depth_file_path, "r") as rh, open(out_depth_path, "w") as wh:
            for line in rh:
                info = line.strip("\n").split("\t")
                num_col = len(info)
                cur_info = []
                for j in range(3, num_col, 2):
                    cur_info.append(info[j])
                wh.write("\t".join([info[0]] + cur_info) + "\n")


def runCONCOCT(
    ori_contig_fasta,
    depth_file_path,
    output_folder,
    num_cpu,
    multi_sample
):
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)
    output_cov_tsv = os.path.join(output_folder, 'coverage_table.tsv')
    output_clust_1000 = os.path.join(output_folder, 'clustering_gt1000.csv')
    output_merged = os.path.join(output_folder, 'clustering_merged.csv')
    bins_folder = os.path.join(output_folder, "output-bins")
    if os.path.exists(bins_folder) is False:
        os.mkdir(bins_folder)
    
    convertMetabat2CONCOCT(depth_file_path, output_cov_tsv, multi_sample)
    
    cmd3 = f"concoct --composition_file {ori_contig_fasta} --coverage_file {output_cov_tsv} -b {output_folder} -t {num_cpu} -l 1000 -i 200"
    print("CONCOCT Step 1.")
    os.system(cmd3)
    
    cmd4 = f"merge_cutup_clustering.py {output_clust_1000} > {output_merged}"
    print("CONCOCT Step 2.")
    os.system(cmd4)
    
    cmd5 = f"extract_fasta_bins.py {ori_contig_fasta} {output_merged} --output_path {bins_folder}"
    print("CONCOCT Step 3.")
    os.system(cmd5)


def bin_concoct(
    contigs_path: str,
    sorted_sorted_bam_file: str,
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
                
    p = Process(target=runCONCOCT, args=(contigs_path, cur_depth_path, res_output_path, num_cpu, multi_sample))
    p.start()
    p.join()


if __name__ == "__main__":
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Dingyi-short-long-reads-data/"
    out_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CONCOCT/Dingyi-data/"
    if os.path.exists(out_folder) is False:
        os.mkdir(out_folder)
    
    for id_name in ["SRR13060973", "SRR13060977"]:
        cur_bin_output_folder = os.path.join(out_folder, id_name)
        cur_contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        depth_file_path = os.path.join(cur_bin_output_folder, "depth.file")
        bam_list = []
        bam_list.append(os.path.join(home_path, f"{id_name}.sorted.bam"))
        bin_concoct(cur_contig_path, bam_list, cur_bin_output_folder, depth_file_path, 86, multi_sample=False)


# if __name__ == "__main__":
#     home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Plant-contigs-bam"
    
#     for i in range(21):
#         id_name = f"plant-sample-{i}"
#         outputpath = os.path.join(f"../CONCOCT-plant/{id_name}/")
#         if os.path.exists(outputpath) is False:
#             os.mkdir(outputpath)
#         contig_file_path=os.path.join(home_path, f"{id_name}.contigs.fasta")
#         sorted_bam_file_path=os.path.join(home_path, f"{id_name}.sorted.bam")
#         depth_file_path = os.path.join(outputpath, "depth.file")
#         bin_concoct(contig_file_path, sorted_bam_file_path,
#                      outputpath,
#                      depth_file_path,
#                      86)