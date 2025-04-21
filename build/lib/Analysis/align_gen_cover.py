

import os
# from DeeperBin.IO import readFasta, readPickle, writePickle, readCheckm2Res, readMetaInfo
import numpy as np

def get_ref_genome_length(ref_fasta_path: str):
    contig2length = {}
    curContig = ""
    curSeq = ""
    with open(ref_fasta_path, mode="r") as rh:
        for line in rh:
            curLine = line.strip("\n")
            if curLine[0] == ">":
                if "plasmid" not in curContig.lower():
                    contig2length[curContig] = len(curSeq.upper())
                    curContig = curLine
                curSeq = ""
            else:
                curSeq += curLine
    if "plasmid" not in curContig.lower():
        contig2length[curContig] = len(curSeq.upper())
    contig2length.pop("")
    genome2length = {}
    for contig, length in contig2length.items():
        genome_name = contig.split("|")[0][1:]
        if genome_name not in genome2length:
            genome2length[genome_name] = length
        else:
            genome2length[genome_name] += length
    return genome2length


def convert_paf2biobox(
    align_paf_path: str,
    output_path: str, 
    genome2length: dict):
    write_handler = open(output_path, "w")
    write_handler.write(f"SEQUENCEID\tSEQ_LENGTH\tGENOMEID\tGEN_LENGTH\n")
    with open(align_paf_path, "r") as rh:
        for line in rh:
            oneline = line.strip("\n").split("\t")
            genome_name = oneline[5].split("|")[0]
            write_handler.write(oneline[0] + "\t"  + oneline[1] + "\t"  + genome_name + "\t"  + str(genome2length[genome_name]) + "\n")
    write_handler.close()


# def calculate_genome_coverage(
#     bin_fasta_path: str,
#     sample_biobox_path: str,
# ):
#     contigname2seq = readFasta(bin_fasta_path)
#     contigname2seq_new = {}
#     total_length = 0
#     for contigname, seq in contigname2seq.items():
#         contigname = contigname.split()[0]
#         contigname2seq_new[contigname] = seq
#         total_length += len(seq)
#     contigname2seq = contigname2seq_new
#     name2aligned_genome = {}
#     with open(sample_biobox_path, "r") as rh:
#         for line in rh:
#             info = line.strip("\n").split("\t")
#             seq_id = info[0]
#             if seq_id not in name2aligned_genome:
#                 name2aligned_genome[">" + seq_id] = ["|".join(info[2:])]
#             else:
#                 name2aligned_genome[">" + seq_id].append("|".join(info[2:]))
#     stat_genome_length = {}
#     for contigname, seq in contigname2seq.items():
#         if contigname not in name2aligned_genome:
#             continue
#         cur_aligned_genomes = name2aligned_genome[contigname]
#         for genome_name in cur_aligned_genomes:
#             if genome_name not in stat_genome_length:
#                 stat_genome_length[genome_name] = len(seq)
#             else:
#                 stat_genome_length[genome_name] += len(seq)
#     stat_list = []
#     for genome_name, mag_align_len in stat_genome_length.items():
#         stat_list.append((mag_align_len, int(genome_name.split("|")[1]), genome_name))
#     sorted_stat_list = list(sorted(stat_list, key=lambda x: x[0], reverse=True))
#     # print(sorted_stat_list)
#     mag_bp = sorted_stat_list[0][0]
#     genome_bp = sorted_stat_list[0][1]
#     coverage = mag_bp / genome_bp + 0.0 
#     # contamination = (total_length - mag_bp) / total_length
#     return coverage, sorted_stat_list[0]



def calculate_genome_coverage_accross_sample(
    category: str,
    sample_id: int,
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/",
    quality_state = "All" # high, medium
):
    binning_tools = ["DeepShortBin", "CONCOCT", "MetaBAT2", "Semibin2", "Comebin"]
    all_genome_name_set = set()
    bin_cov_res = []
    for binning_name in binning_tools:
        cur_sample_res = {}
        cur_cov_path = os.path.join(home_path, f"{binning_name}-{category}-multi-sample", f"{category}-sample-{sample_id}-genome-coverage.tsv")
        i = 0
        with open(cur_cov_path, "r") as rh:
            for line in rh:
                if i == 0:
                    i += 1
                    continue
                _, coverage, _, _, genome_name, mag_quality = line.strip("\n").split("\t")
                if quality_state.lower() == "all":
                    if genome_name not in cur_sample_res:
                        cur_sample_res[genome_name] = [float(coverage)]
                    else:
                        cur_sample_res[genome_name].append(float(coverage))
                elif quality_state.lower() == "high" and mag_quality == "HighQuality":
                    if genome_name not in cur_sample_res:
                        cur_sample_res[genome_name] = [float(coverage)]
                    else:
                        cur_sample_res[genome_name].append(float(coverage))
                elif quality_state.lower() == "medium" and mag_quality == "MediumQuality":
                    if genome_name not in cur_sample_res:
                        cur_sample_res[genome_name] = [float(coverage)]
                    else:
                        cur_sample_res[genome_name].append(float(coverage))
        bin_cov_res.append(cur_sample_res)
    final_res = []
    
    for i, cur_bin_cov_dict in enumerate(bin_cov_res):
        summed_coverage = 0.
        for genome_name, coverage_list in cur_bin_cov_dict.items():
            all_genome_name_set.add(genome_name)
            summed_coverage += list(sorted(coverage_list, reverse=True))[0]
        final_res.append(summed_coverage)
    
    # print(final_res, all_genome_name_set, len(all_genome_name_set))
    final_res = np.array(final_res) / len(all_genome_name_set)
    print(final_res)
    return final_res, bin_cov_res, all_genome_name_set






if __name__ == "__main__":
    
    # all_genomes_find = set()
    # res = [set() for _ in range(5)]
    # for i in range(10):
    #     _, cur_cov_res, all_genomes_set = calculate_genome_coverage_accross_sample(
    #     "marine",
    #     i,
    #     quality_state="all"
    #     )
    #     for j, cur_bin_cov_dict in enumerate(cur_cov_res):
    #         for genome_name, coverage_list in cur_bin_cov_dict.items():
    #             res[j].add(genome_name)
    #             all_genomes_find.add(genome_name)
    # n = len(all_genomes_find)
    # for r in res:
    #     r = len(r)
    #     print(r, n, r / n + 0.)
    
    
    
    # num_samples = 21
    # bin_suffix = "fasta"
    # category = "plant"
    # home_folder_path = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeepShortBin-{category}/"
    
    # for i in range(num_samples):
    #     id_name = f"{category}-sample-{i}"
    #     print(f"{id_name}")
        
    #     # quality_path = os.path.join(home_folder_path, f"{id_name}-checkm2", "quality_report.tsv")
    #     quality_path = os.path.join(home_folder_path, f"{id_name}-750-v2.1.2_final_bin_output", "MetaInfo.tsv")
        
    #     # fasta_folder = os.path.join(home_folder_path, f"{id_name}", "output-bins")
    #     fasta_folder = os.path.join(home_folder_path, f"{id_name}-750-v2.1.2_final_bin_output")
        
    #     sample_biobox_path = f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/{category}-align-contigs-2-ref-genome/{id_name}-biobox.tsv"
        
    #     cur_output_path = os.path.join(home_folder_path, f"{id_name}-genome-coverage.tsv")
    #     wh = open(cur_output_path, "w")
    #     wh.write("MAGName\tGenomeCoverage\tMAGAlignedCoreGenomeLength\tAlignedGenomeLength\tAlignedGenomeName\tGenomeQuality\n")
        
    #     if os.path.exists(quality_path) is False:
    #         print("Not find quality file")
    #         wh.close()
    #         continue
        
    #     # quality_dict, _, _, _ = readCheckm2Res(quality_path, bin_suffix)
    #     quality_dict, _, _, _ = readMetaInfo(quality_path)
        
    #     for file_name, quality_state in quality_dict.items():
    #         if quality_state[-1] != "LowQuality":
    #             cur_fasta_path = os.path.join(fasta_folder, file_name)
    #             cur_genome_coverage, cur_genome_state = calculate_genome_coverage(cur_fasta_path, sample_biobox_path)
    #             wh.write(file_name + "\t" + str(cur_genome_coverage) \
    #                 + "\t" + str(cur_genome_state[0]) + "\t" + str(cur_genome_state[1]) + "\t" + str(cur_genome_state[2]) \
    #                 + "\t" + quality_state[-1] + "\n")
        
    #     # wh.write(str(sum(record_list) / len(record_list) + 0.0))
    #     # print(f"Normalized Weighted Coverage: {weighted_score / k}")
    #     wh.close()
    
    
    ### test case
    # sample_biobox_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/marine-align-contigs-2-ref-genome/marine-sample-0-biobox.tsv"
    # test_fasta_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/DeepShortBin-marine/marine-sample-0-768-v2.1.0_final_bin_output/DeepMetaBin_6.fasta"
    # calculate_genome_coverage(test_fasta_path, sample_biobox_path)
    
    ### align the contig to reference genomes
    output_folder = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine"
    home_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/CAMI-Marine-contigs-bam"
    query_path = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine_ref_genomes.fasta"
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    # genome2length = readPickle("/home/datasets/ZOUbohao/Proj3-DeepMetaBin/marine_ref_genomes2length.pkl")
    for i in range(21):
        id_name = f"plant-sample-{i}"
        cur_contig_path = os.path.join(home_path, f"{id_name}.contigs.fasta")
        cur_output_path = os.path.join(output_folder, f"{id_name}.minimap2")
        cmd = f"minimap2 {query_path} {cur_contig_path} -o {cur_output_path} -t 64"
        print(cmd)
        os.system(cmd)
        # convert_paf2biobox(cur_output_path, os.path.join(output_folder, f"{id_name}-biobox.tsv"), id_name, genome2length)
