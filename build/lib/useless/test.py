


from Analysis.analy_2500vs900 import  readGenome2ID
from CompleteBin.Seqs.seq_utils import base_pair_coverage_calculate
from CompleteBin.IO import readFasta
import numpy as np


def readAbundance(file_path):
    res = {}
    with open(file_path, "r") as rh:
        for line in rh:
            line = line.strip("\n").split("\t")
            if float(line[1]) > 0:
                res[line[0]] = float(line[1])
    return res


def readMinimap2(file_path: str):
    contigname2ref_list = {}
    with open(file_path, "r") as rh:
        for line in rh:
            oneline = line.strip("\n").split("\t")
            contig_name = oneline[0]
            contig_length = oneline[1]
            assert "|" in oneline[5], ValueError(f"The genome name {oneline[5]} does not contain '|' to split genome name and contig name.")
            genome_name, _ = oneline[5].split("|")
            target_length = float(oneline[6])
            start = float(oneline[7])
            end = float(oneline[8])
            cover = float(oneline[10])
            if contig_name not in contigname2ref_list:
                contigname2ref_list[contig_name] = [(genome_name, contig_length)]
            else:
                if float(cover) / float(contig_length) >= 0.99:
                    contigname2ref_list[contig_name].append((genome_name, contig_length))
    return contigname2ref_list


if __name__ == "__main__":
    
    output_file = "/home/comp/21481598/high_low_abundance_contig_length.tsv"
    genome2id = readGenome2ID("/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/genome_to_id.tsv")
    
    wh = open(output_file, "w")
    for i in range(10):
        contigname2ref_list = readMinimap2(f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine/marine-sample-{i}.minimap2")
        id2abundance = readAbundance(f"/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Analysis-data/marine-setup/abundance{i}.tsv")
        high_genome = set()
        low_genome = set()
        filter_genomes = []
        for id_name, abu in id2abundance.items():
            if abu > 0:
                filter_genomes.append((id_name, abu))
        filter_genomes = list(sorted(filter_genomes, key= lambda x: x[-1], reverse=True))
        N = len(filter_genomes)
        # print(filter_genomes)
        for i, (id_name, abu) in enumerate(filter_genomes):
            if i < N // 2:
                high_genome.add(id_name)
            else:
                low_genome.add(id_name)
        for contigname, ref_list in contigname2ref_list.items():
            for genome_name, contig_length in ref_list:
                id_name = genome2id[genome_name]
                # print(id_name)
                if id_name in high_genome:
                    print(id_name, genome_name, "high")
                    wh.write("high_abundance" + "\t" + contig_length + "\n")
                elif id_name in low_genome:
                    print(id_name, genome_name, "low")
                    wh.write("low_abundance" + "\t" + contig_length + "\n")
    wh.close()
    
    # fasta_path = "/home/comp/21481598/Simulated_two_species.fasta"
    # bam_path = "/home/comp/21481598/Simulated.sort.bam"
    # name2bparray_Path = "/home/comp/21481598/Simulated_name2info.csv"
    
    # name2seq = readFasta(fasta_path)
    
    # a = base_pair_coverage_calculate(
    #     name2seq,
    #     bam_path,
    #     name2bparray_Path
    # )
    
    # with open(name2bparray_Path, "w") as wh:
    #     for k, v in a.items():
    #         l_info = k.split("-")[-2]
    #         if "spR57" in k:
    #             name = f"spR57_5-{l_info}bps"
    #         else:
    #             name = f"Elongata_DSM_2581-{l_info}bps"
    #         wh.write(",".join([name, f"{np.mean(v):.10f}", f"{np.std(v):.10f}"]) + "\n")
    
