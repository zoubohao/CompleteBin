
import os
from sys import prefix

from Src.IO import readFasta


def convert_paf2vamb(
    align_paf_path: str,
    output_path: str):
    """_summary_
    we applied minimap2's results
    Args:
        align_paf_path (str): _description_
        output_path (str): _description_
    """
    write_handler = open(output_path, "w")
    # write_handler.write(f"contigname\tgenomename\tsubjectname\tstart\tend\n")
    have_write = set()
    with open(align_paf_path, "r") as rh:
        for line in rh:
            oneline = line.strip("\n").split("\t")
            genome_name, subjectname = oneline[5].split("|")
            length = oneline[1] # 6
            start = oneline[7]
            end = oneline[8]
            cover = oneline[10]
            if oneline[0] not in have_write:
                have_write.add(oneline[0])
                write_handler.write(oneline[0] + "\t"  + genome_name + "\t"  + oneline[5] + "\t"  + start + "\t" + end + "\n")
            else:
                if float(cover) / float(length) >= 0.8:
                    write_handler.write(oneline[0] + "\t"  + genome_name + "\t"  + oneline[5] + "\t"  + start + "\t" + end + "\n")
    write_handler.close()


def get_vamb_cluster_tsv(
    bins_folder: str,
    output_cluster_tsv_path: str,
    suffix: str = ".fasta"
):
    with open(output_cluster_tsv_path, "w") as wh:
        bins_files = os.listdir(bins_folder)
        for file in bins_files:
            prefix, cur_suffix = os.path.splitext(file)
            if cur_suffix == suffix:
                name2seq = readFasta(os.path.join(bins_folder, file))
                for contigname, seq in name2seq.items():
                    wh.write(f"{prefix}\t{contigname[1:]}\n")

