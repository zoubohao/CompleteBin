

import os
from CompleteBin.logger import get_logger

logger = get_logger()


## perl version 5.34.0
def gen_seed( 
    contig_file: str, 
    threads: int, 
    output_folder: str,
    marker40_path: str,
    marker_perl_path: str,
    contig_length_threshold: int = 768
    ):
    """
    Generate seed sequences from contigs using FragGeneScan, HMMsearch, and custom markers.
    """
    
    seedURL = os.path.join(output_folder, "bacar_marker.2quarter.seed")
    fragResultURL = os.path.join(output_folder, "contigs.frag.faa")
    hmmResultURL = os.path.join(output_folder, "bacar_marker.hmmout")

    if not (os.path.exists(fragResultURL)):
        fragCmd = "run_FragGeneScan.pl -genome=" + contig_file + " -out=" + f"{os.path.join(output_folder, 'contigs.frag')}" \
            + f" -complete=0 -train=complete -thread={threads}"
        logger.info(f"--> exec cmd: {fragCmd}")
        os.system(fragCmd)

    if not (os.path.exists(hmmResultURL)):
        hmmCmd = "hmmsearch --domtblout " + hmmResultURL + " --cut_tc --cpu " + str(threads) + " " + marker40_path + " " + fragResultURL + \
            " 1>" + hmmResultURL + ".out 2>" + hmmResultURL + ".err"
        logger.info(f"--> exec cmd: {hmmCmd}")
        os.system(hmmCmd)

    if not (os.path.exists(seedURL)):
        markerCmd = marker_perl_path + " " + hmmResultURL + " " + contig_file + " " + str(contig_length_threshold) + " " + seedURL
        logger.info(f"--> exec cmd: {markerCmd}")
        os.system(markerCmd)


# if __name__ == "__main__":
#     contig_file = "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/Data-CAMI2-Marine-contigs-bam/marine-sample-9.contigs.fasta"
#     gen_seed(
#         contig_file,
#         16,
#         "/home/datasets/ZOUbohao/Proj3-DeepMetaBin/test_gen_seed"
#     )