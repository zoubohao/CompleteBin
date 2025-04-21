import os
import subprocess
from multiprocessing.pool import Pool
from shutil import copy


def bwa_index(contigs_fasta_path):
    cmd1 = f"bwa index {contigs_fasta_path}"
    res = subprocess.Popen(
        cmd1,
        shell=True,
    )
    res.wait()
    res.kill()

# cd /datahome/datasets/ericteam/csbhzou/Deepurify_review/data/soil/
# bwa mem -t 100 SRR25158244_megahit/final.contigs.fa SRR25158244_1.fastq.gz SRR25158244_2.fastq.gz | samtools sort -@ 100 -o SRR25158244.sorted.bam
def bwa_align_reads(contigs_fasta_path, reads1_path, reads2_path, output_sorted_bam_path, cpu_nums):
    if reads2_path is None:
        cmd2 = f"bwa mem -t {cpu_nums} {contigs_fasta_path} {reads1_path} | samtools sort -@ {cpu_nums} -o {output_sorted_bam_path}"
    else:
        cmd2 = f"bwa mem -t {cpu_nums} {contigs_fasta_path} {reads1_path} {reads2_path} | samtools sort -@ {cpu_nums} -o {output_sorted_bam_path}"
    print(cmd2)
    res = subprocess.Popen(
        cmd2,
        shell=True,
    )
    res.wait()
    res.kill()


# /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/reads/Airways/short_read/2017.12.04_18.56.22_sample_10/reads/anonymous_reads.fq.gz
# /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/mapping_results/Oral/2017.12.04_18.45.54_sample_15/mapped.sorted.bam
# /datahome/datasets/ericteam/zmzhang/csmxrao/DeepMetaBin/CAMI2/spades/Airways/2017.12.04_18.56.22_sample_10/contigs.fasta


if __name__ == "__main__":
    
    input_folder = "/datahome/datasets/ericteam/csbhzou/HD-marine-final-results"
    output_folder = "/datahome/datasets/ericteam/csbhzou/HD-multi-sample"
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    
    ids = ["CS01", "CS03", "CS05", "CS08", "CS10", "CS12"]
    
    # with Pool(6) as p:
    #     for i in ids:
    #         output_fasta_path = os.path.join(output_folder, i + ".contigs.fasta")
    #         contig_path = f"{input_folder}/{i}.contigs.fasta"
    #         print(f"Copy fasta file. {contig_path}")
    #         copy(contig_path, output_fasta_path)
    #         res = p.apply_async(bwa_index, args=(contig_path,))
    #     p.close()
    #     p.join()
    
    with Pool(6) as p:
        res_list = []
        for i in ids:
            output_fasta_path = os.path.join(output_folder, i + ".contigs.fasta")
            contig_path = f"{input_folder}/{i}.contigs.fasta"
            # print(contig_path)
            for j in ids:
                reads1_path = f"/datahome/datasets/ericteam/csbhzou/HD-marine-fq/{j}_Fast_PCR_10_1.fq.gz"
                reads2_path = f"/datahome/datasets/ericteam/csbhzou/HD-marine-fq/{j}_Fast_PCR_10_2.fq.gz"
                output_bam_path = os.path.join(output_folder, f"HD-c{i}-r{j}" + ".sorted.bam")
                # print(output_bam_path)
                res = p.apply_async(bwa_align_reads, args=(contig_path, reads1_path, reads2_path, output_bam_path, 64))
                res_list.append(res)
        p.close()
        p.join()
                
                
                
                
    
    # with Pool(3) as p:
    #     ress = []
    #     for name in ["Airways", "Gastrointestinal_tract", "Oral", "Skin", "Urogenital_tract"]:
    #         contigs_path = os.path.join(folder, name, "contigs.fasta")
    #         reads1_path = os.path.join(reads_folder, name + "_Fast_PCR_10_1.fq.gz")
    #         reads2_path = os.path.join(reads_folder, name + "_Fast_PCR_10_2.fq.gz")
    #         output_bam_path = os.path.join(output_folder, name + ".sorted.bam")
    #         output_fasta_path = os.path.join(output_folder, name + ".contigs.fasta")
    #         copy(contigs_path, output_fasta_path)
    #         res = p.apply_async(bwa_func, args=(contigs_path, reads1_path, reads2_path, output_bam_path))
    #         ress.append(res)
    #     p.close()
    #     p.join()
    
    

