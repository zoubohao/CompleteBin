
import multiprocessing
import os

from Src.IO import readDiamond, readFasta
from Src.logger import get_logger

logger = get_logger()

class CheckM_Profile():
    """Profile genomes across different binning methods."""

    def __init__(self, cpus, bac_ms_path, arc_ms_path, db_path, bin_suffix="fasta"):
        """Initialization."""

        self.logger = get_logger()
        self.cpus = cpus
        self.bac_ms_path = bac_ms_path
        self.arc_ms_path = arc_ms_path
        self.bin_suffix = bin_suffix
        self.db_path = db_path
        self.MARKER_GENE_TABLE = 'marker_gene_table.tsv'

    def run(self, bin_dir, output_dir):
        """Profile genomes in each bin directory.

        Parameters
        ----------
        bin_dir : str
            Directories containing bins from different binning methods.
        output_dir : str
            Output directory.
        """
        files = os.listdir(bin_dir)
        self.logger.info(f'--> Start to run checkm analysis.')
        for d, ms_file in [("bac", self.bac_ms_path), ("arc", self.arc_ms_path)]:
            cur_output_dir = os.path.join(output_dir, d)
            cmd = 'checkm analyze -t %d -x %s %s %s %s' % (self.cpus,
                                                           self.bin_suffix,
                                                           ms_file,
                                                           bin_dir,
                                                           cur_output_dir)
            os.system(f"checkm data setRoot {self.db_path}")
            os.system(cmd)
            marker_gene_table = os.path.join(cur_output_dir, self.MARKER_GENE_TABLE)
            cmd = 'checkm qa -t %d -o 5 --tab_table -f %s %s %s' % (self.cpus,
                                                                    marker_gene_table,
                                                                    ms_file,
                                                                    cur_output_dir)
            os.system(f"checkm data setRoot {self.db_path}")
            os.system(cmd)


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


def runCheckm2SingleResume(output_faa_folder, modified_checkm2_tmp_folder, cpu_num, db_path):
    cmd = f"checkm2 predict -x faa --threads {cpu_num} --resume --genes -i {output_faa_folder} -o {modified_checkm2_tmp_folder} --database_path {db_path}"
    os.system(cmd)


def process_one_method(output_faa_folder: str,
                       output_dimond_folder: str,
                       cur_method_bins_folder: str,
                       cur_index: int,
                       gene_info: dict,
                       diamond_info: dict,
                       bin_suffix: str = "fasta"):

    output_dimond_file = os.path.join(output_dimond_folder, f"DIAMOND_RESULTS_{cur_index}.tsv")
    modified_bin_names = os.listdir(cur_method_bins_folder)

    wdh = open(output_dimond_file, "w", encoding="utf-8")
    N = len(modified_bin_names)

    for j, modified_bin_name in enumerate(modified_bin_names):
        # progressBar(j, N)
        bin_name, suffix = os.path.splitext(modified_bin_name)
        if suffix[1:] != bin_suffix:
            continue
        modified_contig2seq = readFasta(os.path.join(cur_method_bins_folder, modified_bin_name))
        modified_contig_names = set(list(modified_contig2seq.keys()))
        with open(os.path.join(output_faa_folder, bin_name + ".faa"), "w") as wfh:
            for modified_contig_name in modified_contig_names:
                ## faa write
                if modified_contig_name in gene_info:
                    for cur_faa_contig_name, cur_seq in gene_info[modified_contig_name]:
                        wfh.write(cur_faa_contig_name + "\n")
                        wfh.write(cur_seq + "\n")
                ## diamond write
                if modified_contig_name in diamond_info:
                    for dia_contig_name, dia_info in diamond_info[modified_contig_name]:
                        wdh.write("\t".join([bin_name + "Ω" + dia_contig_name] + dia_info) + "\n")
    wdh.close()


def buildCheckm2TmpFiles(
    original_checkm2_res_folder: str,
    temp_flspp_bin_output,
    ensemble_list,
    modified_checkm2_tmp_folder: str,
    bin_suffix: str,
    cpus: int):

    if os.path.exists(modified_checkm2_tmp_folder) is False:
        os.mkdir(modified_checkm2_tmp_folder)

    faa_files_folder = os.path.join(original_checkm2_res_folder, "protein_files")
    diam_file = os.path.join(original_checkm2_res_folder, "diamond_output")

    diamond_info = {} # contig name (with ">") with its genes info list
    for file in os.listdir(diam_file):
        readDiamond(os.path.join(diam_file, file), diamond_info)

    gene_info = {}
    for faa_file in os.listdir(faa_files_folder):
        faa_contig2seq = readFasta(os.path.join(faa_files_folder, faa_file))
        for faa_contig_name, seq in faa_contig2seq.items():
            true_contig_name = "_".join(faa_contig_name.split(" ")[0].split("_")[0:-1])
            if true_contig_name not in gene_info:
                gene_info[true_contig_name] = [(faa_contig_name, seq)]
            else:
                gene_info[true_contig_name].append((faa_contig_name, seq))
    
    output_dimond_folder = os.path.join(modified_checkm2_tmp_folder, "diamond_output")
    output_faa_folder = os.path.join(modified_checkm2_tmp_folder, "protein_files")
    if os.path.exists(output_dimond_folder) is False:
        os.mkdir(output_dimond_folder)
    if os.path.exists(output_faa_folder) is False:
        os.mkdir(output_faa_folder)
    pro_list = []
    if cpus > len(ensemble_list):
        cpus = len(ensemble_list)
    logger.info(f"--> Use utf-8 to write.")
    with multiprocessing.Pool(cpus) as multiprocess:
        for i, item in enumerate(ensemble_list):
            cur_method_name = item[0]
            p = multiprocess.apply_async(process_one_method,
                                        (output_faa_folder,
                                        output_dimond_folder,
                                        os.path.join(temp_flspp_bin_output, cur_method_name),
                                        i,
                                        gene_info,
                                        diamond_info,
                                        bin_suffix,
                                        ))
            pro_list.append(p)
        multiprocess.close()
        for p in pro_list:
            p.get()
    return output_faa_folder



def build_checkm2_quality_report_for_galah(
    temp_file_folder_path: str,
    temp_flspp_bin_output,
    ensemble_list,
    split_input_folder: str,
    db_path,
    gmm_flspp,
    cpu_num:int,
    bin_suffix: str = "fasta"
):
    logger.info("--> Start to Run CheckM2.")
    split_input_checkm2_temp_folder = os.path.join(temp_file_folder_path, "split_contigs_checkm2_temp")
    split_input_quality_path = os.path.join(split_input_checkm2_temp_folder, "quality_report.tsv")
    if os.path.exists(split_input_checkm2_temp_folder) is False:
        os.mkdir(split_input_checkm2_temp_folder)
    if os.path.exists(split_input_quality_path) is False:
        runCheckm2Single(split_input_folder, split_input_checkm2_temp_folder, bin_suffix, 
                         os.path.join(db_path, "checkm", "checkm2_db.dmnd"), cpu_num)
    
    selected_temp = os.path.join(temp_file_folder_path, f"selected_bins_checkm2_{gmm_flspp}")
    logger.info(f"--> Start to Reuse the CheckM2's Tmp Files.")
    if os.path.exists(os.path.join(selected_temp, "quality_report.tsv")) is False:
        output_faa_folder = buildCheckm2TmpFiles(split_input_checkm2_temp_folder, 
                                                temp_flspp_bin_output,
                                                ensemble_list,
                                                selected_temp,
                                                bin_suffix,
                                                cpu_num)
        output_faa_folder = os.path.join(selected_temp, "protein_files")
        
        logger.info("--> Start to Run CheckM2 Resume.")
        runCheckm2SingleResume(output_faa_folder, selected_temp, cpu_num, os.path.join(db_path, "checkm", "checkm2_db.dmnd"))
    return os.path.join(selected_temp, "quality_report.tsv")


