import os
from shutil import copy

from CompleteBin.IO import readCheckm2Res, readGalahClusterTSV, readMetaInfo
from CompleteBin.logger import get_logger

from .checkm_utils import build_checkm2_quality_report_for_galah

logger = get_logger()


def getScore(
    qualityValues,
    apply_quality = True
) -> float:
    if apply_quality:
        if qualityValues[-1] == "HighQuality":
            score = qualityValues[0] - 4. * qualityValues[1]  + 100.
        elif qualityValues[-1] == "MediumQuality":
            score = qualityValues[0] - 4. * qualityValues[1] + 50.
        else:
            score = qualityValues[0] - 4. * qualityValues[1]
    else:
        score = qualityValues[0] - 4. * qualityValues[1]
    return score


def runGalah(galah_out_folder,
            temp_flspp_bin_output: str,
            ensemble_list: list,
            cpu_num,
            bin_suffix):
    if not os.path.exists(galah_out_folder):
        os.mkdir(galah_out_folder)
    cur_out_files_txt = os.path.join(galah_out_folder, "files_path.txt")
    with open(cur_out_files_txt, "w") as wh:
        for i, item in enumerate(ensemble_list):
            cur_method_name = item[0]
            cur_method_bin_folder = os.path.join(temp_flspp_bin_output, cur_method_name)
            for j, file_name in enumerate(os.listdir(cur_method_bin_folder)):
                _, suffix = os.path.splitext(file_name)
                if suffix[1:] != bin_suffix:
                    continue
                wh.write(os.path.join(cur_method_bin_folder, file_name) + "\n")
    cmd = f"galah cluster --ani 99 --precluster-ani 95 --genome-fasta-list {cur_out_files_txt}  " + \
        f"  --output-cluster-definition {os.path.join(galah_out_folder, 'clusters.tsv')}  -t {cpu_num}"
    os.system(cmd)


def process_galah_result(
    checkm_quality_path,
    scg_quality_report_path,
    ensemble_with_SCGs,
    galah_tsv_path: str,
    output_folder: str,
    filter_huge_gap,
    bin_suffix = "fasta"
):
    collect = {}
    checkm2_meta_info = readCheckm2Res(checkm_quality_path, bin_suffix)[0]
    clu_res_info = readGalahClusterTSV(galah_tsv_path)
    scg_meta_info = readMetaInfo(scg_quality_report_path)[0]
    wh = open(os.path.join(output_folder, "MetaInfo.tsv"), "w")
    # n: name, q: quality, v: path of file
    for c, vals in clu_res_info.items():
        for v in vals:
            n = os.path.split(v)[-1]
            q = checkm2_meta_info[n]
            q_scg = scg_meta_info[n]
            if c not in collect:
                if not ensemble_with_SCGs:
                    collect[c] = [(n, q, v, getScore(q), q_scg)]
                else:
                    collect[c] = [(n, q, v, getScore(q) + getScore(q_scg, False), q_scg)]
            else:
                if not ensemble_with_SCGs:
                    collect[c].append((n, q, v, getScore(q), q_scg))
                else:
                    collect[c].append((n, q, v, getScore(q) + getScore(q_scg, False), q_scg))
    res = []
    for _, q_l in collect.items():
        res.append(list(sorted(q_l, key=lambda x: x[3], reverse=True))[0])
    
    # print(f"The number of clusters is {len(res)}")
    for i, r in enumerate(res):
        if filter_huge_gap:
            abs_val = abs(r[1][0] - r[-1][0])
            if abs_val > 40 and r[-1][0] < 10: 
                continue ## this case means the checkm2's completeness has a huge gap with the SCGs' completeness
        outName = f"CompleteBin_{i}.fasta"
        wh.write(outName
                 + "\t"
                 + "Checkm2(Comp,Cont,Quality)" 
                 + "\t"
                 + str(r[1][0])
                 + "\t"
                 + str(r[1][1])
                 + "\t"
                 + str(r[1][-1])
                 + "\n")
        copy(r[2], os.path.join(output_folder, outName))
    wh.close()


def process_galah(
    temp_file_folder_path: str,
    temp_flspp_bin_output: str,
    ensemble_list: list,
    split_input_folder,
    db_path,
    outputBinFolder,
    ensemble_with_SCGs,
    scg_quality_report_path,
    filter_huge_gap,
    gmm_flspp,
    cpus=64
):
    logger.info("--> Start to Use Galah to Ensemble the Results.")
    checkm_quality_path = build_checkm2_quality_report_for_galah(
        temp_file_folder_path,
        temp_flspp_bin_output,
        ensemble_list,
        split_input_folder,
        db_path,
        gmm_flspp,
        cpus,
        "fasta"
    )
    checkm_quality_path = os.path.join(temp_file_folder_path, f"selected_bins_checkm2_{gmm_flspp}", "quality_report.tsv")
    ########
    galah_out = os.path.join(temp_file_folder_path, f"galah_out_info_{gmm_flspp}")
    if os.path.exists(galah_out) is False:
        os.mkdir(galah_out)
    # # Drep gather and filter results
    galah_tsv = os.path.join(galah_out, "clusters.tsv")
    if os.path.exists(galah_tsv) is False:
        runGalah(galah_out, temp_flspp_bin_output, ensemble_list, cpus, "fasta")
    if os.path.exists(outputBinFolder) is False:
        os.makedirs(outputBinFolder)
    logger.info("--> Start to Process Galah Results.")
    process_galah_result(
        checkm_quality_path,
        scg_quality_report_path,
        ensemble_with_SCGs,
        galah_tsv,
        outputBinFolder,
        filter_huge_gap,
    )
