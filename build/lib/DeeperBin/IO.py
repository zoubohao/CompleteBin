import os
import pickle
import sys
from typing import Dict, List, Tuple

from DeeperBin.CallGenes.hmm_utils import (HmmerHitDOM, addHit,
                                           identifyAdjacentMarkerGenes)
from DeeperBin.logger import get_logger

logger = get_logger()


def readVocab(vocab_path):
    res = {}
    with open(vocab_path, "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            res[info[0]] = int(info[1])
    return res


def loadTaxonomyTree(pkl_path: str) -> Dict:
    with open(pkl_path, mode="rb") as rb:
        tree = pickle.load(rb)
    return tree


def readFasta(path: str) -> Dict[str, str]:
    """This function is used to read fasta file and
    it will return a dict, which key is the name of seq and the value is the sequence.
    the plasmid sequence would not be read.
    Args:
        path (str): _description_

    Returns:
        Dict[str, str]: _description_
    """
    contig2Seq = {}
    curContig = ""
    curSeq = ""
    with open(path, mode="r", encoding="utf-8") as rh:
        for line in rh:
            curLine = line.strip("\n")
            if curLine[0] == ">":
                if "plasmid" not in curContig.lower():
                    contig2Seq[curContig] = curSeq.upper()
                    curContig = curLine
                curSeq = ""
            else:
                curSeq += curLine
    if "plasmid" not in curContig.lower():
        contig2Seq[curContig] = curSeq.upper()
    contig2Seq.pop("")
    return contig2Seq


def readBinName2Annot(binName2LineagePath: str) -> Dict[str, str]:
    res = {}
    with open(binName2LineagePath, "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            name, suffix = os.path.splitext(info[0])
            res[name] = info[1]
    return res


def readCheckm2Res(file_path: str, bin_suffix):
    res = {}
    h = 0
    m = 0
    l = 0
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            if "Name" not in line:
                info = line.strip("\n").split("\t")
                comp = float(info[1])
                conta = float(info[2])
                if comp >= 90 and conta <= 5:
                    state = "HighQuality"
                    h += 1
                elif comp >= 50 and conta <= 10:
                    state = "MediumQuality"
                    m += 1
                else:
                    state = "LowQuality"
                    l += 1
                res[info[0] + "." + bin_suffix] = (comp, conta, state)
    return res, h, m, l


def readPickle(readPath: str) -> object:
    with open(readPath, "rb") as rh:
        obj = pickle.load(rh)
    return obj


def readHMMFile(file_path: str, hmmAcc2model, accs_set: set, phy_name=None) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    gene2contigNames = {}
    contigName2_gene2num = {}

    if os.path.exists(file_path) is False:
        raise ValueError("HMM file does not exist.")

    markerHits = {}
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            if line[0] != "#":
                info = line.strip("\n").split(" ")
                newInfo = [ele for ele in info if ele != ""]
                pre = newInfo[0: 22]
                aft = "_".join(newInfo[22:])
                try:
                    hit = HmmerHitDOM(pre + [aft])
                except:
                    hit = None
                if hit is not None and hit.query_accession in accs_set:
                    addHit(hit, markerHits, hmmAcc2model)
    identifyAdjacentMarkerGenes(markerHits)

    for query_accession, hitDoms in markerHits.items():
        geneName = query_accession
        for hit in hitDoms:
            contigName = ">" + "_".join(hit.target_name.split("_")[0:-1])
            assert hit.query_accession == geneName, ValueError("The hit query accession is not equal with gene name.")
            assert hit.contig_name == contigName, ValueError(f"hit contig name: {hit.contig_name}, cur contigName: {contigName}")

            if geneName not in gene2contigNames:
                gene2contigNames[geneName] = [contigName]
            else:
                gene2contigNames[geneName].append(contigName)

            if contigName not in contigName2_gene2num:
                newDict = {geneName: 1}
                contigName2_gene2num[contigName] = newDict
            else:
                curDict = contigName2_gene2num[contigName]
                if geneName not in curDict:
                    curDict[geneName] = 1
                else:
                    curDict[geneName] += 1
    return gene2contigNames, contigName2_gene2num


def readHMMFileReturnDict(
    file_path: str
):
    contigName2hits = {}
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            if line[0] != "#":
                info = line.strip("\n").split(" ")
                newInfo = [ele for ele in info if ele != ""]
                pre = newInfo[0: 22]
                aft = "_".join(newInfo[22:])
                hit = HmmerHitDOM(pre + [aft])
                cur_contigName = ">" + "_".join(hit.target_name.split("_")[0:-1])
                assert hit.contig_name == cur_contigName, ValueError(f"hit contig name: {hit.contig_name}, cur contigName: {cur_contigName}")
                if cur_contigName in contigName2hits:
                    contigName2hits[cur_contigName].append(hit)
                else:
                    contigName2hits[cur_contigName] = [hit]
    return contigName2hits


def progressBar(j, N, add_str = ""):
    statusStr = add_str + "          " + "{} / {}".format(j + 1, N)
    cn = len(statusStr)
    if cn < 50:
        statusStr += "".join([" " for _ in range(50 - cn)])
    statusStr += "\r"
    sys.stderr.write("%s\r" % statusStr)
    sys.stderr.flush()


def readClusterResult(
    clus_path: str,
    contigname2seq: dict,
    threshold_MAG: int = 0
):
    clu2contigs = {}
    clu2summed_val = {}
    with open(clus_path, "r", encoding="utf-8") as rh:
        for line in rh:
            contigname, clu = line.strip("\n").split("\t")
            if clu not in clu2contigs:
                clu2contigs[clu] = set([contigname])
                clu2summed_val[clu] = len(contigname2seq[contigname])
            else:
                clu2contigs[clu].add(contigname)
                clu2summed_val[clu] += len(contigname2seq[contigname])
    res = {}
    for clu, contigs in clu2contigs.items():
        if clu2summed_val[clu] >= threshold_MAG:
            res[clu] = contigs
    return res


def readMarkersetTSV(file_path: str):
    index = 0
    gene2contigNames = {}
    contigName2_gene2num = {}
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            if index == 0:
                index += 1
                continue
            info = line.strip("\n").split("\t")
            _, geneName, contigName_pre = info
            if "&&" not in contigName_pre:
                contigName = ">" + "_".join(contigName_pre.split("_")[0:-1])
            else:
                contigName = ">" + "_".join(contigName_pre.split("&&")[0].split("_")[0:-1])
            if geneName not in gene2contigNames:
                gene2contigNames[geneName] = [contigName]
            else:
                gene2contigNames[geneName].append(contigName)
            if contigName not in contigName2_gene2num:
                newDict = {geneName: 1}
                contigName2_gene2num[contigName] = newDict
            else:
                curDict = contigName2_gene2num[contigName]
                if geneName not in curDict:
                    curDict[geneName] = 1
                else:
                    curDict[geneName] += 1
    return gene2contigNames, contigName2_gene2num


def readGalahClusterTSV(tsv_path: str):
    res = {}
    with open(tsv_path, "r", encoding="utf-8") as rh:
        for line in rh:
            info = line.strip("\n").split("\t")
            if info[0] in res:
                res[info[0]].append(info[1])
            else:
                res[info[0]] = [info[1]]
    return res


def readMarkerSets(ms_file_path):
    """Construct bin marker set data from line."""
    taxon2markerset = {}
    with open(ms_file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            lineSplit = line.strip("\n").split("\t")
            markerSet = eval(lineSplit[-1])
            taxon2markerset[lineSplit[0]] = markerSet
    return taxon2markerset


def readMetaInfo(file_path: str, comp_i = 1, cont_i = 2, skip_first_line=False):
    res = {}
    h = 0
    m = 0
    l = 0
    with open(file_path, "r", encoding="utf-8") as rh:
        for i, line in enumerate(rh):
            if skip_first_line and i == 0:
                continue
            info = line.strip("\n").split("\t")
            comp = float(info[comp_i])
            conta = float(info[cont_i])
            if comp >= 90 and conta <= 5:
                state = "HighQuality"
                h += 1
            elif comp >= 50 and conta <= 10:
                state = "MediumQuality"
                m += 1
            else:
                state = "LowQuality"
                l += 1
            res[info[0]] = (comp, conta, state)
    return res, h, m, l


def readCSV(file_path):
    csv = []
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            oneline = line.strip("\n").split(",")
            csv.append(oneline)
    return csv


def readTSV(file_path):
    csv = []
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            oneline = line.strip("\n").split("\t")
            csv.append(oneline)
    return csv


def readDiamond(file_path: str, res):
    with open(file_path, "r", encoding="utf-8") as rh:
        for line in rh:
            thisline = line.strip("\n").split("\t")
            _, contig_name = thisline[0].split("Ω")
            true_contig_name = ">" + "_".join(contig_name.split("_")[0:-1])
            if true_contig_name not in res:
                res[true_contig_name] = [(contig_name, thisline[1:])]
            else:
                res[true_contig_name].append((contig_name, thisline[1:]))


def writePickle(writePath: str, obj: object) -> None:
    with open(writePath, "wb") as wh:
        pickle.dump(obj, wh, pickle.HIGHEST_PROTOCOL)
        wh.flush()


def write_result(
        outputBinFolder,
        collected_list,
        wh):
    for i, (qualityValues, cor_path) in enumerate(collected_list):
        outName = f"Deepurify_Bin_{i}.fasta"
        wh.write(
            outName
            + "\t"
            + str(qualityValues[0])
            + "\t"
            + str(qualityValues[1])
            + "\t"
            + str(qualityValues[2])
            + "\n"
        )
        writeFasta(readFasta(cor_path), os.path.join(outputBinFolder, outName))


def writeAnnot2BinNames(annot2binNames: Dict[str, List[str]], outputPath: str):
    with open(outputPath, "w", encoding="utf-8") as wh:
        for annot, binList in annot2binNames.items():
            for binName in binList:
                wh.write(binName + "\t" + annot + "\n")


def writeFasta(name2seq: Dict, writePath: str, change_name=False):
    index = 0
    with open(writePath, "w", encoding="utf-8") as wh:
        for key, val in name2seq.items():
            if change_name:
                wh.write(f">Contig_{index}_{len(val)}\n")
            else:
                if key[0] != ">":
                    wh.write(f">{key}\n")
                else:
                    wh.write(key + "\n")
            index += 1
            for i in range(0, len(val), 60):
                wh.write(val[i: i + 60] + "\n")


def writeAnnotResult(outputPath: str, name2annotated: Dict, name2maxList: Dict):
    with open(outputPath, "w", encoding="utf-8") as wh:
        for key, val in name2annotated.items():
            wh.write(key + "\t" + val + "\t")
            for prob in name2maxList[key]:
                wh.write(str(prob)[:10] + "\t")
            wh.write("\n")
