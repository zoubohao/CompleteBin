from typing import Dict, List

from CompleteBin.CallGenes.pfam import PFAM
from CompleteBin.logger import get_logger

logger = get_logger()

class HmmModel(object):
    """Store HMM parameters."""

    def __init__(self, keys):
        self.acc = keys["acc"]
        self.ga = keys["ga"]
        self.tc = keys["tc"]
        self.nc = keys["nc"]


class HmmerHitDOM():
    """Encapsulate a HMMER hit given in domtblout format."""

    def __init__(self, values):
        if len(values) == 23:
            self.contig_name = ">" + "_".join(values[0].split("_")[0:-1])
            self.target_name = values[0]
            self.target_accession = values[1]
            self.target_length = int(values[2])
            self.query_name = values[3]

            self.query_accession = values[4]
            if self.query_accession == '-':
                self.query_accession = self.query_name

            self.query_length = int(values[5])
            self.full_e_value = float(values[6])
            self.full_score = float(values[7])
            self.full_bias = float(values[8])
            self.dom = int(values[9])
            self.ndom = int(values[10])
            self.c_evalue = float(values[11])
            self.i_evalue = float(values[12])
            self.dom_score = float(values[13])
            self.dom_bias = float(values[14])
            self.hmm_from = int(values[15])
            self.hmm_to = int(values[16])
            self.ali_from = int(values[17])
            self.ali_to = int(values[18])
            self.env_from = int(values[19])
            self.env_to = int(values[20])
            self.acc = float(values[21])
            self.target_description = values[22]
        else:
            raise ValueError("The input infomation is not correct.")

    def __str__(self):
        return "\t".join(
            [self.target_name,
            self.target_accession,
            str(self.target_length),
            self.query_name,
            self.query_accession,
            str(self.query_length),
            str(self.full_e_value),
            str(self.full_score),
            str(self.full_bias),
            str(self.dom),
            str(self.ndom),
            str(self.c_evalue),
            str(self.i_evalue),
            str(self.dom_score),
            str(self.dom_bias),
            str(self.hmm_from),
            str(self.hmm_to),
            str(self.ali_from),
            str(self.ali_to),
            str(self.env_from),
            str(self.env_to),
            str(self.acc),
            self.target_description]
            )


def vetHit(hit: HmmerHitDOM, bin_models: Dict[str, HmmModel]):
    """Check if hit meets required thresholds."""
    if bin_models is not None:
        if hit.query_accession in bin_models:
            model = bin_models[hit.query_accession]
        else:
            raise ValueError(f"hit.query_accession not in hmm models.")
    else:
        model = None

    # preferentially use model specific bit score thresholds, before
    # using the user specified e-value and length criteria

    # Give preference to the gathering threshold unless the model
    # is marked as TIGR (i.e., TIGRFAM model)

    bIgnoreThresholds = False

    alignment_length = float(hit.ali_to - hit.ali_from)
    length_perc = alignment_length / float(hit.query_length)
    if length_perc < 0.3:
        return False

    if model is not None:
        if model.nc != None and not bIgnoreThresholds and 'TIGR' in model.acc:
            if model.nc[0] <= hit.full_score and model.nc[1] <= hit.dom_score:
                return True
        elif model.ga != None and not bIgnoreThresholds:
            if model.ga[0] <= hit.full_score and model.ga[1] <= hit.dom_score:
                return True
        elif model.tc != None and not bIgnoreThresholds:
            if model.tc[0] <= hit.full_score and model.tc[1] <= hit.dom_score:
                return True
        elif model.nc != None and not bIgnoreThresholds:
            if model.nc[0] <= hit.full_score and model.nc[1] <= hit.dom_score:
                return True
        else:
            if hit.full_e_value > 1e-10:
                return False
            alignment_length = float(hit.ali_to - hit.ali_from)
            length_perc = alignment_length / float(hit.query_length)
            if length_perc >= 0.7:
                return True
        return False
    else:
        if hit.full_e_value > 1e-10:
            return False
        alignment_length = float(hit.ali_to - hit.ali_from)
        length_perc = alignment_length / float(hit.query_length)
        if length_perc >= 0.7:
            return True
        return False


def addHit(hit, markerHits: Dict[str, List[HmmerHitDOM]], hmmAcc2model):
    """Process hit and add it to the set of markers if it passes filtering criteria."""
    if vetHit(hit, hmmAcc2model):
        if hit.query_accession in markerHits:
            # retain only the best domain hit for a given marker to a specific ORF
            previousHitToORF = None
            for h in markerHits[hit.query_accession]:
                if h.target_name == hit.target_name:
                    previousHitToORF = h
                    break
            if previousHitToORF is None:
                markerHits[hit.query_accession].append(hit)
            else:
                if previousHitToORF.dom_score < hit.dom_score:
                    markerHits[hit.query_accession].append(hit)
                    markerHits[hit.query_accession].remove(previousHitToORF)
        else:
            markerHits[hit.query_accession] = [hit]


def identifyAdjacentMarkerGenes(markerHits):
    """Identify adjacent marker genes and exclude these from the contamination estimate."""

    # check for adjacent ORFs with hits to the same marker gene
    for markerId, hits in markerHits.items():

        bCombined = True
        while bCombined:
            for i in range(0, len(hits)):
                orfI = hits[i].target_name
                scaffoldIdI = orfI[0:orfI.rfind('_')]

                bCombined = False
                for j in range(i + 1, len(hits)):
                    orfJ = hits[j].target_name
                    scaffoldIdJ = orfJ[0:orfJ.rfind('_')]

                    # check if hits are on adjacent ORFs
                    if scaffoldIdI == scaffoldIdJ:
                        try:
                            orfNumI = int(orfI[orfI.rfind('_') + 1:])
                            orfNumJ = int(orfJ[orfJ.rfind('_') + 1:])
                        except:
                            # it appears called genes are not labeled
                            # according to the prodigal format, so
                            # it is not possible to perform this correction
                            break

                        if abs(orfNumI - orfNumJ) == 1:
                            # check if hits are to different parts of the HMM
                            sI = hits[i].hmm_from
                            eI = hits[i].hmm_to

                            sJ = hits[j].hmm_from
                            eJ = hits[j].hmm_to

                            if (sI <= sJ and eI > sJ) or (sJ <= sI and eJ > sI):
                                # models overlap so this could represent contamination,
                                # but it seems more likely that adjacent genes hitting
                                # the same marker represent legitimate gene duplication,
                                # a gene calling error, or an assembly error and thus
                                # should not be treated as contamination
                                bCombined = True
                                break
                            else:
                                # combine the two hits
                                bCombined = True
                                break

                if bCombined:
                    newHit = hits[i]
                    
                    # produce concatenated label indicating the two genes being combined
                    last_A = orfI.split("_")[-1]
                    last_B = orfJ.split("_")[-1]
                    orfA, orfB = sorted([last_A, last_B])
                    newHit.target_name = "_".join(orfI.split("_")[0:-1]) + "_" + orfA + orfB

                    newHit.target_length = hits[i].target_length + \
                        hits[j].target_length
                    newHit.hmm_from = min(
                        hits[i].hmm_from, hits[j].hmm_from)
                    newHit.hmm_to = min(hits[i].hmm_to, hits[j].hmm_to)

                    newHit.ali_from = min(
                        hits[i].ali_from, hits[j].ali_from)
                    newHit.ali_to = min(hits[i].ali_to, hits[j].ali_to)

                    newHit.env_from = min(
                        hits[i].env_from, hits[j].env_from)
                    newHit.env_to = min(hits[i].env_to, hits[j].env_to)

                    hits.remove(hits[j])
                    hits.remove(hits[i])

                    hits.append(newHit)

                    break

        markerHits[markerId] = hits


def getHMMModels(input_hmm_file: str):
    hmmAcc2model = {}
    cur_keys = None
    with open(input_hmm_file, "r") as rh:
        for line in rh:
            info = line.strip("\n").split(" ")
            if "HMMER3/f" == info[0] or "HMMER3/b" == info[0]:
                if cur_keys is not None:
                    hmmAcc2model[cur_keys["acc"]] = HmmModel(cur_keys)
                cur_keys = {}
            if "ACC" == info[0]:
                cur_keys["acc"] = info[-1]
            if "GA" == info[0]:
                cur_keys["ga"] = (float(info[-2]), float(info[-1]))
            if "TC" == info[0]:
                cur_keys["tc"] = (float(info[-2]), float(info[-1]))
            if "NC" == info[0]:
                cur_keys["nc"] = (float(info[-2]), float(info[-1]))
    hmmAcc2model[cur_keys["acc"]] = HmmModel(cur_keys)
    return hmmAcc2model


def findSubHits(
    contigName2seq: Dict[str, str],
    contigName2hits: Dict
):
    sub_contigName2hits = {}
    for contigName, _ in contigName2seq.items():
        if contigName in contigName2hits:
            sub_contigName2hits[contigName] = contigName2hits[contigName]
    return sub_contigName2hits


def processHits(
    sub_contigName2hits: Dict[str, List],
    hmmAcc2model = None, 
    pfma_file_path = None,
    accs_set: set = None,
):
    gene2contigNames = {}
    contigName2_gene2num = {}
    markerHits = {}

    for _, hits in sub_contigName2hits.items():
        for hit in hits:
            if accs_set is not None:
                if hit.query_accession in set(accs_set):
                    addHit(hit, markerHits, hmmAcc2model)
            else:
                addHit(hit, markerHits, hmmAcc2model)
    if pfma_file_path is not None:
        pfam = PFAM(pfma_file_path)
        markerHits = pfam.filterHitsFromSameClan(markerHits)
    identifyAdjacentMarkerGenes(markerHits)

    for query_accession, hitDoms in markerHits.items():
        geneName = query_accession
        for hit in hitDoms:
            contigName = ">" + "_".join(hit.target_name.split("_")[0:-1])
            assert hit.query_accession == geneName, ValueError("The hit query accession is not equal with gene name.")
            assert hit.contig_name == contigName, ValueError(f"hit contig name: {hit.contig_name}, cur contigName: {contigName}")

            if geneName not in gene2contigNames:
                gene2contigNames[geneName] = set([contigName])
            else:
                gene2contigNames[geneName].add(contigName)

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


def process_subset(
    sub_contignames: list,
    contigName2_gene2num: dict
):
    sub_gene2contignames = {}
    sub_contigname2_gene2num = {}
    for contigname in sub_contignames:
        if contigname in contigName2_gene2num:
            sub_contigname2_gene2num[contigname] = contigName2_gene2num[contigname]
    for contigname, gene2num in sub_contigname2_gene2num.items():
        for gene_name in gene2num.keys():
            if gene_name not in sub_gene2contignames:
                sub_gene2contignames[gene_name] = set([contigname])
            else:
                sub_gene2contignames[gene_name].add(contigname)
    return sub_gene2contignames, sub_contigname2_gene2num