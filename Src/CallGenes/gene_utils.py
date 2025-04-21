import gzip
import logging
import os
import shutil
import stat
import subprocess
import sys
from multiprocessing import Process
from subprocess import Popen
from typing import List

import numpy as np

from Src.Dereplication.checkm_utils import CheckM_Profile
from Src.IO import readMarkersetTSV, writePickle
from Src.logger import get_logger

logger = get_logger()


def readFasta(fastaFile, trimHeader=True):
    '''Read sequences from FASTA file.'''
    try:
        openFile = gzip.open if fastaFile.endswith('.gz') else open
        seqs = {}
        for line in openFile(fastaFile, 'rt'):
            # skip blank lines
            if not line.strip():
                continue

            if line[0] == '>':
                seqId = line[1:].split(None, 1)[0] if trimHeader else line[1:].rstrip()
                seqs[seqId] = []
            else:
                seqs[seqId].append(line[:-1])

        for seqId, seq in seqs.items():
            seqs[seqId] = ''.join(seq)
    except Exception as e:
        print(e)
        logger = logging.getLogger('timestamp')
        logger.error(f"Failed to process sequence file: {fastaFile}")
        sys.exit(1)

    return seqs


def checkFileExists(inputFile):
    """Check if file exists."""
    if not os.path.exists(inputFile):
        logger = logging.getLogger('timestamp')
        logger.error(f'Input file does not exists: {inputFile}' + '\n')
        sys.exit(1)


class ProdigalGeneFeatureParser():
    """Parses prodigal FASTA output."""

    def __init__(self, filename):
        checkFileExists(filename)

        self.genes = {}
        self.lastCodingBase = {}

        self._parseGFF(filename)

        self.codingBaseMasks = {}
        for seqId in self.genes:
            self.codingBaseMasks[seqId] = self._buildCodingBaseMask(seqId)

    def _parseGFF(self, filename):
        """Parse genes from GFF file."""
        self.translationTable = None
        for line in open(filename):
            if line.startswith('# Model Data') and not self.translationTable:
                lineSplit = line.split(';')
                for token in lineSplit:
                    if 'transl_table' in token:
                        self.translationTable = int(
                            token[token.find('=') + 1:])

            if line[0] == '#' or line.strip() == '"':
                # work around for Prodigal having lines with just a
                # quotation on it when FASTA files have Windows style
                # line endings
                continue

            lineSplit = line.split('\t')
            seqId = lineSplit[0]
            if seqId not in self.genes:
                geneCounter = 0
                self.genes[seqId] = {}
                self.lastCodingBase[seqId] = 0

            geneId = f'{seqId}_{str(geneCounter)}'
            geneCounter += 1

            start = int(lineSplit[3])
            end = int(lineSplit[4])

            self.genes[seqId][geneId] = [start, end]
            self.lastCodingBase[seqId] = max(self.lastCodingBase[seqId], end)

    def _buildCodingBaseMask(self, seqId):
        """Build mask indicating which bases in a sequences are coding."""

        # safe way to calculate coding bases as it accounts
        # for the potential of overlapping genes; indices adjusted
        # to account for GFF file using 1-based indexing
        codingBaseMask = np.zeros(self.lastCodingBase[seqId])
        for pos in self.genes[seqId].values():
            codingBaseMask[pos[0]-1:pos[1]] = 1

        return codingBaseMask

    def codingBases(self, seqId, start=0, end=None):
        """Calculate number of coding bases in sequence between [start, end)."""

        # check if sequence has any genes
        if seqId not in self.genes:
            return 0

        # set end to last coding base if not specified
        if end is None:
            end = self.lastCodingBase[seqId]

        return np.sum(self.codingBaseMasks[seqId][start:end])


class ProdigalRunner():
    """Wrapper for running prodigal."""

    def __init__(self, bin_profix, outDir):
        self.logger = logging.getLogger('timestamp')

        # make sure prodigal is installed
        self.checkForProdigal()

        self.aaGeneFile = os.path.join(outDir, f'{bin_profix}.faa')
        self.gffFile = os.path.join(outDir, f'{bin_profix}.gff')

    def run(self, query):

        prodigal_input = query

        # gather statistics about query file
        seqs = readFasta(prodigal_input)
        totalBases = sum(len(seq) for seqId, seq in seqs.items())
        # call ORFs with different translation tables and select the one with the highest coding density
        tableCodingDensity = {}
        for translationTable in [4, 11]:
            aaGeneFile = f'{self.aaGeneFile}.{str(translationTable)}'
            gffFile = f'{self.gffFile}.{str(translationTable)}'

            # check if there is sufficient bases to calculate prodigal parameters
            procedureStr = 'meta' if totalBases < 100000 else 'single'
            cmd = ('prodigal -p %s -q -m -f gff -g %d -a %s -i %s > %s 2> /dev/null' % (procedureStr,
                                                                                        translationTable,
                                                                                        aaGeneFile,
                                                                                        prodigal_input,
                                                                                        gffFile))

            os.system(cmd)

            if not self._areORFsCalled(aaGeneFile) and procedureStr == 'single':
                # prodigal will fail to learn a model if the input genome has a large number of N's
                # so try gene prediction with 'meta'
                cmd = cmd.replace('-p single', '-p meta')
                os.system(cmd)

            # determine coding density
            prodigalParser = ProdigalGeneFeatureParser(gffFile)

            codingBases = sum(
                prodigalParser.codingBases(seqId) for seqId, seq in seqs.items()
            )
            codingDensity = float(codingBases) / totalBases if totalBases != 0 else 0
            tableCodingDensity[translationTable] = codingDensity

        # determine best translation table
        bestTranslationTable = 11
        if (tableCodingDensity[4] - tableCodingDensity[11] > 0.05) and tableCodingDensity[4] > 0.7:
            bestTranslationTable = 4

        shutil.copyfile(f'{self.aaGeneFile}.{bestTranslationTable}', self.aaGeneFile)
        shutil.copyfile(f'{self.gffFile}.{bestTranslationTable}', self.gffFile)

        # clean up redundant prodigal results
        for translationTable in [4, 11]:
            os.remove(f'{self.aaGeneFile}.{str(translationTable)}')
            os.remove(f'{self.gffFile}.{str(translationTable)}')

        return bestTranslationTable

    def _areORFsCalled(self, aaGeneFile):
        return os.path.exists(aaGeneFile) and os.stat(aaGeneFile)[stat.ST_SIZE] != 0

    def checkForProdigal(self):
        """Check to see if Prodigal is on the system before we try to run it."""

        # Assume that a successful prodigal -h returns 0 and anything
        # else returns something non-zero
        try:
            subprocess.call(
                ['prodigal', '-h'], stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
        except:
            self.logger.error("Make sure prodigal is on your system path.")
            sys.exit(1)


def splitListEqually(input_list: List, num_parts: int) -> List[List[object]]:
    n = len(input_list)
    step = n // num_parts + 1
    out_list = []
    for i in range(num_parts):
        if curList := input_list[i * step: (i + 1) * step]:
            out_list.append(curList)
    return out_list


def runProgidalSingle(binName, bin_path: str, output_faa_folder_path: str) -> None:
    outFAA_path = os.path.join(output_faa_folder_path, f"{binName}.faa")
    if os.path.exists(outFAA_path):
        return
    runner = ProdigalRunner(binName, output_faa_folder_path)
    runner.run(bin_path)


def subProcessProgidal(files: List[str], bin_folder_path: str, output_faa_folder_path: str) -> None:
    for file in files:
        binName = os.path.splitext(file)[0]
        bin_path = os.path.join(bin_folder_path, file)
        runProgidalSingle(binName, bin_path, output_faa_folder_path)


def runProgidalFolder(bin_folder_path: str, output_faa_folder_path: str, num_cpu: int, bin_suffix: str) -> None:
    files = os.listdir(bin_folder_path)
    bin_files = [
        file for file in files if os.path.splitext(file)[-1][1:] == bin_suffix
    ]
    splited_files = splitListEqually(bin_files, num_cpu)
    n = len(splited_files)
    ps = []
    for i in range(n):
        p = Process(
            target=subProcessProgidal,
            args=(
                splited_files[i],
                bin_folder_path,
                output_faa_folder_path,
            ),
        )
        ps.append(p)
        p.start()
    for p in ps:
        p.join()


def runHMMsearchSingle(faa_path: str, ouput_path: str, hmm_model_path, num_worker: int) -> None:
    if os.path.getsize(faa_path) == 0 or os.path.exists(faa_path) is False:
        wh = open(ouput_path, "w")
        wh.close()
        return
    if os.path.exists(ouput_path):
        return
    res = Popen(
        f"hmmsearch --domtblout {ouput_path} --cpu {num_worker} --notextw -E 0.1 --domE 0.1 --noali {hmm_model_path} {faa_path} > /dev/null",
        shell=True,
    )
    res.wait()
    res.kill()


def subProcessHMM(hmm_model_path: str, files: List[str], faa_folder_path: str, output_folder_path: str, num_worker) -> None:
    for file in files:
        binName = os.path.splitext(file)[0]
        faa_path = os.path.join(faa_folder_path, file)
        output_path = os.path.join(output_folder_path, f"{binName}.HMM.txt")
        runHMMsearchSingle(faa_path, output_path, hmm_model_path, num_worker)


def runHMMsearchFolder(faa_folder_path: str, output_folder_path: str, hmm_model_path: str, num_cpu: int, faa_suffix: str) -> None:
    files = os.listdir(faa_folder_path)
    faa_files = [
        file for file in files if os.path.splitext(file)[-1][1:] == faa_suffix
    ]
    splited_files = splitListEqually(faa_files, num_cpu)
    n = len(splited_files)
    ps = []
    for i in range(n):
        p = Process(
            target=subProcessHMM,
            args=(
                hmm_model_path,
                splited_files[i],
                faa_folder_path,
                output_folder_path,
                num_cpu // len(splited_files)
            ),
        )
        ps.append(p)
        p.start()
    for p in ps:
        p.join()


def callMarkerGenes(bin_folder_path: str, temp_folder_path: str, num_cpu: int, hmm_model_path: str, bin_suffix: str) -> None:
    if os.path.exists(temp_folder_path) is False:
        os.mkdir(temp_folder_path)
    logger.info("--> Running Prodigal...")
    runProgidalFolder(bin_folder_path, temp_folder_path, num_cpu, bin_suffix)
    logger.info("--> Running Hmm-Search...")
    runHMMsearchFolder(temp_folder_path, temp_folder_path, hmm_model_path, num_cpu, "faa")


def callMarkerGenesByCheckm(
    temp_folder_path,
    bac_ms_path,
    arc_ms_path,
    input_bins_folder,
    call_genes_folder,
    db_path,
    num_workers
    ):
    checkm_profile = CheckM_Profile(num_workers, bac_ms_path, arc_ms_path, db_path, bin_suffix="fasta")
    checkm_profile.run(input_bins_folder, call_genes_folder)
    bac_marker_set_path = os.path.join(call_genes_folder, "bac", "marker_gene_table.tsv")
    arc_marker_set_path = os.path.join(call_genes_folder, "arc", "marker_gene_table.tsv")
    bac_gene2contigNames, bac_contigName2_gene2num = readMarkersetTSV(bac_marker_set_path)
    arc_gene2contigNames, arc_contigName2_gene2num = readMarkersetTSV(arc_marker_set_path)
    writePickle(os.path.join(temp_folder_path, "bac_gene_info.pkl"), (bac_gene2contigNames, bac_contigName2_gene2num))
    writePickle(os.path.join(temp_folder_path, "arc_gene_info.pkl"), (arc_gene2contigNames, arc_contigName2_gene2num))