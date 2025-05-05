

import os
import collections
from typing import List

from shutil import rmtree

import numpy as np

from .utils import get_vamb_cluster_tsv


class Reference:
    
    def __init__(self, minimap2_paf_path: str):
        """_summary_

        Args:
            ref_file_path (str): contigname\tref_genome_name\tref_genome_contig\tref_genome_contig_length\talign_start\talign_end
        """
        self.contigname2length = {}
        self.contigname2ref_list = dict()
        self.refname2contigname_set = collections.defaultdict(set)
        self.refname2refcontig2length = dict()
        self._read_ref_file(minimap2_paf_path)
    
    def _read_ref_file(self, minimap2_paf_path):
        with open(minimap2_paf_path, "r") as rh:
            for line in rh:
                oneline = line.strip("\n").split("\t")
                contig_name = oneline[0]
                contig_length = oneline[1]
                self.contigname2length[contig_name] = float(contig_length)
                
                if "|" in oneline[5]:
                    genome_name, _ = oneline[5].split("|")
                else:
                    genome_name = oneline[5]
                
                # omit mobile elements
                if "RNODE" in oneline[5] and "length" in oneline[5]:
                    continue
                
                target_length = float(oneline[6])
                start = float(oneline[7])
                end = float(oneline[8])
                cover = float(oneline[10])
                
                if genome_name not in self.refname2refcontig2length:
                    cur_contig_dict = {oneline[5]: target_length}
                    self.refname2refcontig2length[genome_name] = cur_contig_dict
                else:
                    self.refname2refcontig2length[genome_name][oneline[5]] = target_length
                
                if contig_name not in self.contigname2ref_list:
                    self.contigname2ref_list[contig_name] = [(genome_name, oneline[5], target_length, start, end)]
                    self.refname2contigname_set[genome_name].add(contig_name)
                else:
                    if float(cover) / float(contig_length) >= 0.99:
                        self.contigname2ref_list[contig_name].append((genome_name, oneline[5], target_length, start, end))
                        self.refname2contigname_set[genome_name].add(contig_name)


class BinningF1:
    
    def __init__(
        self, 
        binning_cluster_tsv_path: str = None,
        minimap2_paf_path: str = None, 
        binning_res_folder_path: str = None,
        bin_suffix: str = "fasta"):
        """
        Calculate the binning F1.
        Use 'get_f1' function to get the results.
        This code was modified from VAMB

        Args:
            binning_cluster_tsv_path (str): This file contains the results of binning.
            This first column is the MAG name, the second column is contig name. Split with \\t. 
            You can set this param as None but you should give the third and fourth params.
            
            minimap2_paf_path (str): The minimap2 paf outputs. 
            Step1: You should put all of contigs in all reference genomes into a concatenated fasta file with a given contig name format.
            The contig name format in this concatenated fasta file should be like this: >ref_1.name|contig_0, >ref_1.name|contig_1, ..., >ref_k.name|contig_n,
            The genome name and the contigs in this genome should be splited with '|'.
            Step2: You should align ALL of binning contigs (Contigs in all of MAGs) to this concatenated fasta file with minimap2 once time.
            We would use this minimap2 alignment outputs.
            
            binning_res_folder_path (str): The path of binning result folder.
            
            bin_suffix (str): The bin suffix. Default if 'fasta'.
        """
        
        self.ref_obj = Reference(minimap2_paf_path)
        self.contigname2ref_list = self.ref_obj.contigname2ref_list
        self.contigname2length = self.ref_obj.contigname2length
        self.refname2refcontig2length = self.ref_obj.refname2refcontig2length
        self.refname2contigname_set = self.ref_obj.refname2contigname_set
        
        self.binname2contigname_set = collections.defaultdict(set)
        self.contigname2binname_set = collections.defaultdict(set)
        self.binname2breadth = {}
        
        if binning_cluster_tsv_path is not None:
            self._read_file(binning_cluster_tsv_path)
        else:
            binning_cluster_tsv_path = "~/binning_res.cluster.tsv"
            get_vamb_cluster_tsv(binning_res_folder_path, binning_cluster_tsv_path, bin_suffix)
            self._read_file(binning_cluster_tsv_path)
        
        self.benckmark_breadth = 0
        self.genome2breadth = {}
        self.get_ref_breadth()
        
        intersectionsof = dict()
        for genome in self.refname2refcontig2length.keys():
            intersectionsof[genome] = dict()
            for bin_name, intersection in self._iter_intersections(genome):
                intersectionsof[genome][bin_name] = intersection
        self.intersectionsof = intersectionsof
        
    
    def _read_file(self, binning_cluster_tsv_path: str):
        with open(binning_cluster_tsv_path, "r") as rh:
            for line in rh:
                bin_name, contigname = line.strip("\n").split("\t")
                self.binname2contigname_set[bin_name].add(contigname)
                self.contigname2binname_set[contigname].add(bin_name)
        for bin_name, contigset in self.binname2contigname_set.items():
            cur_breadth = 0
            for contigname in contigset:
                if contigname in self.contigname2length:
                    cur_breadth += self.contigname2length[contigname]
            self.binname2breadth[bin_name] = cur_breadth
    
    def get_ref_breadth(self):
        bencmark_breadth = 0
        for ref_name, refcontig2length in self.refname2refcontig2length.items():
            cur_ref_summed_length = sum(refcontig2length.values())
            self.genome2breadth[ref_name] = cur_ref_summed_length
            bencmark_breadth += cur_ref_summed_length
        self.benckmark_breadth = bencmark_breadth

    def getbreadth(self, contigs, genome):
        "This calculates the total number of bases covered at least 1x in ANY Genome."
        bysubject = collections.defaultdict(list)
        for contig in contigs:
            for genome_name, genome_contig, target_length, start, end in self.contigname2ref_list[contig]:
                if genome_name == genome:
                    bysubject[genome_contig].append((float(start), float(end)))

        breadth = 0
        for contiglist in bysubject.values():
            contiglist.sort(key=lambda x: x[0])
            rightmost_end = float('-inf')

            for start, end in contiglist:
                breadth += max(end, rightmost_end) - max(start, rightmost_end)
                rightmost_end = max(end, rightmost_end)

        return breadth

    def _iter_intersections(self, genome):
        """Given a genome, return a generator of (bin_name, intersection) for
        all binning bins with a nonzero recall and precision.
        """
        # Get set of all binning bin names with contigs from that genome
        bin_names = set()
        for contig in self.refname2contigname_set[genome]:
            bin_name = self.contigname2binname_set.get(contig)
            if bin_name is None:
                continue
            elif isinstance(bin_name, str):
                bin_names.add(bin_name)
            else:
                bin_names.update(bin_name)
        for bin_name in bin_names:
            intersecting_contigs = self.refname2contigname_set[genome].intersection(self.binname2contigname_set[bin_name])
            intersection = self.getbreadth(intersecting_contigs, genome)
            assert intersection != 0, ValueError(f"bin_name: {bin_name} to genome: {genome}, intersecting_contigs: {intersecting_contigs}")
            yield bin_name, intersection
    
    def confusion_matrix(self, genome, bin_name):
        "Given a genome and a binname, returns TP, TN, FP, FN"
        true_positives = self.intersectionsof[genome].get(bin_name, 0)
        false_positives = self.binname2breadth[bin_name] - true_positives
        ## OK
        false_negatives = self.genome2breadth[genome] - true_positives
        true_negatives = self.benckmark_breadth - false_negatives - false_positives + true_positives
        return true_positives, true_negatives, false_positives, false_negatives
    
    def _get_prec_rec_dict(self):
        recprecof = collections.defaultdict(dict)
        for genome, intersectiondict in self.intersectionsof.items():
            for binname in intersectiondict:
                tp, tn, fp, fn = self.confusion_matrix(genome, binname)
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                recprecof[genome][binname] = (recall, precision, tp, fp, fn)
        return recprecof
    
    def get_f1(self, f1_list: List[float]):
        """
        Get the f1 of binning results.
        Args:
            f1_list (List[float]): The f1 list. Like [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        Returns:
            tuple: This tuple contains two results:
            res (OrderedDict[float, Int]): This is an ordered dict. It records how many MAGs has higher F1 score than the thresholds in the f1_list.\n
            bin2f1 (Dict[float, Tuple]): This is an dict. 
            It records the MAG name to this tuple: '(F1_score, Recall_score, Precision_score, Reference_genome_name)'
        """
        recprecof = self._get_prec_rec_dict()
        bin2f1s = dict()
        genome2bins = dict()
        for genome, _dict in recprecof.items():
            for binname, (recall, precision, tp, fp, fn) in _dict.items():
                # print(f"genome: {genome}, binname: {binname}, recall: {recall}, precision: {precision}, tp, fp, fn: {(tp, fp, fn)}")
                if binname not in bin2f1s:
                    bin2f1s[binname] = [(2. * (recall * precision) / (recall + precision), recall, precision, genome)]
                else:
                    bin2f1s[binname].append((2. * (recall * precision) / (recall + precision), recall, precision, genome))
                if genome not in genome2bins:
                    genome2bins[genome] = [binname]
                else:
                    genome2bins[genome].append(binname)
        res = collections.OrderedDict()
        for f1 in f1_list:
            res[f1] = 0
        r90p90_mag = 0
        nc_mag = 0
        medium_mag = 0
        bin2f1 = {}
        for binname, f1s in bin2f1s.items():
            max_f1 = list(sorted(f1s, key=lambda x: x[0], reverse=True))[0]
            f1, re, pre, cur_genome = max_f1
            bin2f1[binname] = (str(f1), str(re), str(pre), cur_genome)
            if re >= 0.9 and pre >= 0.9:
                r90p90_mag += 1
            if re >= 0.9 and pre >= 0.95:
                nc_mag += 1
            if re >= 0.5 and pre >= 0.9:
                medium_mag += 1
            for thre_f1 in f1_list:
                if f1 >= thre_f1:
                    res[thre_f1] += 1
        return res, nc_mag, r90p90_mag, medium_mag, bin2f1
