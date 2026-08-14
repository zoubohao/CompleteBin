
import os
import random
from typing import List

import numpy as np
from torch.utils.data import Dataset

from CompleteBin.DataProcess.data_utils import get_features_of_one_seq
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_utils import (generate_feature_mapping_reverse, random_generate_view, sampleSeqFromFasta)
from CompleteBin.DataProcess.data_utils import split_seq_randomly


logger = get_logger()


class TrainingDataset(Dataset):

    def __init__(self,
                 data,
                 data_name,
                 n_views: int,
                 min_contig_len: int,
                 count_kmer: int,
                 split_parts_list,
                 N50,
                 batch_size,
                 train_valid_test: str = None,
                 dropout_p=0.2) -> None:
        super().__init__()
        self.n_view = n_views
        assert n_views >= 3, ValueError(f"The number of n_views must bigger or equal with 3.")
        self.N50 = N50
        self.min_contig_len = min_contig_len
        if train_valid_test.lower() == "train":
            logger.info(f"--> The min contig length for training is {self.min_contig_len}.")
        self.train_valid_test = train_valid_test
        self.count_kmer_dict_rev, self.count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)
        self.count_kmer = count_kmer
        self.split_parts_list = split_parts_list
        self.data = data
        self.data_name = data_name
        self.dropout_p = dropout_p  # 0.15

        # ensure
        if train_valid_test.lower() == "train" and len(data) % batch_size != 0:
            gaps = len(data) // batch_size * batch_size
            assert gaps != 0, ValueError(
                f"The number of contigs is {len(data)}, but the batch size is {batch_size}. It can not consist a batch for training.")
            new_data = []
            for i, item in enumerate(self.data):
                # progressBar(i, N)
                ori_seq, cov_bp_array_list,  cur_seq_rad_tokens, cur_mean_tokens, cur_std_tokens, cur_whole_bp_cov_tnf_array = item
                new_data.append((ori_seq, cov_bp_array_list, cur_seq_rad_tokens, cur_mean_tokens,
                                cur_std_tokens, cur_whole_bp_cov_tnf_array, self.data_name[i], len(ori_seq)))
            new_data = list(sorted(new_data, key=lambda x: x[-1], reverse=True))
            gap_data = []
            gap_data_name = []
            for i in range(gaps):
                if i < 3:
                    logger.info(
                        f"--> The top {i + 1} contig length for training is {new_data[i][-1]}. Its mean and std are {(new_data[i][3], new_data[i][4])}")
                gap_data.append(tuple(new_data[i][0: 6]))
                gap_data_name.append(new_data[i][6])
            logger.info(f"--> There are {len(gap_data)} contigs for training.")
            self.data = gap_data
            self.data_name = gap_data_name

    def __len__(self):
        return len(self.data)

    def generate_view(self, seq, cov_bp_array_list, seed=None):
        cur_view_seq, start_i, end_i = random_generate_view(seq, min_contig_len=self.min_contig_len, seed=seed)
        # if seed is None and random.random() <= 0.5:
        #     cur_view_seq = sequence_data_augmentation(cur_view_seq)
        # bp array
        cur_bp_array_list = cov_bp_array_list[:, start_i: end_i]
        cur_seq_tokens, cur_cov_mean, cur_cov_var_sqrt, cur_whole_bp_cov_tnf_array = get_features_of_one_seq(cur_view_seq,
                                                                                                             cur_bp_array_list,
                                                                                                             self.count_kmer,
                                                                                                             self.count_kmer_dict_rev,
                                                                                                             self.count_nr_feature_rev,
                                                                                                             self.split_parts_list,
                                                                                                             self.min_contig_len)
        return cur_seq_tokens, cur_cov_mean, cur_cov_var_sqrt, cur_whole_bp_cov_tnf_array

    def get_view(self, sub_seq, sub_cov_bp_array_list):
        return get_features_of_one_seq(sub_seq,
                                       sub_cov_bp_array_list,
                                       self.count_kmer,
                                       self.count_kmer_dict_rev,
                                       self.count_nr_feature_rev,
                                       self.split_parts_list,
                                       self.min_contig_len)
    # cur_seq_eql_tokens: [12, 136]
    # cur_seq_rad_tokens: [5 136]
    # mean_tokens shape: (5, 2) for 2 bam files
    # std_tokens shape: (5, 2)
    # whole bp_ cov_ tnf_array shape: (2, 136)

    def __getitem__(self, index):
        # [(deepurify_nparray, seq_kmer_freq, cov_kmer_freq, mean_val, var_val), ...,
        # (deepurify_nparray, seq_kmer_freq, cov_kmer_freq, mean_val, var_val)]
        ori_seq, cov_bp_array_list, seq_rad_tokens, mean_tokens, std_tokens, whole_bp_cov_tnf_array = self.data[index]
        ori_view_tuple = (seq_rad_tokens, mean_tokens, std_tokens, whole_bp_cov_tnf_array)
        l, _ = seq_rad_tokens.shape
        # whole_bp_cov_tnf_array_shape = whole_bp_cov_tnf_array.shape
        # for differnet cases
        if self.train_valid_test.lower() == "train":
            # generate n_views of original sequence for contrastiv learning
            n_view_list = [ori_view_tuple]
            n_views_info = split_seq_randomly(ori_seq, self.min_contig_len, n_views=self.n_view - 1)
            # n_views_info.append(generate_view_with_fixed_len(ori_seq, self.min_contig_len + 20))
            for sub_seq, start_i, end_i in n_views_info:
                cur_seq_rad_tokens, cur_mean_tokens, cur_std_tokens, cur_whole_bp_cov_tnf_array = \
                    self.get_view(sub_seq, cov_bp_array_list[:, start_i: end_i])
                if random.random() < 0.25:
                    mask = np.array(np.random.random(size=[l, 1]) > self.dropout_p, dtype=np.float32)
                    cur_seq_rad_tokens = cur_seq_rad_tokens * mask
                    cur_mean_tokens = cur_mean_tokens * mask
                    cur_std_tokens = cur_std_tokens * mask
                n_view_list.append((cur_seq_rad_tokens, cur_mean_tokens, cur_std_tokens, cur_whole_bp_cov_tnf_array))
            return n_view_list
        elif self.train_valid_test.lower() == "valid":
            return [ori_view_tuple] + [self.generate_view(ori_seq, cov_bp_array_list, index),
                                       self.generate_view(ori_seq, cov_bp_array_list, index + 1)], self.data_name[index]
        else:
            return [ori_view_tuple], self.data_name[index]


class PretrainDataset(Dataset):

    def __init__(self,
                 genome_db_folder_path: str,
                 count_kmer: int,
                 split_parts_list: List,
                 train_valid: str,
                 min_contig_len: int,
                 max_seq_length: int = 50000):
        self.genome_db_folder_path = genome_db_folder_path
        self.min_contig_len = min_contig_len
        self.max_seq_length = max_seq_length
        self.split_parts_list = split_parts_list
        self.count_kmer = count_kmer
        genomes_file_names = os.listdir(genome_db_folder_path)
        self.genome_data = []
        for i, file_name in enumerate(genomes_file_names):
            self.genome_data.append((os.path.join(genome_db_folder_path, file_name), i))
        if train_valid == "train":
            self.fixed = False
            self.genome_data = self.genome_data * 40
        else:
            self.fixed = True
        self.short_prob = 0.1
        self.count_kmer_dict_rev, self.count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)

    def __len__(self):
        return len(self.genome_data)

    def __getitem__(self, index):
        # get sequences from representitive genomes
        while True:
            cur_genome_seq = sampleSeqFromFasta(
                self.genome_data[index][0],
                seq_min_len=self.min_contig_len,
                seq_max_len=self.max_seq_length,
                short_prob=self.short_prob,
                fixed=False)
            n_count = cur_genome_seq.count("N")
            if self.fixed:
                break
            if float(n_count) / len(cur_genome_seq) + 0. > 0.2 or len(cur_genome_seq) < self.min_contig_len:
                continue
            else:
                break
        genome_seq_tokens, _, _, _ = get_features_of_one_seq(
            cur_genome_seq,
            None,
            self.count_kmer,
            self.count_kmer_dict_rev,
            self.count_nr_feature_rev,
            self.split_parts_list,
            self.min_contig_len
        )
        return genome_seq_tokens, self.genome_data[index][1]


# def mask_view(self, view_tuple, l, cov_shape):
#     if random.random() < 0.5:
#         mask_dim = 1
#     else:
#         mask_dim = self.count_nr_feature_rev
#     mask = np.array(np.random.random(size=[l, mask_dim]) > self.dropout_p, dtype=np.float32)
#     bp_shape = view_tuple[3].shape
#     if random.random() < 0.5:
#         bp_aug = view_tuple[3] * np.array(np.random.random(size=bp_shape) > self.dropout_p, dtype=np.float32)
#     else:
#         bp_aug = np.clip(np.random.randn(*bp_shape) * 0.001 + view_tuple[3], a_min=0, a_max=None)
#     # print(view_tuple[0], mask, view_tuple[0] * mask)
#     mask_view = (view_tuple[0] * mask,
#                 np.clip(np.random.randn(*cov_shape) * 0.01 + view_tuple[1], a_min=0, a_max=None),
#                 np.clip(np.random.randn(*cov_shape) * 0.01 + view_tuple[2], a_min=0, a_max=None),
#                 bp_aug)
#     return mask_view


# def generate_view(self, seq, cov_bp_array_list, seed=None):
# cur_view_seq, start_i, end_i = random_generate_view(seq, min_contig_len=self.min_contig_len, seed=seed)
# cur_bp_array_list = cov_bp_array_list[:, start_i: end_i]
# cur_seq_3mer_tokens, cur_seq_rad_tokens, cur_mean_tokens, cur_std_tokens, cur_whole_bp_cov_tnf_array = get_features_of_one_seq(
#     cur_view_seq, cur_bp_array_list,
#     self.count_3mer_nr_features, self.count_3mer_dict,
#     self.count_kmer, self.count_kmer_dict_rev, self.count_nr_feature_rev,
#     self.split_parts_list,
#     self.min_contig_len)
# return cur_seq_3mer_tokens, cur_seq_rad_tokens, cur_mean_tokens, cur_std_tokens, cur_whole_bp_cov_tnf_array
