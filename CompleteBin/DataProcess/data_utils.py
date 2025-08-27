
from collections import defaultdict, OrderedDict
import multiprocessing
import os
import random

import numpy as np
import psutil

from CompleteBin.CallGenes.gene_utils import splitListEqually
from CompleteBin.IO import readPickle, writePickle
from CompleteBin.logger import get_logger
from CompleteBin.Seqs.seq_utils import (generate_feature_mapping_reverse,
                                generate_feature_mapping_whole_tokens, random_generate_view)

logger = get_logger()


def get_kmer_count_from_seq(seq: str, kmer_dict: dict, kmer_len: int, nr_features: int):
    kmers_seq = []
    seq = seq.upper()
    N = len(seq)
    N = N - kmer_len + 1
    for i in range(N):
        cur_mer = seq[i: i + kmer_len]
        if cur_mer in kmer_dict:
            kmers_seq.append(kmer_dict[cur_mer])
    kmers_seq.append(nr_features - 1)
    cur_composition_v = np.bincount(np.array(kmers_seq, dtype=np.int64))
    cur_composition_v[-1] -= 1
    return cur_composition_v


def get_global_kmer_feature_vector_reverse(
    contigname2seq,
    kmer_len=4
):
    kmer_dict, nr_features = generate_feature_mapping_reverse(kmer_len)
    composition_v = np.zeros(shape=[nr_features], dtype=np.float32)
    i = 1
    N = len(contigname2seq)
    for _, seq in contigname2seq.items():
        # progressBar(i, N)
        composition_v += get_kmer_count_from_seq(seq, kmer_dict, kmer_len, nr_features)
        i += 1
    return composition_v / np.sum(composition_v)


def get_global_kmer_feature_vector_whole_tokens(
    contigname2seq,
    kmer_len=4
):
    kmer_dict, nr_features = generate_feature_mapping_whole_tokens(kmer_len)
    composition_v = np.zeros(shape=[nr_features], dtype=np.float32)
    i = 1
    N = len(contigname2seq)
    for _, seq in contigname2seq.items():
        # progressBar(i, N)
        composition_v += get_kmer_count_from_seq(seq, kmer_dict, kmer_len, nr_features)
        i += 1
    return composition_v / np.sum(composition_v)


BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
def get_tuple_kmer(kmer: str):
    rev_kmer = "".join([BASE_COMPLEMENT[x] for x in reversed(kmer)])
    return tuple(sorted([kmer, rev_kmer]))


def get_normlized_count_vec_of_seq(
        seq: str,
        kmer_dict: dict,
        nr_features: int,
        kmer_len: int,
        bparray_list: np.ndarray,
        cal_bp_tnf = False
    ):
    seq = seq.upper()
    bam_num = None
    kmer2cov_list = OrderedDict()
    if cal_bp_tnf:
        bam_num = len(bparray_list)
        assert len(seq) == len(bparray_list[0]), ValueError(f"The len of seq is: {len(seq)}, but its bparray's length is {len(bparray_list[0])}")
        for kmer, _ in kmer_dict.items():
            kmer2cov_list[get_tuple_kmer(kmer)] = []
    kmers = []
    N = len(seq)
    div_val = kmer_len * N + 0.
    for i in range(N):
        cur_mer = seq[i: i + kmer_len]
        if cal_bp_tnf:
            cur_bp_cov = bparray_list[:, i: i + kmer_len] # bam_num, 4
            if cur_mer in kmer_dict:
                bp_mer = get_tuple_kmer(cur_mer)
                kmer2cov_list[bp_mer].append(cur_bp_cov)
        if cur_mer in kmer_dict:
            kmers.append(kmer_dict[cur_mer])
    kmers.append(nr_features-1)
    composition_v = np.bincount(np.array(kmers, dtype=np.int64))
    composition_v[-1] -= 1
    assert np.sum(composition_v) != 0, ValueError(f"the ori seq is {seq}")
    composition_v = np.array(composition_v, dtype=np.float32) / np.sum(composition_v)
    ### return bp-tnf-array-list
    bp_cov_tnf_array = None
    if cal_bp_tnf:
        bp_cov_tnf_array_list = []
        for _, bp_value_array_list in kmer2cov_list.items():
            if len(bp_value_array_list) != 0:
                bp_values_N = np.concatenate(bp_value_array_list, axis=1).sum(axis=1, keepdims=True) / div_val # bam_num, 1 
            else:
                bp_values_N = np.zeros(shape=[bam_num, 1], dtype=np.float32)
            bp_cov_tnf_array_list.append(bp_values_N)
        bp_cov_tnf_array = np.concatenate(bp_cov_tnf_array_list, axis=1)
        # print("bp_cov_tnf_array shape: ", bp_cov_tnf_array.shape, bp_cov_tnf_array, bparray_list[0], len(bparray_list[0]))
    return composition_v, bp_cov_tnf_array ### L, C


def split_seq_equally(seq: str, num_parts: int, count_kmer: int):
    N = len(seq)
    gap = N // num_parts
    if gap == 0:
        gap = 1
    res = []
    for i in range(0, N, gap):
        cur_seq = seq[i: i + gap]
        if len(cur_seq) < count_kmer:
            cur_seq = seq[i: i + count_kmer]
        if len(cur_seq) < count_kmer:
            cur_seq = seq[-count_kmer:]
        res.append(cur_seq)
    return res[0: num_parts]


def get_features_of_one_seq(seq: str,
                            bp_nparray_list,
                            count_kmer,
                            count_kmer_dict,
                            count_nr_features,
                            subparts_list):
    if bp_nparray_list is not None:
        mean = np.mean(bp_nparray_list, axis=1, keepdims=False)
        sqrt_var = np.std(bp_nparray_list, axis=1, keepdims=False)
    else:
        mean, sqrt_var = None, None
    seq_tokens = []
    seq = seq.upper().replace("N", "A")
    cal_tnf_cov = False
    whole_bp_cov_tnf_array = None
    for sub_parts in subparts_list:
        if sub_parts == 1 and bp_nparray_list is not None:
            cal_tnf_cov = True
        else:
            cal_tnf_cov = False
        sub_seqs_list = split_seq_equally(seq, sub_parts, count_kmer)
        assert len(sub_seqs_list) == sub_parts, ValueError(f"The seq is {seq}")
        for sub_seq in sub_seqs_list:
            sub_composition_v, cur_bp_cov_tnf_array = get_normlized_count_vec_of_seq(sub_seq, count_kmer_dict, count_nr_features, count_kmer, 
                                                               bp_nparray_list, cal_tnf_cov)
            seq_tokens.append(sub_composition_v)
            if sub_parts == 1 and bp_nparray_list is not None:
                whole_bp_cov_tnf_array = cur_bp_cov_tnf_array
    assert len(seq_tokens) == sum(subparts_list), ValueError(f"seq: {seq}, len seq tokens: {len(seq_tokens)}")
    
    seq_tokens = np.stack(seq_tokens, axis=0) # L, C
    return seq_tokens, mean, sqrt_var, whole_bp_cov_tnf_array


def process_data_one_thread_return_list(
    contigname2seq,
    contigname2bp_nparray_list,
    count_kmer,
    split_parts_list,
):
    j = 0
    n = len(contigname2seq)
    output_list = []
    count_kmer_dict, count_nr_features = generate_feature_mapping_reverse(count_kmer)
    for contigname, seq in contigname2seq.items():
        # progressBar(j, n)
        cur_tuple = (seq,
                    contigname2bp_nparray_list[contigname],
                    *get_features_of_one_seq(seq, 
                                            contigname2bp_nparray_list[contigname],
                                            count_kmer, 
                                            count_kmer_dict, 
                                            count_nr_features,
                                            split_parts_list)
        )
        ###
        output_list.append((contigname[1:], cur_tuple))
        j += 1
    return output_list


def build_training_seq_data_numpy_save(
    contigname2seq_path: str,
    contigname2bp_nparray_list_path: str,
    data_output_path: str,
    count_kmer,
    split_parts_list,
    num_workers: int = None
):
    contigname2seq = readPickle(contigname2seq_path)
    contigname2bp_array_list = readPickle(contigname2bp_nparray_list_path)
    if os.path.exists(data_output_path) is False:
        os.mkdir(data_output_path)
    if num_workers is None:
        num_workers = psutil.cpu_count()
    contignames = list(contigname2seq.keys())
    random.shuffle(contignames)
    contignames_list = splitListEqually(contignames, num_workers)
    split_list = []
    for names in contignames_list:
        c2s = {}
        c2b = {}
        for one_name in names:
            c2s[one_name] = contigname2seq[one_name]
            c2b[one_name] = contigname2bp_array_list[one_name]
        split_list.append((c2s, c2b))
    pro_list = []
    res = []
    logger.info("--> Start to generate data for training.") # len(split_list)
    with multiprocessing.Pool(len(split_list)) as multiprocess:
        for i, item in enumerate(split_list):
            p = multiprocess.apply_async(process_data_one_thread_return_list,
                                         (item[0],
                                          item[1],
                                          count_kmer,
                                          split_parts_list,
                                          ))
            pro_list.append(p)
        multiprocess.close()
        for p in pro_list:
            res.append(p.get())
    
    save_list = []
    for cur_thread_list in res:
        for item in cur_thread_list:
            save_list.append(item)
    np.save(os.path.join(data_output_path, "training_data.npy"), np.array(save_list, dtype=object), allow_pickle=True)



### useless codes

# def aug_contigs(
#     contigname2seq: dict,
#     contigname2bp_array_list: dict,
#     min_contig_len: int,
#     default_aug_contig_num = 35000
# ):
#     gap_num = default_aug_contig_num - len(contigname2seq)
#     if gap_num <= 0:
#         return contigname2seq, contigname2bp_array_list
#     new_contigname2seq = {}
#     new_contigname2bp_array_list = {}
#     N = 0
#     tmp_store_list = []
#     judge_aug = True
#     aug_times = 0
#     for contigname, seq in contigname2seq.items():
#         N += len(seq)
#         tmp_store_list.append((contigname, seq, len(seq)))
#     tmp_store_list = list(sorted(tmp_store_list, key=lambda x: x[-1], reverse=True))
#     for contigname, seq, seq_len in tmp_store_list:
#         cur_aug_num = int((seq_len * 1.0 / N + 0.0) * gap_num) + 1
#         new_contigname2seq[contigname] = seq
#         new_contigname2bp_array_list[contigname] = contigname2bp_array_list[contigname]
#         for j in range(cur_aug_num):
#             if judge_aug:
#                 cur_contigname = contigname + f"_augcontig_{j}"
#                 aug_seq, aug_start, aug_end = random_generate_view(seq, min_contig_len)
#                 new_contigname2seq[cur_contigname] = aug_seq
#                 ## bp array aug
#                 cur_new_bp_array_list = []
#                 for cur_bp_array in contigname2bp_array_list[contigname]:
#                     cur_new_bp_array_list.append(cur_bp_array[aug_start: aug_end])
#                 new_contigname2bp_array_list[cur_contigname] = cur_new_bp_array_list
#             else:
#                 break
#         aug_times += cur_aug_num
#         if aug_times > gap_num:
#             judge_aug = False
#     return new_contigname2seq, new_contigname2bp_array_list


# def get_features_one_seq_iter_once_time(
#         seq: str,
#         bp_array,
#         count_kmer,
#         count_kmer_dict,
#         count_nr_features,
#         subparts_list):
#     if bp_array is not None:
#         mean, sqrt_var = np.sum(bp_array) / (len(seq) - 2. * 75.), np.sqrt(np.var(bp_array, dtype=np.float32))
#     else:
#         mean, sqrt_var = None, None
#     seq = seq.upper().replace("N", "")
#     num_sub = len(subparts_list)
#     kmers_list = [[[]] for _ in range(num_sub)]
#     N = len(seq)
#     gap_list = [N // subparts_list[i] for i in range(num_sub)]
#     index_list = [0 for _ in range(num_sub)]
#     for i in range(N):
#         cur_mer = seq[i: i + count_kmer]
#         for j, gap in enumerate(gap_list):
#             if i != 0 and i % gap == 0:
#                 index_list[j] += 1
#                 kmers_list[j].append([])
#             if cur_mer in count_kmer_dict:
#                 # print(f"index_list[j]: {index_list[j]}, j: {j}")
#                 kmers_list[j][index_list[j]].append(count_kmer_dict[cur_mer])
#     res= []
#     for j, sub_kmer_list in enumerate(kmers_list):
#         # print(len(sub_kmer_list))
#         for kmers in sub_kmer_list[0: subparts_list[j]]:
#             kmers.append(count_nr_features-1)
#             composition_v = np.bincount(np.array(kmers, dtype=np.int64))
#             composition_v[-1] -= 1
#             assert np.sum(composition_v) != 0, ValueError(f"the ori seq is {seq}")
#             composition_v = np.array(composition_v, dtype=np.float32) / np.sum(composition_v)
#             res.append(composition_v)
#     seq_tokens = np.stack(res, axis=0) # L, C
#     return seq_tokens, mean, sqrt_var


# def process_data_one_thread(
#     contigname2seq,
#     contigname2bp_nparray_list,
#     count_kmer,
#     data_output_path,
#     split_parts_list,
# ):
#     j = 0
#     n = len(contigname2seq)
#     count_kmer_dict, count_nr_features = generate_feature_mapping_reverse(count_kmer)
#     for contigname, seq in contigname2seq.items():
#         # progressBar(j, n)
#         if os.path.exists(os.path.join(data_output_path, f"{contigname[1:]}.pkl")) is False:
#             cur_tuple = (seq,
#                         contigname2bp_nparray_list[contigname],
#                         *get_features_of_one_seq(seq, 
#                                                 contigname2bp_nparray_list[contigname],
#                                                 count_kmer, 
#                                                 count_kmer_dict, 
#                                                 count_nr_features,
#                                                 split_parts_list)
#             )
#             ###
#             writePickle(os.path.join(data_output_path, f"{contigname[1:]}.pkl"), cur_tuple)
#         j += 1


# def build_training_seq_data(
#     contigname2seq_path: str,
#     contigname2bp_nparray_list_path: str,
#     data_output_path: str,
#     count_kmer,
#     split_parts_list,
#     num_workers: int = None
# ):
#     contigname2seq = readPickle(contigname2seq_path)
#     contigname2bp_array_list = readPickle(contigname2bp_nparray_list_path)
#     if os.path.exists(data_output_path) is False:
#         os.mkdir(data_output_path)
#     if num_workers is None:
#         num_workers = psutil.cpu_count()
#     contignames = list(contigname2seq.keys())
#     random.shuffle(contignames)
#     contignames_list = splitListEqually(contignames, num_workers)
#     split_list = []
#     for names in contignames_list:
#         c2s = {}
#         c2b = {}
#         for one_name in names:
#             c2s[one_name] = contigname2seq[one_name]
#             c2b[one_name] = contigname2bp_array_list[one_name]
#         split_list.append((c2s, c2b))
#     pro_list = []
#     logger.info("--> Start to generate data for training.")
#     with multiprocessing.Pool(len(split_list)) as multiprocess:
#         for i, item in enumerate(split_list):
#             p = multiprocess.apply_async(process_data_one_thread,
#                                          (item[0],
#                                           item[1],
#                                           count_kmer,
#                                           data_output_path,
#                                           split_parts_list,
#                                           ))
#             pro_list.append(p)
#         multiprocess.close()
#         for p in pro_list:
#             p.get()