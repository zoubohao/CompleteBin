

import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# from CompleteBin.CallGenes.gene_utils import splitListEqually
from CompleteBin.IO import progressBar, readPickle
from CompleteBin.logger import get_logger
from CompleteBin.Model.model import DeeperBinModel
from CompleteBin.Seqs.seq_utils import generate_feature_mapping_reverse, generate_feature_mapping_whole_tokens
from CompleteBin.Trainer.dataset import TrainingDataset
# from CompleteBin.Trainer.optimizer import get_optimizer
from CompleteBin.Trainer.trainer import Trainer
from CompleteBin.Trainer.warmup import GradualWarmupScheduler

logger = get_logger()


def read_list(contignames_list, training_data_path):
    i = 0
    N = len(contignames_list)
    data = []
    data_name = []
    for key in contignames_list:
        progressBar(i, N)
        seq_features_tuple = readPickle(os.path.join(training_data_path, key[1:] + ".pkl"))
        data.append(seq_features_tuple)
        data_name.append(key[1:] + ".pkl")
        i += 1
    return data, data_name

class SelfSupervisedMethodsTrainer(object):

    def __init__(
        self,
        feature_dim: int,
        n_views: int,
        drop_p: float,
        device: str,
        temperature_simclr: float,
        min_contig_len: int,
        batch_size: int,
        lr: float,
        lr_multiple: int,
        lr_warmup_epoch: int,
        sampler: int,
        train_epoch: int,
        weight_decay: float,
        training_data_path: str,
        model_save_folder: str,
        emb_output_folder: str,
        count_kmer: int,
        split_parts_list: list,
        N50: int,
        large_model: bool,
        num_bam_files: int,
        std_val: np.ndarray,
        pretrain_model_weight_path: str,
        log_every_n_steps: int = 10,
        multi_contrast = False
    ) -> None:
        self.emb_output_folder = emb_output_folder
        self.model_save_folder = model_save_folder
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.count_kmer = count_kmer
        self.count_kmer_dict_rev, self.count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)

        torch.manual_seed(3407)
        torch.cuda.manual_seed_all(3407)

        if large_model:
            hidden_dim = 768
            layers = 4
        else:
            hidden_dim = 512
            layers = 3
    
        model = DeeperBinModel(
            kmer_dim=self.count_nr_feature_rev,
            whole_kmer_dim=self.count_nr_feature_rev,
            feature_dim=feature_dim,
            num_bam_files = num_bam_files,
            split_parts_list = split_parts_list,
            dropout=drop_p,
            device=device,
            hidden_dim=hidden_dim,
            layers=layers,
            multi_contrast=multi_contrast
        ).to(device)

        model.load_weight_for_model(pretrain_model_weight_path)
        model.fix_param_in_pretrain_model()
        parameter_list = []
        ### needs to add code
        i = 0
        j = 0
        for name, parameters in model.named_parameters():
            if "pretrain_model" not in name:
                parameter_list.append(parameters)
                i += 1
            j += 1
        logger.info(f"--> Total number of parameters: {j}. Number of {i} parameters need to be trained.")
        optimizer = AdamW(
            parameter_list,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.8, 0.95),
            eps=1e-10
        )
        # optimizer = get_optimizer("muon", parameter_list, lr, weight_decay)
        warmUpScheduler = GradualWarmupScheduler(
            optimizer,
            lr_multiple,
            lr_warmup_epoch,
            train_epoch - lr_warmup_epoch,
        )
        i = 0
        # N = len(contigname2seq)
        logger.info("--> Start to read training data to memory.")
        data = []
        data_name = []
        save_array = np.load(training_data_path, allow_pickle=True)
        max_val_list = [[] for _ in range(num_bam_files)]
        for cur_contigname, cur_tuples in save_array:
            data.append(cur_tuples)
            data_name.append(cur_contigname)
            cur_whole_bp_cov_tnf_array = cur_tuples[-1]
            # assert whole_bp_cov_tnf_array.shape[0] == len(bp_nparray_list) and \
            #     whole_bp_cov_tnf_array.shape[1] == len(count_kmer_dict)
            for j in range(num_bam_files):
                max_val_list[j].append(np.max(cur_whole_bp_cov_tnf_array[j]))
        max_val_list = np.array(max_val_list)
        max_val = np.max(np.array(max_val_list, dtype=np.float32), axis=1, keepdims=False) / 2.
        self.training_set = TrainingDataset(data,
                                            data_name, 
                                            n_views, 
                                            min_contig_len,
                                            count_kmer,
                                            split_parts_list,
                                            N50,
                                            batch_size,
                                            train_valid_test = "train",
                                            dropout_p=drop_p)
        self.training_loader = DataLoader(self.training_set,
                                          batch_size,
                                          num_workers=32,
                                          pin_memory=True,
                                          sampler=sampler,
                                          prefetch_factor=2,
                                          persistent_workers=True,
                                          drop_last=False)
        self.valid_set = TrainingDataset(data,
                                        data_name, 
                                        n_views, 
                                        min_contig_len,
                                        count_kmer,
                                        split_parts_list,
                                        N50,
                                        batch_size,
                                        train_valid_test = "valid")
        self.valid_loader = DataLoader(self.valid_set,
                                       batch_size,
                                       shuffle=False,
                                       num_workers=32,
                                       pin_memory=True,
                                       drop_last=True)
        ########### testing dataloader #############
        self.testing_set = TrainingDataset(data,
                                        data_name, 
                                        n_views, 
                                        min_contig_len,
                                        count_kmer,
                                        split_parts_list,
                                        N50,
                                        batch_size,
                                        train_valid_test = "test")
        self.infer_loader = DataLoader(self.testing_set,
                                       batch_size,
                                       shuffle=False,
                                       num_workers=32,
                                       pin_memory=True,
                                       drop_last=False)
        # trainer class
        self.trainer = Trainer(
            model,
            optimizer,
            warmUpScheduler,
            device,
            train_epoch,
            model_save_folder,
            n_views,
            batch_size,
            drop_p,
            (max_val, std_val),
            temperature_simclr=temperature_simclr,
            log_every_n_steps=log_every_n_steps,
            multi_contrast = multi_contrast
        )
        self.loss_record = {}

    def train(self, load_epoch_set = None):
        if load_epoch_set is not None:
            model_path = os.path.join(self.model_save_folder, f'checkpoint_{load_epoch_set}.pth')
        else:
            model_path = None
        logger.info(f"--> The load epoch is {load_epoch_set}, the path of it is {model_path}.")
        self.loss_record = self.trainer.train(self.training_loader, self.valid_loader, model_weight_path=model_path)

    def inference(self, min_epoch_set = None):
        min_epoch = 0
        min_loss = 100000000.
        for epoc, loss in self.loss_record.items():
            if epoc > self.train_epoch // 2 and loss < min_loss:
                min_epoch = epoc
                min_loss = loss
        if min_epoch_set is None:
            min_epoch_set = min_epoch
        logger.info(f"--> The best epoch is {min_epoch_set}, the loss of it is {min_loss}.")
        self.trainer.inference(
            self.infer_loader,
            self.emb_output_folder,
            os.path.join(self.model_save_folder, f'checkpoint_{min_epoch_set}.pth')
            )
