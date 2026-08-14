

import os
import random

import numpy as np
import torch.multiprocessing as tmp
from torch.optim import AdamW
from torch.utils.data import DataLoader

from CompleteBin.logger import get_logger
from CompleteBin.Model.model import CompleteBinModel
from CompleteBin.Seqs.seq_utils import generate_feature_mapping_reverse
from CompleteBin.Trainer.dataset import TrainingDataset
from CompleteBin.Trainer.trainer import Trainer
from CompleteBin.Trainer.warmup import GradualWarmupScheduler

logger = get_logger()


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
        num_bam_files: int,
        mean_std_val,
        pretrain_model_weight_path: str,
        log_every_n_steps: int = 10,
        dataloader_workers: int = 32,
        num_classes=15434,
        layers=4,
        seed: int = 2048,
        use_pretrained: bool = True,
    ) -> None:
        self.emb_output_folder = emb_output_folder
        self.model_save_folder = model_save_folder
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.count_kmer = count_kmer
        self.count_kmer_dict_rev, self.count_nr_feature_rev = generate_feature_mapping_reverse(count_kmer)
        seq_length = 1  # change from 6 to 3
        hidden_dim = 768
        model = CompleteBinModel(
            kmer_dim=self.count_nr_feature_rev,
            whole_kmer_dim=self.count_nr_feature_rev,
            feature_dim=feature_dim,
            num_bam_files=num_bam_files,
            split_parts_list=split_parts_list,
            dropout=drop_p,
            device=device,
            n_views=n_views,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            layers=layers,
            num_classes=num_classes,
            use_pretrained=use_pretrained,
        ).to(device)

        model.load_weight_for_model(pretrain_model_weight_path)
        model.fix_param_in_pretrain_model()

        parameter_list = []
        # needs to add code
        i = 0
        j = 0
        for name, parameters in model.named_parameters():
            if use_pretrained and "pretrain_model" in name:
                pass  # frozen pretrained params excluded
            else:
                parameter_list.append(parameters)
                i += 1
            j += 1
        logger.info(f"--> Total number of parameters: {j}. Number of {i} parameters need to be trained.")
        optimizer = AdamW(
            parameter_list,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-10
        )
        # optimizer = get_optimizer("muon", parameter_list, lr, weight_decay)
        warmUpScheduler = GradualWarmupScheduler(
            optimizer,
            lr_multiple,
            lr_warmup_epoch,
            train_epoch - lr_warmup_epoch,
        )

        # ── device-dependent DataLoader settings ─────────────────
        # CPU training: set num_workers=0 and pin_memory=False to avoid
        # fork-based COW memory explosion (each worker copies the full
        # training-data backing store, which can reach 60–100 GB for
        # large multi-sample datasets).
        # GPU training: keep the original behaviour (user-specified
        # workers + pin_memory).
        if device is not None and (device == "cpu" or not device.startswith("cuda")):
            effective_workers = 0
            use_pin_memory = False
            logger.warning(
                "--> CPU mode detected: setting DataLoader num_workers=0 and "
                "pin_memory=False to avoid fork-based COW memory explosion."
            )
        else:
            effective_workers = dataloader_workers
            use_pin_memory = True

        # Use file_system sharing strategy for multiprocessing to reduce
        # COW pressure when num_workers > 0 (safe for both CPU and GPU).
        try:
            if tmp.get_sharing_strategy() != 'file_system':
                tmp.set_sharing_strategy('file_system')
                logger.info("--> Set torch multiprocessing sharing strategy to 'file_system'.")
        except RuntimeError:
            # set_sharing_strategy can only be called once; if already set
            # by another module, silently keep the existing strategy.
            pass

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
            cur_whole_bp_cov_tnf_array = cur_tuples[5]
            for j in range(num_bam_files):
                max_val_list[j].append(np.max(cur_whole_bp_cov_tnf_array[j]))
        max_val_list = np.array(max_val_list)
        max_val = np.max(np.array(max_val_list, dtype=np.float32), axis=1, keepdims=False)
        self.training_set = TrainingDataset(data,
                                            data_name,
                                            n_views,
                                            min_contig_len,
                                            count_kmer,
                                            split_parts_list,
                                            N50,
                                            batch_size,
                                            train_valid_test="train",
                                            dropout_p=drop_p)
        # prefetch_factor is only valid when num_workers > 0
        _prefetch = 2 if effective_workers > 0 else None
        self.training_loader = DataLoader(self.training_set,
                                          batch_size,
                                          num_workers=effective_workers,
                                          pin_memory=use_pin_memory,
                                          sampler=sampler,
                                          prefetch_factor=_prefetch,
                                          persistent_workers=False,
                                          worker_init_fn=lambda worker_id: (
                                              np.random.seed(seed + worker_id),
                                              random.seed(seed + worker_id)
                                          ) if effective_workers > 0 else None,
                                          drop_last=False)
        self.valid_set = TrainingDataset(data,
                                         data_name,
                                         n_views,
                                         min_contig_len,
                                         count_kmer,
                                         split_parts_list,
                                         N50,
                                         batch_size,
                                         train_valid_test="valid")
        self.valid_loader = DataLoader(self.valid_set,
                                       batch_size,
                                       shuffle=False,
                                       num_workers=effective_workers,
                                       pin_memory=use_pin_memory,
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
                                           train_valid_test="test")
        self.infer_loader = DataLoader(self.testing_set,
                                       batch_size,
                                       shuffle=False,
                                       num_workers=effective_workers,
                                       pin_memory=use_pin_memory,
                                       drop_last=False)
        # trainer class
        mean_val, std_val = mean_std_val
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
            (max_val, mean_val, std_val),
            temperature_simclr=temperature_simclr,
            log_every_n_steps=log_every_n_steps,
            seq_length=seq_length,
            use_pretrained=use_pretrained,
        )
        self.loss_record = {}

    def train(self, load_epoch_set=None):
        if load_epoch_set is not None:
            model_path = os.path.join(self.model_save_folder, f'checkpoint_{load_epoch_set}.pth')
        else:
            model_path = None
        self.loss_record = self.trainer.train(self.training_loader, self.valid_loader, model_weight_path=model_path)

    def inference(self, min_epoch_set=None):
        min_epoch = 0
        min_loss = 100000000.
        for epoc, loss in self.loss_record.items():
            if epoc > (self.train_epoch - 9) and loss < min_loss:
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
        return min_epoch_set
