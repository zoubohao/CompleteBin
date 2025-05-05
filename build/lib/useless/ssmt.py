
import multiprocessing
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from Src.CallGenes.gene_utils import splitListEqually
from Src.IO import progressBar, readPickle
from Src.logger import get_logger
from Src.Model.model import DeeperBinModel
from Src.Seqs.seq_utils import generate_feature_mapping_reverse
from Src.Trainer.dataset import TrainingDataset
from Src.Trainer.optimizer import get_optimizer
from Src.Trainer.trainer import Trainer
from Src.Trainer.warmup import GradualWarmupScheduler

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
        train_epoch: int,
        weight_decay: float,
        training_data_path: str,
        model_save_folder: str,
        emb_output_folder: str,
        count_kmer: int,
        contigname2seq,
        split_parts_list,
        N50,
        large_model,
        num_bam_files,
        max_cov_mean,
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
            hidden_dim = 2048
            layers = 4
        else:
            hidden_dim = 512
            layers = 3
    
        model = DeeperBinModel(
            kmer_dim=self.count_nr_feature_rev,
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
            betas=(0.9, 0.95),
            eps=1e-6
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
        contignames = list(contigname2seq.keys())
        contignames_list = splitListEqually(contignames, 64)
        pro_list = []
        res = []
        with multiprocessing.Pool(len(contignames_list)) as multiprocess:
            for i, item in enumerate(contignames_list):
                p = multiprocess.apply_async(read_list,
                                            (item,
                                             training_data_path,
                                            ))
                pro_list.append(p)
            multiprocess.close()
            for p in pro_list:
                res.append(p.get())
        for cur_data, cur_data_name in res:
            data += cur_data
            data_name += cur_data_name

        self.training_set = TrainingDataset(data,
                                            data_name, 
                                            n_views, 
                                            min_contig_len,
                                            count_kmer,
                                            split_parts_list,
                                            N50,
                                            batch_size,
                                            train_valid_test = "train")
        self.training_loader = DataLoader(self.training_set,
                                          batch_size,
                                          shuffle=True,
                                          num_workers=16,
                                          pin_memory=True,
                                          drop_last=True)
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
                                       num_workers=16,
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
                                       num_workers=16,
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
            max_cov_mean,
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
            if loss < min_loss:
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
