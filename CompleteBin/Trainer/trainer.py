
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from CompleteBin.IO import writePickle
from CompleteBin.logger import get_logger
from CompleteBin.Trainer.loss import info_nce_loss, info_nce_loss_for_loop

logger = get_logger()


# bsz : batch size (number of positive pairs)
# d : latent dim
# x : Tensor, shape=[bsz, d]
# latents for one side of positive pairs
# y : Tensor, shape=[bsz, d]
# latents for the other side of positive pairs

# Alignment & Uniformity, the lower the better
def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def min_max_norm(input_tensor: torch.Tensor):
    max_vals, _ = torch.max(input_tensor, dim=-1, keepdim=True)
    min_vals, _ = torch.min(input_tensor, dim=-1, keepdim=True)
    dis = max_vals - min_vals
    # print(dis.shape)
    return (input_tensor - min_vals) / dis


def schedule_of_temperature(temp: float, epochs: int):
    start_temp = temp - 0.005
    end_temp = temp + 0.005
    step = 0.01 / epochs
    res = [0 for _ in range(epochs)]
    for i, cur_temp in enumerate(np.arange(start_temp, end_temp, step)):
        if i <= epochs - 1:
            res[i] = float("%.3f" % cur_temp)
    res[-1] = end_temp
    for i, ite in enumerate(res):
        if ite == 0:
            res[i] = temp
    return res


class Trainer(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str,
                 epochs: int,
                 model_save_folder: str,
                 n_views: int,
                 batch_size: int,
                 dropout: float,
                 max_cov_mean = (100., 100.),
                 temperature_simclr=0.123,
                 log_every_n_steps: int = 10,
                 multi_contrast = False
                 ):
        self.device = device
        self.dropout = dropout
        self.model = model.to(device)
        self.optimizer = optimizer
        self.multi_contrast = multi_contrast
        # self.optimizer2 = optimizer2
        self.scheduler = scheduler
        self.epochs = epochs
        self.model_save_folder = model_save_folder
        self.log_every_n_steps = log_every_n_steps
        self.n_views = n_views
        self.batch_size = batch_size
        self.max_cov_mean = torch.tensor(max_cov_mean[0], dtype=torch.float32)[None, :].to(self.device)
        self.max_cov_var = torch.tensor(max_cov_mean[1], dtype=torch.float32)[None, :].to(self.device)
        logger.info(f"--> The max of coverage mean value is {self.max_cov_mean}.")
        logger.info(f"--> The max of coverage std value is {self.max_cov_var}.")
        self.temperature_simclr = temperature_simclr
        self.temperature_schedule = schedule_of_temperature(temperature_simclr, epochs)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def get_model_inputs(self, n_views_tuple_list):
        seq_tokens_n_views = []
        mean_n_views = []
        var_n_views = []
        whole_bp_cov_tnf_array_n_views = []
        for i, (seq_tokens, cov_mean, cov_var_sqrt, whole_bp_cov_tnf_array) in enumerate(n_views_tuple_list):
            seq_tokens_n_views.append(seq_tokens)
            mean_n_views.append(cov_mean)
            var_n_views.append(cov_var_sqrt)
            whole_bp_cov_tnf_array_n_views.append(whole_bp_cov_tnf_array)
        seq_tokens_inputs = torch.cat(seq_tokens_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True)
        whole_bp_cov_tnf_inputs = torch.cat(whole_bp_cov_tnf_array_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True)
        ## [b * nviews, L, C] max_cov_mean: [1, L]
        # print("before", whole_bp_cov_tnf_inputs)
        # print(self.max_cov_mean[..., None].shape)
        whole_bp_cov_tnf_inputs /= self.max_cov_mean[..., None]
        # print("after", whole_bp_cov_tnf_inputs, whole_bp_cov_tnf_inputs.shape)
        mean_inputs = torch.cat(mean_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True) / self.max_cov_mean # max_cov_mean: [1, L]
        var_inputs = torch.cat(var_n_views, dim=0).to(torch.float32).to(self.device, non_blocking=True) / self.max_cov_var # max_cov_mean: [1, L]
        if len(mean_inputs.shape) == 1:
            mean_inputs.unsqueeze(1)
        if len(var_inputs.shape) == 1:
            var_inputs.unsqueeze(1)
        assert len(mean_inputs.shape) == 2 and len(var_inputs.shape) == 2, \
            ValueError(f"The dim mean_input is {mean_inputs.shape}, The dim var_inputs is {var_inputs.shape}. One of them not equal with 2.")
        return seq_tokens_inputs, mean_inputs, var_inputs, whole_bp_cov_tnf_inputs

    def train(self, train_loader: DataLoader, valid_loader: DataLoader = None, model_weight_path = None):
        logger.info(f"--> Start self-supervised training with {self.epochs} epochs.")
        logger.info(f"--> Training with {self.device} device.")
        loss_record = {}
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            logger.info(f"--> Model weight has been load.")
        for epoch_counter in range(1, self.epochs + 1):
            n_iter = 0.
            self.model.train()
            for n_views_tuple_list in tqdm(train_loader):
                n_views = len(n_views_tuple_list)
                assert n_views == self.n_views + 2, ValueError("Views number is not equal with each other.")
                seq_tokens_inputs, mean_inputs, var_inputs, whole_bp_cov_tnf_inputs = self.get_model_inputs(n_views_tuple_list)
                # ============ multi-res forward passes ... ============
                self.optimizer.zero_grad()
                simclr_emb_contrast, seq_emb, _= self.model.forward(seq_tokens_inputs, mean_inputs, var_inputs, whole_bp_cov_tnf_inputs)
                ### SimCLR loss ## changed here
                loss_simclr, logits, labels = info_nce_loss_for_loop(simclr_emb_contrast[0: -self.batch_size * 2], 
                                                            self.batch_size, 
                                                            self.n_views,
                                                            self.temperature_schedule[epoch_counter - 1], 
                                                            self.device, 
                                                            self.criterion)
                if self.multi_contrast:
                    loss_simclr_seq, logits_seq, labels_seq = info_nce_loss_for_loop(seq_emb[0: -self.batch_size * 2], 
                                                            self.batch_size, 
                                                            self.n_views,
                                                            self.temperature_schedule[epoch_counter - 1], 
                                                            self.device, 
                                                            self.criterion)
                else:
                    loss_simclr_seq = 0.
                    logits_seq, labels_seq = None, None
                
                ### SimCSE loss
                cat_two_view = torch.cat([simclr_emb_contrast[0: self.batch_size], 
                                          simclr_emb_contrast[-self.batch_size * 2: -self.batch_size],
                                          simclr_emb_contrast[-self.batch_size:]], dim=0)
                loss_simcse, logits_simces, labels_simces = info_nce_loss(cat_two_view, 
                                                            self.batch_size, 
                                                            3,
                                                            self.temperature_schedule[epoch_counter - 1], 
                                                            self.device, 
                                                            self.criterion)
                ### AE loss
                loss_simclr *= 2.0 ## changed here
                loss_simcse *= 1.0
                loss_simclr_seq *= 0.5
                loss = loss_simclr + loss_simcse + loss_simclr_seq
                loss.backward()
                self.optimizer.step()
                # logger
                if n_iter % self.log_every_n_steps == 0:
                    acc1 = accuracy(logits, labels)[0]
                    acc1_cse = accuracy(logits_simces, labels_simces)[0]
                    if self.multi_contrast:
                        loss_simclr_seq_dis = loss_simclr_seq.item()
                        acc1_seq = accuracy(logits_seq, labels_seq)[0].item()
                    else:
                        loss_simclr_seq_dis = 0.
                        acc1_seq = 0.
                    
                    if self.multi_contrast:
                        logger.info(f"-->Epoch:{epoch_counter}/{self.epochs}" +
                                    f"|LossNViews:{loss_simclr.item():.2f}" +
                                    f"|LossNViews_seq:{loss_simclr_seq_dis:.2f}" +
                                    f"|LossMask:{loss_simcse.item():.2f}" + 
                                    f"|NViews_Acc1:{acc1.item():.2f}" +
                                    f"|Mask_Acc1:{acc1_seq:.2f}" +
                                    f"|LR:{self.optimizer.param_groups[0]['lr']:.8f}" + 
                                    f"|Temp:{self.temperature_schedule[epoch_counter - 1]}")
                    else:
                        logger.info(f"-->Epoch:{epoch_counter}/{self.epochs}|LossSum:{loss.item():.3f}" +
                                    f"|LossNViews:{loss_simclr.item():.3f}" +
                                    f"|LossMask:{loss_simcse.item():.3f}" + 
                                    f"|NViews_Acc1:{acc1.item():.2f}" +
                                    f"|Mask_Acc1:{acc1_cse.item():.2f}" +
                                    f"|LR:{self.optimizer.param_groups[0]['lr']:.8f}" + 
                                    f"|Temp: {self.temperature_schedule[epoch_counter - 1]}")
                n_iter += 1
            self.scheduler.step()
            self.model.eval()
            losses_valid = self.valid(valid_loader)
            self.model.train()
            loss_record[epoch_counter] = losses_valid
            # save model checkpoints
            checkpoint_name = 'checkpoint_{}.pth'.format(epoch_counter)
            torch.save(self.model.state_dict(), os.path.join(self.model_save_folder, checkpoint_name))
            logger.info(f"--> Model checkpoint has been saved at {os.path.join(self.model_save_folder, checkpoint_name)}.")
        logger.info("--> Training has finished.")
        return loss_record

    def valid(self, valid_loader: DataLoader = None):
        logger.info(f"--> Start self-supervised validation.")
        logger.info(f"--> Valid with {self.device} device.")
        self.model.eval()
        loss = 0.
        n_iter = 0.
        combined_x_o = []
        combined_x_p1 = []
        combined_x_p2 = []
        i = 1
        with torch.no_grad():
            for n_views_tuple_list, _  in tqdm(valid_loader):
                n_views = len(n_views_tuple_list)
                assert n_views == 3, ValueError("Views number is not equal with each other.")
                seq_tokens_inputs, mean_inputs, var_inputs, whole_bp_cov_tnf_inputs = self.get_model_inputs(n_views_tuple_list)
                # ============ multi-res forward passes ... ============
                simclr_emb_contrast, _, _ = self.model.forward(seq_tokens_inputs, mean_inputs, var_inputs, whole_bp_cov_tnf_inputs)
                # align loss
                x_o = simclr_emb_contrast[0: self.batch_size]
                x_p1 = simclr_emb_contrast[self.batch_size: self.batch_size * 2]
                x_p2 = simclr_emb_contrast[self.batch_size * 2: self.batch_size * 3]
                combined_x_o.append(x_o)
                combined_x_p1.append(x_p1)
                combined_x_p2.append(x_p2)
                if i % 10 == 0:
                    combined_x_o = torch.cat(combined_x_o, dim=0)
                    combined_x_p1 = torch.cat(combined_x_p1, dim=0)
                    combined_x_p2 = torch.cat(combined_x_p2, dim=0)
                    loss_align = (align_loss(combined_x_o, combined_x_p1) + align_loss(combined_x_o, combined_x_p2) + \
                        align_loss(combined_x_p1, combined_x_p2)) / 3.
                    # uniform loss
                    loss_uni = (uniform_loss(combined_x_o) + uniform_loss(combined_x_p1) + uniform_loss(combined_x_p2)) / 3.
                    cur_loss = loss_align + loss_uni
                    loss += cur_loss
                    n_iter += 1.
                    combined_x_o = []
                    combined_x_p1 = []
                    combined_x_p2 = []
                i += 1
        if len(combined_x_o) != 0:
            combined_x_o = torch.cat(combined_x_o, dim=0)
            combined_x_p1 = torch.cat(combined_x_p1, dim=0)
            combined_x_p2 = torch.cat(combined_x_p2, dim=0)
            loss_align = (align_loss(combined_x_o, combined_x_p1) + align_loss(combined_x_o, combined_x_p2) + \
                align_loss(combined_x_p1, combined_x_p2)) / 3.
            # uniform loss
            loss_uni = (uniform_loss(combined_x_o) + uniform_loss(combined_x_p1) + uniform_loss(combined_x_p2)) / 3.
            cur_loss = loss_align + loss_uni
            loss += cur_loss
            n_iter += 1.
            combined_x_o = []
            combined_x_p1 = []
            combined_x_p2 = []
        self.model.train()
        logger.info(f"--> The validation loss is {loss / n_iter}. Validation has finished.")
        return loss / n_iter

    def inference(self, infer_loader: DataLoader, output_folder_path: str, model_weight_path: str = None):
        logger.info(f"--> Inference with {self.device} device.")
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
        self.model.eval()
        self.optimizer.zero_grad()
        n_iter = 0
        simclr_contigname2emb_norm_ndarray = {}
        with torch.no_grad():
            for n_views_tuple_list, contigname_file_list in tqdm(infer_loader):
                seq_tokens_inputs, mean_inputs, var_inputs, whole_bp_cov_tnf_inputs = self.get_model_inputs(n_views_tuple_list)
                # ============ multi-res forward passes ... ============
                simclr_emb, _, _ = self.model.forward(seq_tokens_inputs, mean_inputs, var_inputs, whole_bp_cov_tnf_inputs)
                for i in range(len(contigname_file_list)):
                    prefix, suffix = os.path.splitext(contigname_file_list[i])
                    if suffix == ".pkl":
                        contigname = ">" + prefix
                    else:
                        contigname = ">" + contigname_file_list[i]
                    simclr_contigname2emb_norm_ndarray[contigname] = simclr_emb[i].detach().to("cpu").numpy()
                n_iter += 1
        writePickle(os.path.join(output_folder_path, f"SimCLR_contigname2emb_norm_ndarray.pkl"), simclr_contigname2emb_norm_ndarray)


class PretrainTrainer(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str,
                 epochs: int,
                 model_save_folder: str,
                 batch_size: int,
                 log_every_n_steps: int = 20
                 ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.model_save_folder = model_save_folder
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0).to(self.device)


    def train(self, train_loader: DataLoader, valid_loader: DataLoader,  model_weight_path = None):
        logger.info(f"--> Start self-supervised training with {self.epochs} epochs.")
        logger.info(f"--> Training with {self.device} device.")
        loss_record = {}
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            logger.info(f"--> Model weight has been load.")
        for epoch_counter in range(1, self.epochs + 1):
            n_iter = 0.
            self.model.train()
            for batch_genome_seq_tokens, batch_taxon_labels in tqdm(train_loader):
                batch_genome_seq_tokens = batch_genome_seq_tokens.to(torch.float32).to(self.device, non_blocking=True)
                batch_taxon_labels = batch_taxon_labels.to(self.device, non_blocking=True)
                # ============ multi-res forward passes ... ============
                self.optimizer.zero_grad()
                genom_seq_fea = self.model.forward(batch_genome_seq_tokens)
                # taxonomic loss
                loss_taxon = self.criterion(genom_seq_fea, batch_taxon_labels)
                loss = loss_taxon
                loss.backward()
                self.optimizer.step()
                # logger
                if n_iter % self.log_every_n_steps == 0:
                    acc1_taxon = accuracy(genom_seq_fea, batch_taxon_labels)[0]
                    logger.info(f"--> Epoch: {epoch_counter} / {self.epochs} LossSum:{loss.item():.4f}" +
                                f"|LossTaxon: {loss_taxon:.3f}" + 
                                f"|Taxon_Acc_1: {acc1_taxon.item():.2f}" +
                                f"|LR: {self.optimizer.param_groups[0]['lr']:.8f}")
                n_iter += 1
            self.scheduler.step()
            self.model.eval()
            losses_valid, f_acc1 = self.valid(valid_loader)
            self.model.train()
            loss_record[epoch_counter] = losses_valid
            # save model checkpoints
            checkpoint_name = f'checkpoint_{epoch_counter}_loss{losses_valid:.3f}_acc{f_acc1:.2f}.pth'
            torch.save(self.model.state_dict(), os.path.join(self.model_save_folder, checkpoint_name))
            logger.info(f"--> Model checkpoint has been saved at {os.path.join(self.model_save_folder, checkpoint_name)}.")
        logger.info("--> Training has finished.")
        return loss_record


    def valid(self, valid_loader):
        losses = 0.
        self.model.eval()
        self.optimizer.zero_grad()
        acc_store = []
        with torch.no_grad():
            index = 0
            for  batch_genome_seq_tokens, batch_taxon_labels in tqdm(valid_loader):
                batch_genome_seq_tokens = batch_genome_seq_tokens.to(torch.float32).to(self.device, non_blocking=True)
                batch_taxon_labels = batch_taxon_labels.to(self.device, non_blocking=True)
                # ============ multi-res forward passes ... ============
                genom_seq_fea = self.model.forward(batch_genome_seq_tokens)
                # taxonomic loss
                loss_taxon = self.criterion(genom_seq_fea, batch_taxon_labels)
                loss = loss_taxon
                losses += loss.item()
                acc1 = accuracy(genom_seq_fea, batch_taxon_labels)[0]
                acc_store.append(acc1.item())
                index += 1
        f_acc1 = sum(acc_store) / len(acc_store) + 0.
        logger.info(f"--> Valid loss is {losses / index + 0.}, Acc1 is {f_acc1}.")
        return losses / index + 0., f_acc1


class PretrainVQVAETrainer(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 device: str,
                 epochs: int,
                 model_save_folder: str,
                 batch_size: int,
                 log_every_n_steps: int = 20
                 ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.model_save_folder = model_save_folder
        self.log_every_n_steps = log_every_n_steps
        self.batch_size = batch_size
        self.esp = 1e-9
        self.criteria = nn.KLDivLoss(reduction="batchmean")

    def custom_KL(self, y_pred, y_true):
        y_pred = torch.flatten(y_pred, end_dim=1)
        y_pred = F.log_softmax(y_pred, dim=1)
        y_true = torch.flatten(y_true, end_dim=1)
        # print(y_true, y_true.sum(dim=-1))
        return self.criteria(y_pred, y_true)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader,  model_weight_path = None):
        logger.info(f"--> Start VQ-VAE training with {self.epochs} epochs. Cross entropy loss. 1024.")
        logger.info(f"--> Training with {self.device} device.")
        loss_record = {}
        if model_weight_path is not None:
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            logger.info(f"--> Model weight has been load.")
        for epoch_counter in range(1, self.epochs + 1):
            n_iter = 0.
            self.model.train()
            for batch_genome_seq_tokens, _, _ in tqdm(train_loader):
                batch_genome_seq_tokens = batch_genome_seq_tokens.to(torch.float32).to(self.device, non_blocking=True)
                # ============ multi-res forward passes ... ============
                self.optimizer.zero_grad()
                x_hat, commit_loss, indices, logit = self.model(batch_genome_seq_tokens)
                # rec_loss = F.mse_loss(batch_genome_seq_tokens, x_hat, reduction="sum") / self.batch_size * 20.
                rec_loss = self.custom_KL(logit, batch_genome_seq_tokens) * 4.
                commit_loss *= 10.0
                loss = rec_loss + commit_loss
                loss.backward()
                self.optimizer.step()
                # logger
                if n_iter % self.log_every_n_steps == 0:
                    logger.info(f"--> Epoch: {epoch_counter} / {self.epochs} LossSum:{loss.item():.4f}" +
                                f"|Rec Loss: {rec_loss.item():.4f}" + 
                                f"|Commit Loss: {commit_loss.item():.4f}" + 
                                f"|LR: {self.optimizer.param_groups[0]['lr']:.8f}")
                n_iter += 1
            self.scheduler.step()
            print("#######")
            print(batch_genome_seq_tokens)
            print(x_hat)
            print(indices)
            print("#######")
            self.model.eval()
            losses_valid = self.valid(valid_loader)
            self.model.train()
            loss_record[epoch_counter] = losses_valid
            # save model checkpoints
            checkpoint_name = f'checkpoint_{epoch_counter}_loss_{losses_valid:.4f}.pth'
            torch.save(self.model.state_dict(), os.path.join(self.model_save_folder, checkpoint_name))
            logger.info(f"--> Model checkpoint has been saved at {os.path.join(self.model_save_folder, checkpoint_name)}.")
        logger.info("--> Training has finished.")
        return loss_record


    def valid(self, valid_loader):
        losses = 0.
        self.model.eval()
        self.optimizer.zero_grad()
        with torch.no_grad():
            index = 0
            for  batch_genome_seq_tokens, _, seq_len in tqdm(valid_loader):
                batch_genome_seq_tokens = batch_genome_seq_tokens.to(torch.float32).to(self.device, non_blocking=True)
                # ============ multi-res forward passes ... ============
                x_hat, commit_loss, indices, logit = self.model(batch_genome_seq_tokens)
                rec_loss = self.custom_KL(logit, batch_genome_seq_tokens) * 4.
                commit_loss *= 10.0
                loss = rec_loss + commit_loss
                losses += loss.item()
                index += 1
        logger.info(f"--> Valid loss is {losses / index + 0.}")
        return losses / index + 0.