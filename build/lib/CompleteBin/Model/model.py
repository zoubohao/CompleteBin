

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from CompleteBin.Model.layers import (
    MLP,
    CompleteBinBaseModel,   # 用于 pretrain_model，保持不变
)
from CompleteBin.logger import get_logger

logger = get_logger()


class CompleteBinModel(nn.Module):

    def __init__(self,
                 kmer_dim,
                 whole_kmer_dim,
                 feature_dim,
                 num_bam_files,
                 split_parts_list: List,
                 dropout: float,
                 device: str,
                 n_views: int,
                 seq_length: int,
                 hidden_dim=512,
                 layers=4,
                 num_classes=15781,
                 use_pretrained: bool = True) -> None:
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_views = n_views
        self.seq_length = seq_length
        logger.info(f"--> Transformer Model hidden dim: {hidden_dim}, layers: {layers}, {5}, device: {device}. Fix")
        self.cov_mean_model = nn.Sequential(MLP(num_bam_files, 256, 512, p=dropout),
                                            nn.Linear(512, hidden_dim // 2, bias=False)).to(device)
        self.cov_mean_linear = nn.Linear(num_bam_files, hidden_dim // 2, bias=False)
        
        self.cov_var_model = nn.Sequential(MLP(num_bam_files, 256, 512, p=dropout),
                                           nn.Linear(512, hidden_dim // 2, bias=False)).to(device)
        self.cov_var_linear = nn.Linear(num_bam_files, hidden_dim // 2, bias=False)
        
        self.cov_tnf_model = nn.Sequential(MLP(num_bam_files * whole_kmer_dim, 2048, 1024, p=dropout),
                                           nn.Linear(1024, hidden_dim, bias=False)).to(device)
        
        self.pure_cov_encoder = nn.Sequential(
            MLP(num_bam_files * 2, 256, 512, p=dropout),
            nn.Linear(512, hidden_dim, bias=False),
        ).to(device)
        
        self.token_proj = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)

        self.use_pretrained = use_pretrained
        if use_pretrained:
            self.pretrain_model = CompleteBinBaseModel(kmer_dim, num_classes, split_parts_list, dropout, False, hidden_dim, layers, True).to(device)
        else:
            self.pretrain_model = None
        self.train_model = CompleteBinBaseModel(kmer_dim, None, split_parts_list, dropout, False, hidden_dim, layers, True).to(device)
        self.null_taxon_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.projector_simclr = nn.Linear(hidden_dim, feature_dim, bias=False).to(device)

    def fix_param_in_pretrain_model(self):
        if self.pretrain_model is not None:
            logger.info("--> Fixed the weights of pretrain model.")
            i = 0
            for _, v in self.pretrain_model.named_parameters():
                v.requires_grad = False
                i += 1
            logger.info(f"--> Number of {i} parameters have been fixed.")

    def load_weight_for_model(self, pretrain_model_weight_path: str):
        if self.pretrain_model is not None:
            self.pretrain_model.load_state_dict(torch.load(pretrain_model_weight_path, map_location=self.device), strict=True)
        self.train_model.load_state_dict(torch.load(pretrain_model_weight_path, map_location=self.device), strict=False)
        logger.info(f"--> Loaded the pretrain weight.")

    # cur_seq_eql_tokens: [12, 136]
    # cur_seq_rad_tokens: [5 136]
    # mean_tokens shape: (5, 2) for 2 bam files
    # std_tokens shape: (5, 2)
    # whole bp_ cov_ tnf_array shape: (2, 136)
    def forward(self, seq_rad_tokens_n_views, mean_tokens_n_views, var_tokens_n_views, whole_bp_cov_tnf_inputs, n_views):
        # get the seq taxon embedding from pretrain model (or learnable null token)
        if self.use_pretrained:
            with torch.no_grad():
                seq_taxon_enc = self.pretrain_model(seq_rad_tokens_n_views)[:, None, :]  # b, hidden_dim
        else:
            b_n_view = seq_rad_tokens_n_views.shape[0]
            seq_taxon_enc = self.null_taxon_token.expand(b_n_view, -1, -1)  # b, 1, hidden_dim

        cov_mean_fea_enc = self.cov_mean_model(mean_tokens_n_views)  # b, l, hidden_dim / 2
        cov_mean_linear = self.cov_mean_linear(mean_tokens_n_views)
        
        cov_var_fea_enc = self.cov_var_model(var_tokens_n_views)  # b, l, hidden_dim / 2
        cov_var_linear = self.cov_var_linear(var_tokens_n_views)

        global_mean_cov = mean_tokens_n_views[:, 0, :]   # (B, num_bam_files)
        global_var_cov = var_tokens_n_views[:, 0, :]      # (B, num_bam_files)
        global_cov = torch.cat([global_mean_cov, global_var_cov], dim=-1)
        pure_cov_token = self.pure_cov_encoder(global_cov)[:, None, :] # (B, num_bam_files * 2)

        seq_rad_tokens_n_views = self.train_model.get_token_proj(seq_rad_tokens_n_views)  # b, l, hidden_dim
        g_kmer = seq_rad_tokens_n_views[:, 0: 1, :]

        seq_tokens_inputs = torch.cat([seq_rad_tokens_n_views, 
                                       cov_mean_fea_enc, cov_mean_linear, 
                                       cov_var_fea_enc, cov_var_linear], dim=-1)  # b, l, hidden_dim * 3
        seq_tokens_inputs = self.token_proj(seq_tokens_inputs)

        whole_bp_cov_tnf_inputs = torch.flatten(whole_bp_cov_tnf_inputs, 1)

        # concanated these tokens
        seq_tokens_inputs = torch.cat([seq_tokens_inputs[:, 0: 1, :], pure_cov_token, g_kmer,
                                      seq_tokens_inputs[:, 1:, :], seq_taxon_enc], dim=1)  # b, l+1, hidden_dim
        b_n_view, _ = whole_bp_cov_tnf_inputs.shape
        bs = b_n_view // n_views

        # input to model
        seq_fea_enc = self.train_model.get_feature_of_tokens(seq_tokens_inputs)[:, 0, :]  # b, l, hidden_dim

        all_info_seq = F.normalize(self.projector_simclr(seq_fea_enc))  # b, c

        return all_info_seq, bs, seq_taxon_enc.squeeze(1)
