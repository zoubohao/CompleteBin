

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
                 num_classes=15324,
                 use_pretrained: bool = True) -> None:
        """kmer_dim       : token 宽度，原始 k-mer 组成（4-mer → 136 类）。
        whole_kmer_dim : coverage 分箱那一路用的宽度，同样是原始 k-mer 宽度。
        """
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_views = n_views
        self.seq_length = seq_length
        self.kmer_dim = kmer_dim
        logger.info(f"--> Transformer Model hidden dim: {hidden_dim}, layers: {layers}, {5}, device: {device}. Fix")
        logger.info(f"--> Token width: {self.kmer_dim} (coverage stream {whole_kmer_dim}).")
        self.cov_mean_model = nn.Sequential(MLP(num_bam_files, 256, 512, p=dropout),
                                            nn.Linear(512, hidden_dim // 2, bias=False)).to(device)
        self.cov_mean_linear = nn.Linear(num_bam_files, hidden_dim // 2, bias=False)
        
        self.cov_std_model = nn.Sequential(MLP(num_bam_files, 256, 512, p=dropout),
                                           nn.Linear(512, hidden_dim // 2, bias=False)).to(device)
        self.cov_std_linear = nn.Linear(num_bam_files, hidden_dim // 2, bias=False)
        
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
        # 冻结编码器和可训练编码器吃的是同一份 token（原始 4-mer 组成）。
        self.train_model = CompleteBinBaseModel(kmer_dim, None, split_parts_list, dropout, False, hidden_dim, layers, True).to(device)
        self.null_taxon_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.projector_simclr = nn.Linear(hidden_dim, feature_dim, bias=False).to(device)

    def train(self, mode: bool = True):
        """Keep the frozen pretrained encoder in eval mode.

        ``nn.Module.train`` recurses into submodules, so without this the
        pretrained model's attention dropout stays active during downstream
        training. That injects noise into the taxon token and into the
        same-genome mask derived from it.
        """
        super().train(mode)
        if self.pretrain_model is not None:
            self.pretrain_model.eval()
        return self

    def fix_param_in_pretrain_model(self):
        if self.pretrain_model is not None:
            logger.info("--> Fixed the weights of pretrain model.")
            i = 0
            for _, v in self.pretrain_model.named_parameters():
                v.requires_grad = False
                i += 1
            logger.info(f"--> Number of {i} parameters have been fixed.")

    def load_weight_for_model(self, pretrain_model_weight_path: str):
        """加载预训练权重。

        分类头现在是裸线性层（输入是主干原始向量，不做归一化），新旧 checkpoint
        会有对不上的键：
          · 旧 checkpoint 有 ``out_linear.logit_scale``，新模型没有
          · 更早的 checkpoint 有 ``out_norm.weight/bias``（LayerNorm 时代），
            新模型已经没有这层
        形状或名字对不上的键直接跳过（strict=False + 手动筛），
        transformer 主体照常加载；分类头那几层由后续训练学出来。

        另外 token 宽度可能不同：预训练流 136（原始 4-mer）、可训练流 128（白化），
        所以 ``token_proj.weight`` 也可能对不上，同样跳过。
        """
        loaded = torch.load(pretrain_model_weight_path, map_location=self.device)
        state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded

        def _compatible(model, sd):
            own = model.state_dict()
            kept, dropped = {}, []
            for k, v in sd.items():
                if k in own and tuple(own[k].shape) == tuple(v.shape):
                    kept[k] = v
                else:
                    dropped.append(k)
            return kept, dropped

        if self.pretrain_model is not None:
            kept, dropped = _compatible(self.pretrain_model, state_dict)
            self.pretrain_model.load_state_dict(kept, strict=False)
            logger.info(f"--> pretrain_model: loaded {len(kept)} tensors, "
                        f"reinitialised {len(dropped)} (shape/name mismatch): "
                        f"{dropped[:5]}")
        kept, dropped = _compatible(self.train_model, state_dict)
        self.train_model.load_state_dict(kept, strict=False)
        logger.info(f"--> train_model: loaded {len(kept)} tensors, "
                    f"reinitialised {len(dropped)}: {dropped[:5]}")
        logger.info(f"--> Loaded the pretrain weight.")

    # seq_rad_tokens_n_views: [b*n_views, 6, 136]  4-mer 组成向量（冻结编码器 + 可训练编码器共用）
    # mean_tokens shape: (5, 2) for 2 bam files
    # std_tokens shape: (5, 2)
    # whole bp_ cov_ tnf_array shape: (2, 136)
    def forward(self, seq_rad_tokens_n_views,
                mean_tokens_n_views, std_tokens_n_views, whole_bp_cov_tnf_inputs, n_views):
        # get the seq taxon embedding from pretrain model (or learnable null token)
        if self.use_pretrained:
            with torch.no_grad():
                seq_taxon_enc = self.pretrain_model.encode(seq_rad_tokens_n_views)[:, None, :] # b, hidden_dim
        else:
            b_n_view = seq_rad_tokens_n_views.shape[0]
            seq_taxon_enc = self.null_taxon_token.expand(b_n_view, -1, -1)  # b, 1, hidden_dim

        cov_mean_fea_enc = self.cov_mean_model(mean_tokens_n_views)  # b, l, hidden_dim / 2
        cov_mean_linear = self.cov_mean_linear(mean_tokens_n_views)
        
        cov_std_fea_enc = self.cov_std_model(std_tokens_n_views)  # b, l, hidden_dim / 2
        cov_std_linear = self.cov_std_linear(std_tokens_n_views)

        global_mean_cov = mean_tokens_n_views[:, 0, :]   # (B, num_bam_files)
        global_std_cov = std_tokens_n_views[:, 0, :]      # (B, num_bam_files)
        global_cov = torch.cat([global_mean_cov, global_std_cov], dim=-1)
        pure_cov_token = self.pure_cov_encoder(global_cov)[:, None, :] # (B, num_bam_files * 2)

        # 可训练编码器和冻结编码器吃同一份 token（原始 k-mer 组成）。
        train_tokens = self.train_model.get_token_proj(seq_rad_tokens_n_views)  # b, l, hidden_dim
        g_kmer = train_tokens[:, 0: 1, :]

        seq_tokens_inputs = torch.cat([train_tokens,
                                       cov_mean_fea_enc, cov_mean_linear, 
                                       cov_std_fea_enc, cov_std_linear], dim=-1)  # b, l, hidden_dim * 3
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
