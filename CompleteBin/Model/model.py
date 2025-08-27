
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from CompleteBin.logger import get_logger

logger = get_logger()


def encode_seq2vec(seq_tensor):
    sft = torch.softmax(torch.mean(seq_tensor, dim=-1, keepdim=True), dim=1)
    seq_rep = torch.sum(seq_tensor * sft, dim=1)  # [B, C]
    return seq_rep


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, out_dim: int,  p: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.w1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, p: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.drop_m = nn.Dropout(p)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.drop_m(x)
        x = self.w2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout_prob):
        super().__init__()
        self.atten = nn.MultiheadAttention(dim, n_heads, dropout_prob, batch_first=True)

    def forward(self, x):
        atten_out, _ = self.atten(x, x.clone(), x.clone(),  need_weights = False)
        return atten_out


class TransformerEncoder(nn.Module):

    def __init__(self, feature_dim, hidden_dim, dropout_prob) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(8, feature_dim, dropout_prob)
        self.attention_norm = RMSNorm(feature_dim, 1e-6)
        self.ffw = FeedForward(feature_dim, hidden_dim, multiple_of=256, p=dropout_prob)
        self.ffw_norm = RMSNorm(feature_dim, 1e-6)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.ffw(self.ffw_norm(h))
        return out


class DeeperBinBaseModel(nn.Module):
    
    def __init__(self,
                 kmer_dim,
                 classes_num,
                 split_parts_list: List,
                 dropout: float,
                 hidden_dim: int = 2048,
                 layers = 4,
                 output_embedding = False) -> None:
        super(DeeperBinBaseModel, self).__init__()
        self.layers = layers
        self.output_embedding = output_embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, sum(split_parts_list) + 100, hidden_dim, requires_grad=True), 
            requires_grad = True)
        # seq
        self.token_proj = nn.Linear(kmer_dim, hidden_dim, bias=False)
        self.transformer_model = nn.ModuleList(
            [TransformerEncoder(hidden_dim, hidden_dim, dropout)
             for _ in range(self.layers)]
        )
        if classes_num == 0 or classes_num is None:
            self.out_linear =  None
        else:
            self.out_linear =  nn.Linear(hidden_dim, classes_num, bias=False)
    
    def get_token_proj(self, x):
        return self.token_proj(x)

    def get_feature_of_tokens(self, x):
        _, l, _ = x.shape
        x += self.pos_embedding[:, 0: l]
        for i in range(self.layers):
            x = self.transformer_model[i](x)
        return x

    def forward(self, seq_tokens_inputs):
        seq_fea_enc = encode_seq2vec(self.get_feature_of_tokens(self.get_token_proj(seq_tokens_inputs)))
        if self.output_embedding or self.out_linear is None:
            return seq_fea_enc
        return self.out_linear(seq_fea_enc)


class DeeperBinModel(nn.Module):
    
    def __init__(self,
                 kmer_dim,
                 whole_kmer_dim,
                 feature_dim,
                 num_bam_files,
                 split_parts_list: List,
                 dropout: float,
                 device: str,
                 hidden_dim = 512,
                 layers = 3,
                 multi_contrast=False) -> None:
        super(DeeperBinModel, self).__init__()
        self.device = device
        self.multi_contrast=multi_contrast
        logger.info(f"--> Model hidden dim: {hidden_dim}, layers: {layers}, device: {device}. Fix")
        # self.cov_mean_model = nn.Sequential(MLP(num_bam_files, 256, 512, p=dropout),
        #                                     nn.Linear(512, 512, bias=False)).to(device)
        self.cov_var_model = nn.Sequential(MLP(num_bam_files, 256, 512, p=dropout),
                                            nn.Linear(512, 512, bias=False)).to(device)
        self.cov_tnf_model = nn.Sequential(MLP(num_bam_files * whole_kmer_dim, 2048, 1024, p=dropout),
                                            nn.Linear(1024, 1024, bias=False)).to(device)
        
        self.pretrain_model = DeeperBinBaseModel(kmer_dim, 15781, split_parts_list, dropout, hidden_dim, layers, True).to(device)
        self.train_model = DeeperBinBaseModel(kmer_dim, None, split_parts_list, dropout, hidden_dim, layers, True).to(device)
        
        self.projector_simclr = nn.Sequential(
            MLP(hidden_dim * 2 + 1024 + 512, hidden_dim * 8, hidden_dim * 6, dropout),
            MLP(hidden_dim * 6, hidden_dim * 4, hidden_dim * 2, dropout),
            nn.Linear(hidden_dim * 2, feature_dim, bias=False)
        ).to(device)

    def fix_param_in_pretrain_model(self):
        logger.info("--> Fixed the weights of pretrain model.")
        i = 0
        for _, v in self.pretrain_model.named_parameters():
            v.requires_grad = False
            i += 1
        logger.info(f"--> Number of {i} parameters have been fixed.")

    def load_weight_for_model(self, pretrain_model_weight_path: str):
        self.pretrain_model.load_state_dict(torch.load(pretrain_model_weight_path, map_location=self.device), strict=True)
        self.train_model.load_state_dict(torch.load(pretrain_model_weight_path, map_location=self.device), strict=False)
        logger.info(f"--> Have loaded the pretrain weight.")

    def forward(self, seq_tokens_inputs, mean_val, var_val, whole_bp_cov_tnf_inputs):
        # get the seq taxon embedding from pretrain model
        # print(mean_val, mean_val.shape)
        with torch.no_grad():
            seq_taxon_enc = self.pretrain_model(seq_tokens_inputs)
        # cov_mean_fea_enc = self.cov_mean_model(mean_val)
        cov_var_fea_enc = self.cov_var_model(var_val)
        whole_bp_cov_tnf_inputs = torch.flatten(whole_bp_cov_tnf_inputs, 1)
        cov_tnf_fea_enc = self.cov_tnf_model(whole_bp_cov_tnf_inputs)
        seq_fea_enc = self.train_model.get_feature_of_tokens(self.train_model.get_token_proj(seq_tokens_inputs))
        rep_seq_fea = encode_seq2vec(seq_fea_enc) # seq_fea_enc[:, 0, :]
        all_info_seq = torch.cat([rep_seq_fea, seq_taxon_enc, cov_tnf_fea_enc, cov_var_fea_enc], dim=-1)
        all_info_seq = F.normalize(self.projector_simclr(all_info_seq))
        if self.multi_contrast:
            return all_info_seq, F.normalize(seq_fea_enc), None
        return all_info_seq, None, None


# #### UESLESS ####
# class SimVQVAE(torch.nn.Module):
    
#     def __init__(self, kmer_dim, split_parts_list, infer_mode = False, hidden_dim = 512, dropout = 0.05, layers = 3):
#         super().__init__()
#         self.layers = layers
        
#         self.token_proj = nn.Linear(kmer_dim, hidden_dim, bias=False)
#         self.pos_embedding = nn.Parameter(
#             torch.zeros(1, sum(split_parts_list) + 100, hidden_dim, requires_grad=True), 
#             requires_grad = True)
#         self.enc = nn.ModuleList(
#             [TransformerEncoder(hidden_dim, hidden_dim, dropout)
#             for _ in range(layers)]
#         )
#         self.vq_med_layer = SimVQ(
#             dim = hidden_dim,
#             codebook_size = 1024,
#             codebook_transform = nn.Sequential(
#                 nn.Linear(hidden_dim, 1024),
#                 nn.ReLU(),
#                 nn.Linear(1024, hidden_dim)),
#             rotation_trick = True  # use rotation trick from Fifty et al.
#         )
#         if not infer_mode:
#             self.dec = nn.ModuleList(
#             [TransformerEncoder(hidden_dim, hidden_dim, dropout)
#             for _ in range(layers)]
#             )
#             self.token_rev = nn.Linear(hidden_dim, kmer_dim, bias=False)
#         else:
#             self.dec = None
#             self.token_rev = None
#         self.criteria = nn.KLDivLoss(reduction="batchmean")

#     def custom_KL(self, y_pred, y_true):
#         y_pred = torch.flatten(y_pred, end_dim=1)
#         y_pred = F.log_softmax(y_pred, dim=1)
#         y_true = torch.flatten(y_true, end_dim=1)
#         # print(y_true, y_true.sum(dim=-1))
#         return self.criteria(y_pred, y_true)

#     def get_feature_of_tokens(self, x):
#         _, l, _ = x.shape
#         x += self.pos_embedding[:, 0: l]
#         for i in range(self.layers):
#             x = self.enc[i](x)
#         return x
    
#     def dec_tokens(self, x):
#         _, l, _ = x.shape
#         x += self.pos_embedding[:, 0: l]
#         for i in range(self.layers):
#             x = self.dec[i](x)
#         return x
    
#     def forward(self, x):
#         # print(x.sum(dim=-1))
#         x = self.token_proj(x)
#         z_e = self.get_feature_of_tokens(x)
#         z_q, indices, commit_loss = self.vq_med_layer(z_e)
#         if self.dec is None:
#             return indices
#         logit = self.token_rev(self.dec_tokens(z_q))
#         x_hat = F.softmax(logit, dim=-1)
#         return x_hat, commit_loss, indices, logit
