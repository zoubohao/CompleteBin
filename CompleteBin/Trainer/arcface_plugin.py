"""
ArcFace Margin Plugin for CompleteBin contrastive learning.

Provides ArcFace additive angular margin (cos(θ+m)) on positive pairs,
with per-contig adaptive margin based on sequence length.

Usage (in ssmt_v2.py / trainer.py):
    from CompleteBin.Trainer.arcface_plugin import (
        ArcFaceConfig,
        LengthAugmentedDataset,
        compute_adaptive_margin,
        arcface_debiased_info_nce_loss,
    )

    # Wrap training dataset to inject contig lengths
    contig_lengths_list = [len(cur_tuples[0]) for _, cur_tuples in save_array]
    wrapped_dataset = LengthAugmentedDataset(training_set, contig_lengths_list)

    # In training loop
    margins = compute_adaptive_margin(contig_lengths, config)
    loss, logits, labels = arcface_debiased_info_nce_loss(
        features, batch_size, n_views, temperature, device, criterion,
        sg_mask, margins=margins,
    )

Reference:
    useless/ContrastiveStrategies/instance_loss.py  — ArcFace formula + pairwise InfoNCE
    useless/ContrastiveStrategies/instance_config.py — hyperparameter defaults
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from CompleteBin.logger import get_logger

logger = get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArcFaceConfig:
    """Configuration for ArcFace additive angular margin.

    Attributes
    ----------
    margin_max : float
        Maximum additive angular margin in radians.
        0.06 rad ≈ 3.44° — conservative default from reference implementation.
    use_adaptive_margin : bool
        If True, m_i = margin_max · σ(β·log(L_i) − b), scaling the margin by
        contig length. Short contigs get smaller margins (noisier k-mer features).
    adaptive_margin_beta : float
        Steepness of the sigmoid in the adaptive margin formula.
    adaptive_margin_b : float
        Bias term — log(L) at which sigma = 0.5 (half-max margin).
        Set via ``compute_adaptive_margin_b(contig_lengths, min_len)``,
        i.e. log(median contig length > min_len), so the sigmoid midpoint
        automatically adapts to the dataset's length distribution.
    eps : float
        Numerical stability constant.
    """

    margin_max: float = 0.0525
    use_adaptive_margin: bool = True
    adaptive_margin_beta: float = 1.0
    adaptive_margin_b: float = 7.600902459541082  # log(2000)
    eps: float = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Length-augmented dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class LengthAugmentedDataset(Dataset):
    """Wrap a TrainingDataset to append per-contig lengths to each batch item.

    ``TrainingDataset.__getitem__`` does not return contig names or lengths
    in training mode, so there is no way to look up per-contig metadata
    during the training loop.  This wrapper adds a parallel length tensor
    aligned with the base dataset indices, enabling per-contig adaptive
    margin computation without modifying ``dataset.py``.

    Parameters
    ----------
    base_dataset : Dataset
        The underlying ``TrainingDataset``.
    contig_lengths_list : list of float
        Per-contig lengths, strictly aligned with ``base_dataset`` indices.
    """

    def __init__(self, base_dataset: Dataset, contig_lengths_list: list):
        self.base = base_dataset
        self.lengths = contig_lengths_list

    def __getitem__(self, index):
        views = self.base[index]
        length = torch.tensor(self.lengths[index], dtype=torch.float32)
        return views, length

    def __len__(self):
        return len(self.base)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Adaptive margin computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_adaptive_margin_b(contig_lengths_list: list, min_len: int) -> float:
    """Compute the bias term ``b`` as log(median contig length).

    Only contigs longer than ``min_len`` are considered, since shorter
    contigs are filtered out during data preparation.  This makes the
    sigmoid midpoint data-dependent: at L = median, σ = 0.5 and the
    margin is m_max / 2.

    Parameters
    ----------
    contig_lengths_list : list of float
        Per-contig lengths (all contigs in the dataset).
    min_len : int
        Minimum contig length used for training.

    Returns
    -------
    b : float
        log(median) — ready for use as ``adaptive_margin_b`` in ``ArcFaceConfig``.
    """
    import math

    valid = [l for l in contig_lengths_list if l > min_len]
    if not valid:
        return math.log(2000.0)  # fallback
    median = sorted(valid)[len(valid) // 2]
    return math.log(max(median, 1.0))


def compute_adaptive_margin(contig_lengths: torch.Tensor, config: ArcFaceConfig) -> torch.Tensor:
    """Compute per-contig adaptive ArcFace margin.

    m_i = margin_max · σ(β · log(L_i) − b)

    Why log(L)?
    — k-mer frequency estimation variance scales as 1/L (Central Limit
      Theorem for Markov chains).  Using log(L) compresses the vast
      range of contig lengths (500 bp – 500 kbp) into a manageable [6, 13].

    Why sigmoid?
    — Smooth, monotonic, bounded in (0, 1).  Easy to tune via β and b.
    — At L = exp(b/β): σ = 0.5 → m_i = margin_max / 2.

    Why adaptive margin at all?
    — Short contigs (< 1000 bp) have noisy 4-mer frequency estimates.
      Forcing large margins on them would fit noise, not signal.
    — Long contigs (> 5000 bp) have reliable features and benefit from
      the full discriminative power of a large margin.

    Parameters
    ----------
    contig_lengths : torch.Tensor  [B]
        Contig lengths in base pairs.
    config : ArcFaceConfig

    Returns
    -------
    margins : torch.Tensor  [B]
        Per-contig margins in radians.
    """
    if not config.use_adaptive_margin:
        return torch.full_like(contig_lengths, config.margin_max)

    lengths = contig_lengths.float()
    log_L = torch.log(lengths.clamp(min=1.0))
    sigmoid_arg = config.adaptive_margin_beta * log_L - config.adaptive_margin_b
    scale = torch.sigmoid(sigmoid_arg)
    return config.margin_max * scale


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ArcFace-debiased InfoNCE loss
# ═══════════════════════════════════════════════════════════════════════════════

def arcface_debiased_info_nce_loss(
    features: torch.Tensor,
    batch_size: int,
    n_views: int,
    temperature: float,
    device: torch.device,
    criterion: torch.nn.Module,
    sg_mask: torch.Tensor,
    margins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-view debiased InfoNCE loss with ArcFace additive angular margin.

    Identical to ``debiased_info_nce_loss_optimized`` except that positive
    pairs receive the ArcFace margin: cos(θ) → cos(θ + m_i).

    When margins=0, the output is numerically identical to the original
    (no-margin) loss.

    Parameters
    ----------
    features : torch.Tensor  [n_views * batch_size, dim]
        L2-normalised embeddings.  Ordering: contig_0_view_0, …, contig_0_view_{V−1}, …
    batch_size : int
        Number of unique contigs per view.
    n_views : int
        Total number of effective views.
    temperature : float
        Softmax temperature.
    device : torch.device
    criterion : torch.nn.Module
        ``CrossEntropyLoss`` instance.
    sg_mask : torch.Tensor  [batch_size, batch_size]
        Same-genome mask (1.0 = same genome).
    margins : torch.Tensor  [batch_size]
        Per-contig ArcFace margin in radians.

    Returns
    -------
    loss : torch.Tensor  scalar
    logits : torch.Tensor  [total_anchors, 2*batch_size − 1]
    labels : torch.Tensor  [total_anchors]
    """
    bs = batch_size
    two_bs = 2 * bs
    dim = features.shape[-1]
    total_pairs = n_views * (n_views - 1) // 2
    eps = 1e-8

    # =================================================================
    # 1. normalize once
    # =================================================================
    features = F.normalize(features, dim=-1)
    views = features.view(n_views, bs, dim)

    # =================================================================
    # 2. batched self-similarities  [n_views, bs, bs]
    # =================================================================
    all_self = torch.matmul(views, views.transpose(1, 2))

    # =================================================================
    # 3. batched cross-similarities per anchor
    # =================================================================
    cross_sims: list[torch.Tensor] = []
    for a in range(n_views - 1):
        cross = torch.matmul(views[a], views[a + 1:].reshape(-1, dim).T)
        cross_sims.append(cross)

    # =================================================================
    # 4. precompute invariant masks (2D, contiguous)
    # =================================================================
    contig_ids_2v = torch.cat([torch.arange(bs, device=device),
                               torch.arange(bs, device=device)])
    labels_full = (contig_ids_2v.unsqueeze(0) == contig_ids_2v.unsqueeze(1))
    diag_2bs = torch.eye(two_bs, dtype=torch.bool, device=device)
    labels_nodiag = labels_full[~diag_2bs].view(two_bs, two_bs - 1).float()
    pos_mask = labels_nodiag.bool()
    neg_mask = ~pos_mask

    sg_nodiag = (sg_mask.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1))
    sg_nodiag = sg_nodiag[~diag_2bs].view(two_bs, two_bs - 1)
    false_neg_mask = neg_mask & (sg_nodiag > 0.5)

    # =================================================================
    # 5. precompute ArcFace margin trig values
    # =================================================================
    cos_m = torch.cos(margins)  # [bs]
    sin_m = torch.sin(margins)  # [bs]
    # Indices for positive pairs on the two cross-view diagonals
    pos_idx = torch.arange(bs, device=device)

    # =================================================================
    # 6. per-anchor batched processing
    # =================================================================
    logits_list = []
    labels_list = []
    loss_total = 0.0

    for a in range(n_views - 1):
        m = n_views - 1 - a
        n_rows = m * two_bs

        # ---- 6a. assemble [m, 2bs, 2bs] ----
        batch_sim = torch.empty(m, two_bs, two_bs, device=device)
        sim_aa = all_self[a]                                        # [bs, bs]
        sim_ab = cross_sims[a].view(bs, m, bs).permute(1, 0, 2)    # [m, bs, bs]
        batch_sim[:, :bs, :bs] = sim_aa.unsqueeze(0)
        batch_sim[:, :bs, bs:] = sim_ab
        batch_sim[:, bs:, :bs] = sim_ab.transpose(1, 2)
        batch_sim[:, bs:, bs:] = all_self[a + 1: a + 1 + m]

        # ---- 6b. ArcFace margin on positive pairs ----
        # Positive pairs are the diagonals of the two cross-view blocks.
        # batch_sim[:, i, bs+i] = cos(θ between view_a contig_i and view_b contig_i)
        # batch_sim[:, bs+i, i] = same (transposed)
        s_pos_ab = batch_sim[:, pos_idx, bs + pos_idx]    # [m, bs]
        s_pos_ba = batch_sim[:, bs + pos_idx, pos_idx]    # [m, bs]

        s_ab = s_pos_ab.clamp(-1.0 + eps, 1.0 - eps)
        s_ba = s_pos_ba.clamp(-1.0 + eps, 1.0 - eps)

        sin_ab = torch.sqrt(1.0 - s_ab * s_ab + eps)
        sin_ba = torch.sqrt(1.0 - s_ba * s_ba + eps)

        # ArcFace: cos(θ + m) = cos(θ)·cos(m) − sin(θ)·sin(m)
        batch_sim[:, pos_idx, bs + pos_idx] = s_ab * cos_m.unsqueeze(0) - sin_ab * sin_m.unsqueeze(0)
        batch_sim[:, bs + pos_idx, pos_idx] = s_ba * cos_m.unsqueeze(0) - sin_ba * sin_m.unsqueeze(0)

        # ---- 6c. remove diag, clone -> 2D contiguous ----
        sim_2d = batch_sim[:, ~diag_2bs].view(n_rows, two_bs - 1).clone()

        # ---- 6d. adaptive soft margin for masked false negatives ----
        fn_2d = false_neg_mask.unsqueeze(0).expand(m, -1, -1).reshape(n_rows, two_bs - 1)
        n_2d = neg_mask.unsqueeze(0).expand(m, -1, -1).reshape(n_rows, two_bs - 1)
        unmasked_neg = sim_2d[~fn_2d & n_2d]
        if unmasked_neg.numel() == 0:
            soft_neg_sim = sim_2d.min().item()
        else:
            n_sample = min(10_000, unmasked_neg.numel())
            step = max(1, unmasked_neg.numel() // n_sample)
            idx = torch.arange(0, unmasked_neg.numel(), step,
                               device=unmasked_neg.device)[:n_sample]
            soft_neg_sim = torch.quantile(unmasked_neg[idx], 0.05)
        sim_2d[fn_2d] = soft_neg_sim

        # ---- 6e. InfoNCE logits ----
        p_2d = pos_mask.unsqueeze(0).expand(m, -1, -1).reshape(n_rows, two_bs - 1)

        logits = torch.cat([
            sim_2d[p_2d].view(-1, 1),
            sim_2d[n_2d].view(n_rows, two_bs - 2),
        ], dim=1)
        logits = logits / temperature
        ce_labels = torch.zeros(n_rows, dtype=torch.long, device=device)

        logits_list.append(logits)
        labels_list.append(ce_labels)

        # ---- 6f. loss (equal-weight over all view pairs) ----
        subloss = criterion(logits, ce_labels) * m / total_pairs
        loss_total += subloss

    logits_cat = torch.cat(logits_list, dim=0)
    labels_cat = torch.cat(labels_list, dim=0)
    return loss_total, logits_cat, labels_cat
