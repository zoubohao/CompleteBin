"""
Optimized debiased InfoNCE loss — V6 (V2 restored + minimal improvements).

Key optimizations:
1. Normalize once.
2. Batched self-sim matmul (1 call).
3. Batched cross-sim matmuls per anchor (35 calls).
4. Per-anchor batched processing: 35 loop iterations, each processing
   all pairs for one anchor in a single [m, 2*bs, 2*bs] batch.
5. All operations on 2D contiguous tensors (best GPU perf).

Usage:
    from CompleteBin.Trainer.optimized_debiased_loss import (
        debiased_info_nce_loss_optimized,
    )
    loss, logits, labels = debiased_info_nce_loss_optimized(
        features, batch_size, n_views, temperature, device, criterion, sg_mask,
    )
"""


import torch
import torch.nn.functional as F


def debiased_info_nce_loss_optimized(
    features: torch.Tensor,
    batch_size: int,
    n_views: int,
    temperature: float,
    device: torch.device,
    criterion: torch.nn.Module,
    sg_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimized multi-view debiased InfoNCE loss.

    Args:
        features: [n_views * batch_size, dim].
        batch_size: number of unique contigs per view.
        n_views: total number of views.
        temperature: softmax temperature.
        device: torch device.
        criterion: ``CrossEntropyLoss`` instance.
        sg_mask: [batch_size, batch_size] same-genome mask.
    """
    bs = batch_size
    two_bs = 2 * bs
    dim = features.shape[-1]
    total_pairs = n_views * (n_views - 1) // 2  # for equal-weight averaging over all view pairs

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
    pos_mask = labels_nodiag.bool()       # [2bs, 2bs-1]
    neg_mask = ~pos_mask                  # [2bs, 2bs-1]

    sg_nodiag = (sg_mask.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1))
    sg_nodiag = sg_nodiag[~diag_2bs].view(two_bs, two_bs - 1)
    false_neg_mask = neg_mask & (sg_nodiag > 0.5)  # [2bs, 2bs-1], contiguous

    # =================================================================
    # 5. per-anchor batched processing (2D operations throughout)
    # =================================================================
    logits_list = []
    labels_list = []
    loss_total = 0.0

    for a in range(n_views - 1):
        m = n_views - 1 - a
        n_rows = m * two_bs

        # ---- 5a. assemble [m, 2bs, 2bs] ----
        batch_sim = torch.empty(m, two_bs, two_bs, device=device)
        sim_aa = all_self[a]                     # [bs, bs]
        sim_ab = cross_sims[a].view(bs, m, bs).permute(1, 0, 2)  # [m, bs, bs]
        batch_sim[:, :bs, :bs] = sim_aa.unsqueeze(0)
        batch_sim[:, :bs, bs:] = sim_ab
        batch_sim[:, bs:, :bs] = sim_ab.transpose(1, 2)
        batch_sim[:, bs:, bs:] = all_self[a + 1: a + 1 + m]

        # ---- 5b. remove diag, clone → 2D contiguous ----
        sim_2d = batch_sim[:, ~diag_2bs].view(n_rows, two_bs - 1).clone()

        # ---- 5c. adaptive soft margin for masked false negatives ----
        fn_2d = false_neg_mask.unsqueeze(0).expand(m, -1, -1).reshape(n_rows, two_bs - 1)
        n_2d = neg_mask.unsqueeze(0).expand(m, -1, -1).reshape(n_rows, two_bs - 1)
        # Set masked pairs to 5th percentile of unmasked negatives.
        # PyTorch's quantile() internally sorts the entire input tensor,
        # which triggers RuntimeError for large tensors. Cap at 100K samples
        # via deterministic evenly-spaced subsampling.
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

        # ---- 5d. InfoNCE logits ----
        p_2d = pos_mask.unsqueeze(0).expand(m, -1, -1).reshape(n_rows, two_bs - 1)

        logits = torch.cat([
            sim_2d[p_2d].view(-1, 1),
            sim_2d[n_2d].view(n_rows, two_bs - 2),
        ], dim=1)
        logits = logits / temperature
        ce_labels = torch.zeros(n_rows, dtype=torch.long, device=device)

        logits_list.append(logits)
        labels_list.append(ce_labels)

        # ---- 5e. loss (equal-weight over all view pairs) ----
        # criterion already averages over n_rows = m * 2B.
        # Anchor a has m pairs, each pair gets weight 1/total_pairs.
        subloss = criterion(logits, ce_labels) * m / total_pairs
        loss_total += subloss

    # no final division — weights already sum to 1
    logits_cat = torch.cat(logits_list, dim=0)
    labels_cat = torch.cat(labels_list, dim=0)
    return loss_total, logits_cat, labels_cat
