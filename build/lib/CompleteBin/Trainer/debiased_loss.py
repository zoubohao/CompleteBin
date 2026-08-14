"""
Debiased Contrastive Learning for metagenomics binning.

In standard InfoNCE, all contigs in a batch other than the anchor are treated as
negatives.  But in metagenomic samples, multiple contigs from the *same* genome
often appear in one batch — these become false negatives that confuse the model.

This module identifies suspected same-genome contig pairs using the pretrained
model's classification confidence: two contigs are likely from the same genome
if both have top-1 probability >= tau AND share the same top-1 class AND share
the same top-3 classes.

Usage (in trainer.py):
    from CompleteBin.Trainer.debiased_loss import (
        build_same_genome_mask,
        debiased_info_nce_loss_for_loop,
    )

    sg_mask = build_same_genome_mask(
        seq_taxon_enc[:batch_size],
        model.pretrain_model.out_linear,
        epoch=epoch_counter,
        enable_mask=use_pretrained,
    )
"""

import torch
import torch.nn.functional as F
from CompleteBin.logger import get_logger

logger = get_logger()


@torch.no_grad()
def build_same_genome_mask(
    taxon_emb: torch.Tensor,              # [B, hidden_dim]
    out_linear,                           # Linear(hidden_dim, num_classes) or None
    epoch: int = 0,
    n_iter: int = 0,
    total_iters: int = 1,
    warmup_epochs: int = 10,
    tau: float = 0.98,                    # top-1 probability threshold
    enable_mask: bool = True,
) -> torch.Tensor:
    """Identify contig pairs likely from the same genome.

    Uses pretrained classification confidence:
    1. Both contigs must have top-1 probability >= tau (high-confidence prediction)
    2. Their top-1 predicted class must be the same
    3. Their top-3 predicted classes must be identical (as a set, sorted)

    Adaptive scheduling:
    - epoch <= warmup_epochs: mask fully disabled (return zeros)
    - epoch > warmup_epochs: mask enabled

    Args:
        taxon_emb: [B, hidden_dim] pretrained embeddings (first-view only).
        out_linear: pretrained classification head Linear(hidden_dim, num_classes).
        epoch: current training epoch (1-indexed).
        n_iter: current iteration within epoch.
        total_iters: total iterations in this epoch.
        warmup_epochs: epochs during which mask is fully disabled.
        tau: top-1 probability confidence threshold.
        enable_mask: if False, returns all-zero mask.

    Returns:
        [B, B] float mask (1.0 = same genome, 0.0 otherwise). Diagonal is 0.
    """
    B = taxon_emb.shape[0]
    device = taxon_emb.device

    # --- Fast path: mask disabled ---
    if not enable_mask or out_linear is None:
        return torch.zeros(B, B, device=device)

    if epoch <= warmup_epochs:
        return torch.zeros(B, B, device=device)

    # --- Compute classification probabilities ---
    logits = out_linear(taxon_emb)                         # [B, num_classes]
    probs = F.softmax(logits, dim=-1)                      # [B, num_classes]

    # Condition 1: both contigs have confident predictions
    top1_prob, top1_idx = probs.max(dim=-1)                # [B]
    high_conf = top1_prob >= tau                           # [B]
    high_conf_pair = high_conf.unsqueeze(0) & high_conf.unsqueeze(1)  # [B, B]

    # Condition 2: same top-1 predicted class
    same_top1 = (top1_idx.unsqueeze(1) == top1_idx.unsqueeze(0))  # [B, B]

    # Condition 3: identical top-3 classes (sorted to ignore order)
    _, top3_idx = probs.topk(3, dim=-1)                   # [B, 3]
    top3_sorted, _ = top3_idx.sort(dim=-1)                # [B, 3]
    same_top3 = (top3_sorted.unsqueeze(1) == top3_sorted.unsqueeze(0)).all(dim=-1)  # [B, B]

    # Combined: high confidence + same top-1 + same top-3
    sg_mask = high_conf_pair & same_top1 & same_top3       # [B, B]

    # Remove self-pairs
    sg_mask = sg_mask & ~torch.eye(B, dtype=torch.bool, device=device)

    # --- Log ---
    mid_iter = total_iters // 2
    last_iter = total_iters - 1
    if n_iter in (0, mid_iter, last_iter) and total_iters > 1:
        phase = "start" if n_iter == 0 else ("middle" if n_iter == mid_iter else "end")
        n_high = high_conf.sum().item()
        masked_frac = sg_mask.float().sum() / max(B * B - B, 1)
        logger.info(
            f"--> SGM mask [{phase}]: epoch={epoch}, tau={tau}, "
            f"high_conf_contigs={n_high}/{B}, masked_ratio={masked_frac:.6f}"
        )

    return sg_mask.float()


def debiased_info_nce_loss_2_views(
    features_2views: torch.Tensor,
    batch_size: int,
    temperature: float,
    device: torch.device,
    sg_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Debiased pairwise InfoNCE for two views.

    Suspected same-genome contigs (from sg_mask) are excluded from the negative
    set so they do not push apart contigs that likely belong together.

    Args:
        features_2views: [2 * batch_size, dim] — cat(view_a, view_b).
        batch_size: number of unique contigs.
        temperature: softmax temperature.
        device: torch device.
        sg_mask: [batch_size, batch_size] same-genome mask (1.0 = same genome).

    Returns:
        (logits, labels) compatible with ``CrossEntropyLoss``.
    """
    n_total = 2 * batch_size
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    features_2views = F.normalize(features_2views, dim=-1)
    similarity = torch.matmul(features_2views, features_2views.T)  # [2bs, 2bs]

    # Remove diagonal
    diag = torch.eye(n_total, dtype=torch.bool, device=device)
    labels = labels[~diag].view(n_total, -1)
    similarity = similarity[~diag].view(n_total, -1)

    # ---- Expand sg_mask to 2-view space ----
    # sg_mask is [bs, bs]; after repeat → [2bs, 2bs]; then remove diag → [2bs, 2bs-1]
    full_sg = sg_mask.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    full_sg = full_sg[~diag].view(n_total, -1)

    # ---- Mask false negatives ----
    is_negative = (labels == 0)
    false_neg = is_negative & (full_sg > 0.5)

    similarity = similarity.clone()
    similarity[false_neg] = -1e9  # effectively removed from softmax denominator

    # ---- InfoNCE ----
    positives = similarity[labels.bool()].view(-1, 1)
    negatives = similarity[~labels.bool()].view(similarity.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    ce_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    logits = logits / temperature

    return logits, ce_labels


def debiased_info_nce_loss_for_loop(
    features: torch.Tensor,
    batch_size: int,
    n_views: int,
    temperature: float,
    device: torch.device,
    criterion: torch.nn.Module,
    sg_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Debiased InfoNCE — drop-in replacement for ``info_nce_loss_for_loop``.

    Works identically to the original but excludes suspected same-genome negatives.

    Args:
        features: [n_views * batch_size, dim] all views stacked.
        batch_size: number of unique contigs.
        n_views: total number of effective views (e.g. 6 augmented views × 6 segments = 36).
        temperature: softmax temperature.
        device: torch device.
        criterion: ``CrossEntropyLoss``.
        sg_mask: [batch_size, batch_size] same-genome mask from ``build_same_genome_mask``.

    Returns:
        (loss, logits_cat, labels_cat) — same format as ``info_nce_loss_for_loop``.
    """
    n_views_emb = torch.chunk(features, chunks=n_views, dim=0)
    logits_list = []
    labels_list = []
    loss_total = 0.0

    for i in range(n_views):
        subloss = 0.0
        cur_view = n_views_emb[i]
        for v in range(i + 1, n_views):
            other_view = n_views_emb[v]
            cat_two = torch.cat([cur_view, other_view], dim=0)
            logits, ce_labels = debiased_info_nce_loss_2_views(
                cat_two, batch_size, temperature, device, sg_mask,
            )
            logits_list.append(logits)
            labels_list.append(ce_labels)
            subloss += criterion(logits, ce_labels)
        subloss /= max(1, n_views - i)
        loss_total += subloss

    loss_total = loss_total / n_views
    logits_cat = torch.cat(logits_list, dim=0)
    labels_cat = torch.cat(labels_list, dim=0)
    return loss_total, logits_cat, labels_cat
