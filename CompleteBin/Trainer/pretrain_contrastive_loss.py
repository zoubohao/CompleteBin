import torch
import torch.nn.functional as F


def genome_supervised_contrastive_loss(
    projections: torch.Tensor,
    genome_ids: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Multi-positive supervised contrastive loss for genome-labelled contigs."""
    if projections.ndim != 2:
        raise ValueError(f"projections must be [N, D], got {tuple(projections.shape)}")
    if genome_ids.ndim != 1 or genome_ids.shape[0] != projections.shape[0]:
        raise ValueError("genome_ids must be [N] and align with projections")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    n_items = projections.shape[0]
    if n_items < 2:
        raise ValueError("at least two projections are required")

    features = F.normalize(projections.float(), dim=-1)
    logits = features @ features.t()
    logits = logits / float(temperature)

    diagonal = torch.eye(n_items, dtype=torch.bool, device=features.device)
    positive_mask = genome_ids[:, None].eq(genome_ids[None, :]) & ~diagonal
    positive_count = positive_mask.sum(dim=1)
    if torch.any(positive_count == 0):
        bad = torch.nonzero(positive_count == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"every anchor needs a same-genome positive; missing at indices {bad[:8]}")

    # Exclude self-pairs, then stabilize the log-softmax explicitly.
    logits = logits.masked_fill(diagonal, float("-inf"))
    log_denominator = torch.logsumexp(logits, dim=1, keepdim=True)
    log_probability = logits - log_denominator
    positive_log_probability = log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
    positive_log_probability = positive_log_probability / positive_count
    return -positive_log_probability.mean()


@torch.no_grad()
def genome_contrastive_top1_accuracy(
    projections: torch.Tensor,
    genome_ids: torch.Tensor,
    chunk_size: int = 2048,
) -> float:
    """SupCon 的 top-1 命中率（0–1）。

    对 batch 内每一条片段（锚点），在**其它**片段里找余弦相似度最高的那一条，
    看它是不是来自同一个基因组；是就算命中。返回命中的比例。

    为什么要分块：一个 batch 有 6×genomes_per_batch 条片段（默认 9126 条），
    整块算两两相似度是 9126×9126 的矩阵；按 chunk_size 行分块逐块算，
    结果一样，峰值显存只有 chunk_size×N。

    只用于日志/指标，不参与反向传播。
    """
    if projections.ndim != 2:
        raise ValueError(f"projections must be [N, D], got {tuple(projections.shape)}")
    if genome_ids.ndim != 1 or genome_ids.shape[0] != projections.shape[0]:
        raise ValueError("genome_ids must be [N] and align with projections")
    n_items = projections.shape[0]
    if n_items < 2:
        raise ValueError("at least two projections are required")

    features = F.normalize(projections.detach().float(), dim=-1)
    correct = 0
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        similarity = features[start:end] @ features.t()
        rows = torch.arange(end - start, device=features.device)
        # 排除锚点自己，否则最近邻永远是它自己
        similarity[rows, torch.arange(start, end, device=features.device)] = float("-inf")
        nearest = similarity.argmax(dim=1)
        correct += int(genome_ids[nearest].eq(genome_ids[start:end]).sum().item())
    return correct / n_items
