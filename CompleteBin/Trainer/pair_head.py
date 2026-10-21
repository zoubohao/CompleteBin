"""Same-genome pair head over backbone embeddings.

The closed-set taxon head can only express confidence about genomes that exist
in the reference database, which is why its top1_prob is unusable on marine
samples (the database excludes them). A pairwise head answers the question the
same-genome mask actually asks - "do these two contigs come from one genome?" -
on the backbone's L2 embeddings, so it stays meaningful for any sample.

Input features follow the usual sentence-pair recipe: both embeddings, their
absolute difference and their elementwise product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SameGenomePairHead(nn.Module):

    def __init__(self, dim: int = 768, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def features(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat([left, right, (left - right).abs(), left * right], dim=-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(left, right))


def sample_same_genome_pairs(embeddings: torch.Tensor, labels: torch.Tensor,
                             pairs: int, n_views: int = 6, generator=None):
    """Sample balanced positive/negative pairs from one training batch.

    ``embeddings`` is laid out as ``n_views`` consecutive blocks of
    ``batch_size`` rows, one block per view. Positives pair two views of the
    same genome, negatives pair a view with a different genome.
    """
    batch_size = labels.shape[0]
    if batch_size < 2 or pairs <= 0:
        return None
    view_a = embeddings[:batch_size]
    view_b = embeddings[batch_size:2 * batch_size] if n_views > 1 else view_a
    half = max(1, pairs // 2)

    anchor = torch.randint(0, batch_size, (half,), device=embeddings.device, generator=generator)
    # Positives: the same anchor index in another view.
    positive_left = view_a[anchor]
    positive_right = view_b[anchor]

    # Negatives: shuffle anchors until the partner has a different label.
    partner = torch.randint(0, batch_size, (half,), device=embeddings.device, generator=generator)
    same = labels[partner].eq(labels[anchor])
    if same.any():
        partner[same] = (partner[same] + 1 + torch.randint(
            0, batch_size - 1, (int(same.sum()),), device=embeddings.device, generator=generator,
        )) % batch_size
    negative_left = view_a[anchor]
    negative_right = view_b[partner]

    left = torch.cat([positive_left, negative_left], dim=0)
    right = torch.cat([positive_right, negative_right], dim=0)
    target = torch.cat([torch.ones(half, dtype=torch.long, device=embeddings.device),
                        torch.zeros(half, dtype=torch.long, device=embeddings.device)])
    return left, right, target


def pair_loss(pair_head: SameGenomePairHead, embeddings: torch.Tensor, labels: torch.Tensor,
              pairs: int, n_views: int = 6):
    """Return (loss, accuracy) for the sampled pairs, or (None, None)."""
    sampled = sample_same_genome_pairs(embeddings.detach(), labels, pairs, n_views)
    if sampled is None:
        return None, None
    left, right, target = sampled
    logits = pair_head(left, right)
    loss = F.cross_entropy(logits, target)
    accuracy = logits.argmax(dim=-1).eq(target).float().mean()
    return loss, accuracy
