"""
Automatic pretrained model decision logic.

S (K-mer Sufficiency):  S = (avg_len - 768) / 5000
O (Overfitting Pressure): O = train_epochs / log10(num_contigs)

Decision matrix:
           O <= 10    10 < O <= 25    O > 25
S < 0.3    USE         USE              DISABLE
0.3-0.65   USE         USE              DISABLE
S >= 0.65  DISABLE     DISABLE          DISABLE
"""

import math

from CompleteBin.logger import get_logger

logger = get_logger()


def compute_S(train_avg_len: float, min_contig_len: int = 768) -> float:
    """
    Compute k-mer sufficiency metric S.

    S = (avg_len - min_contig_len) / 5000

    S < 0.3:   k-mer signal noisy, pretrained model is complementary
    0.3 <= S < 0.65: moderate
    S >= 0.65:  k-mer signal stable, pretrained model is redundant
    """
    return max(0.0, (train_avg_len - min_contig_len) / 5000.0)


def compute_O(train_epoch: int, num_contigs: int) -> float:
    """
    Compute overfitting pressure metric O.

    O = train_epochs / log10(num_contigs)

    O <= 10:    low overfitting risk
    10 < O <= 25: moderate
    O > 25:     high overfitting risk
    """
    return train_epoch / max(math.log10(num_contigs), 1.0)


def _categorize_S(S: float) -> str:
    if S < 0.3:
        return "noisy"
    elif S < 0.65:
        return "moderate"
    return "stable"


def _categorize_O(O: float) -> str:
    if O <= 10:
        return "low"
    elif O <= 25:
        return "moderate"
    return "high"


def decide_pretrained(S: float, O: float) -> tuple:
    """
    Decision matrix for pretrained model usage.

    Boundary / cautious cases default to DISABLE.

    Args:
        S: k-mer sufficiency metric
        O: overfitting pressure metric

    Returns:
        (use_pretrained: bool, reason: str)
    """
    S_cat = _categorize_S(S)
    O_cat = _categorize_O(O)

    if S_cat == "noisy":
        if O_cat in ("low", "moderate"):
            return True, (
                f"S={S:.3f}({S_cat}), O={O:.3f}({O_cat}) -> USE "
                f"(noisy k-mer needs complementary taxonomic prior)"
            )
        else:
            return False, (
                f"S={S:.3f}({S_cat}), O={O:.3f}({O_cat}) -> DISABLE "
                f"(severe overfitting risk)"
            )

    elif S_cat == "moderate":
        if O_cat in ("low", "moderate"):
            return True, (
                f"S={S:.3f}({S_cat}), O={O:.3f}({O_cat}) -> USE "
                f"(moderate k-mer still benefits from pretrained)"
            )
        else:
            return False, (
                f"S={S:.3f}({S_cat}), O={O:.3f}({O_cat}) -> DISABLE "
                f"(high overfitting + marginal pretrain benefit)"
            )

    else:  # stable
        return False, (
            f"S={S:.3f}({S_cat}), O={O:.3f}({O_cat}) -> DISABLE "
            f"(k-mer signal already sufficient, pretrain redundant)"
        )


def apply_epoch_safety_valve(
    train_epoch: int,
    S: float,
    O: float,
    original_cap: int = 200,
    reduced_cap: int = 150,
) -> tuple:
    """
    Only reduce epoch cap under extreme overfitting: S >= 1.0 AND O >= 28.

    Args:
        train_epoch: current train_epoch value
        S: k-mer sufficiency
        O: overfitting pressure
        original_cap: default epoch cap (200)
        reduced_cap: reduced cap for extreme cases (150)

    Returns:
        (train_epoch: int, reason: str)
    """
    if S >= 1.0 and O >= 28:
        if train_epoch > reduced_cap:
            new_epoch = reduced_cap
            reason = (
                f"Extreme overfit detected (S={S:.2f}>=1.0, O={O:.2f}>=2.0): "
                f"capping epochs {train_epoch} -> {new_epoch}"
            )
            return new_epoch, reason
    return train_epoch, ""


def auto_min_contig_length(
    contigname2seq: dict,
    current_min_len: int,
    min_len_lo: int = 850,
    min_len_hi: int = 1000,
    min_contig_threshold: int = 20000,
    r_threshold_lo: float = 0.75,  # 0.6
    r_threshold_hi: float = 0.95,
) -> int:
    """
    Auto-select min_contig_length based on contig length distribution.

    Uses the ratio R = n(>=min_len_hi) / n(>=min_len_lo) to decide how much
    to raise min_contig_length. High R means contigs are generally long and
    we can afford a stricter filter without losing much training data.

    Args:
        contigname2seq: dict mapping contig name to sequence string.
        current_min_len: current min_contig_length value.
        min_len_lo: lower bound of auto-range (default 850).
        min_len_hi: upper bound of auto-range (default 1000).
        min_contig_threshold: minimum n(>=min_len_lo) required to consider raising.
        r_threshold_lo: R below which no raise occurs.
        r_threshold_hi: R above which full raise (to min_len_hi) occurs.

    Returns:
        int: adjusted min_contig_length in [min_len_lo, min_len_hi].

    Derivation:
        R = n(>=1000) / n(>=850)
        - R >= 0.95 (Plant GS): nearly all contigs pass 1000 → raise to 1000
        - R < 0.75 (Marine): ~32% lost in 850-1000 range → keep 850
        - 0.75 <= R < 0.95: linear interpolation between 850 and 1000

        raise_ratio = max(0, min(1, (R - r_threshold_lo) / (r_threshold_hi - r_threshold_lo)))
        min_len_new = min_len_lo + raise_ratio * (min_len_hi - min_len_lo)
    """
    if current_min_len > min_len_lo:
        logger.info(f"--> current_min_len > min_len_lo, Auto min_contig_length: {current_min_len} -> {current_min_len} ")
        return current_min_len  # already above the auto-range, don't touch

    n_lo = sum(1 for s in contigname2seq.values() if len(s) >= min_len_lo)
    if n_lo < min_contig_threshold:
        logger.info(f"--> n_lo < min_contig_threshold, Auto min_contig_length: {current_min_len} -> {current_min_len} ")
        return current_min_len  # too few contigs, don't raise

    n_hi = sum(1 for s in contigname2seq.values() if len(s) >= min_len_hi)
    R = n_hi / n_lo

    if R < r_threshold_lo:
        logger.info(f"--> R < r_threshold_lo, R: {R}, r_threshold_lo: {r_threshold_lo}, Auto min_contig_length: {current_min_len} -> {current_min_len} ")
        return current_min_len  # too many contigs in [min_len_lo, min_len_hi), don't raise

    raise_ratio = max(0.0, min(1.0, (R - r_threshold_lo) / (r_threshold_hi - r_threshold_lo)))
    new_min_len = int(min_len_lo + raise_ratio * (min_len_hi - min_len_lo))

    logger.info(
        f"--> Auto min_contig_length: {current_min_len} -> {new_min_len} "
        f"(R={R:.3f}, n_{min_len_lo}={n_lo}, n_{min_len_hi}={n_hi}, "
        f"raise_ratio={raise_ratio:.3f})"
    )
    return new_min_len
