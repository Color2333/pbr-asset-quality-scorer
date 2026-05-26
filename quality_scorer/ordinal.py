from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicCoralHead(nn.Module):
    """CORAL ordinal head for scores ordered as 0 < ... < num_classes - 1."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, num_classes: int = 6):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = num_classes
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
        )
        self.rank_score = nn.Linear(hidden_dim, 1)
        self.raw_threshold_gaps = nn.Parameter(torch.zeros(num_classes - 1))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.raw_threshold_gaps) + 1e-4
        offsets = torch.cumsum(gaps, dim=0)
        return offsets - offsets.mean()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        score = self.rank_score(self.trunk(features))
        return score - self.thresholds().view(1, -1)


def make_coral_levels(scores: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=scores.device).view(1, -1)
    return (scores.long().view(-1, 1) > thresholds).float()


def logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    left = 1.0 - cumulative[:, :1]
    middle = cumulative[:, :-1] - cumulative[:, 1:]
    right = cumulative[:, -1:]
    return torch.cat([left, middle, right], dim=1).clamp_min(0.0)


def expected_quality_score(class_probs: torch.Tensor, max_score: float = 5.0) -> torch.Tensor:
    classes = torch.arange(class_probs.shape[1], device=class_probs.device, dtype=class_probs.dtype)
    return (class_probs * classes.view(1, -1)).sum(dim=1) / max_score


class CoralRankLoss(nn.Module):
    """Rank-consistent BCE over cumulative labels y > k."""

    def __init__(self, num_classes: int = 6, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        levels = make_coral_levels(scores, self.num_classes)
        return F.binary_cross_entropy_with_logits(logits, levels, pos_weight=self.pos_weight)


class CoralEntropyLoss(nn.Module):
    """CORAL rank loss + batch-level entropy regularization + label smoothing.

    Entropy regularization prevents prediction collapse (all outputs → 0 or K-1).
    Label smoothing softens binary CORAL targets to reduce pole overconfidence.

    Args:
        entropy_lambda: weight on the entropy bonus (larger = more spread).
        label_smoothing: ε in targets ∈ [ε/2, 1-ε/2] (0 = hard targets).
    """

    def __init__(
        self,
        num_classes: int = 6,
        pos_weight: torch.Tensor | None = None,
        entropy_lambda: float = 0.15,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.entropy_lambda = entropy_lambda
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        levels = make_coral_levels(scores, self.num_classes)
        if self.label_smoothing > 0:
            levels = levels * (1.0 - self.label_smoothing) + self.label_smoothing * 0.5
        coral_loss = F.binary_cross_entropy_with_logits(logits, levels, pos_weight=self.pos_weight)

        # Entropy regularization: maximize entropy of mean predicted distribution
        # to prevent the model from collapsing all predictions to the extreme classes.
        class_probs = logits_to_class_probs(logits.detach())
        mean_probs = class_probs.mean(dim=0)
        entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum()
        return coral_loss - self.entropy_lambda * entropy

