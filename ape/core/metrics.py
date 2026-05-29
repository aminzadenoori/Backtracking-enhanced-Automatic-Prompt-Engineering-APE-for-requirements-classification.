"""Evaluation metrics: per-class precision/recall/F1 and macro-F1."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class Metrics:
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    weighted_f2: float = 0.0

    def as_dict(self) -> dict:
        return {
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "weighted_f2": self.weighted_f2,
            "per_class": {
                k: {"precision": v.precision, "recall": v.recall, "f1": v.f1, "support": v.support}
                for k, v in self.per_class.items()
            },
        }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def compute_metrics(preds: list[str], golds: list[str], labels: list[str]) -> Metrics:
    """Per-class P/R/F1 (one-vs-rest), plus macro-F1 and support-weighted F1/F2.

    The paper reports weighted F1 (wF1) and weighted F2 (wF2); both are
    computed here. Comparison is case-insensitive and whitespace-trimmed.
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds ({len(preds)}) and golds ({len(golds)}) length mismatch")

    norm = lambda s: (s or "").strip().lower()
    per_class: dict[str, ClassMetrics] = {}

    for label in labels:
        ll = norm(label)
        tp = fp = fn = 0
        for p, g in zip(preds, golds):
            np_, ng = norm(p), norm(g)
            if np_ == ll and ng == ll:
                tp += 1
            elif np_ == ll and ng != ll:
                fp += 1
            elif np_ != ll and ng == ll:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = _fbeta(precision, recall, 1.0)
        support = sum(1 for g in golds if norm(g) == ll)
        per_class[label] = ClassMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            support=support,
        )

    total = sum(c.support for c in per_class.values()) or 1
    macro_f1 = round(sum(c.f1 for c in per_class.values()) / len(labels), 4) if labels else 0.0
    weighted_f1 = round(sum(c.f1 * c.support for c in per_class.values()) / total, 4)
    weighted_f2 = round(
        sum(_fbeta(c.precision, c.recall, 2.0) * c.support for c in per_class.values()) / total, 4
    )
    return Metrics(per_class=per_class, macro_f1=macro_f1,
                   weighted_f1=weighted_f1, weighted_f2=weighted_f2)


def majority_vote(runs: list[list[str]]) -> list[str]:
    """Element-wise majority vote across several prediction runs.

    `runs` is a list of equally long prediction lists. Ties are broken by
    first-seen order (stable)."""
    if not runs:
        return []
    n = len(runs[0])
    out: list[str] = []
    for i in range(n):
        counts: dict[str, int] = defaultdict(int)
        order: list[str] = []
        for r in runs:
            v = r[i]
            if v not in counts:
                order.append(v)
            counts[v] += 1
        # pick max count, tie-break by first appearance
        best = max(order, key=lambda v: (counts[v], -order.index(v)))
        out.append(best)
    return out
