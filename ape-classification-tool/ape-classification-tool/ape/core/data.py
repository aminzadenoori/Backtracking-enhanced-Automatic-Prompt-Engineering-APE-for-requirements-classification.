"""Data types and the three-way pool/val/test split from Algorithm 1, Phase 1."""
from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    text: str
    label: str


@dataclass
class Split:
    """Disjoint subsets from a single random partition (Algorithm 1, Phase 1)."""
    pool: list[Example]   # D_pool  : in-context demonstrations + optimization feedback
    val: list[Example]    # D_val   : in-loop candidate scoring / ranking / backtracking
    test: list[Example]   # D_test  : single terminal held-out evaluation
    labels: list[str]


def load_csv(path_or_text: str, is_text: bool = False) -> tuple[list[Example], list[str]]:
    """Load a dataset whose first column is text and last column is the label.

    Returns (examples, sorted_labels). One header row is required.
    """
    raw = path_or_text if is_text else open(path_or_text, encoding="utf-8").read()
    rows = list(csv.reader(io.StringIO(raw)))
    if len(rows) < 2:
        raise ValueError("CSV needs a header row and at least one data row.")
    examples: list[Example] = []
    for line in rows[1:]:
        if len(line) < 2:
            continue
        text, label = line[0].strip(), line[-1].strip()
        if text and label:
            examples.append(Example(text=text, label=label))
    if not examples:
        raise ValueError("No valid rows. Expected: text column first, label column last.")
    labels = sorted({e.label for e in examples})
    return examples, labels


def three_way_split(
    examples: list[Example],
    labels: list[str],
    pool_frac: float = 0.30,
    val_frac: float = 0.30,
    seed: int = 42,
) -> Split:
    """Single random split into D_pool / D_val / D_test (no cross-validation).

    Test fraction is the remainder (default 0.40). The split is held fixed so
    F1 values entered into the ranked list are directly comparable across
    iterations (paper: "data partitioning").
    """
    if pool_frac + val_frac >= 1.0:
        raise ValueError("pool_frac + val_frac must be < 1.0 (test set is the remainder).")

    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_pool = max(1, int(round(n * pool_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_pool + n_val >= n:
        raise ValueError("Not enough data for a non-empty test set with these fractions.")

    pool = shuffled[:n_pool]
    val = shuffled[n_pool : n_pool + n_val]
    test = shuffled[n_pool + n_val :]
    return Split(pool=pool, val=val, test=test, labels=labels)
