"""Dynamic example selection (Step 2.6) and APE candidate generation (Step 2.1).

The paper frames classification as one-vs-rest per requirement category, so
"positive" / "negative" are defined relative to a focus label. For multi-class
datasets we pick the focus label as the one with the most support in the pool;
this keeps the balanced 1/1/1/1 sampling well-defined while remaining faithful
to the algorithm's intent (mix of correct and misclassified, positive and
negative demonstrations).
"""
from __future__ import annotations

import random
from collections import Counter

from .data import Example
from .prompting import classify


def _focus_label(pool: list[Example], labels: list[str]) -> str:
    counts = Counter(e.label for e in pool)
    # most frequent label present in the pool; fall back to labels[0]
    for lbl, _ in counts.most_common():
        if lbl in labels:
            return lbl
    return labels[0]


def initial_examples(
    pool: list[Example],
    labels: list[str],
    seed: int = 42,
    focus: str | None = None,
) -> list[Example]:
    """Phase 1: 2 random positive + 2 random negative examples from D_pool."""
    rng = random.Random(seed)
    focus = focus or _focus_label(pool, labels)
    positives = [e for e in pool if e.label == focus]
    negatives = [e for e in pool if e.label != focus]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    chosen = positives[:2] + negatives[:2]
    # if pool is tiny, top up with whatever is available
    if len(chosen) < 4:
        extra = [e for e in pool if e not in chosen]
        rng.shuffle(extra)
        chosen += extra[: 4 - len(chosen)]
    return chosen


def dynamic_examples(
    pool: list[Example],
    pool_preds: list[str],
    labels: list[str],
    seed: int = 42,
    focus: str | None = None,
) -> list[Example]:
    """Step 2.6: select 1 correct-pos, 1 correct-neg, 1 mis-pos, 1 mis-neg.

    `pool_preds` are predictions for `pool` (same order) produced by the current
    prompt under 3-run voting. Selection is one-vs-rest w.r.t. `focus`.
    """
    rng = random.Random(seed)
    focus = focus or _focus_label(pool, labels)

    correct_pos, correct_neg, mis_pos, mis_neg = [], [], [], []
    for ex, pred in zip(pool, pool_preds):
        is_pos = ex.label == focus
        correct = pred == ex.label
        if is_pos and correct:
            correct_pos.append(ex)
        elif (not is_pos) and correct:
            correct_neg.append(ex)
        elif is_pos and not correct:
            mis_pos.append(ex)
        else:
            mis_neg.append(ex)

    for bucket in (correct_pos, correct_neg, mis_pos, mis_neg):
        rng.shuffle(bucket)

    chosen: list[Example] = []
    for bucket in (correct_pos, correct_neg, mis_pos, mis_neg):
        if bucket:
            chosen.append(bucket[0])

    # If any quadrant is empty (common on small/easy pools), backfill from the
    # remaining pool so the prompt still receives 4 demonstrations.
    if len(chosen) < 4:
        remaining = [e for e in pool if e not in chosen]
        rng.shuffle(remaining)
        chosen += remaining[: 4 - len(chosen)]
    return chosen


# --------------------------------------------------------------------------- #
# APE candidate generation (Step 2.1) — meta-prompting the model to rewrite
# only the optimizable section, conditioned on the current example set.
# --------------------------------------------------------------------------- #

META_SYSTEM = (
    "You are an expert NLP prompt engineer. You improve text-classification "
    "prompts. Return ONLY the rewritten definitions/instructions section, with "
    "no preamble, no explanation, and no markdown code fences."
)


def generate_candidate(
    model_fn,
    current_optimizable: str,
    current_full_prompt: str,
    labels: list[str],
    examples: list[Example],
) -> str:
    """Step 2.1: ask the model to propose an improved optimizable section."""
    ex_block = "\n".join(f'  - "{e.text}"  ->  {e.label}' for e in examples)
    meta_user = f"""Improve the following text-classification prompt to increase macro-F1.

Categories: {', '.join(labels)}

Current full prompt (for context):
---
{current_full_prompt}
---

The section you must rewrite (label definitions / disambiguation rules):
---
{current_optimizable}
---

Representative labeled examples to learn from (mix of correctly and
incorrectly classified, positive and negative):
{ex_block}

Rewrite the definitions/rules so the classifier better separates the
categories above. Sharpen boundaries, add disambiguation rules for commonly
confused pairs, and keep it concise.

Return ONLY the rewritten definitions/instructions section."""

    out = model_fn(META_SYSTEM, meta_user)
    out = out.strip()
    # strip accidental fences
    if out.startswith("```"):
        out = out.split("\n", 1)[-1]
    if out.endswith("```"):
        out = out.rsplit("```", 1)[0]
    out = out.strip()
    return out if len(out) >= 20 else current_optimizable
