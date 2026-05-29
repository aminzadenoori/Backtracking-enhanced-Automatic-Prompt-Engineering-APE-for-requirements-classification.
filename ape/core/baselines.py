"""Run the four reference prompting strategies on the held-out test set.

All baselines draw their in-context demonstrations (where applicable) from
D_pool and are scored on the SAME D_test as the optimized prompt, so no method
enjoys an information advantage (paper: "data partitioning").
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Split
from .metrics import Metrics, compute_metrics
from .prompting import build_prompt, classify
from .examples import initial_examples

STRATEGIES = ["zero_shot", "few_shot", "cot", "cot_few_shot"]
STRATEGY_NAMES = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "cot": "Chain-of-Thought",
    "cot_few_shot": "CoT + Few-shot",
}


@dataclass
class BaselineResult:
    strategy: str
    prompt: str
    metrics: Metrics


def run_baseline(
    *,
    model_fn,
    split: Split,
    strategy: str,
    fixed_part: str,
    optimizable_part: str,
    voting_runs: int = 3,
    seed: int = 42,
) -> BaselineResult:
    examples = initial_examples(split.pool, split.labels, seed=seed)
    prompt = build_prompt(strategy, fixed_part, optimizable_part, split.labels, examples)
    preds = classify(model_fn, prompt, split.test, split.labels, voting_runs)
    metrics = compute_metrics(preds, [e.label for e in split.test], split.labels)
    return BaselineResult(strategy=strategy, prompt=prompt, metrics=metrics)


def run_all_baselines(
    *,
    model_fn,
    split: Split,
    fixed_part: str,
    optimizable_part: str,
    voting_runs: int = 3,
    seed: int = 42,
    on_event=None,
) -> dict[str, BaselineResult]:
    results: dict[str, BaselineResult] = {}
    for s in STRATEGIES:
        if on_event:
            on_event({"type": "baseline_start", "strategy": s})
        r = run_baseline(
            model_fn=model_fn, split=split, strategy=s,
            fixed_part=fixed_part, optimizable_part=optimizable_part,
            voting_runs=voting_runs, seed=seed,
        )
        results[s] = r
        if on_event:
            on_event({"type": "baseline_done", "strategy": s, "macro_f1": r.metrics.macro_f1})
    return results
