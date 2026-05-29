"""Tests for the Algorithm 1 implementation using a deterministic mock backend.

Run with:  pytest -q   (or: python -m tests.test_optimizer)
"""
from __future__ import annotations

import random

from ape.core import (
    Example, three_way_split, compute_metrics, majority_vote,
    build_prompt, extract_label, optimize, run_all_baselines,
)
from ape.core.examples import dynamic_examples, initial_examples


# --------------------------------------------------------------------------- #
# Mock backend: a keyword classifier whose accuracy improves as the prompt
# accumulates the magic token "REFINED". This lets us deterministically test
# the optimization loop without any network calls.
# --------------------------------------------------------------------------- #
KEYWORDS = {
    "Performance": ["fast", "latency", "throughput", "seconds", "response"],
    "Security":    ["encrypt", "auth", "password", "access", "secure"],
    "Functional":  ["display", "export", "feature", "endpoint", "generate"],
}


def make_mock_backend():
    rng = random.Random(0)

    def model_fn(system: str, user: str) -> str:
        # meta-prompt call (Step 2.1): return an "improved" optimizable section
        if "rewrite" in system.lower() or "prompt engineer" in system.lower():
            # each refinement appends REFINED, raising the simulated skill
            base = "REFINED definitions. " * (system.count("REFINED") + 1)
            return base + "Performance/Security/Functional disambiguated."

        text = user.lower()
        skill = system.count("REFINED")  # more REFINED -> more accurate
        # score each label by keyword hits
        best, best_hits = "Performance", -1
        for label, kws in KEYWORDS.items():
            hits = sum(1 for k in kws if k in text)
            if hits > best_hits:
                best, best_hits = label, hits
        # inject noise that shrinks as skill grows
        if rng.random() < max(0.0, 0.4 - 0.1 * skill):
            best = rng.choice(list(KEYWORDS))
        return best

    return model_fn


def toy_dataset() -> tuple[list[Example], list[str]]:
    samples = [
        ("The system shall respond within two seconds", "Performance"),
        ("Search results return in under one second", "Performance"),
        ("Throughput must reach 5000 requests per second", "Performance"),
        ("Latency stays low under heavy load", "Performance"),
        ("Passwords must be encrypted at rest", "Security"),
        ("All access requires authentication", "Security"),
        ("Data is secured with strong encryption", "Security"),
        ("Login uses a secure password policy", "Security"),
        ("The app shall display a dashboard", "Functional"),
        ("Users can export reports to PDF", "Functional"),
        ("The API exposes a create-user endpoint", "Functional"),
        ("The system shall generate monthly invoices", "Functional"),
    ]
    examples = [Example(t, l) for t, l in samples]
    labels = sorted({l for _, l in samples})
    return examples, labels


def test_metrics_basic():
    preds = ["A", "A", "B", "B"]
    golds = ["A", "B", "B", "B"]
    m = compute_metrics(preds, golds, ["A", "B"])
    assert 0.0 <= m.macro_f1 <= 1.0
    assert m.per_class["A"].support == 1
    assert m.per_class["B"].support == 3


def test_majority_vote():
    runs = [["A", "B"], ["A", "C"], ["B", "B"]]
    assert majority_vote(runs) == ["A", "B"]


def test_extract_label_cot():
    labels = ["Performance", "Security", "Functional"]
    assert extract_label("...reasoning...\nLABEL: Security", labels) == "Security"
    assert extract_label("performance", labels) == "Performance"
    assert extract_label("I think this is about security stuff", labels) == "Security"


def test_three_way_split_disjoint():
    examples, labels = toy_dataset()
    split = three_way_split(examples, labels, 0.30, 0.30, seed=1)
    all_back = split.pool + split.val + split.test
    assert len(all_back) == len(examples)
    # disjoint
    texts = [e.text for e in all_back]
    assert len(texts) == len(set(texts))
    assert len(split.test) > 0


def test_dynamic_examples_returns_four():
    examples, labels = toy_dataset()
    # pretend all predictions are correct
    preds = [e.label for e in examples]
    chosen = dynamic_examples(examples, preds, labels, seed=2)
    assert len(chosen) == 4


def test_optimize_end_to_end():
    examples, labels = toy_dataset()
    split = three_way_split(examples, labels, 0.34, 0.33, seed=7)
    model_fn = make_mock_backend()

    result = optimize(
        model_fn=model_fn,
        split=split,
        fixed_part="You are a precise text classifier.",
        initial_optimizable="Performance: speed. Security: protection. Functional: features.",
        n_max=6,
        backtrack_threshold=3,
        voting_runs=3,
        seed=7,
    )

    # structural guarantees from Algorithm 1
    assert 0.0 <= result.best_val_f1 <= 1.0
    assert 0.0 <= result.test_f1 <= 1.0
    assert len(result.history) == 6
    # ranked list contains the initial prompt (iter 0) + 6 candidates
    assert len(result.ranked) == 7
    # ranked list is sorted descending by F1
    f1s = [r.f1 for r in result.ranked]
    assert f1s == sorted(f1s, reverse=True)
    # best prompt's val F1 equals the top of the ranked list
    assert abs(result.ranked[0].f1 - result.best_val_f1) < 1e-9


def test_baselines_run():
    examples, labels = toy_dataset()
    split = three_way_split(examples, labels, 0.34, 0.33, seed=3)
    model_fn = make_mock_backend()
    results = run_all_baselines(
        model_fn=model_fn, split=split,
        fixed_part="You are a precise text classifier.",
        optimizable_part="Performance/Security/Functional.",
        voting_runs=3, seed=3,
    )
    assert set(results) == {"zero_shot", "few_shot", "cot", "cot_few_shot"}
    for r in results.values():
        assert 0.0 <= r.metrics.macro_f1 <= 1.0


if __name__ == "__main__":
    # allow running without pytest
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All tests passed.")
