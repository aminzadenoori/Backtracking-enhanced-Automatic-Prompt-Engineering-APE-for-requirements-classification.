#!/usr/bin/env python3
"""Command-line runner that reproduces the paper's experiment protocol:
baselines + backtracking-enhanced APE, on a single fixed pool/val/test split.

Examples
--------
# Requirements classification with fine-grained Security definitions, Llama-3:
python -m scripts.run_experiment \
    --dataset datasets/security_requirements.csv \
    --definitions definitions/finegrained/security.txt \
    --backend ollama --model llama3:8b-instruct \
    --n-max 20 --backtrack 3 --voting 3

# Use an OpenAI-compatible endpoint instead of Ollama:
python -m scripts.run_experiment --dataset my.csv \
    --backend openai-compat --base-url https://my-host/v1 --api-key $KEY \
    --model my-model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ape.core import load_csv, three_way_split, run_all_baselines, optimize, STRATEGY_NAMES
from ape.backends import make_backend, PAPER_MODELS

DEFAULT_FIXED = (
    "You are a precise text classifier. Classify each input text into exactly "
    "one of the provided categories. Output only the exact label name."
)


def fmt_metrics(m, title):
    lines = [f"\n{title}  (macro-F1 = {m.macro_f1*100:.1f}%)"]
    lines.append(f"  {'label':<16}{'P':>7}{'R':>7}{'F1':>7}{'n':>5}")
    for lbl, c in m.per_class.items():
        lines.append(f"  {lbl:<16}{c.precision*100:>6.1f}%{c.recall*100:>6.1f}%{c.f1*100:>6.1f}%{c.support:>5}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run baselines + APE on a classification dataset.")
    ap.add_argument("--dataset", required=True, help="CSV: text column first, label column last")
    ap.add_argument("--definitions", help="Text file with label definitions (the optimizable part)")
    ap.add_argument("--fixed-prompt", default=DEFAULT_FIXED, help="Fixed task-framing prompt")

    ap.add_argument("--backend", default="ollama", choices=["ollama", "openai-compat"])
    ap.add_argument("--model", default="llama3:8b-instruct")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--api-key", default="")

    ap.add_argument("--pool-frac", type=float, default=0.30)
    ap.add_argument("--val-frac", type=float, default=0.30)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--backtrack", type=int, default=3)
    ap.add_argument("--voting", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--out", help="Optional JSON file to write results to")
    args = ap.parse_args(argv)

    examples, labels = load_csv(args.dataset)
    split = three_way_split(examples, labels, args.pool_frac, args.val_frac, seed=args.seed)
    definitions = Path(args.definitions).read_text(encoding="utf-8") if args.definitions else ""

    print(f"Dataset: {len(examples)} samples, {len(labels)} classes: {', '.join(labels)}")
    print(f"Split  : pool={len(split.pool)}  val={len(split.val)}  test={len(split.test)}")
    if args.model in PAPER_MODELS:
        info = PAPER_MODELS[args.model]
        print(f"Model  : {info['name']} ({info['params']}, {info['attn']}, {info['org']})")
    else:
        print(f"Model  : {args.model} via {args.backend}")
    print(f"Config : N_max={args.n_max}  X={args.backtrack}  voting={args.voting}\n")

    backend = make_backend(args.backend, args.model, args.base_url, args.api_key)

    out = {"dataset": args.dataset, "model": args.model, "labels": labels,
           "split": {"pool": len(split.pool), "val": len(split.val), "test": len(split.test)},
           "baselines": {}, "ape": {}}

    # ---- Baselines (scored on D_test) ------------------------------------
    if not args.skip_baselines:
        def bl_log(ev):
            if ev["type"] == "baseline_start":
                print(f"[baseline] running {STRATEGY_NAMES[ev['strategy']]} ...")
            elif ev["type"] == "baseline_done":
                print(f"[baseline] {STRATEGY_NAMES[ev['strategy']]}: macro-F1 = {ev['macro_f1']*100:.1f}%")
        baselines = run_all_baselines(
            model_fn=backend.model_fn, split=split,
            fixed_part=args.fixed_prompt, optimizable_part=definitions,
            voting_runs=args.voting, seed=args.seed, on_event=bl_log,
        )
        for s, r in baselines.items():
            print(fmt_metrics(r.metrics, STRATEGY_NAMES[s]))
            out["baselines"][s] = r.metrics.as_dict()

    # ---- APE optimization (val-driven, single test eval) -----------------
    def ape_log(ev):
        t = ev["type"]
        if t == "init":
            print(f"\n[APE] init  val-F1 = {ev['val_f1']*100:.1f}%")
        elif t == "iter_start":
            print(f"[APE] iter {ev['n']}  (best {ev['best_f1']*100:.1f}%)")
        elif t == "improved":
            print(f"      ^ improved -> {ev['val_f1']*100:.1f}%")
        elif t == "no_improve":
            print(f"      . {ev['val_f1']*100:.1f}% (best {ev['best_f1']*100:.1f}%)")
        elif t == "backtrack":
            print(f"      < backtrack to rank-{ev['to_rank']} (val-F1 {ev['to_f1']*100:.1f}%)")
        elif t == "final_test":
            print(f"\n[APE] FINAL TEST macro-F1 = {ev['test_f1']*100:.1f}%")

    result = optimize(
        model_fn=backend.model_fn, split=split,
        fixed_part=args.fixed_prompt, initial_optimizable=definitions,
        n_max=args.n_max, backtrack_threshold=args.backtrack,
        voting_runs=args.voting, seed=args.seed, on_event=ape_log,
    )
    print(fmt_metrics(result.test_metrics, "APE Optimized (held-out test)"))
    out["ape"] = {
        "best_val_f1": result.best_val_f1,
        "test_f1": result.test_f1,
        "test_metrics": result.test_metrics.as_dict(),
        "best_prompt": result.best_prompt,
        "history": [
            {"n": h.n, "val_f1": h.val_f1, "improved": h.improved, "backtracked": h.backtracked}
            for h in result.history
        ],
    }

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
