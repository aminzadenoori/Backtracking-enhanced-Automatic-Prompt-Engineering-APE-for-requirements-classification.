#!/usr/bin/env python3
"""Run all four research-question statistical analyses on the paper result tables.

    python -m scripts.run_stats

By default it loads:
  analysis/results/per_cell_f1.csv   (RQ1: APE vs each baseline)
  analysis/results/rq2_ape_wf1.csv   (RQ2: dataset/LLM effect on APE wF1)
  analysis/results/rq4_final_wf1.csv (RQ4 Dim 1)
  analysis/results/rq4_gains.csv     (RQ4 Dim 2)

For RQ3 a per-iteration prompt-feature log is required; the script accepts
`--rq3 path/to/file.csv` with columns SC, WC, PM, LD, VB, SCx, AS, SD, wf1.
If omitted, a small synthetic example is used so the pipeline is runnable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.stats import rq1, rq2, rq3, rq4, BASELINES


def fmt_rq1(rows):
    print("\nRQ1  APE vs each baseline (Wilcoxon, Holm-Bonferroni; pooled over 15 cells)")
    print(f"  {'Baseline':<14}{'n':>3}{'mean Δ':>9}{'95% CI':>22}"
          f"{'r':>7} {'label':<10}{'p_raw':>9}{'p_corr':>9}  sig")
    for r in rows:
        ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        print(f"  {r.baseline:<14}{r.n:>3}{r.mean_delta:>+9.4f}{ci:>22}"
              f"{r.effect_r:>7.3f} {r.effect_label:<10}{r.p_raw:>9.4f}{r.p_corrected:>9.4f}"
              f"  {'YES' if r.significant else 'no'}")


def fmt_rq2(results):
    print("\nRQ2  Friedman tests + Kendall's W (one value per (dataset, LLM) cell)")
    print(f"  {'Effect':<10}{'chi^2':>9}{'df':>4}{'p':>9}{'W':>9}  sig")
    for r in results:
        print(f"  {r.effect:<10}{r.chi2:>9.3f}{r.df:>4}{r.pvalue:>9.4f}{r.kendall_w:>9.3f}"
              f"  {'YES' if r.significant else 'no'}")


def fmt_rq3(result):
    print(f"\nRQ3  Prompt-feature regression  (n={result.n}, R^2={result.r2})")
    print(f"  {'Feature':<6}{'beta':>9}{'se':>8}{'t':>8}{'p':>9}{'95% CI':>22}")
    for f, c in result.coefficients.items():
        ci = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
        print(f"  {f:<6}{c['beta']:>+9.4f}{c['se']:>8.4f}"
              f"{c['t']:>+8.3f}{c['p']:>9.4f}{ci:>22}")


def fmt_rq4(results):
    print("\nRQ4  APE-Informed vs APE-Uninformed")
    print(f"  {'Dimension':<40}{'n':>4}{'mean Δ':>9}{'95% CI':>22}{'p':>9}  sig")
    for r in results:
        ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
        print(f"  {r.dimension:<40}{r.n_nonzero:>4}{r.mean_diff:>+9.4f}{ci:>22}"
              f"{r.pvalue:>9.4f}  {'YES' if r.significant else 'no'}")


def synthetic_rq3() -> pd.DataFrame:
    """Tiny synthetic feature log that produces a regression structurally similar to the paper:
    higher VB / PM / SD raise wf1; higher SC / WC / LD / SCx lower it.

    For real analysis, supply a CSV via --rq3.
    """
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "SC":  rng.integers(2, 10, n),
        "WC":  rng.integers(40, 220, n),
        "PM":  rng.integers(4, 25, n),
        "LD":  rng.uniform(0.45, 0.85, n),
        "VB":  rng.integers(4, 20, n),
        "SCx": rng.uniform(2.5, 5.5, n),
        "AS":  rng.uniform(0.0, 0.06, n),
        "SD":  rng.uniform(0.0, 0.5, n),
    })
    # paper-style sign structure + noise
    df["wf1"] = (
        0.65
        - 0.010 * df["SC"]
        - 0.001 * df["WC"]
        + 0.005 * df["PM"]
        - 0.10 * df["LD"]
        + 0.012 * df["VB"]
        - 0.020 * df["SCx"]
        - 0.30 * df["AS"]
        + 0.18 * df["SD"]
        + rng.normal(0, 0.03, n)
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1", default="analysis/results/per_cell_f1.csv")
    ap.add_argument("--rq2", default="analysis/results/rq2_ape_wf1.csv")
    ap.add_argument("--rq3", default="", help="optional per-prompt feature log CSV")
    ap.add_argument("--rq4-final", default="analysis/results/rq4_final_wf1.csv")
    ap.add_argument("--rq4-gains", default="analysis/results/rq4_gains.csv")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args()

    out = {}

    # RQ1
    if Path(args.rq1).exists():
        df1 = pd.read_csv(args.rq1)
        rq1_rows = rq1(df1)
        fmt_rq1(rq1_rows)
        print("  (recomputed from public Tables 3-5 average-F1; same test/direction)")
        out["rq1"] = [r.__dict__ for r in rq1_rows]

    # RQ1 — paper's officially reported Table 6 values (computed on weighted-F1)
    ref = Path("analysis/results/rq1_paper_reported.csv")
    if ref.exists():
        dref = pd.read_csv(ref)
        print("\nRQ1  Paper-reported (Table 6, weighted-F1)")
        print(f"  {'Baseline':<14}{'n':>3}{'mean Δ':>9}{'95% CI':>22}"
              f"  {'effect':<8}{'p_corr':>9}  sig")
        for _, r in dref.iterrows():
            ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
            print(f"  {r.baseline:<14}{int(r.n):>3}{r.mean_delta:>+9.4f}{ci:>22}"
                  f"  {r.effect_label:<8}{r.p_corrected:>9.4f}  YES")
        out["rq1_paper_reported"] = dref.to_dict("records")

    # RQ2
    if Path(args.rq2).exists():
        df2 = pd.read_csv(args.rq2)
        rq2_rows = rq2(df2)
        fmt_rq2(rq2_rows)
        out["rq2"] = [r.__dict__ for r in rq2_rows]

    # RQ3
    if args.rq3 and Path(args.rq3).exists():
        df3 = pd.read_csv(args.rq3)
        note = f"(loaded {args.rq3})"
    else:
        df3 = synthetic_rq3()
        note = "(synthetic example — pass --rq3 PATH for real data)"
    print(f"\n--- RQ3 input {note} ---")
    rq3_res = rq3(df3)
    fmt_rq3(rq3_res)
    out["rq3"] = {"r2": rq3_res.r2, "n": rq3_res.n, "coefficients": rq3_res.coefficients}

    # RQ4
    if Path(args.rq4_final).exists() and Path(args.rq4_gains).exists():
        df4f = pd.read_csv(args.rq4_final)
        df4g = pd.read_csv(args.rq4_gains)
        rq4_rows = rq4(df4f, df4g)
        fmt_rq4(rq4_rows)
        out["rq4"] = [r.__dict__ for r in rq4_rows]

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
