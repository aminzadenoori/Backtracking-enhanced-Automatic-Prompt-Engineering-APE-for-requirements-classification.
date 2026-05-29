"""Statistical analyses for the four research questions (Section 4 of the paper).

RQ1  Wilcoxon signed-rank (APE vs each baseline) + Holm-Bonferroni, effect size r = Z/sqrt(N).
RQ2  Friedman tests (dataset effect blocked on LLM; LLM effect blocked on dataset) + Kendall's W.
RQ3  Multiple linear regression of wF1 on prompt features (standardized betas, R^2).
RQ4  Wilcoxon signed-rank on (a) final wF1 difference and (b) gains-over-baseline difference.

All tests use alpha = 0.05. Implemented with scipy/numpy/pandas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
BASELINES = ["Zero-shot", "Few-shot", "CoT", "CoT+Few-shot"]


def effect_size_r(stat_z: float, n: int) -> float:
    return abs(stat_z) / math.sqrt(n) if n > 0 else 0.0


def interpret_r(r: float) -> str:
    if r < 0.10:
        return "negligible"
    if r < 0.30:
        return "small"
    if r < 0.50:
        return "moderate"
    return "large"


def holm_bonferroni(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """Return per-test significance decisions under Holm-Bonferroni."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    decisions = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvals[idx] <= threshold:
            decisions[idx] = True
        else:
            break  # once one fails, all larger p-values fail
    return decisions


# --------------------------------------------------------------------------- #
# RQ1 — APE vs baselines
# --------------------------------------------------------------------------- #
@dataclass
class RQ1Row:
    baseline: str
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    effect_r: float
    effect_label: str
    p_raw: float
    p_corrected: float
    significant: bool


def rq1(df: pd.DataFrame, metric: str = "avg_f1") -> list[RQ1Row]:
    """df columns: dataset, model, strategy, <metric>. APE compared to each baseline.

    Delta = metric_APE - metric_baseline, pooled over all (dataset, model) cells.
    One-tailed Wilcoxon (APE > baseline), Holm-Bonferroni corrected.
    """
    wide = df.pivot_table(index=["dataset", "model"], columns="strategy", values=metric)
    deltas, raws, stats_z = {}, [], {}
    for b in BASELINES:
        d = (wide["APE"] - wide[b]).dropna().to_numpy()
        deltas[b] = d
        # one-tailed: alternative greater
        try:
            res = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
            p = res.pvalue
        except ValueError:
            p = 1.0
        raws.append(p)
        # z from normal approximation for effect size
        n = len(d)
        if n > 0:
            # use two-sided statistic to recover |z|
            try:
                w_two = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
                z = stats.norm.isf(w_two.pvalue / 2)
            except ValueError:
                z = 0.0
        else:
            z = 0.0
        stats_z[b] = z

    decisions = holm_bonferroni(raws)
    # corrected p-values (Holm)
    m = len(raws)
    order = sorted(range(m), key=lambda i: raws[i])
    corrected = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * raws[idx]
        running = max(running, val)
        corrected[idx] = min(running, 1.0)

    rows = []
    for i, b in enumerate(BASELINES):
        d = deltas[b]
        n = len(d)
        mean = float(np.mean(d)) if n else 0.0
        if n > 1:
            se = float(np.std(d, ddof=1) / math.sqrt(n))
            ci = stats.t.ppf(0.975, n - 1) * se
        else:
            ci = 0.0
        r = effect_size_r(stats_z[b], n)
        rows.append(RQ1Row(
            baseline=b, n=n, mean_delta=round(mean, 4),
            ci_low=round(mean - ci, 4), ci_high=round(mean + ci, 4),
            effect_r=round(r, 3), effect_label=interpret_r(r),
            p_raw=round(raws[i], 6), p_corrected=round(corrected[i], 6),
            significant=decisions[i],
        ))
    return rows


# --------------------------------------------------------------------------- #
# RQ2 — Friedman tests + Kendall's W
# --------------------------------------------------------------------------- #
@dataclass
class FriedmanResult:
    effect: str
    chi2: float
    df: int
    pvalue: float
    kendall_w: float
    significant: bool


def _kendall_w(chi2: float, n_blocks: int, k_treatments: int) -> float:
    return chi2 / (n_blocks * (k_treatments - 1)) if n_blocks and k_treatments > 1 else 0.0


def rq2(df_ape: pd.DataFrame, metric: str = "wf1") -> list[FriedmanResult]:
    """df_ape columns: dataset, model, <metric> (APE final wF1 per cell).

    Dataset effect: treatments = datasets, blocks = models.
    LLM effect:     treatments = models,   blocks = datasets.
    """
    table = df_ape.pivot_table(index="model", columns="dataset", values=metric)

    # Dataset effect (each model is a block; columns are datasets)
    cols = [table[c].to_numpy() for c in table.columns]
    chi2_d, p_d = stats.friedmanchisquare(*cols)
    w_d = _kendall_w(chi2_d, n_blocks=table.shape[0], k_treatments=table.shape[1])

    # LLM effect (each dataset is a block; rows are models)
    tt = table.T
    rows = [tt[c].to_numpy() for c in tt.columns]
    chi2_l, p_l = stats.friedmanchisquare(*rows)
    w_l = _kendall_w(chi2_l, n_blocks=tt.shape[0], k_treatments=tt.shape[1])

    return [
        FriedmanResult("Dataset", round(float(chi2_d), 3), table.shape[1] - 1,
                       round(float(p_d), 4), round(float(w_d), 3), p_d < ALPHA),
        FriedmanResult("LLM", round(float(chi2_l), 3), table.shape[0] - 1,
                       round(float(p_l), 4), round(float(w_l), 3), p_l < ALPHA),
    ]


# --------------------------------------------------------------------------- #
# RQ3 — prompt-feature regression
# --------------------------------------------------------------------------- #
@dataclass
class RegressionResult:
    coefficients: dict[str, dict]   # feature -> {beta, se, t, p, ci_low, ci_high}
    r2: float
    n: int


def rq3(df: pd.DataFrame, target: str = "wf1", features: list[str] | None = None) -> RegressionResult:
    """OLS regression of wF1 on standardized prompt features.

    df has one row per logged prompt with columns for each feature + target.
    Features default to the paper's set: SC, WC, PM, LD, VB, SCx, AS, SD.
    """
    if features is None:
        features = ["SC", "WC", "PM", "LD", "VB", "SCx", "AS", "SD"]
    data = df.dropna(subset=features + [target])
    X = data[features].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)

    # standardize predictors (and target) -> standardized betas
    Xs = (X - X.mean(0)) / (X.std(0, ddof=0) + 1e-12)
    ys = (y - y.mean()) / (y.std(ddof=0) + 1e-12)

    n, p = Xs.shape
    Xd = np.column_stack([np.ones(n), Xs])     # design matrix with intercept
    beta, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
    resid = ys - Xd @ beta
    dof = max(n - p - 1, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    tvals = beta / (se + 1e-12)
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    tcrit = stats.t.ppf(0.975, dof)

    ss_res = float(resid @ resid)
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    coeffs = {}
    for i, f in enumerate(features):
        j = i + 1   # skip intercept
        coeffs[f] = {
            "beta": round(float(beta[j]), 4),
            "se": round(float(se[j]), 4),
            "t": round(float(tvals[j]), 3),
            "p": round(float(pvals[j]), 4),
            "ci_low": round(float(beta[j] - tcrit * se[j]), 4),
            "ci_high": round(float(beta[j] + tcrit * se[j]), 4),
        }
    return RegressionResult(coefficients=coeffs, r2=round(r2, 3), n=n)


# --------------------------------------------------------------------------- #
# RQ4 — Informed vs Uninformed
# --------------------------------------------------------------------------- #
@dataclass
class RQ4Result:
    dimension: str
    n_nonzero: int
    mean_diff: float
    ci_low: float
    ci_high: float
    wilcoxon_w: float
    pvalue: float
    significant: bool


def _wilcoxon_block(diffs: np.ndarray, alternative: str) -> tuple[float, float]:
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(nz, alternative=alternative, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return 0.0, 1.0


def rq4(df_final: pd.DataFrame, df_gains: pd.DataFrame) -> list[RQ4Result]:
    """Dimension 1: final wF1 difference (Informed - Uninformed), one-tailed (>0).
       Dimension 2: gains difference delta_gain = d_informed - d_uninformed, two-tailed.
    """
    out = []

    d1 = (df_final["ape_informed"] - df_final["ape_uninformed"]).to_numpy(dtype=float)
    nz1 = d1[d1 != 0]
    w1, p1 = _wilcoxon_block(d1, alternative="greater")
    mean1 = float(np.mean(d1))
    n1 = len(nz1)
    if n1 > 1:
        se1 = float(np.std(d1, ddof=1) / math.sqrt(len(d1)))
        ci1 = stats.t.ppf(0.975, len(d1) - 1) * se1
    else:
        ci1 = 0.0
    out.append(RQ4Result(
        "Final wF1 (Informed - Uninformed)", n1, round(mean1, 4),
        round(mean1 - ci1, 4), round(mean1 + ci1, 4),
        round(w1, 1), round(p1, 4), p1 < ALPHA,
    ))

    dg = (df_gains["delta_informed"] - df_gains["delta_uninformed"]).to_numpy(dtype=float)
    nzg = dg[dg != 0]
    wg, pg = _wilcoxon_block(dg, alternative="two-sided")
    meang = float(np.mean(dg))
    ng = len(nzg)
    if ng > 1:
        seg = float(np.std(dg, ddof=1) / math.sqrt(len(dg)))
        cig = stats.t.ppf(0.975, len(dg) - 1) * seg
    else:
        cig = 0.0
    out.append(RQ4Result(
        "Gain difference (delta_gain)", ng, round(meang, 4),
        round(meang - cig, 4), round(meang + cig, 4),
        round(wg, 1), round(pg, 4), pg < ALPHA,
    ))
    return out
