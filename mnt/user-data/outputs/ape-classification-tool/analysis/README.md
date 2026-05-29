# Statistical analyses (RQ1–RQ4)

`stats.py` implements the four research-question analyses from the paper
(Section 4); `../scripts/run_stats.py` runs them on the CSVs in `results/`.

```bash
python -m scripts.run_stats                      # RQ1, RQ2, RQ4 (+ RQ3 demo)
python -m scripts.run_stats --rq3 my_features.csv   # real RQ3 regression
```

## Tests used
- **RQ1** — Wilcoxon signed-rank (one-tailed, APE > baseline), Holm–Bonferroni
  correction, effect size r = Z/√N.
- **RQ2** — Friedman test (dataset effect blocked on LLM; LLM effect blocked on
  dataset), Kendall's W effect size.
- **RQ3** — multiple linear regression of wF1 on eight prompt features
  (standardized β, R²).
- **RQ4** — Wilcoxon signed-rank on (1) final-wF1 difference and (2)
  gain-over-baseline difference (Informed vs Uninformed).

## Data files and provenance
| File | Source in paper | Used by |
|------|-----------------|---------|
| `results/per_cell_f1.csv` | Tables 3, 4, 5 (per-cell **average** F1) | RQ1 (recomputation) |
| `results/rq1_paper_reported.csv` | **Table 6** (official RQ1 on **weighted** F1) | RQ1 reference |
| `results/rq2_ape_wf1.csv` | Table 8 (APE wF1 per dataset×LLM) | RQ2 |
| `results/rq4_final_wf1.csv` | Table 10 | RQ4 Dim 1 |
| `results/rq4_gains.csv` | Table 11 | RQ4 Dim 2 |

### Note on RQ1 numbers
`per_cell_f1.csv` holds the per-cell **average-F1** values transcribed directly
from Tables 3–5, which are fully public. The paper's Table 6, however, reports
RQ1 on **weighted-F1 (wF1)**, whose per-cell values are not all tabulated in the
paper. Running `run_stats` therefore reports two RQ1 blocks: the *recomputed*
one (reproducible from the public tables, same test and direction) and the
*paper-reported* one (Table 6 verbatim, `rq1_paper_reported.csv`). Both agree on
the conclusion — APE beats every baseline with large effect sizes at α = 0.05.
The RQ2 and RQ4 inputs are wF1 and reproduce the paper's statistics essentially
exactly.
