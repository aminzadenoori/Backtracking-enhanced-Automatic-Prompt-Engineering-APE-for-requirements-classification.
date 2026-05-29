# Datasets

This repository ships with three small **sample** CSVs (`sample_*.csv`) so the
tool is runnable out of the box. The paper uses three established benchmarks
that are **not redistributed here** — please download them from their original
sources and place them in this directory.

## Sample datasets (ship with the repo)
- `sample_security.csv`    — 16-row Security vs Non-security toy set
- `sample_functional.csv`  — 16-row Functional vs Non-functional toy set
- `sample_quality.csv`     — 16-row Quality vs Non-quality toy set

These reproduce the algorithm end-to-end on a few examples for testing and
demonstration. They are **not** the datasets used in the paper.

## Paper benchmarks (download separately)

| Dataset           | # Reqs | Classes                                | Source |
|-------------------|:------:|----------------------------------------|--------|
| **PROMISE (NFR)** | 625    | 255 FR, 370 NFR                        | Cleland-Huang et al. (2007) — Zenodo: <https://doi.org/10.5281/zenodo.268542> |
| **PROMISE Refined** | 625  | F/onlyF/Q/onlyQ relabelling of PROMISE | Dalpiaz et al. (2019) — see https://doi.org/10.1109/RE.2019.00026 |
| **SecReq**        | 510    | 187 security, 323 non-security         | Knauss et al. (2011) — Zenodo: <https://doi.org/10.5281/zenodo.4530183> |

Format expected by the tool (CSV, header row, **text first, label last**):
```
text,label
"The system shall respond within 2 seconds",Non-security
"All passwords must be encrypted at rest",Security
```

Once you have downloaded the originals, save them as:
```
datasets/promise_nfr.csv
datasets/promise_refined.csv
datasets/secreq.csv
```
and run, e.g.:
```bash
python -m scripts.run_experiment \
    --dataset datasets/secreq.csv \
    --definitions definitions/finegrained/security.txt \
    --backend ollama --model llama3:8b-instruct \
    --n-max 20 --backtrack 3 --voting 3
```
