#!/usr/bin/env python3
"""Gradio GUI for the APE classification tool.

Launches without any LLM running; the backend is only contacted when you click
Run. Uses the same `ape.core` algorithm as the CLI, so the GUI and the paper
experiments share one implementation.

    python -m web.app                       # local
    python -m web.app -- --share            # public link
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import gradio as gr

from ape.core import (
    load_csv, three_way_split, run_all_baselines, optimize,
    build_prompt, STRATEGY_NAMES,
)
from ape.core.examples import initial_examples
from ape.backends import make_backend, PAPER_MODELS


# --------------------------------------------------------------------------- #
# Session state (single-user; fine for a research demo)
# --------------------------------------------------------------------------- #
class S:
    examples = []
    labels = []
    split = None
    results = {}      # strategy -> Metrics
    ape = None        # OptimizationResult


def check_connection(base_url, backend):
    import requests
    try:
        if backend == "ollama":
            r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=4)
            ms = [m["name"] for m in r.json().get("models", [])]
            return True, (f"Connected — {len(ms)} model(s)." if ms else
                          "Connected — no models pulled yet (`ollama pull <model>`).")
        requests.get(base_url.rstrip("/") + "/v1/models", timeout=4)
        return True, "Connected to OpenAI-compatible endpoint."
    except Exception as e:
        return False, f"Cannot reach backend at {base_url} ({e})."


def metrics_md(m, title=""):
    lines = [f"### {title}" if title else "", f"**Macro F1: {m.macro_f1*100:.1f}%**\n",
             "| Label | P | R | F1 | n |", "|---|---|---|---|---|"]
    for lbl, c in m.per_class.items():
        lines.append(f"| {lbl} | {c.precision*100:.0f}% | {c.recall*100:.0f}% | **{c.f1*100:.1f}%** | {c.support} |")
    return "\n".join(lines)


def comparison_table():
    if not S.results and not S.ape:
        return ""
    head = "| Method | Macro F1 |" + "".join(f" {l} |" for l in S.labels)
    sep = "|---|---|" + "".join("---|" for _ in S.labels)
    rows = [head, sep]
    for s in ["zero_shot", "few_shot", "cot", "cot_few_shot"]:
        if s in S.results:
            m = S.results[s]
            r = f"| {STRATEGY_NAMES[s]} | **{m.macro_f1*100:.1f}%** |"
            r += "".join(f" {m.per_class[l].f1*100:.0f}% |" for l in S.labels)
            rows.append(r)
    if S.ape:
        m = S.ape.test_metrics
        r = f"| APE Optimized | **{m.macro_f1*100:.1f}%** |"
        r += "".join(f" {m.per_class[l].f1*100:.0f}% |" for l in S.labels)
        rows.append(r)
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def cb_load(file_obj, pool_pct, val_pct):
    if file_obj is None:
        return "No file selected.", "", ""
    if pool_pct + val_pct >= 100:
        return "Pool % + Validation % must be under 100 (test is the remainder).", "", ""
    try:
        examples, labels = load_csv(file_obj)
    except Exception as e:
        return f"Error: {e}", "", ""
    S.examples, S.labels = examples, labels
    S.split = three_way_split(examples, labels, pool_pct / 100, val_pct / 100)
    S.results, S.ape = {}, None
    counts = defaultdict(int)
    for e in examples:
        counts[e.label] += 1
    summary = (f"✓ **{len(examples)}** samples · **{len(labels)}** classes · "
               f"pool **{len(S.split.pool)}** · val **{len(S.split.val)}** · test **{len(S.split.test)}**")
    detail = "\n".join(f"- **{l}**: {counts[l]}" for l in labels)
    return summary, detail, ", ".join(labels)


def cb_fetch_models(base_url, backend):
    if backend != "ollama":
        return gr.update(choices=["(type model name)"], value="(type model name)")
    try:
        import requests
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=4)
        ms = [m["name"] for m in r.json().get("models", [])]
        return gr.update(choices=ms or ["(no models — ollama pull <model>)"],
                         value=(ms[0] if ms else "(no models — ollama pull <model>)"))
    except Exception as e:
        return gr.update(choices=[f"(Ollama not running: {e})"], value=f"(Ollama not running: {e})")


def cb_check(base_url, backend):
    ok, msg = check_connection(base_url, backend)
    return ("🟢 " if ok else "🔴 ") + msg


def _guard(base_url, backend):
    if S.split is None:
        return "⚠ Load a dataset first."
    ok, msg = check_connection(base_url, backend)
    return "" if ok else f"⛔ {msg}"


def cb_run_baselines(fixed, opt, model, url, backend, api_key, voting):
    err = _guard(url, backend)
    if err:
        yield err, "", ""
        return
    be = make_backend(backend, model, url, api_key)
    log, blocks = [], []
    def ev(e):
        if e["type"] == "baseline_start":
            log.append(f"▶ {STRATEGY_NAMES[e['strategy']]} …")
        elif e["type"] == "baseline_done":
            log.append(f"  ✓ macro-F1 {e['macro_f1']*100:.1f}%")
    res = run_all_baselines(model_fn=be.model_fn, split=S.split,
                            fixed_part=fixed, optimizable_part=opt,
                            voting_runs=int(voting), on_event=ev)
    for s, r in res.items():
        S.results[s] = r.metrics
        blocks.append(metrics_md(r.metrics, STRATEGY_NAMES[s]))
    yield "\n".join(log) + "\n\n✅ done", "\n\n---\n\n".join(blocks), comparison_table()


def cb_run_ape(fixed, opt, model, url, backend, api_key, voting, n_max, backtrack):
    err = _guard(url, backend)
    if err:
        yield err, "", "", ""
        return
    be = make_backend(backend, model, url, api_key)
    log = []
    def ev(e):
        t = e["type"]
        if t == "init": log.append(f"APE init · val-F1 {e['val_f1']*100:.1f}%")
        elif t == "iter_start": log.append(f"iter {e['n']} (best {e['best_f1']*100:.1f}%)")
        elif t == "improved": log.append(f"  ↑ improved → {e['val_f1']*100:.1f}%")
        elif t == "no_improve": log.append(f"  → {e['val_f1']*100:.1f}% (best {e['best_f1']*100:.1f}%)")
        elif t == "backtrack": log.append(f"  ↩ backtrack → rank-{e['to_rank']} ({e['to_f1']*100:.1f}%)")
        elif t == "final_test": log.append(f"✓ FINAL TEST macro-F1 {e['test_f1']*100:.1f}%")
    result = optimize(model_fn=be.model_fn, split=S.split,
                      fixed_part=fixed, initial_optimizable=opt,
                      n_max=int(n_max), backtrack_threshold=int(backtrack),
                      voting_runs=int(voting), on_event=ev)
    S.ape = result
    hist = "| Iter | Val F1 | Status |\n|---|---|---|\n"
    for h in result.history:
        icon = "✅" if h.improved else ("↩" if h.backtracked else "—")
        hist += f"| {h.n} | {h.val_f1*100:.1f}% | {icon} |\n"
    yield ("\n".join(log), result.best_prompt,
           hist + "\n\n" + metrics_md(result.test_metrics, "Final test (held-out)"),
           comparison_table())


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
DESC = """# APE Classification Tool
Backtracking-enhanced automatic prompt engineering for text classification.
Load a dataset, choose definitions, and run baselines + APE. The GUI opens
without any model running — a backend is contacted only when you click Run.
"""

with gr.Blocks(title="APE Classification Tool") as demo:
    gr.Markdown(DESC)
    with gr.Tabs():
        with gr.Tab("1 · Data & Config"):
            with gr.Row():
                file_in = gr.File(label="Upload CSV (text first, label last)", file_types=[".csv"])
                pool_pct = gr.Slider(10, 80, value=30, step=5, label="Example pool %")
                val_pct = gr.Slider(10, 80, value=30, step=5, label="Validation %")
            load_btn = gr.Button("Load dataset", variant="primary")
            ds_status = gr.Markdown(); label_detail = gr.Markdown()
            label_list = gr.Textbox(label="Detected labels", interactive=False)
            gr.Markdown("Test set = remainder (100 − pool − val).")
            with gr.Row():
                backend_sel = gr.Radio(["ollama", "openai-compat"], value="ollama", label="Backend")
                base_url = gr.Textbox(value="http://localhost:11434", label="Base URL")
                api_key = gr.Textbox(label="API key (if needed)", type="password")
            conn = gr.Markdown("*(connection not checked)*")
            with gr.Row():
                check_btn = gr.Button("Check connection")
                fetch_btn = gr.Button("↺ Fetch models")
            model_dd = gr.Dropdown(choices=list(PAPER_MODELS), value="llama3:8b-instruct",
                                   allow_custom_value=True, label="Model")
            with gr.Row():
                voting = gr.Slider(1, 5, value=3, step=1, label="Voting runs")
                n_max = gr.Slider(1, 20, value=20, step=1, label="APE max iterations (N_max)")
                backtrack = gr.Slider(1, 10, value=3, step=1, label="Backtrack threshold (X)")
            load_btn.click(cb_load, [file_in, pool_pct, val_pct], [ds_status, label_detail, label_list])
            check_btn.click(cb_check, [base_url, backend_sel], [conn])
            fetch_btn.click(cb_fetch_models, [base_url, backend_sel], [model_dd])

        with gr.Tab("2 · Prompt"):
            gr.Markdown("**Fixed part** frames the task. **Optimisable part** holds the label "
                        "definitions — APE rewrites only this. Load one from `definitions/`.")
            with gr.Row():
                fixed_prompt = gr.Textbox(label="Fixed prompt", lines=6,
                    value="You are a precise text classifier. Classify each text into exactly "
                          "one of the provided categories. Output only the exact label name.")
                opt_prompt = gr.Textbox(label="Optimisable prompt (APE rewrites this)", lines=6,
                    placeholder="Paste a definitions file, e.g. definitions/finegrained/security.txt")

        with gr.Tab("3 · Baselines"):
            run_bl = gr.Button("Run all 4 baselines", variant="primary")
            bl_log = gr.Markdown(); bl_res = gr.Markdown(); bl_cmp = gr.Markdown()
            run_bl.click(cb_run_baselines,
                         [fixed_prompt, opt_prompt, model_dd, base_url, backend_sel, api_key, voting],
                         [bl_log, bl_res, bl_cmp])

        with gr.Tab("4 · APE Optimisation"):
            run_ape = gr.Button("Run APE (Algorithm 1)", variant="primary")
            with gr.Row():
                ape_log = gr.Markdown()
                ape_best = gr.Textbox(label="Best prompt (p*)", lines=12, interactive=False)
            ape_hist = gr.Markdown(); ape_cmp = gr.Markdown()
            run_ape.click(cb_run_ape,
                          [fixed_prompt, opt_prompt, model_dd, base_url, backend_sel, api_key,
                           voting, n_max, backtrack],
                          [ape_log, ape_best, ape_hist, ape_cmp])

        with gr.Tab("5 · Results"):
            refresh = gr.Button("Refresh", variant="primary")
            out = gr.Markdown()
            refresh.click(lambda: comparison_table(), [], [out])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_name", default="127.0.0.1")
    ap.add_argument("--server_port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args(argv)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main(sys.argv[1:] if "--" not in sys.argv else sys.argv[sys.argv.index("--") + 1:])
