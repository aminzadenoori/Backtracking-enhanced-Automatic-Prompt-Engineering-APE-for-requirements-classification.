"""Prompt construction (baselines + APE), label extraction, and voted classification."""
from __future__ import annotations

from .data import Example
from .metrics import majority_vote


# --------------------------------------------------------------------------- #
# Prompt assembly
# --------------------------------------------------------------------------- #
# A prompt has two regions:
#   - fixed_part      : task framing + output format (never optimized)
#   - optimizable_part: label definitions / disambiguation (APE rewrites this)
# --------------------------------------------------------------------------- #

def build_prompt(
    strategy: str,
    fixed_part: str,
    optimizable_part: str,
    labels: list[str],
    examples: list[Example],
) -> str:
    label_list = ", ".join(labels)
    base = fixed_part.strip() or (
        f"You are a precise text classifier. "
        f"Classify text into exactly one of: {label_list}."
    )
    defs = optimizable_part.strip()
    p = base
    if defs:
        p += f"\n\n{defs}"

    if strategy == "zero_shot":
        p += "\n\nOutput only the exact label name, nothing else."

    elif strategy == "few_shot":
        ex = "\n\n".join(f"Text: {e.text}\nLabel: {e.label}" for e in examples)
        p += f"\n\nExamples:\n{ex}\n\nOutput only the exact label name."

    elif strategy == "cot":
        p += (
            "\n\nThink step by step:\n"
            "1. Identify the core concern of the text.\n"
            "2. Match it to the most fitting category.\n"
            "3. On the final line write exactly: LABEL: <category>"
        )

    elif strategy == "cot_few_shot":
        ex = "\n\n".join(
            f"Text: {e.text}\n"
            f'Reasoning: This text concerns "{e.label}" because it directly addresses '
            f"that category.\nLABEL: {e.label}"
            for e in examples
        )
        p += f"\n\nThink step by step, then write LABEL: <category>\n\nExamples:\n{ex}"

    else:
        p += "\n\nOutput only the exact label name."

    return p


# --------------------------------------------------------------------------- #
# Label extraction (robust to CoT and chatty models)
# --------------------------------------------------------------------------- #

def extract_label(response: str, labels: list[str]) -> str:
    t = (response or "").strip().lower()

    # 1. CoT style: trailing "LABEL: x"
    for line in reversed(t.splitlines()):
        s = line.strip()
        if s.startswith("label:"):
            cand = s[len("label:"):].strip()
            for lbl in labels:
                if cand == lbl.lower():
                    return lbl

    # 2. exact whole-string match
    for lbl in labels:
        if t == lbl.lower():
            return lbl

    # 3. substring fallback (first label found in text)
    for lbl in labels:
        if lbl.lower() in t:
            return lbl

    # 4. give up -> first label (counts as a wrong prediction unless it matches)
    return labels[0]


# --------------------------------------------------------------------------- #
# Voted classification (3-run majority voting in the paper)
# --------------------------------------------------------------------------- #

def classify(
    model_fn,
    system_prompt: str,
    examples: list[Example],
    labels: list[str],
    voting_runs: int = 3,
) -> list[str]:
    """Classify `examples` with `voting_runs` passes and majority-vote per item.

    `model_fn(system_prompt, user_message) -> str` is supplied by a backend.
    """
    runs: list[list[str]] = []
    for _ in range(voting_runs):
        preds: list[str] = []
        for ex in examples:
            try:
                resp = model_fn(system_prompt, f"Text: {ex.text}")
                preds.append(extract_label(resp, labels))
            except Exception:
                preds.append(labels[0])
        runs.append(preds)
    return majority_vote(runs)
