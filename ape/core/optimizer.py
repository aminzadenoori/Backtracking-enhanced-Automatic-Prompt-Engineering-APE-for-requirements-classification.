"""Reference implementation of Algorithm 1:
   Backtracking-Enhanced Prompt Optimization with Dynamic Example Selection.

Each step in `optimize()` is annotated with the corresponding line/step of the
algorithm in the paper. The implementation is intentionally explicit (rather
than clever) so it can be read alongside the pseudocode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .data import Split, Example
from .metrics import Metrics, compute_metrics
from .prompting import build_prompt, classify
from .examples import initial_examples, dynamic_examples, generate_candidate


@dataclass
class RankedEntry:
    prompt: str
    f1: float
    iteration: int


@dataclass
class IterationRecord:
    n: int
    val_f1: float
    improved: bool
    backtracked: bool
    optimizable: str


@dataclass
class OptimizationResult:
    best_prompt: str
    best_optimizable: str
    best_val_f1: float
    test_f1: float
    test_metrics: Metrics
    history: list[IterationRecord] = field(default_factory=list)
    ranked: list[RankedEntry] = field(default_factory=list)


def optimize(
    *,
    model_fn,                       # backend callable: (system, user) -> str
    split: Split,                   # D_pool / D_val / D_test
    fixed_part: str,                # task framing (never optimized)
    initial_optimizable: str,       # label definitions (APE rewrites this)
    n_max: int = 20,                # N_max
    backtrack_threshold: int = 3,   # X
    voting_runs: int = 3,           # 3-run majority voting
    strategy: str = "few_shot",     # prompt template used during the search
    seed: int = 42,
    on_event=None,                  # optional callback(event: dict) for logging/UI
) -> OptimizationResult:
    """Run Algorithm 1 and return the best prompt plus a single held-out test F1."""

    labels = split.labels

    def emit(event_type: str, **kw):
        if on_event:
            on_event({"type": event_type, **kw})

    def val_f1(prompt: str) -> tuple[float, Metrics]:
        preds = classify(model_fn, prompt, split.val, labels, voting_runs)
        m = compute_metrics(preds, [e.label for e in split.val], labels)
        return m.macro_f1, m

    # ===================================================================== #
    # Phase 1: Initialization
    # ===================================================================== #
    # E <- 2 random positive + 2 random negative from D_pool
    E: list[Example] = initial_examples(split.pool, labels, seed=seed)

    # p_1 = initial prompt (fixed + initial optimizable, under chosen strategy)
    cur_optimizable = initial_optimizable
    p1 = build_prompt(strategy, fixed_part, cur_optimizable, labels, E)

    # F* <- F_val(p_1)   ; score initial prompt on the validation set
    best_f1, _ = val_f1(p1)
    emit("init", val_f1=best_f1, prompt=p1)

    # p* <- p_1, p_curr <- p_1, c <- 0, ptr <- 1
    p_star = p1
    best_optimizable = cur_optimizable
    p_curr = p1
    c = 0                       # consecutive non-improving iterations
    ptr = 1                     # pointer into the ranked list for backtracking

    # Insert (p_1, F*, 0) into ranked list R (kept sorted by F1, descending)
    ranked: list[RankedEntry] = [RankedEntry(prompt=p1, f1=best_f1, iteration=0)]
    history: list[IterationRecord] = []

    # ===================================================================== #
    # Phase 2: Iterative Refinement (validation-driven)
    # ===================================================================== #
    for n in range(1, n_max + 1):
        emit("iter_start", n=n, best_f1=best_f1)

        # ---- Step 2.1: Generate candidate p_n via APE from p_curr | E ----
        new_optimizable = generate_candidate(
            model_fn,
            current_optimizable=cur_optimizable,
            current_full_prompt=p_curr,
            labels=labels,
            examples=E,
        )
        p_n = build_prompt(strategy, fixed_part, new_optimizable, labels, E)

        # ---- Step 2.2: Evaluate candidate on D_val (3-run voting) --------
        f_pn, _ = val_f1(p_n)

        # ---- Step 2.3: Update ranked list (sorted by F1) -----------------
        ranked.append(RankedEntry(prompt=p_n, f1=f_pn, iteration=n))
        ranked.sort(key=lambda r: r.f1, reverse=True)

        # ---- Step 2.4: Compare to best -----------------------------------
        improved = f_pn >= best_f1
        if improved:
            p_star = p_n
            best_f1 = f_pn
            best_optimizable = new_optimizable
            p_curr = p_n
            cur_optimizable = new_optimizable
            c = 0
            ptr = ptr + 1
            emit("improved", n=n, val_f1=f_pn)
        else:
            c = c + 1
            emit("no_improve", n=n, val_f1=f_pn, best_f1=best_f1)

        # ---- Step 2.5: Backtracking check (c == X) -----------------------
        backtracked = False
        if c == backtrack_threshold:
            idx = min(max(ptr, 0), len(ranked) - 1)
            p_curr = ranked[idx].prompt
            # recover the optimizable section that produced p_curr if known;
            # otherwise keep current (p_curr still drives generation)
            ptr = ptr - 1
            c = 0
            backtracked = True
            emit("backtrack", n=n, to_rank=idx + 1, to_f1=ranked[idx].f1)

        history.append(
            IterationRecord(
                n=n, val_f1=f_pn, improved=improved,
                backtracked=backtracked, optimizable=new_optimizable,
            )
        )

        # ---- Step 2.6: Select examples for next iteration ----------------
        # Classify D_pool with p_curr (3-run voting), then pick
        # 1 correct-pos, 1 correct-neg, 1 mis-pos, 1 mis-neg.
        pool_preds = classify(model_fn, p_curr, split.pool, labels, voting_runs)
        E = dynamic_examples(split.pool, pool_preds, labels, seed=seed + n)

    # ===================================================================== #
    # Phase 3: Final held-out evaluation (test set touched exactly once)
    # ===================================================================== #
    test_preds = classify(model_fn, p_star, split.test, labels, voting_runs)
    test_metrics = compute_metrics(test_preds, [e.label for e in split.test], labels)
    emit("final_test", test_f1=test_metrics.macro_f1)

    return OptimizationResult(
        best_prompt=p_star,
        best_optimizable=best_optimizable,
        best_val_f1=best_f1,
        test_f1=test_metrics.macro_f1,
        test_metrics=test_metrics,
        history=history,
        ranked=ranked,
    )
