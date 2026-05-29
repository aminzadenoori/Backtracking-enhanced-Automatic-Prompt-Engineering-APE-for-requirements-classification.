"""
Re-render the five tool UI panels as crisp vector PDFs (and high-res PNG previews).
Matches the dark theme of the real screenshots.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import matplotlib.font_manager as fm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- palette (from screenshots) ---------------------------------------------
BG       = "#1e1e1f"
PANEL    = "#262627"
PANEL_E  = "#3a3a3c"
INPUT_BG = "#2c2c2e"
INPUT_E  = "#46464a"
TXT      = "#e6e6e6"
TXT_DIM  = "#9a9a9e"
TXT_FAINT= "#6f6f74"
HEADER   = "#cfcfd2"
ACCENT   = "#7b7bf0"      # purple slider / active
PURPLE_BAR = "#6f6cf0"
GREEN    = "#39b878"
GREEN_DIM= "#4a7a5c"
AMBER    = "#d99b3a"
RED      = "#d4604f"
ORANGE   = "#cf8a3a"
WHITE_BTN= "#f2f2f0"

LABEL_COLORS = {
    "Performance": "#8f8ff2",
    "Security":    "#4fcaa0",
    "Functional":  "#e0a85a",
    "Scalability": "#6ba8ee",
    "Availability":"#e89a82",
}

def new_canvas(w_in, h_in, W, H):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    return fig, ax

def rrect(ax, x, y, w, h, fc, ec=None, lw=1.0, r=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec or fc, linewidth=lw))

def rect(ax, x, y, w, h, fc, ec=None, lw=0):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc,
        edgecolor=ec or fc, linewidth=lw))

def txt(ax, x, y, s, color=TXT, size=10, weight="normal", ha="left", va="center", family=None):
    ax.text(x, y, s, color=color, fontsize=size, fontweight=weight,
            ha=ha, va=va, family=family)

def text_width(ax, s, size, weight="normal", family=None):
    """Measure rendered text width in DATA coordinates."""
    fig = ax.figure
    fig.canvas.draw()
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight, family=family, alpha=0)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    inv = ax.transData.inverted()
    x0, _ = inv.transform((0, 0))
    x1, _ = inv.transform((bb.width, 0))
    return abs(x1 - x0)

def button(ax, x, y, label, size=10, weight="bold", h=3.4, pad=2.6,
           fc=PANEL, ec=INPUT_E, tc=TXT, lw=1.0, align="left"):
    """Auto-sized pill/button that always fits its text. Returns total width."""
    w = text_width(ax, label, size, weight) + pad*2
    bx = x if align == "left" else x - w/2 if align == "center" else x - w
    ax.add_patch(FancyBboxPatch((bx, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        facecolor=fc, edgecolor=ec, linewidth=lw))
    txt(ax, bx + w/2, y + h/2, label, color=tc, size=size, weight=weight, ha="center")
    return w

def tab_bar(ax, W, H, active_idx):
    tabs = ["1 · Data & Config", "2 · Prompt", "3 · Baselines",
            "4 · APE Optimisation", "5 · Results"]
    rect(ax, 0, H-5.5, W, 5.5, "#212122")
    rect(ax, 0, H-5.55, W, 0.12, PANEL_E)
    x = 3
    for i, t in enumerate(tabs):
        active = (i == active_idx)
        tw = text_width(ax, t, 10.5, "bold" if active else "normal")
        txt(ax, x, H-2.75, t, color=TXT if active else TXT_FAINT,
            size=10.5, weight="bold" if active else "normal")
        if active:
            rect(ax, x, H-5.4, tw, 0.45, "#f5f5f3")
        x += tw + 4.0

def slider(ax, x, y, w, frac, val, label, W):
    txt(ax, 2.6, y, label, color=TXT, size=10)
    rect(ax, x, y-0.35, w, 0.7, "#3a3a3c", r=0) if False else \
        ax.add_patch(FancyBboxPatch((x, y-0.32), w, 0.64,
            boxstyle="round,pad=0,rounding_size=0.3", facecolor="#3a3a3c", edgecolor="#3a3a3c"))
    ax.add_patch(FancyBboxPatch((x, y-0.32), w*frac, 0.64,
        boxstyle="round,pad=0,rounding_size=0.3", facecolor=PURPLE_BAR, edgecolor=PURPLE_BAR))
    ax.add_patch(Circle((x + w*frac, y), 0.95, facecolor=ACCENT, edgecolor="#1e1e1f", linewidth=1.2, zorder=5))
    txt(ax, W-2.5, y, str(val), color=TXT, size=11, weight="bold", ha="right")

def hbar(ax, x, y, w, frac, color, pct, lblcolor):
    ax.add_patch(FancyBboxPatch((x, y-0.55), w, 1.1,
        boxstyle="round,pad=0,rounding_size=0.4", facecolor="#2f2f31", edgecolor="#2f2f31"))
    ax.add_patch(FancyBboxPatch((x, y-0.55), w*frac, 1.1,
        boxstyle="round,pad=0,rounding_size=0.4", facecolor=color, edgecolor=color))
    txt(ax, x+w+1.3, y, pct, color=lblcolor, size=10.5, weight="bold", ha="left")

def save(fig, name):
    fig.savefig(f"figures/{name}.pdf",
                facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"figures/{name}.png",
                facecolor=BG, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {name}")

# =============================================================================
# PANEL 1 — Data & Config
# =============================================================================
W, H = 100, 80
fig, ax = new_canvas(8.2, 6.5, W, H)
tab_bar(ax, W, H, 0)

txt(ax, 2.6, H-8.5, "DATASET", color=TXT_DIM, size=9, weight="bold")
# upload box (dashed)
ax.add_patch(FancyBboxPatch((2.6, H-20.5), W-5.2, 9,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL,
    edgecolor=INPUT_E, linewidth=1.0, linestyle=(0,(4,3))))
txt(ax, W/2, H-14.3, "⬆", color=TXT_DIM, size=13, ha="center")
txt(ax, W/2, H-17.2, "Upload CSV — text column first, label column last",
    color=TXT_DIM, size=10, ha="center")
# success bar
ax.add_patch(FancyBboxPatch((2.6, H-26.5), W-5.2, 4.2,
    boxstyle="round,pad=0,rounding_size=1.0", facecolor="#e9f0d8", edgecolor="#e9f0d8"))
txt(ax, 4.5, H-24.4, "✓", color="#3b7a1e", size=11, weight="bold")
parts = [("16", True), ("  samples   ·   ", False), ("5", True),
         ("  classes   ·   pool  ", False), ("5", True), ("   ·   val  ", False), ("5", True),
         ("   ·   test  ", False), ("6", True)]
xx = 7.5
for s, b in parts:
    txt(ax, xx, H-24.4, s, color="#2f5a14", size=10.0, weight="bold" if b else "normal")
    xx += len(s) * (1.18 if b else 0.88)
# label chips
chips = [("Performance", 4), ("Security", 4), ("Functional", 3), ("Scalability", 3), ("Availability", 2)]
cx = 2.6
for name, n in chips:
    c = LABEL_COLORS[name]
    label = f"{name} ({n})"
    cw = text_width(ax, label, 8.7, "bold") + 2.6
    ax.add_patch(FancyBboxPatch((cx, H-31.5), cw, 2.9,
        boxstyle="round,pad=0,rounding_size=1.45", facecolor=c+"22", edgecolor=c, linewidth=1.0))
    txt(ax, cx+cw/2, H-30.05, label, color=c, size=8.7, weight="bold", ha="center")
    cx += cw + 1.4
# divider
rect(ax, 2.6, H-33.5, W-5.2, 0.1, PANEL_E)

txt(ax, 2.6, H-36.5, "LLM BACKEND", color=TXT_DIM, size=9, weight="bold")
txt(ax, 2.6, H-40.0, "Backend", color=TXT, size=10, weight="bold")
w_ol = button(ax, 2.6, H-44.0, "ollama", size=9.0, weight="bold", h=3.2, pad=2.0,
              fc="#e9e9f6", ec=ACCENT, tc="#3a3a8a", lw=1.2)
w_oc = button(ax, 2.6 + w_ol + 1.6, H-44.0, "openai-compat", size=9.0, weight="normal", h=3.2, pad=2.0,
       fc=INPUT_BG, ec=INPUT_E, tc=TXT_DIM, lw=1.0)
backend_right = 2.6 + w_ol + 1.6 + w_oc          # x where backend group ends
url_x = backend_right + 4.0                       # Base URL column starts after it
url_w = 26
txt(ax, url_x, H-40.0, "Base URL", color=TXT, size=10, weight="bold")
ax.add_patch(FancyBboxPatch((url_x, H-44.5), url_w, 3.6,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=INPUT_BG, edgecolor=INPUT_E, linewidth=1.0))
txt(ax, url_x+1.5, H-42.6, "http://localhost:11434", color=TXT_DIM, size=9.3)
key_x = url_x + url_w + 4.0                        # API key column after Base URL
txt(ax, key_x, H-40.0, "API key", color=TXT, size=10, weight="bold")
ax.add_patch(FancyBboxPatch((key_x, H-44.5), W-2.5-key_x, 3.6,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=INPUT_BG, edgecolor=INPUT_E, linewidth=1.0))
txt(ax, key_x+1.5, H-42.5, "••••••••", color=TXT_DIM, size=9.5)

txt(ax, 2.6, H-48.0, "Model", color=TXT, size=10, weight="bold")
# Fetch button is auto-sized; the model field fills the space to its left
w_fetch = text_width(ax, "↻ Fetch models", 9.3, "normal") + 5.0
fetch_x = W - 2.5 - w_fetch
model_w = fetch_x - 2.6 - 1.5
ax.add_patch(FancyBboxPatch((2.6, H-52.5), model_w, 3.6,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=INPUT_BG, edgecolor=INPUT_E, linewidth=1.0))
txt(ax, 4.2, H-50.6, "llama3:8b-instruct", color=TXT, size=10, weight="bold")
txt(ax, 2.6 + model_w - 2.5, H-50.6, "⌄", color=TXT_DIM, size=11, ha="center")
button(ax, fetch_x, H-52.5, "↻ Fetch models", size=9.3, weight="normal", h=3.6, pad=2.5,
       fc=PANEL, ec=INPUT_E, tc=TXT, lw=1.0)
rect(ax, 2.6, H-55.5, W-5.2, 0.1, PANEL_E)

txt(ax, 2.6, H-58.5, "RUN SETTINGS", color=TXT_DIM, size=9, weight="bold")
slider(ax, 38, H-62.0, 57, 0.30, 30, "Example pool %", W)
slider(ax, 38, H-65.5, 57, 0.30, 30, "Validation %", W)
slider(ax, 38, H-69.0, 57, 0.16, 1,  "Voting runs", W)
slider(ax, 38, H-72.5, 57, 0.40, 8,  "APE max iterations", W)
slider(ax, 38, H-76.0, 57, 0.30, 3,  "Backtrack threshold X", W)
txt(ax, 2.6, H-79.0, "Test set = remainder (40%)", color=TXT_FAINT, size=8.5)
save(fig, "tool_setup")

# =============================================================================
# PANEL 2 — Prompt
# =============================================================================
W, H = 100, 30
fig, ax = new_canvas(8.2, 2.55, W, H)
tab_bar(ax, W, H, 1)
txt(ax, 2.6, H-8.5, "Fixed prompt", color=TXT, size=11, weight="bold")
txt(ax, 21, H-8.5, "(never changes)", color=TXT_DIM, size=9.5)
ax.add_patch(FancyBboxPatch((2.6, 2.5), 45, H-13,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=INPUT_E, linewidth=1.0))
fixed_lines = ["You are a precise text classifier.",
               "Classify text into exactly one of the",
               "provided categories.",
               "Output only the exact label name."]
for i, ln in enumerate(fixed_lines):
    txt(ax, 4.5, H-12.5 - i*2.6, ln, color=TXT_DIM, size=9.8)

txt(ax, 52, H-8.5, "Optimisable prompt", color=TXT, size=11, weight="bold")
txt(ax, 79.5, H-8.5, "(APE rewrites this)", color=ACCENT, size=9.5)
ax.add_patch(FancyBboxPatch((52, 2.5), 45.5, H-13,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=INPUT_E, linewidth=1.0))
rect(ax, 52, 2.5, 0.55, H-13, ACCENT)
opt_lines = ["Performance: speed, latency, throughput.",
             "Security: authentication, encryption.",
             "Functional: user-visible features.",
             "Scalability: load, concurrent users.",
             "Availability: uptime, reliability."]
for i, ln in enumerate(opt_lines):
    txt(ax, 54, H-12.5 - i*2.6, ln, color=TXT_DIM, size=9.6)
save(fig, "tool_prompt")

# =============================================================================
# PANEL 3 — Baselines
# =============================================================================
W, H = 100, 44
fig, ax = new_canvas(8.2, 3.65, W, H)
tab_bar(ax, W, H, 2)
txt(ax, 2.6, H-9, "Strategy", color=TXT, size=11, weight="bold")
strat = [("zero_shot", True), ("few_shot", False), ("cot", False), ("cot_few_shot", False)]
for i, (s, active) in enumerate(strat):
    y = H-15.0 - i*4.2
    if active:
        button(ax, 2.6, y, s, size=9.8, weight="bold", h=3.4, pad=2.6,
               fc="#e9e9f6", ec=ACCENT, tc="#3a3a8a", lw=1.2)
    else:
        button(ax, 2.6, y, s, size=9.8, weight="normal", h=3.4, pad=2.6,
               fc=INPUT_BG, ec=INPUT_E, tc=TXT, lw=1.0)
# run buttons (auto-fit)
button(ax, 2.6, 6.5, "Run selected strategy", size=9.8, weight="bold", h=3.8, pad=3.0,
       fc=WHITE_BTN, ec=WHITE_BTN, tc="#1e1e1f")
button(ax, 2.6, 1.8, "Run all 4 baselines", size=9.8, weight="bold", h=3.8, pad=3.0,
       fc=PANEL, ec=INPUT_E, tc=TXT, lw=1.0)

# status card
ax.add_patch(FancyBboxPatch((36, H-23, ), 0, 0, boxstyle="round", facecolor=PANEL)) if False else None
ax.add_patch(FancyBboxPatch((36, H-22), 61, 12.5,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=PANEL_E, linewidth=1.0))
txt(ax, 38, H-12, "Status", color=TXT_DIM, size=10, weight="bold")
txt(ax, 38, H-15.0, "✓ Zero-shot — Macro F1: 54.2%", color=GREEN_DIM, size=9.3, family="DejaVu Sans Mono")
txt(ax, 38, H-17.6, "✓ Few-shot — Macro F1: 61.8%", color=GREEN_DIM, size=9.3, family="DejaVu Sans Mono")
txt(ax, 38, H-20.2, "▶ Running Chain-of-Thought…", color="#5a8fce", size=9.3, family="DejaVu Sans Mono")

# results card
ax.add_patch(FancyBboxPatch((36, 1.8), 61, 18,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=PANEL_E, linewidth=1.0))
txt(ax, 38, 17.5, "Results — few-shot", color=TXT, size=10, weight="bold")
res = [("Performance", 0.75, "75%"), ("Security", 0.68, "68%"), ("Functional", 0.55, "55%"),
       ("Scalability", 0.60, "60%"), ("Availability", 0.50, "50%")]
for i, (lbl, frac, pct) in enumerate(res):
    y = 14.0 - i*2.6
    txt(ax, 62, y, lbl, color=TXT, size=9.3, ha="right")
    hbar(ax, 63.5, y, 24, frac, PURPLE_BAR, pct, ACCENT)
save(fig, "tool_baselines")

# =============================================================================
# PANEL 4 — APE Optimisation
# =============================================================================
W, H = 100, 68
fig, ax = new_canvas(8.2, 5.6, W, H)
tab_bar(ax, W, H, 3)
# buttons (auto-fit, chained)
bx = 2.6
w1 = button(ax, bx, H-11, "① Init APE", size=10, weight="bold", h=3.8, pad=2.8,
            fc=WHITE_BTN, ec=WHITE_BTN, tc="#1e1e1f")
bx += w1 + 1.8
w2 = button(ax, bx, H-11, "② Next iteration", size=10, weight="bold", h=3.8, pad=2.8,
            fc=PANEL, ec="#5a5a5e", tc=TXT, lw=1.0)
bx += w2 + 1.8
button(ax, bx, H-11, "Run all automatically", size=10, weight="bold", h=3.8, pad=2.8,
       fc=PANEL, ec=INPUT_E, tc=TXT, lw=1.0)

# iteration log card (left)
ax.add_patch(FancyBboxPatch((2.6, 20.5), 46, H-34,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=PANEL_E, linewidth=1.0))
txt(ax, 4.5, H-14.5, "Iteration log", color=TXT, size=10.5, weight="bold")
log = [("APE init. Val F1: 61.8%", GREEN_DIM),
       ("Iteration 1/8 — best: 61.8%", "#5a8fce"),
       ("↑ Improved: 61.8% → 67.3%", GREEN_DIM),
       ("Iteration 2/8 — best: 67.3%", "#5a8fce"),
       ("→ 64.1% (best: 67.3%)", TXT_DIM),
       ("Iteration 3/8 — best: 67.3%", "#5a8fce"),
       ("→ 65.0% (best: 67.3%)", TXT_DIM),
       ("↩ Backtrack → rank-2 prompt", ORANGE),
       ("Iteration 4/8 — best: 67.3%", "#5a8fce"),
       ("↑ Improved: 67.3% → 72.1%", GREEN_DIM),
       ("✓ Final Test F1: 70.4%", GREEN)]
for i, (ln, c) in enumerate(log):
    w = "bold" if ln.startswith("✓ Final") else "normal"
    txt(ax, 4.5, H-17.5 - i*2.35, ln, color=c, size=8.5, weight=w, family="DejaVu Sans Mono")

# iteration history
txt(ax, 2.6, 17.5, "Iteration history", color=TXT, size=10.5, weight="bold")
cells = [("1", "67%", GREEN, "#e9f0d8"), ("2", "64%", TXT_DIM, INPUT_BG),
         ("3", "↩", AMBER, "#f0e6cf"), ("4", "72%", GREEN, "#e9f0d8"),
         ("5", "70%", TXT_DIM, INPUT_BG)]
for i, (n, v, tc, bg) in enumerate(cells):
    x = 2.6 + i*8.5
    ax.add_patch(FancyBboxPatch((x, 8.5), 7, 6,
        boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=bg, edgecolor=tc if v != "64%" and v != "70%" else INPUT_E, linewidth=1.0))
    txt(ax, x+3.5, 12.7, f"#{n}", color="#555", size=7.5, ha="center")
    txt(ax, x+3.5, 10.4, v, color=("#2f5a14" if bg=="#e9f0d8" else ("#7a5310" if bg=="#f0e6cf" else TXT_DIM)),
        size=11, ha="center", weight="bold")
ax.add_patch(Circle((4, 5.5), 0.7, facecolor="#cfe0b0", edgecolor="none"))
txt(ax, 5.5, 5.5, "improved", color=TXT_DIM, size=8.5)
ax.add_patch(Circle((18, 5.5), 0.7, facecolor="#e8d6a8", edgecolor="none"))
txt(ax, 19.5, 5.5, "backtrack", color=TXT_DIM, size=8.5)
txt(ax, 2.6, 3.0, "Examples / iter: 1 correct-pos, 1 correct-neg,",
    color=TXT_FAINT, size=7.8)
txt(ax, 2.6, 1.2, "1 mis-pos, 1 mis-neg (balanced from pool)",
    color=TXT_FAINT, size=7.8)

# best prompt (right top)
txt(ax, 51, H-14.0, "Best prompt so far", color=TXT, size=10.5, weight="bold")
ax.add_patch(FancyBboxPatch((51, H-31.5), 46.5, 15,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=PANEL_E, linewidth=1.0))
rect(ax, 51, H-31.5, 0.5, 15, GREEN)
txt(ax, 53, H-19.5, "You are a precise text classifier…", color=TXT_DIM, size=9)
txt(ax, 53, H-24.5, "Performance: requirements specifying", color=TXT_DIM, size=9)
txt(ax, 53, H-27.0, "measurable speed or latency thresholds…", color=TXT_DIM, size=9)

# current optimisable section
txt(ax, 51, H-34.5, "Current optimisable section", color=TXT, size=10.5, weight="bold")
txt(ax, 51, H-37.0, "(evolves each iteration)", color=TXT_DIM, size=8.8)
ax.add_patch(FancyBboxPatch((51, 20.5), 46.5, 7.2,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=PANEL_E, linewidth=1.0))
rect(ax, 51, 20.5, 0.5, 7.2, ACCENT)
txt(ax, 53, 25.7, "Performance: requirements with explicit", color=TXT_DIM, size=9)
txt(ax, 53, 23.6, "numeric thresholds for speed, latency,", color=TXT_DIM, size=9)
txt(ax, 53, 21.5, "or throughput. Security: access control…", color=TXT_DIM, size=9)

# latest metrics
ax.add_patch(FancyBboxPatch((51, 1.8), 46.5, 16.5,
    boxstyle="round,pad=0,rounding_size=1.2", facecolor=PANEL, edgecolor=PANEL_E, linewidth=1.0))
txt(ax, 53, 16.0, "Validation F1 — iteration 4", color=TXT, size=10, weight="bold")
mets = [("Performance", 0.82, "82%"), ("Security", 0.76, "76%"), ("Functional", 0.65, "65%"),
        ("Scalability", 0.70, "70%"), ("Availability", 0.68, "68%")]
for i, (lbl, frac, pct) in enumerate(mets):
    y = 12.7 - i*2.5
    txt(ax, 70, y, lbl, color=TXT, size=9, ha="right")
    hbar(ax, 71.5, y, 18, frac, GREEN, pct, GREEN)
save(fig, "tool_ape")

# =============================================================================
# PANEL 5 — Results
# =============================================================================
W, H = 100, 40
fig, ax = new_canvas(8.2, 3.3, W, H)
tab_bar(ax, W, H, 4)
button(ax, 2.6, H-12, "Refresh table", size=9.8, weight="bold", h=4.0, pad=3.0,
       fc=WHITE_BTN, ec=WHITE_BTN, tc="#1e1e1f")

cols = ["Method", "Macro F1", "Perf.", "Sec.", "Func.", "Scale", "Avail."]
colx = [3, 24, 42, 55, 68, 81, 93]
hy = H-16
for c, x in zip(cols, colx):
    txt(ax, x, hy, c, color=TXT_DIM, size=10, weight="bold")
rows = [
    ("Zero-shot",    "54.2%", [("42%",RED),("58%",RED),("51%",ORANGE),("60%",ORANGE),("60%",RED)], False),
    ("Few-shot",     "61.8%", [("67%",ORANGE),("72%",GREEN),("55%",ORANGE),("60%",ORANGE),("55%",ORANGE)], False),
    ("CoT",          "63.4%", [("70%",GREEN),("71%",GREEN),("58%",ORANGE),("58%",ORANGE),("60%",ORANGE)], False),
    ("CoT + Few-shot","66.1%",[("74%",GREEN),("75%",GREEN),("62%",ORANGE),("62%",ORANGE),("57%",ORANGE)], False),
    ("APE Optimized","72.1%", [("82%",GREEN),("76%",GREEN),("65%",GREEN),("70%",GREEN),("68%",GREEN)], True),
]
for ri, (method, macro, cells, hl) in enumerate(rows):
    y = hy - 3.5 - ri*4.0
    if hl:
        rect(ax, 1.5, y-1.7, W-3, 3.6, "#2e2e30")
    txt(ax, colx[0], y, method, color=TXT, size=10, weight="bold")
    txt(ax, colx[1], y, macro, color=(GREEN if hl else TXT), size=10.5, weight="bold")
    for ci, (pct, c) in enumerate(cells):
        txt(ax, colx[2+ci], y, pct, color=c, size=10, weight="bold")
    rect(ax, 1.5, y-2.0, W-3, 0.06, PANEL_E)
save(fig, "tool_results")

print("ALL DONE")
