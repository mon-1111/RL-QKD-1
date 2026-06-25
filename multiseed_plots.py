"""
All Paper Figures — Multi-Seed, Single Folder, No Titles
========================================================================
Produces every figure used in the paper, all sourced from 5-seed data,
written WITHOUT figure titles into ONE folder using the paper's
\\includegraphics filenames.

Figures produced
----------------
Computed here from data (multi-seed):
    fig2_detection_comparison.png    (paper Fig 2)  detection means +/- std
    fig4_tradeoff_frontier.png       (paper Fig 5)  detection vs bits, mean +/- std
    fig6_heatmap.png                 (paper Fig 8)  detection + bits means
    fig7_endgame_burst.png           (paper Fig 6)  burst, 5-seed mean + per-seed overlay
    ext_fig2_burst_comparison.png    (paper Fig 7)  fixed vs random, multi-seed

Copied in (already multi-seed, produced by multiseed.py):
    ms_fig3_learning_curves_shaded.png   (paper Fig 1)
    ms_fig1_detection_errorbars.png      (paper Fig 3)
    ms_fig4_statistical_tests.png        (paper Fig 4)

Everything lands in OUTPUT_DIR with no figure titles. Fig 7 has no red
text and no red highlight bars (block 9 annotated in plain text only).

Usage
-----
    python multiseed_plots.py

Edit the CONFIG block paths to match your local layout.
"""

import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# CONFIG  — edit these paths to match your local layout
# ---------------------------------------------------------------------------

SUMMARY_MAIN  = "results/multiseed/summary_main.json"   # 5-seed means (detection, bits)
BASELINE_DIR  = "results/baselines"                     # baseline JSONs
RAW_MAIN_DIR  = "results/multiseed/main"                # raw per-seed logs (fixed checkpoints)
RAW_EXT_DIR   = "results/multiseed/extension"           # raw per-seed logs (random checkpoints)

# Filename prefixes inside the raw dirs (extension files use 'randckpt_').
RAW_MAIN_PREFIX = ""
RAW_EXT_PREFIX  = "randckpt_"

# Where the 3 already-multi-seed figures live. Use the notitle/ subfolder so
# the copied figures are titleless like the rest.
MS_FIG_DIR    = "results/multiseed/figures/notitle"

# Single destination folder for ALL paper figures (no titles).
OUTPUT_DIR    = "results/paper_figures"

SEEDS         = [42, 123, 456, 789, 1337]
NOISE_LEVELS  = [0.01, 0.03, 0.05]
AGENTS        = ["qlearning", "sarsa", "doubleq"]
BURST_NOISE   = 0.03
MAX_BLOCKS    = 15        # random-checkpoint upper bound (extension)

AGENT_LABELS  = {
    "qlearning"    : "Q-Learning",
    "sarsa"        : "SARSA",
    "doubleq"      : "Double Q-Learning",
    "always_attack": "Always Attack",
    "fixed_rate"   : "Fixed Rate",
}
AGENT_COLORS  = {
    "qlearning"    : "#3A6EA5",   # slate blue
    "sarsa"        : "#C08552",   # ochre
    "doubleq"      : "#5A8A6B",   # sage green
    "always_attack": "#9C5A5A",   # brick / terracotta
    "fixed_rate"   : "#8C6C94",   # muted plum
}
NOISE_LABELS  = {0.01: "1%", 0.03: "3%", 0.05: "5%"}
NOISE_MARKERS = {0.01: "o", 0.03: "s", 0.05: "^"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary():
    with open(SUMMARY_MAIN) as f:
        return json.load(f)


def summary_mean(summary, agent, noise, metric):
    """metric in {'detection_rate','mean_correct'}; returns (mean, std)."""
    s = summary[agent][str(noise)]
    return s[metric][0], s[metric][1]


def load_baseline(policy, noise):
    fname = f"{policy}_noise{int(round(noise*100)):02d}.json"
    with open(os.path.join(BASELINE_DIR, fname)) as f:
        data = json.load(f)
    return data["detection_rate_last_500"], data["mean_correct_last_500"]


def load_seed_logs(agent, noise, raw_dir, prefix=""):
    """
    Load all per-seed raw logs for one agent/noise from raw_dir.
    `prefix` is prepended to the filename (e.g. 'randckpt_' for extension).
    Tries _seed{idx} then _seed{actualSeed} naming. Returns list of 'logs'.
    """
    noise_str = f"{int(round(noise*100)):02d}"
    out = []
    for idx in range(len(SEEDS)):
        for cand in (f"{prefix}{agent}_noise{noise_str}_seed{idx}.json",
                     f"{prefix}{agent}_noise{noise_str}_seed{SEEDS[idx]}.json"):
            p = os.path.join(raw_dir, cand)
            if os.path.exists(p):
                with open(p) as f:
                    out.append(json.load(f)["logs"])
                break
    return out


def save_fig(fig, fname):
    """Save a single titleless figure (pdf + png) into OUTPUT_DIR."""
    path = os.path.join(OUTPUT_DIR, fname + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _fmt_pct(value):
    """
    Format a percentage value with adaptive precision:
    two decimals when |value| < 10, one decimal otherwise.
    Surfaces precision on the headline low-detection cells
    (0.28%, 1.04%, 3.52%) while keeping high bars readable.
    """
    return f"{value:.2f}%" if abs(value) < 10 else f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Fig 2 — Detection comparison (multi-seed mean +/- std), no title
# ---------------------------------------------------------------------------

def plot_detection_comparison(summary):
    fig, ax = plt.subplots(figsize=(13, 6))
    all_agents = AGENTS + ["fixed_rate", "always_attack"]
    n_agents   = len(all_agents)
    width      = 0.13
    x          = np.arange(len(NOISE_LEVELS))

    for i, agent in enumerate(all_agents):
        rates, errs = [], []
        for noise in NOISE_LEVELS:
            if agent in AGENTS:
                m, sd = summary_mean(summary, agent, noise, "detection_rate")
                rates.append(m * 100); errs.append(sd * 100)
            else:
                det, _ = load_baseline(agent, noise)
                rates.append(det * 100); errs.append(0.0)

        offset = (i - n_agents / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, yerr=errs, capsize=3,
                      label=AGENT_LABELS[agent], color=AGENT_COLORS[agent],
                      alpha=0.85, edgecolor="white", linewidth=0.5,
                      error_kw={"linewidth": 1.0})
        for bar, rate in zip(bars, rates):
            if rate > 80:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 4,
                        _fmt_pct(rate), ha="center", va="top",
                        fontsize=6.5, fontweight="bold", color="white")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
                        _fmt_pct(rate), ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold", color=AGENT_COLORS[agent])

    ax.set_xlabel("Channel Noise Level (μ_ch)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"μ_ch = {NOISE_LABELS[n]}" for n in NOISE_LEVELS], fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    save_fig(fig, "fig2_detection_comparison")


# ---------------------------------------------------------------------------
# Fig 5 — Trade-off frontier (multi-seed mean +/- std), no title
# ---------------------------------------------------------------------------

def plot_tradeoff_frontier(summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for idx, noise in enumerate(NOISE_LEVELS):
        ax = axes[idx]
        for baseline in ["always_attack", "fixed_rate"]:
            det, correct = load_baseline(baseline, noise)
            ax.scatter(det * 100, correct, color=AGENT_COLORS[baseline],
                       marker="x", s=120, linewidths=2.5, zorder=5,
                       label=AGENT_LABELS[baseline] if idx == 0 else "")
        for agent in AGENTS:
            dm, ds = summary_mean(summary, agent, noise, "detection_rate")
            cm, cs = summary_mean(summary, agent, noise, "mean_correct")
            ax.errorbar(dm * 100, cm, xerr=ds * 100, yerr=cs,
                        fmt=NOISE_MARKERS[noise], color=AGENT_COLORS[agent],
                        markersize=9, capsize=3, elinewidth=1.0, zorder=6,
                        label=AGENT_LABELS[agent] if idx == 0 else "")
        ax.set_xlabel("Detection Rate (%)", fontsize=11)
        ax.set_ylabel("Avg Correct Bits Gained", fontsize=11)
        ax.set_title(f"μ_ch = {NOISE_LABELS[noise]}", fontsize=12, fontweight="bold")
        ax.set_xlim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.axvspan(0, 30, alpha=0.05, color="green")
        ylim = ax.get_ylim()
        ax.text(2, ylim[1] * 0.95 if ylim[1] > 0 else 5, "Low\ndetection\nzone",
                fontsize=7.5, color="green", alpha=0.7, va="top")

    handles = [mpatches.Patch(color=AGENT_COLORS[a], label=AGENT_LABELS[a])
               for a in AGENTS + ["fixed_rate", "always_attack"]]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    save_fig(fig, "fig4_tradeoff_frontier")


# ---------------------------------------------------------------------------
# Fig 8 — Summary heatmap (multi-seed means), no title
# ---------------------------------------------------------------------------

def plot_summary_heatmap(summary):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    all_agents = AGENTS + ["fixed_rate", "always_attack"]
    labels     = [AGENT_LABELS[a] for a in all_agents]
    noise_lbls = [f"μ_ch={NOISE_LABELS[n]}" for n in NOISE_LEVELS]

    for ax_idx, metric in enumerate(["detected", "correct_bits"]):
        ax = axes[ax_idx]
        matrix = np.zeros((len(all_agents), len(NOISE_LEVELS)))
        for i, agent in enumerate(all_agents):
            for j, noise in enumerate(NOISE_LEVELS):
                if agent in AGENTS:
                    key = "detection_rate" if metric == "detected" else "mean_correct"
                    matrix[i, j] = summary_mean(summary, agent, noise, key)[0]
                else:
                    det, bits = load_baseline(agent, noise)
                    matrix[i, j] = det if metric == "detected" else bits

        if metric == "detected":
            cmap, title = "RdYlGn_r", "Detection Rate (%)"
            fmt = lambda v: f"{v*100:.1f}%"
            vmin, vmax = 0, 1
        else:
            cmap, title = "YlOrRd", "Avg Correct Bits"
            fmt = lambda v: f"{v:.1f}"
            vmin, vmax = matrix.min(), matrix.max()

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if metric == "detected":
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

        ax.set_xticks(np.arange(len(NOISE_LEVELS)))
        ax.set_yticks(np.arange(len(all_agents)))
        ax.set_xticklabels(noise_lbls, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")  # subplot label, not figure title

        for i in range(len(all_agents)):
            for j in range(len(NOISE_LEVELS)):
                val = matrix[i, j]
                color = "white" if (metric == "detected" and val > 0.6) or \
                                   (metric != "detected" and val > (vmin + vmax) * 0.6) \
                        else "black"
                ax.text(j, i, fmt(val), ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)
        ax.axhline(y=len(AGENTS) - 0.5, color="white", linewidth=2.5, linestyle="--")

    fig.text(0.5, -0.02,
             "— — —  Dashed line separates RL agents (above) from baselines (below)",
             ha="center", fontsize=9, color="gray", style="italic")
    plt.tight_layout()
    save_fig(fig, "fig6_heatmap")


# ---------------------------------------------------------------------------
# Per-seed block means helper (shared by Fig 6 and Fig 7)
# ---------------------------------------------------------------------------

def _per_seed_block_means(agent, noise, raw_dir, n_blocks, random_cond, prefix=""):
    """
    Per seed, mean attacks per block over that seed's last-500 full episodes,
    padded to n_blocks. random_cond=True selects undetected episodes (variable
    length); False selects exactly-10-block episodes.
    Returns (list_of_curves, list_of_n_full).
    """
    seed_logs = load_seed_logs(agent, noise, raw_dir, prefix=prefix)
    curves, n_full = [], []
    for logs in seed_logs:
        last500 = logs[-500:]
        if random_cond:
            eps = [e["attacks_per_block"] for e in last500 if not e["detected"]]
        else:
            eps = [e["attacks_per_block"] for e in last500
                   if len(e["attacks_per_block"]) == 10]
        if not eps:
            continue
        padded = []
        for blocks in eps:
            b = list(blocks)
            while len(b) < n_blocks:
                b.append(0)
            padded.append(b[:n_blocks])
        curves.append(np.mean(np.array(padded), axis=0))
        n_full.append(len(eps))
    return curves, n_full


# ---------------------------------------------------------------------------
# Fig 6 — End-game burst (5-seed mean + per-seed overlay), no title, no red
# ---------------------------------------------------------------------------

def plot_endgame_burst():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    noise = BURST_NOISE
    blocks = np.arange(10)

    for col, agent in enumerate(AGENTS):
        ax = axes[col]
        curves, n_full = _per_seed_block_means(agent, noise, RAW_MAIN_DIR,
                                               n_blocks=10, random_cond=False)
        if not curves:
            ax.text(0.5, 0.5, "No per-seed data found\n(check RAW_MAIN_DIR)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            ax.set_title(AGENT_LABELS[agent], fontsize=12,
                         fontweight="bold", color=AGENT_COLORS[agent])
            continue

        stacked = np.vstack(curves)
        mean_curve = stacked.mean(axis=0)
        std_curve  = stacked.std(axis=0)
        n_seeds    = stacked.shape[0]

        # All bars in agent color (no red highlight bar)
        ax.bar(blocks, mean_curve, yerr=std_curve, capsize=3,
               color=AGENT_COLORS[agent], alpha=0.85,
               edgecolor="white", linewidth=0.5,
               error_kw={"linewidth": 1.0, "ecolor": "#444444"}, zorder=2)

        # Per-seed scatter overlay (jittered)
        rng = np.random.default_rng(0)
        for s in range(n_seeds):
            jitter = (rng.random(10) - 0.5) * 0.45
            ax.scatter(blocks + jitter, stacked[s], s=16, color="#222222",
                       alpha=0.55, zorder=4, edgecolors="white", linewidths=0.3)

        mean_early = mean_curve[:9].mean()
        ax.axhline(y=mean_early, color=AGENT_COLORS[agent], linestyle="--",
                   alpha=0.6, linewidth=1.5,
                   label=f"B0–B8 mean: {mean_early:.1f}", zorder=3)

        ratio = mean_curve[9] / mean_early if mean_early > 0 else float("inf")
        # Plain (non-red) annotation box
        ax.text(0.72, 0.97,
                f"Block 9:\n{mean_curve[9]:.1f} attacks\n({ratio:.1f}× B0–B8)",
                transform=ax.transAxes, fontsize=8.5, color="black",
                fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#888888", linewidth=1.0))
        ax.text(0.02, 0.97, f"{n_seeds} seeds\n(n={'/'.join(map(str, n_full))})",
                transform=ax.transAxes, fontsize=7.5, va="top", color="gray")

        ax.set_xticks(blocks)
        ax.set_xticklabels([f"B{i}" for i in range(10)], fontsize=9)
        ax.set_xlabel("Block Index", fontsize=11)
        ax.set_ylabel("Avg Attacks", fontsize=11)
        ax.set_title(AGENT_LABELS[agent], fontsize=12,
                     fontweight="bold", color=AGENT_COLORS[agent])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    save_fig(fig, "fig7_endgame_burst")


# ---------------------------------------------------------------------------
# Fig 7 — Fixed vs random burst comparison (multi-seed), no title, no red
# ---------------------------------------------------------------------------

def plot_burst_comparison():
    noise = BURST_NOISE
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for row, agent in enumerate(AGENTS):
        color = AGENT_COLORS[agent]
        label = AGENT_LABELS[agent]

        for col, (raw_dir, prefix, cond_label, random_cond, n_blocks) in enumerate([
            (RAW_MAIN_DIR, RAW_MAIN_PREFIX, "Fixed Checkpoints",         False, 10),
            (RAW_EXT_DIR,  RAW_EXT_PREFIX,  "Random Checkpoints [5–15]", True,  MAX_BLOCKS),
        ]):
            ax = axes[row][col]
            ax.set_title(f"{label} — {cond_label}", fontsize=10)  # panel label

            curves, n_full = _per_seed_block_means(agent, noise, raw_dir,
                                                   n_blocks=n_blocks,
                                                   random_cond=random_cond,
                                                   prefix=prefix)
            if not curves:
                ax.text(0.5, 0.5, "No per-seed data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="gray")
                continue

            stacked = np.vstack(curves)
            mean_curve = stacked.mean(axis=0)
            std_curve  = stacked.std(axis=0)
            n_seeds    = stacked.shape[0]

            # All bars agent color — no red highlight
            ax.bar(range(n_blocks), mean_curve, yerr=std_curve, capsize=2,
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
                   error_kw={"linewidth": 0.8, "ecolor": "#444444"}, zorder=2)

            rng = np.random.default_rng(0)
            for s in range(n_seeds):
                jitter = (rng.random(n_blocks) - 0.5) * 0.45
                ax.scatter(np.arange(n_blocks) + jitter, stacked[s],
                           s=12, color="#222222", alpha=0.5, zorder=4,
                           edgecolors="white", linewidths=0.25)

            early = mean_curve[:9].mean() if col == 0 else mean_curve[:-1].mean()
            ax.axhline(early, color=color, linestyle="--", linewidth=1.2,
                       alpha=0.7, label=f"B0–B{n_blocks-2} mean: {early:.1f}")

            # Plain-text block-9 annotation on the fixed panel only (no red).
            # Placed mid-panel via axes-fraction coords to avoid the title.
            if col == 0:
                b9 = mean_curve[9]
                ratio = b9 / early if early > 0 else float("inf")
                ax.text(0.30, 0.88,
                        f"Block 9: {b9:.1f}\n({ratio:.1f}× B0–B8)",
                        transform=ax.transAxes, fontsize=8,
                        color="black", fontweight="bold",
                        va="top", ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec="#888888", alpha=0.85, linewidth=1.0))

            ax.set_xlabel("Block Index")
            ax.set_ylabel("Avg Attacks")
            ax.set_xticks(range(n_blocks))
            ax.set_xticklabels([f"B{i}" for i in range(n_blocks)], fontsize=7)
            ax.legend(fontsize=8)
            ax.text(0.02, 0.96,
                    f"{n_seeds} seeds (n={'/'.join(map(str, n_full))})",
                    transform=ax.transAxes, fontsize=7.5, va="top", color="gray")
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "ext_fig2_burst_comparison")


# ---------------------------------------------------------------------------
# Copy the 3 already-multi-seed figures (Fig 1, 3, 4) into the same folder
# ---------------------------------------------------------------------------

def copy_existing_multiseed_figures():
    """
    These are already multi-seed; copied from MS_FIG_DIR. If you generated
    titleless versions (multiseed.py notitle/ folder), point MS_FIG_DIR there.
    Copies both .png and .pdf when present.
    """
    bases = [
        "ms_fig3_learning_curves_shaded",  # Fig 1
        "ms_fig1_detection_errorbars",     # Fig 3
        "ms_fig4_statistical_tests",       # Fig 4
    ]
    for base in bases:
        copied = False
        for ext in (".png", ".pdf"):
            src = os.path.join(MS_FIG_DIR, base + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUTPUT_DIR, base + ext))
                copied = True
        if copied:
            print(f"  Copied: {base} (from {MS_FIG_DIR})")
        else:
            print(f"  WARNING: {base}.png/.pdf not found in {MS_FIG_DIR} "
                  f"— run multiseed.py plot first")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("All paper figures -> single folder, no titles, multi-seed")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

    summary = load_summary()

    print("\n[Fig 2] Detection comparison");  plot_detection_comparison(summary)
    print("[Fig 5] Trade-off frontier");      plot_tradeoff_frontier(summary)
    print("[Fig 8] Summary heatmap");         plot_summary_heatmap(summary)
    print("[Fig 6] End-game burst");          plot_endgame_burst()
    print("[Fig 7] Fixed vs random burst");   plot_burst_comparison()
    print("[Fig 1/3/4] Copying existing multi-seed figures")
    copy_existing_multiseed_figures()

    print("\n" + "=" * 70)
    print("Done. All paper figures in:", os.path.abspath(OUTPUT_DIR))
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            print("   ", f)
    print("=" * 70)


if __name__ == "__main__":
    main()