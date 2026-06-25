"""
Plots and Experiments for BB84 RL Eavesdropping Paper
======================================================
Produces all figures needed for the paper.
Run: python plots.py
Figures saved to: results/figures/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAIN_DIR        = "results/main/"
BASELINE_DIR    = "results/baselines/"
SAVE_DIR        = "results/figures"          # titled versions (original)
SAVE_DIR_NOTITLE = "results/figures/notitle" # same filenames, no titles
NOISE_LEVELS  = [0.01, 0.03, 0.05]
AGENTS        = ["qlearning", "sarsa", "doubleq"]
AGENT_LABELS  = {
    "qlearning"    : "Q-Learning",
    "sarsa"        : "SARSA",
    "doubleq"      : "Double Q-Learning",
    "always_attack": "Always Attack",
    "fixed_rate"   : "Fixed Rate",
}
AGENT_COLORS  = {
    "qlearning"    : "#2196F3",
    "sarsa"        : "#FF9800",
    "doubleq"      : "#4CAF50",
    "always_attack": "#F44336",
    "fixed_rate"   : "#9C27B0",
}
NOISE_LABELS  = {0.01: "1%", 0.03: "3%", 0.05: "5%"}
NOISE_MARKERS = {0.01: "o", 0.03: "s", 0.05: "^"}

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR_NOTITLE, exist_ok=True)


def save_figure(fig, fname, show_title):
    """
    Save a figure to the appropriate folder, keeping the same filename.

    show_title=True  -> results/figures/        (original, with titles)
    show_title=False -> results/figures/notitle/ (titles removed)

    Saves both .pdf and .png, matching the original behavior.
    """
    out_dir = SAVE_DIR if show_title else SAVE_DIR_NOTITLE
    path = os.path.join(out_dir, fname + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load(fname):
    for directory in [MAIN_DIR, BASELINE_DIR]:
        path = os.path.join(directory, fname + ".json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return {"meta": data, "logs": data["logs"]}
    raise FileNotFoundError(f"Could not find {fname}.json")

def smooth(values, window=200):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i+1]))
    return np.array(result)

def get_per_episode(logs, key):
    return np.array([e[key] for e in logs])

def last_n_mean(logs, key, n=500):
    return np.mean([e[key] for e in logs[-n:]])


# ---------------------------------------------------------------------------
# Figure 1 — Learning Curves
# ---------------------------------------------------------------------------

def plot_learning_curves(show_title=True):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    if show_title:
        fig.suptitle("Learning Curves (Detection Rate over Training Episodes)",
                     fontsize=14, fontweight="bold", y=1.01)

    for row, noise in enumerate(NOISE_LEVELS):
        for col, agent in enumerate(AGENTS):
            ax = axes[row][col]
            fname = f"{agent}_noise{int(round(noise*100)):02d}"
            d = load(fname)
            detected = get_per_episode(d["logs"], "detected").astype(float)
            smoothed = smooth(detected, window=300)
            episodes = np.arange(len(detected))

            ax.plot(episodes, detected * 100, color=AGENT_COLORS[agent],
                    alpha=0.08, linewidth=0.5)
            ax.plot(episodes, smoothed * 100, color=AGENT_COLORS[agent],
                    linewidth=2.0)
            final = last_n_mean(d["logs"], "detected") * 100
            ax.axhline(y=final, color=AGENT_COLORS[agent],
                       linestyle="--", alpha=0.5, linewidth=1.0)

            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

            if row == 0:
                ax.set_title(AGENT_LABELS[agent], fontsize=11,
                             fontweight="bold", color=AGENT_COLORS[agent])
            if col == 0:
                ax.set_ylabel(f"μ_ch = {NOISE_LABELS[noise]}\nDetection Rate (%)",
                              fontsize=10)
            if row == 2:
                ax.set_xlabel("Episode", fontsize=10)

            # Black text with white box — always readable over colored background
            ax.annotate(f"{final:.1f}%",
                        xy=(len(detected)-1, final),
                        xytext=(-50, 8), textcoords="offset points",
                        fontsize=9, color="black", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.7,
                                  edgecolor="none"))

    plt.tight_layout()
    save_figure(fig, "fig1_learning_curves", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Detection Rate Comparison
# ---------------------------------------------------------------------------

def plot_detection_comparison(show_title=True):
    fig, ax = plt.subplots(figsize=(13, 6))

    all_agents = AGENTS + ["fixed_rate", "always_attack"]
    n_agents   = len(all_agents)
    width      = 0.13
    x          = np.arange(len(NOISE_LEVELS))

    for i, agent in enumerate(all_agents):
        rates = []
        for noise in NOISE_LEVELS:
            fname = f"{agent}_noise{int(round(noise*100)):02d}"
            d = load(fname)
            rates.append(last_n_mean(d["logs"], "detected") * 100)

        offset = (i - n_agents / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width,
                      label=AGENT_LABELS[agent],
                      color=AGENT_COLORS[agent],
                      alpha=0.85, edgecolor="white", linewidth=0.5)

        for bar, rate in zip(bars, rates):
            if rate > 80:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() - 4,
                        f"{rate:.1f}%", ha="center", va="top",
                        fontsize=6.5, fontweight="bold", color="white")
            else:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1.0,
                        f"{rate:.1f}%", ha="center", va="bottom",
                        fontsize=6.5, fontweight="bold",
                        color=AGENT_COLORS[agent])

    ax.set_xlabel("Channel Noise Level (μ_ch)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    if show_title:
        ax.set_title("Detection Rate (RL Agents vs Baselines)",
                     fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"μ_ch = {NOISE_LABELS[n]}" for n in NOISE_LEVELS],
                       fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    save_figure(fig, "fig2_detection_comparison", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Temporal Pacing
# Median only, no error bars, noise=5% excluded (too few full episodes)
# ---------------------------------------------------------------------------

def plot_temporal_pacing(show_title=True):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    if show_title:
        fig.suptitle("Temporal Pacing (Median Attacks per Block)",
                     fontsize=13, fontweight="bold")

    plot_noises = [0.01, 0.03]
    # Note: noise=5% excluded — too few full episodes (<30/500) for reliable pacing stats

    # Precompute shared y-axis max per row
    row_ymaxes = {}
    for row, noise in enumerate(plot_noises):
        row_max = 0
        for agent in AGENTS:
            fname = f"{agent}_noise{int(round(noise*100)):02d}"
            d = load(fname)
            full_eps = [e for e in d["logs"][-500:]
                        if len(e["attacks_per_block"]) == 10]
            if full_eps:
                blocks_arr = np.array([e["attacks_per_block"] for e in full_eps])
                med = np.median(blocks_arr, axis=0)
                row_max = max(row_max, float(np.max(med)))
        row_ymaxes[row] = row_max * 1.6 if row_max > 0 else 10

    for row, noise in enumerate(plot_noises):
        for col, agent in enumerate(AGENTS):
            ax = axes[row][col]
            fname = f"{agent}_noise{int(round(noise*100)):02d}"
            d = load(fname)
            last500  = d["logs"][-500:]
            full_eps = [e for e in last500
                        if len(e["attacks_per_block"]) == 10]
            blocks   = np.arange(10)

            if full_eps:
                blocks_arr = np.array([e["attacks_per_block"] for e in full_eps])
                median = np.median(blocks_arr, axis=0)

                ax.bar(blocks, median, color=AGENT_COLORS[agent],
                       alpha=0.75, edgecolor="white", linewidth=0.5)

                n_full = len(full_eps)
            else:
                ax.text(0.5, 0.5, "No full episodes",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                n_full = 0

            ax.set_xticks(blocks)
            ax.set_xticklabels([f"B{i}" for i in blocks], fontsize=8)
            ax.set_ylim(0, row_ymaxes[row])
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(labelsize=9)

            if row == 0:
                ax.set_title(AGENT_LABELS[agent], fontsize=11,
                             fontweight="bold", color=AGENT_COLORS[agent])
            if col == 0:
                ax.set_ylabel(f"μ_ch = {NOISE_LABELS[noise]}\nMedian Attacks",
                              fontsize=10)
            if row == 1:
                ax.set_xlabel("Block Index", fontsize=10)

            ax.text(0.02, 0.97, f"n={n_full}/500",
                    transform=ax.transAxes, fontsize=8,
                    va="top", color="gray")

    plt.tight_layout()
    save_figure(fig, "fig3_temporal_pacing", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — Information Gain vs Stealth Trade-off
# ---------------------------------------------------------------------------

def plot_tradeoff_frontier(show_title=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    if show_title:
        fig.suptitle("Information Gain vs. Stealth: The Eavesdropper's Trade-off",
                     fontsize=13, fontweight="bold")

    for idx, noise in enumerate(NOISE_LEVELS):
        ax = axes[idx]

        for baseline in ["always_attack", "fixed_rate"]:
            fname   = f"{baseline}_noise{int(round(noise*100)):02d}"
            d       = load(fname)
            detect  = last_n_mean(d["logs"], "detected") * 100
            correct = last_n_mean(d["logs"], "correct_bits")
            ax.scatter(detect, correct, color=AGENT_COLORS[baseline],
                       marker="x", s=120, linewidths=2.5, zorder=5,
                       label=AGENT_LABELS[baseline] if idx == 0 else "")

        for agent in AGENTS:
            fname   = f"{agent}_noise{int(round(noise*100)):02d}"
            d       = load(fname)
            detect  = last_n_mean(d["logs"], "detected") * 100
            correct = last_n_mean(d["logs"], "correct_bits")
            ax.scatter(detect, correct, color=AGENT_COLORS[agent],
                       marker=NOISE_MARKERS[noise], s=100, zorder=6,
                       label=AGENT_LABELS[agent] if idx == 0 else "")

        ax.set_xlabel("Detection Rate (%)", fontsize=11)
        ax.set_ylabel("Avg Correct Bits Gained", fontsize=11)
        ax.set_title(f"μ_ch = {NOISE_LABELS[noise]}", fontsize=12,
                     fontweight="bold")
        ax.set_xlim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.axvspan(0, 30, alpha=0.05, color="green")
        ylim = ax.get_ylim()
        ax.text(2, ylim[1] * 0.95 if ylim[1] > 0 else 5,
                "Low\ndetection\nzone", fontsize=7.5,
                color="green", alpha=0.7, va="top")

    handles = [mpatches.Patch(color=AGENT_COLORS[a], label=AGENT_LABELS[a])
               for a in AGENTS + ["fixed_rate", "always_attack"]]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    save_figure(fig, "fig4_tradeoff_frontier", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5 — Performance by Noise Level
# Solid bars = detection rate, hatched = correct bits
# Y-axis labels only on outermost subplots
# ---------------------------------------------------------------------------

def plot_generalization(show_title=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    if show_title:
        fig.suptitle("Performance by Channel Noise Level",
                     fontsize=13, fontweight="bold")

    x     = np.arange(len(NOISE_LEVELS))
    width = 0.35

    for col, agent in enumerate(AGENTS):
        ax  = axes[col]
        ax2 = ax.twinx()

        detection_vals = []
        correct_vals   = []
        for noise in NOISE_LEVELS:
            fname = f"{agent}_noise{int(round(noise*100)):02d}"
            d = load(fname)
            detection_vals.append(last_n_mean(d["logs"], "detected") * 100)
            correct_vals.append(last_n_mean(d["logs"], "correct_bits"))

        bars1 = ax.bar(x - width/2, detection_vals, width,
                       color=AGENT_COLORS[agent], alpha=0.9,
                       label="Detection Rate (%)", zorder=3)
        bars2 = ax2.bar(x + width/2, correct_vals, width,
                        color=AGENT_COLORS[agent], alpha=0.4,
                        hatch="///", edgecolor=AGENT_COLORS[agent],
                        label="Avg Correct Bits", zorder=3)

        ax.set_xlabel("Channel Noise Level", fontsize=11)
        # Y-axis labels only on outermost subplots — legend explains the bars
        ax.set_ylabel("" if col > 0 else "Detection Rate (%)", fontsize=11,
                      color=AGENT_COLORS[agent])
        ax2.set_ylabel("" if col < 2 else "Avg Correct Bits", fontsize=11,
                       color=AGENT_COLORS[agent])
        ax.set_title(AGENT_LABELS[agent], fontsize=12,
                     fontweight="bold", color=AGENT_COLORS[agent])
        ax.set_xticks(x)
        ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS], fontsize=11)
        ax.set_ylim(0, 115)
        ax2.set_ylim(0, 60)
        ax.tick_params(axis="y", labelcolor=AGENT_COLORS[agent],
                       labelleft=(col == 0))
        ax2.tick_params(axis="y", labelcolor=AGENT_COLORS[agent],
                        labelright=(col == 2))
        ax.grid(True, axis="y", alpha=0.3, zorder=0)

        for bar, val in zip(bars1, detection_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color=AGENT_COLORS[agent])

        for bar, val in zip(bars2, correct_vals):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=8, color=AGENT_COLORS[agent])

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, ["Detection Rate (%)", "Avg Correct Bits"],
                  fontsize=8, loc="upper left")

    plt.tight_layout()
    save_figure(fig, "fig5_noise_sensitivity", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 6 — Summary Heatmap
# Colorbar shows 0%-100%, separator label below figure
# ---------------------------------------------------------------------------

def plot_summary_heatmap(show_title=True):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    if show_title:
        fig.suptitle("Detection Rate Heatmap (All Conditions)",
                     fontsize=13, fontweight="bold")

    all_agents_ordered = AGENTS + ["fixed_rate", "always_attack"]
    labels_ordered     = [AGENT_LABELS[a] for a in all_agents_ordered]
    noise_labels       = [f"μ_ch={NOISE_LABELS[n]}" for n in NOISE_LEVELS]

    for ax_idx, metric in enumerate(["detected", "correct_bits"]):
        ax = axes[ax_idx]

        matrix = np.zeros((len(all_agents_ordered), len(NOISE_LEVELS)))
        for i, agent in enumerate(all_agents_ordered):
            for j, noise in enumerate(NOISE_LEVELS):
                fname = f"{agent}_noise{int(round(noise*100)):02d}"
                d = load(fname)
                matrix[i, j] = last_n_mean(d["logs"], metric)

        if metric == "detected":
            cmap       = "RdYlGn_r"
            title      = "Detection Rate (%)"
            fmt        = lambda v: f"{v*100:.1f}%"
            vmin, vmax = 0, 1
        else:
            cmap       = "YlOrRd"
            title      = "Avg Correct Bits"
            fmt        = lambda v: f"{v:.1f}"
            vmin       = matrix.min()
            vmax       = matrix.max()

        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if metric == "detected":
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

        ax.set_xticks(np.arange(len(NOISE_LEVELS)))
        ax.set_yticks(np.arange(len(all_agents_ordered)))
        ax.set_xticklabels(noise_labels, fontsize=10)
        ax.set_yticklabels(labels_ordered, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for i in range(len(all_agents_ordered)):
            for j in range(len(NOISE_LEVELS)):
                val   = matrix[i, j]
                text  = fmt(val)
                color = "white" if (metric == "detected" and val > 0.6) or \
                                   (metric != "detected" and
                                    val > (vmin + vmax) * 0.6) \
                        else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

        ax.axhline(y=len(AGENTS) - 0.5, color="white",
                   linewidth=2.5, linestyle="--")

    fig.text(0.5, -0.02,
             "— — —  Dashed line separates RL agents (above) from baselines (below)",
             ha="center", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    save_figure(fig, "fig6_heatmap", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Figure 7 — End-game Burst Detail
# Error bars removed, annotation inside subplot with black text
# ---------------------------------------------------------------------------

def plot_endgame_burst(show_title=True):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    if show_title:
        fig.suptitle("End-Game Burst Strategy of Block 9 vs Earlier Blocks\n"
                     "(noise = 3%, last 500 episodes, full episodes only)",
                     fontsize=13, fontweight="bold")

    noise = 0.03

    for col, agent in enumerate(AGENTS):
        ax    = axes[col]
        fname = f"{agent}_noise{int(round(noise*100)):02d}"
        d     = load(fname)
        last500  = d["logs"][-500:]
        full_eps = [e for e in last500
                    if len(e["attacks_per_block"]) == 10]

        if not full_eps:
            ax.text(0.5, 0.5, "No full episodes",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        avg    = np.mean([e["attacks_per_block"] for e in full_eps], axis=0)
        colors = [AGENT_COLORS[agent]] * 9 + ["#F44336"]
        alphas = [0.6] * 9 + [1.0]

        for i, (a, c, al) in enumerate(zip(avg, colors, alphas)):
            ax.bar(i, a, color=c, alpha=al,
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(range(10))
        ax.set_xticklabels([f"B{i}" for i in range(10)], fontsize=9)
        ax.set_xlabel("Block Index", fontsize=11)
        ax.set_ylabel("Avg Attacks", fontsize=11)
        ax.set_title(AGENT_LABELS[agent], fontsize=12,
                     fontweight="bold", color=AGENT_COLORS[agent])
        ax.grid(True, axis="y", alpha=0.3)

        mean_early = np.mean(avg[:9])
        ax.axhline(y=mean_early, color=AGENT_COLORS[agent],
                   linestyle="--", alpha=0.5, linewidth=1.5,
                   label=f"B0–B8 avg: {mean_early:.1f}")
        ax.legend(fontsize=8, loc="lower right")

        # Annotation inside subplot — black text, red border box
        ax.text(0.72, 0.97,
                f"Block 9:\n{avg[9]:.1f} attacks\n({avg[9]/mean_early:.1f}× avg)",
                transform=ax.transAxes,
                fontsize=8.5, color="black", fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#F44336", linewidth=1.5))

        ax.text(0.02, 0.97, f"n={len(full_eps)}/500 full eps",
                transform=ax.transAxes, fontsize=8,
                va="top", color="gray")

    plt.tight_layout()
    save_figure(fig, "fig7_endgame_burst", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Results table (LaTeX-ready)
# ---------------------------------------------------------------------------

def print_results_table():
    print("\n" + "=" * 75)
    print("RESULTS TABLE (LaTeX-ready)")
    print("=" * 75)
    all_agents = AGENTS + ["fixed_rate", "always_attack"]
    print("\n\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Agent & \\multicolumn{2}{c}{$\\mu_{ch}=1\\%$} & "
          "\\multicolumn{2}{c}{$\\mu_{ch}=3\\%$} & "
          "\\multicolumn{2}{c}{$\\mu_{ch}=5\\%$} \\\\")
    print(" & Det.\\% & Bits & Det.\\% & Bits & Det.\\% & Bits \\\\")
    print("\\hline")
    for agent in all_agents:
        row = f"{AGENT_LABELS[agent]:20s}"
        for noise in NOISE_LEVELS:
            fname = f"{agent}_noise{int(round(noise*100)):02d}"
            d     = load(fname)
            det   = last_n_mean(d["logs"], "detected") * 100
            bits  = last_n_mean(d["logs"], "correct_bits")
            row  += f" & {det:.1f} & {bits:.1f}"
        row += " \\\\"
        print(row)
        if agent == "doubleq":
            print("\\hline")
    print("\\hline")
    print("\\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Generating all paper figures...")
    print(f"  Titled   -> {os.path.abspath(SAVE_DIR)}")
    print(f"  No-title -> {os.path.abspath(SAVE_DIR_NOTITLE)}")
    print("=" * 65)

    figures = [
        ("Figure 1", "Learning curves",          plot_learning_curves),
        ("Figure 2", "Detection rate comparison", plot_detection_comparison),
        ("Figure 3", "Temporal pacing",           plot_temporal_pacing),
        ("Figure 4", "Trade-off",                 plot_tradeoff_frontier),
        ("Figure 5", "Noise sensitivity",         plot_generalization),
        ("Figure 6", "Summary heatmap",           plot_summary_heatmap),
        ("Figure 7", "End-game burst detail",     plot_endgame_burst),
    ]

    for tag, desc, fn in figures:
        print(f"\n[{tag}] {desc}...")
        fn(show_title=True)    # original, with titles
        fn(show_title=False)   # no titles, same filename in notitle/

    print_results_table()

    print("\n" + "=" * 65)
    print("All figures generated.")
    print(f"  Titled   -> {os.path.abspath(SAVE_DIR)}")
    print(f"  No-title -> {os.path.abspath(SAVE_DIR_NOTITLE)}")
    for label, d in [("titled", SAVE_DIR), ("no-title", SAVE_DIR_NOTITLE)]:
        print(f"\n  [{label}] {d}")
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp) // 1024
                print(f"    {f}  ({size} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()