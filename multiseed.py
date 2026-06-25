"""
Multi-Seed Experiment Runner
==============================
Runs all experiments (main + extension) across multiple random seeds
and computes mean ± std for all key metrics. Required for publication.

Why this matters:
    A single-seed result like "Q-Learning achieves 0.2% detection" could
    be lucky or unlucky. With 5 seeds we can report "0.2% ± 0.1%" and
    test whether differences between agents are statistically significant.

Usage:
    python multiseed.py verify          # smoke test (~2 min)
    python multiseed.py run_main        # main experiment, 5 seeds (~4 hrs)
    python multiseed.py run_extension   # random checkpoint, 5 seeds (~4 hrs)
    python multiseed.py run_all         # both (~8 hrs)
    python multiseed.py analyze         # stats + tables (needs results)
    python multiseed.py plot            # figures with error bars (needs results)

Results saved to:
    results/multiseed/main/
        qlearning_noise01_seed0.json  ...  qlearning_noise01_seed4.json
        ...
    results/multiseed/extension/
        randckpt_qlearning_noise01_seed0.json  ...

Summary tables saved to:
    results/multiseed/summary_main.json
    results/multiseed/summary_extension.json

Figures saved to:
    results/multiseed/figures/
        ms_fig1_detection_errorbars.png
        ms_fig2_correct_bits_errorbars.png
        ms_fig3_learning_curves_shaded.png
        ms_fig4_statistical_tests.png
"""

import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional

from bb84_environment import BB84Environment, RewardConfig, N_STATES, N_ACTIONS
from agents import make_agent
from training import (
    run_training, load_result, TrainingResult, EpisodeLog, AGENT_TYPES, NOISE_LEVELS
)
from extension_random_checkpoints import (
    run_training_randckpt, save_result as save_ext_result,
    MIN_BLOCKS, MAX_BLOCKS
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SEEDS      = 5
SEEDS        = [42, 123, 456, 789, 1337]   # fixed, reproducible
N_EPISODES   = 10000

MAIN_DIR      = "results/multiseed/main"
EXT_DIR       = "results/multiseed/extension"
FIGURES_DIR   = "results/multiseed/figures"
FIGURES_DIR_NOTITLE = os.path.join(FIGURES_DIR, "notitle")
SUMMARY_DIR   = "results/multiseed"

AGENT_LABELS  = {
    "qlearning" : "Q-Learning",
    "sarsa"     : "SARSA",
    "doubleq"   : "Double Q-Learning",
}
AGENT_COLORS  = {
    "qlearning" : "#3A6EA5",   # slate blue
    "sarsa"     : "#C08552",   # ochre
    "doubleq"   : "#5A8A6B",   # sage green
}
NOISE_LABELS  = {0.01: "1%", 0.03: "3%", 0.05: "5%"}


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_result(result: TrainingResult, filepath: str):
    """Save a TrainingResult to JSON (mirrors training._save_result)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "agent_type"              : result.agent_type,
        "noise_level"             : result.noise_level,
        "alpha"                   : result.alpha,
        "gamma"                   : result.gamma,
        "n_episodes"              : result.n_episodes,
        "mean_reward_last_500"    : result.mean_reward_last_500,
        "detection_rate_last_500" : result.detection_rate_last_500,
        "mean_correct_last_500"   : result.mean_correct_last_500,
        "mean_attacks_last_500"   : result.mean_attacks_last_500,
        "logs": [
            {
                "episode"           : e.episode,
                "total_reward"      : e.total_reward,
                "detected"          : e.detected,
                "correct_bits"      : e.correct_bits,
                "total_attacks"     : e.total_attacks,
                "episode_length"    : e.episode_length,
                "attacks_per_block" : e.attacks_per_block,
            }
            for e in result.logs
        ]
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def _load_seed_results(
    agent_type : str,
    noise      : float,
    results_dir: str,
    prefix     : str = "",
) -> list:
    """
    Load all seed results for one agent-noise combination.
    Returns a list of TrainingResult objects (one per seed found).
    """
    noise_str = f"{int(round(noise * 100)):02d}"
    loaded    = []
    for seed_idx, seed in enumerate(SEEDS):
        fname = f"{prefix}{agent_type}_noise{noise_str}_seed{seed_idx}.json"
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            loaded.append(load_result(fpath))
    return loaded


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_seed_stats(results: list) -> dict:
    """
    Given a list of TrainingResult objects (one per seed),
    compute mean and std for all key metrics over the last 500 episodes.

    Returns
    -------
    dict with keys:
        detection_rate  : (mean, std)
        mean_reward     : (mean, std)
        mean_correct    : (mean, std)
        mean_attacks    : (mean, std)
        n_seeds         : int
    """
    det_rates   = [r.detection_rate_last_500 for r in results]
    rewards     = [r.mean_reward_last_500    for r in results]
    corrects    = [r.mean_correct_last_500   for r in results]
    attacks     = [r.mean_attacks_last_500   for r in results]

    return {
        "detection_rate" : (float(np.mean(det_rates)), float(np.std(det_rates))),
        "mean_reward"    : (float(np.mean(rewards)),   float(np.std(rewards))),
        "mean_correct"   : (float(np.mean(corrects)),  float(np.std(corrects))),
        "mean_attacks"   : (float(np.mean(attacks)),   float(np.std(attacks))),
        "n_seeds"        : len(results),
        "raw_detection"  : det_rates,
        "raw_reward"     : rewards,
        "raw_correct"    : corrects,
        "raw_attacks"    : attacks,
    }


def compute_learning_curve_stats(results: list, window: int = 200) -> dict:
    """
    Compute mean ± std of the smoothed detection rate learning curve
    across seeds. Used for shaded learning curve plots.

    Returns arrays of length n_episodes for mean and std.
    """
    n_eps = min(r.n_episodes for r in results)

    # Smooth each seed's detection curve with a rolling window
    smoothed = []
    for r in results:
        raw = np.array([e.detected for e in r.logs[:n_eps]], dtype=float)
        # Rolling mean via cumsum trick
        cumsum    = np.cumsum(np.insert(raw, 0, 0))
        half      = window // 2
        padded    = np.pad(raw, half, mode="edge")
        cum2      = np.cumsum(np.insert(padded, 0, 0))
        rolled    = (cum2[window:] - cum2[:-window]) / window
        smoothed.append(rolled[:n_eps])

    smoothed = np.array(smoothed)   # shape: (n_seeds, n_eps)
    return {
        "episodes" : np.arange(n_eps),
        "mean"     : smoothed.mean(axis=0),
        "std"      : smoothed.std(axis=0),
        "min"      : smoothed.min(axis=0),
        "max"      : smoothed.max(axis=0),
    }


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

def run_significance_tests(
    summary: dict,
    condition: str = "main",
) -> dict:
    """
    For each noise level, test whether differences between agent pairs
    are statistically significant using Mann-Whitney U test
    (non-parametric, appropriate for small N=5 samples).

    Comparisons: Q-Learning vs SARSA, Q-Learning vs Double Q,
                 SARSA vs Double Q.

    Returns nested dict: tests[noise][pair] = {u, p, significant}
    """
    pairs  = [
        ("qlearning", "sarsa"),
        ("qlearning", "doubleq"),
        ("sarsa",     "doubleq"),
    ]
    tests  = {}

    for noise in NOISE_LEVELS:
        noise_key    = str(noise)
        tests[noise] = {}

        for a1, a2 in pairs:
            key = f"{a1}_vs_{a2}"
            try:
                d1 = summary[condition][a1][noise_key]["raw_detection"]
                d2 = summary[condition][a2][noise_key]["raw_detection"]

                if len(d1) < 2 or len(d2) < 2:
                    tests[noise][key] = {
                        "u": None, "p": None, "significant": None,
                        "note": "insufficient data"
                    }
                    continue

                u_stat, p_val = stats.mannwhitneyu(
                    d1, d2, alternative="two-sided"
                )
                tests[noise][key] = {
                    "u"           : float(u_stat),
                    "p"           : float(p_val),
                    "significant" : bool(p_val < 0.05),
                    "mean_diff"   : float(np.mean(d1) - np.mean(d2)),
                }
            except Exception as e:
                tests[noise][key] = {
                    "u": None, "p": None,
                    "significant": None, "note": str(e)
                }

    return tests


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_main(rng_seed_list: list = SEEDS):
    """
    Run main experiment (fixed checkpoints) across all seeds.
    Skips combinations that already have a saved result file.
    """
    os.makedirs(MAIN_DIR, exist_ok=True)

    hp_path = ("results/best_hyperparams.json"
               if os.path.exists("results/best_hyperparams.json")
               else "best_hyperparams.json")
    with open(hp_path) as f:
        best_hp = json.load(f)

    total     = len(AGENT_TYPES) * len(NOISE_LEVELS) * len(rng_seed_list)
    run_count = 0
    t_start   = time.time()

    print("=" * 65)
    print("Multi-Seed Main Experiment (Fixed Checkpoints)")
    print(f"  {len(AGENT_TYPES)} agents × {len(NOISE_LEVELS)} noise × "
          f"{len(rng_seed_list)} seeds = {total} runs")
    print(f"  Seeds: {rng_seed_list}")
    print(f"  Saving to: {os.path.abspath(MAIN_DIR)}")
    print("=" * 65)

    for agent_type in AGENT_TYPES:
        for noise in NOISE_LEVELS:
            noise_key = str(noise)
            noise_str = f"{int(round(noise * 100)):02d}"
            alpha     = best_hp[agent_type][noise_key]["alpha"]
            gamma     = best_hp[agent_type][noise_key]["gamma"]

            for seed_idx, seed in enumerate(rng_seed_list):
                run_count += 1
                fname     = f"{agent_type}_noise{noise_str}_seed{seed_idx}.json"
                fpath     = os.path.join(MAIN_DIR, fname)

                if os.path.exists(fpath):
                    print(f"  [{run_count}/{total}] SKIP (exists): {fname}")
                    continue

                print(f"\n[{run_count}/{total}] {agent_type} | "
                      f"noise={noise} | α={alpha} | γ={gamma} | "
                      f"seed={seed} (idx={seed_idx})")

                result = run_training(
                    agent_type  = agent_type,
                    noise_level = noise,
                    alpha       = alpha,
                    gamma       = gamma,
                    n_episodes  = N_EPISODES,
                    rng_seed    = seed,
                    verbose     = True,
                    log_interval= 2000,
                )
                _save_result(result, fpath)
                print(f"  Saved → {fpath}  "
                      f"(det={result.detection_rate_last_500:.3f}, "
                      f"correct={result.mean_correct_last_500:.1f})")

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*65}")
    print(f"Main multi-seed complete in {elapsed:.1f} minutes.")
    print("=" * 65)


def run_extension(rng_seed_list: list = SEEDS):
    """
    Run random-checkpoint extension across all seeds.
    Skips combinations that already have a saved result file.
    """
    os.makedirs(EXT_DIR, exist_ok=True)

    hp_path = ("results/best_hyperparams.json"
               if os.path.exists("results/best_hyperparams.json")
               else "best_hyperparams.json")
    with open(hp_path) as f:
        best_hp = json.load(f)

    total     = len(AGENT_TYPES) * len(NOISE_LEVELS) * len(rng_seed_list)
    run_count = 0
    t_start   = time.time()

    print("=" * 65)
    print("Multi-Seed Extension Experiment (Random Checkpoints)")
    print(f"  Blocks: Uniform[{MIN_BLOCKS}, {MAX_BLOCKS}]")
    print(f"  {len(AGENT_TYPES)} agents × {len(NOISE_LEVELS)} noise × "
          f"{len(rng_seed_list)} seeds = {total} runs")
    print(f"  Seeds: {rng_seed_list}")
    print(f"  Saving to: {os.path.abspath(EXT_DIR)}")
    print("=" * 65)

    for agent_type in AGENT_TYPES:
        for noise in NOISE_LEVELS:
            noise_key = str(noise)
            noise_str = f"{int(round(noise * 100)):02d}"
            alpha     = best_hp[agent_type][noise_key]["alpha"]
            gamma     = best_hp[agent_type][noise_key]["gamma"]

            for seed_idx, seed in enumerate(rng_seed_list):
                run_count += 1
                fname     = f"randckpt_{agent_type}_noise{noise_str}_seed{seed_idx}.json"
                fpath     = os.path.join(EXT_DIR, fname)

                if os.path.exists(fpath):
                    print(f"  [{run_count}/{total}] SKIP (exists): {fname}")
                    continue

                print(f"\n[{run_count}/{total}] {agent_type} | "
                      f"noise={noise} | α={alpha} | γ={gamma} | "
                      f"seed={seed} (idx={seed_idx})")

                result = run_training_randckpt(
                    agent_type  = agent_type,
                    noise_level = noise,
                    alpha       = alpha,
                    gamma       = gamma,
                    n_episodes  = N_EPISODES,
                    min_blocks  = MIN_BLOCKS,
                    max_blocks  = MAX_BLOCKS,
                    rng_seed    = seed,
                    verbose     = True,
                    log_interval= 2000,
                )
                _save_result(result, fpath)
                print(f"  Saved → {fpath}  "
                      f"(det={result.detection_rate_last_500:.3f}, "
                      f"correct={result.mean_correct_last_500:.1f})")

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*65}")
    print(f"Extension multi-seed complete in {elapsed:.1f} minutes.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Analysis: build summary dicts and print tables
# ---------------------------------------------------------------------------

def analyze(save: bool = True) -> dict:
    """
    Load all seed results, compute stats, run significance tests,
    print formatted tables, and optionally save summary JSONs.

    Returns
    -------
    summary dict:
        summary["main"][agent_type][noise_key]  = stats_dict
        summary["extension"][agent_type][noise_key] = stats_dict
        summary["tests"]["main"]  = significance_tests
        summary["tests"]["extension"] = significance_tests
    """
    summary = {"main": {}, "extension": {}, "tests": {}}

    for condition, results_dir, prefix in [
        ("main",      MAIN_DIR, ""),
        ("extension", EXT_DIR,  "randckpt_"),
    ]:
        summary[condition] = {}
        for agent_type in AGENT_TYPES:
            summary[condition][agent_type] = {}
            for noise in NOISE_LEVELS:
                noise_key = str(noise)
                results   = _load_seed_results(
                    agent_type, noise, results_dir, prefix)
                if not results:
                    summary[condition][agent_type][noise_key] = None
                    continue
                summary[condition][agent_type][noise_key] = \
                    compute_seed_stats(results)

        summary["tests"][condition] = run_significance_tests(
            summary, condition)

    _print_summary_table(summary, "main")
    _print_summary_table(summary, "extension")
    _print_significance_table(summary)

    if save:
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        for condition in ["main", "extension"]:
            # Strip raw arrays before saving (too large)
            saveable = {}
            for agent_type in AGENT_TYPES:
                saveable[agent_type] = {}
                for noise in NOISE_LEVELS:
                    noise_key = str(noise)
                    s = summary[condition][agent_type].get(noise_key)
                    if s is None:
                        saveable[agent_type][noise_key] = None
                        continue
                    saveable[agent_type][noise_key] = {
                        k: v for k, v in s.items()
                        if not k.startswith("raw_")
                    }
            fpath = os.path.join(SUMMARY_DIR, f"summary_{condition}.json")
            with open(fpath, "w") as f:
                json.dump(saveable, f, indent=2)
            print(f"\nSaved summary → {fpath}")

        # Save significance tests
        tests_serializable = {}
        for condition in ["main", "extension"]:
            tests_serializable[condition] = {
                str(noise): tests
                for noise, tests in summary["tests"][condition].items()
            }
        fpath = os.path.join(SUMMARY_DIR, "significance_tests.json")
        with open(fpath, "w") as f:
            json.dump(tests_serializable, f, indent=2)
        print(f"Saved significance tests → {fpath}")

    return summary


def _print_summary_table(summary: dict, condition: str):
    label = "Main (Fixed Checkpoints)" if condition == "main" \
            else "Extension (Random Checkpoints)"
    print(f"\n{'='*70}")
    print(f"Results: {label}")
    print(f"{'='*70}")
    print(f"  {'Agent':18s} | {'Noise':6s} | "
          f"{'Det Rate (mean±std)':22s} | "
          f"{'Correct Bits (mean±std)':24s} | "
          f"{'Seeds':5s}")
    print("  " + "-" * 82)

    for agent_type in AGENT_TYPES:
        for noise in NOISE_LEVELS:
            noise_key = str(noise)
            s = summary[condition][agent_type].get(noise_key)
            if s is None:
                print(f"  {AGENT_LABELS[agent_type]:18s} | "
                      f"{NOISE_LABELS[noise]:6s} | NO DATA")
                continue
            det_m, det_s = s["detection_rate"]
            cor_m, cor_s = s["mean_correct"]
            print(f"  {AGENT_LABELS[agent_type]:18s} | "
                  f"{NOISE_LABELS[noise]:6s} | "
                  f"{det_m*100:6.2f}% ± {det_s*100:5.2f}%        | "
                  f"{cor_m:7.2f}  ± {cor_s:5.2f}            | "
                  f"{s['n_seeds']:5d}")


def _print_significance_table(summary: dict):
    print(f"\n{'='*70}")
    print("Statistical Significance (Mann-Whitney U, α=0.05)")
    print(f"{'='*70}")

    pair_labels = {
        "qlearning_vs_sarsa"  : "QL vs SARSA",
        "qlearning_vs_doubleq": "QL vs DQ",
        "sarsa_vs_doubleq"    : "SARSA vs DQ",
    }

    for condition in ["main", "extension"]:
        label = "Fixed" if condition == "main" else "Random"
        print(f"\n  [{label} Checkpoints]")
        print(f"  {'Comparison':16s} | {'Noise':6s} | "
              f"{'p-value':10s} | {'Significant':12s} | {'Mean Diff':10s}")
        print("  " + "-" * 60)

        tests = summary["tests"].get(condition, {})
        for noise in NOISE_LEVELS:
            for pair_key, pair_label in pair_labels.items():
                t = tests.get(noise, {}).get(pair_key, {})
                if not t or t.get("p") is None:
                    print(f"  {pair_label:16s} | {NOISE_LABELS[noise]:6s} | N/A")
                    continue
                sig = "✓ YES" if t["significant"] else "✗ no"
                print(f"  {pair_label:16s} | {NOISE_LABELS[noise]:6s} | "
                      f"p={t['p']:.4f}   | {sig:12s} | "
                      f"{t['mean_diff']*100:+.2f}pp")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_figure(fig, fname, show_title):
    out_dir = FIGURES_DIR if show_title else FIGURES_DIR_NOTITLE
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fpath}")


def _fmt_pct(value):
    """
    Format a percentage value with adaptive precision:
    two decimals when |value| < 10, one decimal otherwise.
    Surfaces precision on the headline low-detection cells while
    keeping high bars readable.
    """
    return f"{value:.2f}%" if abs(value) < 10 else f"{value:.1f}%"


def plot_all(summary: dict = None):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR_NOTITLE, exist_ok=True)
    if summary is None:
        summary = analyze(save=False)
    for show_title in (True, False):
        _plot_detection_errorbars(summary, show_title=show_title)
        _plot_correct_bits_errorbars(summary, show_title=show_title)
        _plot_learning_curves_shaded(summary, show_title=show_title)
        _plot_significance_heatmap(summary, show_title=show_title)
    print(f"\nFigures saved to:")
    print(f"  Titled   -> {os.path.abspath(FIGURES_DIR)}/")
    print(f"  No-title -> {os.path.abspath(FIGURES_DIR_NOTITLE)}/")


def _plot_detection_errorbars(summary: dict, show_title=True):
    """
    Detection rate mean ± std across seeds for all agents and noise levels.
    One subplot per noise level. Replaces original fig2_detection_comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    if show_title:
        fig.suptitle(
            "Detection Rate: Mean ± Std Across 5 Seeds",
            fontsize=14, fontweight="bold",
        )

    x         = np.arange(len(AGENT_TYPES))
    bar_width  = 0.5

    for ax, noise in zip(axes, NOISE_LEVELS):
        means = []
        stds  = []
        for agent_type in AGENT_TYPES:
            s = summary["main"][agent_type].get(str(noise))
            if s:
                m, sd = s["detection_rate"]
                means.append(m * 100)
                stds.append(sd * 100)
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x, means, bar_width, yerr=stds,
                      color=[AGENT_COLORS[a] for a in AGENT_TYPES],
                      capsize=6, alpha=0.85, error_kw={"linewidth": 1.5})

        for bar, m, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + sd + 1.5,
                    _fmt_pct(m), ha="center", va="bottom", fontsize=8.5)

        ax.set_title(f"μ_ch = {NOISE_LABELS[noise]}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS[a].replace(" ", "\n")
                            for a in AGENT_TYPES], fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Detection Rate (%)" if ax == axes[0] else "")
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "ms_fig1_detection_errorbars.png", show_title)
    plt.close()


def _plot_correct_bits_errorbars(summary: dict, show_title=True):
    """
    Correct bits gained mean ± std across seeds.
    Side by side: main vs extension.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey="row")
    if show_title:
        fig.suptitle(
            "Avg Correct Bits Gained: Mean ± Std Across 5 Seeds",
            fontsize=14, fontweight="bold",
        )
    CORRECT_BITS_YMAX = 50   # fixed ceiling for both rows

    for row, (condition, row_label) in enumerate([
        ("main",      "Fixed Checkpoints"),
        ("extension", "Random Checkpoints [5–15]"),
    ]):
        for ax, noise in zip(axes[row], NOISE_LEVELS):
            means = []
            stds  = []
            for agent_type in AGENT_TYPES:
                s = summary[condition][agent_type].get(str(noise))
                if s:
                    m, sd = s["mean_correct"]
                    means.append(m)
                    stds.append(sd)
                else:
                    means.append(0)
                    stds.append(0)

            x = np.arange(len(AGENT_TYPES))
            bars = ax.bar(x, means, 0.5, yerr=stds,
                          color=[AGENT_COLORS[a] for a in AGENT_TYPES],
                          capsize=6, alpha=0.85,
                          error_kw={"linewidth": 1.5})

            for bar, m, sd in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + sd + 0.3,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=8)

            if row == 0:
                ax.set_title(f"μ_ch = {NOISE_LABELS[noise]}", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([AGENT_LABELS[a].replace(" ", "\n")
                                for a in AGENT_TYPES], fontsize=8.5)
            ax.set_ylabel(f"Avg Correct Bits\n({row_label})"
                          if ax == axes[row][0] else "")
            ax.set_ylim(0, CORRECT_BITS_YMAX)
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "ms_fig2_correct_bits_errorbars.png", show_title)
    plt.close()


def _plot_learning_curves_shaded(summary: dict, show_title=True):
    """
    Learning curves (detection rate vs episode) with shaded ± std band.
    3×3 grid: rows = noise levels, columns = agents.
    Only uses main (fixed checkpoint) condition.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    if show_title:
        fig.suptitle(
            "Learning Curves with Seed Variance (Detection Rate, smoothed)",
            fontsize=14, fontweight="bold",
        )

    for col, agent_type in enumerate(AGENT_TYPES):
        for row, noise in enumerate(NOISE_LEVELS):
            ax      = axes[row][col]
            results = _load_seed_results(agent_type, noise, MAIN_DIR)
            color   = AGENT_COLORS[agent_type]

            if row == 0:
                ax.set_title(AGENT_LABELS[agent_type],
                             fontsize=11, color=color, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"μ_ch={NOISE_LABELS[noise]}\nDetection Rate (%)",
                              fontsize=9)

            if not results:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            curve = compute_learning_curve_stats(results, window=300)
            eps   = curve["episodes"]
            mean  = curve["mean"] * 100
            std   = curve["std"]  * 100

            ax.plot(eps, mean, color=color, linewidth=1.2, alpha=0.9)
            ax.fill_between(eps, mean - std, mean + std,
                            color=color, alpha=0.20)
            ax.fill_between(eps, curve["min"]*100, curve["max"]*100,
                            color=color, alpha=0.08,
                            label="seed range")

            # Annotation uses last-500 summary stats (matches reported numbers)
            # not the smoothed curve endpoint, to stay consistent with tables.
            s = summary["main"][agent_type].get(str(noise))
            if s:
                ann_mean, ann_std = s["detection_rate"]
                ann_mean *= 100
                ann_std  *= 100
            else:
                ann_mean = mean[-1]
                ann_std  = std[-1]

            # At μ_ch=5% all agents are near 100% — place annotation lower
            # to avoid overlapping the curve at the top of the panel.
            if noise == 0.05:
                ann_y = ann_mean - 18
                ann_x = eps[-1] * 0.55
            else:
                ann_y = ann_mean + 8
                ann_x = eps[-1] * 0.78

            # Match std precision to mean precision so the annotation reads
            # consistently (e.g. "29.8% ± 6.4%" rather than "29.8% ± 6.40%").
            decimals = 2 if abs(ann_mean) < 10 else 1
            ax.annotate(
                f"{ann_mean:.{decimals}f}%\n±{ann_std:.{decimals}f}%\n(last 500 eps)",
                xy=(eps[-1], ann_mean),
                xytext=(ann_x, ann_y),
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=color, alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=0.8) if noise == 0.05 else None,
            )

            ax.set_xlim(0, len(eps))
            ax.set_ylim(0, 105)
            ax.set_xlabel("Episode" if row == 2 else "")
            ax.grid(alpha=0.25)

    plt.tight_layout()
    _save_figure(fig, "ms_fig3_learning_curves_shaded.png", show_title)
    plt.close()


def _plot_significance_heatmap(summary: dict, show_title=True):
    """
    Heatmap of p-values for pairwise agent comparisons.
    Rows = comparisons, columns = noise levels.
    Separate heatmaps for main and extension conditions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if show_title:
        fig.suptitle(
            "Statistical Significance: Pairwise Agent Comparisons\n"
            "(Mann-Whitney U test, p-values; green = significant at α=0.05)",
            fontsize=12, fontweight="bold",
        )

    pair_labels = ["QL vs SARSA", "QL vs DQ", "SARSA vs DQ"]
    pair_keys   = ["qlearning_vs_sarsa", "qlearning_vs_doubleq",
                   "sarsa_vs_doubleq"]
    noise_lbls  = [NOISE_LABELS[n] for n in NOISE_LEVELS]

    for ax, condition, title in zip(
        axes,
        ["main", "extension"],
        ["Fixed Checkpoints", "Random Checkpoints"],
    ):
        tests = summary["tests"].get(condition, {})
        matrix    = np.ones((len(pair_keys), len(NOISE_LEVELS)))
        annot     = np.empty_like(matrix, dtype=object)

        for j, noise in enumerate(NOISE_LEVELS):
            for i, pair_key in enumerate(pair_keys):
                t = tests.get(noise, {}).get(pair_key, {})
                p = t.get("p")
                if p is not None:
                    matrix[i, j] = p
                    annot[i, j]  = f"p={p:.3f}" + (" *" if p < 0.05 else "")
                else:
                    annot[i, j]  = "N/A"

        # Flipped colormap: low p-value (significant) → green, high → red
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1,
                       aspect="auto")
        ax.set_xticks(range(len(NOISE_LEVELS)))
        ax.set_xticklabels(noise_lbls)
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Noise Level")

        for i in range(len(pair_keys)):
            for j in range(len(NOISE_LEVELS)):
                ax.text(j, i, annot[i, j], ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if matrix[i, j] < 0.03 else "black")

        plt.colorbar(im, ax=ax, label="p-value")

    plt.tight_layout()
    _save_figure(fig, "ms_fig4_statistical_tests.png", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    print("=" * 65)
    print("Multi-Seed Verification")
    print("=" * 65)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"

    def check(condition, label, observed=None, expected=None):
        status = PASS if condition else FAIL
        msg    = f"  {status} | {label}"
        if observed is not None:
            msg += f"  (observed={observed}, expected={expected})"
        print(msg)

    # V1: Two seeds produce different results (not identical)
    print("\n[V1] Different seeds produce different results")
    r1 = run_training("qlearning", 0.03, 0.5, 0.5,
                      n_episodes=200, rng_seed=42)
    r2 = run_training("qlearning", 0.03, 0.5, 0.5,
                      n_episodes=200, rng_seed=123)
    det1 = np.mean([e.detected for e in r1.logs])
    det2 = np.mean([e.detected for e in r2.logs])
    check(det1 != det2, "Seeds 42 and 123 yield different detection rates",
          round(det1, 3), f"!= {round(det2, 3)}")

    # V2: compute_seed_stats works correctly
    print("\n[V2] compute_seed_stats")
    results = [r1, r2]
    r1.compute_summaries()
    r2.compute_summaries()
    stats_out = compute_seed_stats(results)
    m, s = stats_out["detection_rate"]
    expected_mean = np.mean([r1.detection_rate_last_500,
                             r2.detection_rate_last_500])
    check(abs(m - expected_mean) < 0.001,
          "Mean detection rate correct",
          round(m, 4), round(expected_mean, 4))
    check(s >= 0, "Std >= 0", round(s, 4), ">= 0")
    check(stats_out["n_seeds"] == 2, "n_seeds = 2",
          stats_out["n_seeds"], 2)

    # V3: Mann-Whitney U runs without error
    print("\n[V3] Mann-Whitney U test")
    from scipy import stats as scipy_stats
    d1 = [0.1, 0.2, 0.15, 0.18, 0.12]
    d2 = [0.8, 0.9, 0.85, 0.88, 0.82]
    u, p = scipy_stats.mannwhitneyu(d1, d2, alternative="two-sided")
    check(p < 0.05, "Clearly different distributions: p < 0.05",
          round(p, 4), "< 0.05")

    # V4: Learning curve stats shape is correct
    print("\n[V4] Learning curve stats shape")
    curve = compute_learning_curve_stats([r1, r2], window=50)
    check(len(curve["mean"]) == 200,
          "Curve length == n_episodes", len(curve["mean"]), 200)
    check(len(curve["std"]) == 200,
          "Std curve length == n_episodes", len(curve["std"]), 200)
    check(all(curve["std"] >= 0), "All std values >= 0")

    # V5: Save/load round-trip
    print("\n[V5] Save/load round-trip")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test_seed.json")
        _save_result(r1, fpath)
        r1_loaded = load_result(fpath)
        check(abs(r1_loaded.detection_rate_last_500
                  - r1.detection_rate_last_500) < 0.001,
              "Detection rate preserved after save/load",
              round(r1_loaded.detection_rate_last_500, 4),
              round(r1.detection_rate_last_500, 4))

    print("\n" + "=" * 65)
    print("Multi-seed verification complete. Ready for full run.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if mode == "verify":
        verify()

    elif mode == "run_main":
        run_main()

    elif mode == "run_extension":
        run_extension()

    elif mode == "run_all":
        run_main()
        run_extension()

    elif mode == "analyze":
        summary = analyze(save=True)

    elif mode == "plot":
        summary = analyze(save=False)
        plot_all(summary)

    elif mode == "all":
        verify()
        run_main()
        run_extension()
        summary = analyze(save=True)
        plot_all(summary)

    else:
        print(f"Unknown mode '{mode}'.")
        print("Usage: python multiseed.py "
              "[verify|run_main|run_extension|run_all|analyze|plot|all]")