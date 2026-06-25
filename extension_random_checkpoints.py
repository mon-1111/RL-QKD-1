"""
Extension: Random Checkpoint Interval Experiment
=================================================
Tests whether the end-game burst strategy discovered by RL agents
persists when checkpoint intervals are randomized (min=5, max=15 blocks).

Central hypothesis:
    The burst emerges because agents identify block 9 as terminal via
    backward induction. Randomizing episode length removes that signal.
    If the burst disappears, it confirms the exploit is structural —
    a direct consequence of fixed, predictable checkpoint intervals.

Usage:
    python extension_random_checkpoints.py verify   # quick checks (~1 min)
    python extension_random_checkpoints.py run      # full experiment (~40 min)
    python extension_random_checkpoints.py plot     # generate figures
    python extension_random_checkpoints.py all      # verify + run + plot

Results saved to: results/extension/
    randckpt_qlearning_noise01.json   (and noise03, noise05)
    randckpt_sarsa_noise01.json       ...
    randckpt_doubleq_noise01.json     ...

Figures saved to: results/extension/figures/
    ext_fig1_detection_comparison.png
    ext_fig2_burst_comparison.png
    ext_fig3_episode_lengths.png
"""

import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

# Import from the modified modules
from bb84_environment import BB84Environment, RewardConfig, N_STATES, N_ACTIONS
from agents import make_agent, SARSAAgent
from training import EpisodeLog, TrainingResult, run_training, load_result


# ---------------------------------------------------------------------------
# Extension constants
# ---------------------------------------------------------------------------

MIN_BLOCKS   = 5
MAX_BLOCKS   = 15
NOISE_LEVELS = [0.01, 0.03, 0.05]
AGENT_TYPES  = ["qlearning", "sarsa", "doubleq"]
N_EPISODES   = 10000

RESULTS_DIR = "results/extension"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")           # titled (original)
FIGURES_DIR_NOTITLE = os.path.join(FIGURES_DIR, "notitle")   # same names, no titles
ORIG_DIR    = "results/main"

AGENT_LABELS = {
    "qlearning" : "Q-Learning",
    "sarsa"     : "SARSA",
    "doubleq"   : "Double Q-Learning",
}
AGENT_COLORS = {
    "qlearning" : "#2196F3",
    "sarsa"     : "#FF9800",
    "doubleq"   : "#4CAF50",
}
NOISE_LABELS = {0.01: "1%", 0.03: "3%", 0.05: "5%"}


# ---------------------------------------------------------------------------
# Modified training loop (uses random-checkpoint environment)
# ---------------------------------------------------------------------------

def run_training_randckpt(
    agent_type    : str,
    noise_level   : float,
    alpha         : float,
    gamma         : float,
    n_episodes    : int    = N_EPISODES,
    min_blocks    : int    = MIN_BLOCKS,
    max_blocks    : int    = MAX_BLOCKS,
    reward_config : RewardConfig = None,
    rng_seed      : Optional[int] = None,
    verbose       : bool   = False,
    log_interval  : int    = 500,
) -> TrainingResult:
    """
    Train one agent with randomized checkpoint intervals.
    Identical to training.run_training() except the environment
    uses min_blocks/max_blocks instead of fixed 10 blocks.
    """
    env = BB84Environment(
        channel_error_rate=noise_level,
        reward_config=reward_config or RewardConfig(),
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        rng_seed=rng_seed,
    )

    agent = make_agent(
        agent_type,
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_episodes=n_episodes,
        rng_seed=rng_seed,
    )

    result = TrainingResult(
        agent_type=agent_type,
        noise_level=noise_level,
        alpha=alpha,
        gamma=gamma,
        n_episodes=n_episodes,
    )

    for ep in range(n_episodes):
        ep_log         = _run_episode_randckpt(env, agent, agent_type)
        ep_log.episode = ep
        result.logs.append(ep_log)
        agent.decay_epsilon()

        if verbose and (ep + 1) % log_interval == 0:
            recent      = result.logs[-log_interval:]
            avg_reward  = np.mean([e.total_reward for e in recent])
            detect_rate = np.mean([e.detected for e in recent])
            avg_correct = np.mean([e.correct_bits for e in recent])
            print(f"  Ep {ep+1:6d}/{n_episodes} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"avg_reward={avg_reward:7.2f} | "
                  f"detect_rate={detect_rate:.3f} | "
                  f"avg_correct={avg_correct:.1f}")

    result.compute_summaries()
    return result


def _run_episode_randckpt(env, agent, agent_type: str) -> EpisodeLog:
    """
    Run one episode. Identical structure to training._run_episode().
    The variable episode length is handled entirely inside the environment.
    """
    state             = env.reset()
    done              = False
    total_reward      = 0.0
    attacks_per_block = []

    if agent_type == "sarsa":
        action = agent.select_action(state)
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state,
                         next_action=next_action, done=done)
            state  = next_state
            action = next_action
            total_reward += reward
            if info["completed_block_attacks"] is not None:
                attacks_per_block.append(info["completed_block_attacks"])
    else:
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done=done)
            state = next_state
            total_reward += reward
            if info["completed_block_attacks"] is not None:
                attacks_per_block.append(info["completed_block_attacks"])

    return EpisodeLog(
        episode           = 0,
        total_reward      = total_reward,
        detected          = info["detected"],
        correct_bits      = info["total_correct_bits"],
        total_attacks     = info["total_attacks"],
        episode_length    = info["qubit_idx"] + 1,
        attacks_per_block = attacks_per_block,
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_result(result: TrainingResult, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "agent_type"              : result.agent_type,
        "noise_level"             : result.noise_level,
        "alpha"                   : result.alpha,
        "gamma"                   : result.gamma,
        "n_episodes"              : result.n_episodes,
        "min_blocks"              : MIN_BLOCKS,
        "max_blocks"              : MAX_BLOCKS,
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
    print(f"  Saved → {filepath}  ({os.path.getsize(filepath)//1024} KB)")


def load_extension_result(filepath: str) -> TrainingResult:
    return load_result(filepath)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_all(rng_seed: int = 42):
    """
    Train all 9 agent-noise combinations with randomized checkpoints.
    Uses the same best hyperparameters as the original experiment.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load best hyperparams from original sensitivity analysis
    hp_path = "results/best_hyperparams.json"
    if not os.path.exists(hp_path):
        hp_path = "best_hyperparams.json"
    with open(hp_path) as f:
        best_hp = json.load(f)

    total = len(AGENT_TYPES) * len(NOISE_LEVELS)
    print("=" * 65)
    print("Extension: Random Checkpoint Experiment")
    print(f"  Blocks per episode: Uniform[{MIN_BLOCKS}, {MAX_BLOCKS}]")
    print(f"  Expected episode length: "
          f"{(MIN_BLOCKS + MAX_BLOCKS) // 2 * 100} qubits")
    print(f"  {len(AGENT_TYPES)} agents × {len(NOISE_LEVELS)} noise levels "
          f"= {total} runs × {N_EPISODES} episodes")
    print(f"  Saving to: {os.path.abspath(RESULTS_DIR)}")
    print("=" * 65)

    run_count = 0
    t_start   = time.time()

    for agent_type in AGENT_TYPES:
        for noise in NOISE_LEVELS:
            run_count += 1
            noise_key  = str(noise)
            alpha      = best_hp[agent_type][noise_key]["alpha"]
            gamma      = best_hp[agent_type][noise_key]["gamma"]

            print(f"\n[{run_count}/{total}] {agent_type} | "
                  f"noise={noise} | α={alpha} | γ={gamma}")

            result = run_training_randckpt(
                agent_type=agent_type,
                noise_level=noise,
                alpha=alpha,
                gamma=gamma,
                n_episodes=N_EPISODES,
                min_blocks=MIN_BLOCKS,
                max_blocks=MAX_BLOCKS,
                rng_seed=rng_seed,
                verbose=True,
                log_interval=1000,
            )

            noise_str = f"{int(round(noise * 100)):02d}"
            filename  = f"randckpt_{agent_type}_noise{noise_str}.json"
            filepath  = os.path.join(RESULTS_DIR, filename)
            save_result(result, filepath)

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*65}")
    print(f"Extension complete in {elapsed:.1f} minutes.")
    _print_summary()
    print("=" * 65)


def _print_summary():
    """Print a comparison table of original vs extension results."""
    print("\nDetection Rate Comparison (last 500 episodes):")
    print(f"  {'Agent':15s} | {'Noise':6s} | "
          f"{'Original':10s} | {'RandCkpt':10s} | {'Δ':8s}")
    print("  " + "-" * 60)

    for agent_type in AGENT_TYPES:
        for noise in NOISE_LEVELS:
            noise_str = f"{int(round(noise * 100)):02d}"

            orig_path = os.path.join(
                ORIG_DIR, f"{agent_type}_noise{noise_str}.json")
            ext_path  = os.path.join(
                RESULTS_DIR, f"randckpt_{agent_type}_noise{noise_str}.json")

            orig_det = "N/A"
            ext_det  = "N/A"
            delta    = "N/A"

            if os.path.exists(orig_path):
                with open(orig_path) as f:
                    d = json.load(f)
                orig_det = d["detection_rate_last_500"]

            if os.path.exists(ext_path):
                with open(ext_path) as f:
                    d = json.load(f)
                ext_det = d["detection_rate_last_500"]

            if orig_det != "N/A" and ext_det != "N/A":
                delta = f"{(ext_det - orig_det)*100:+.1f}pp"
                orig_det_s = f"{orig_det*100:.1f}%"
                ext_det_s  = f"{ext_det*100:.1f}%"
            else:
                orig_det_s = str(orig_det)
                ext_det_s  = str(ext_det)

            print(f"  {AGENT_LABELS[agent_type]:15s} | "
                  f"{NOISE_LABELS[noise]:6s} | "
                  f"{orig_det_s:10s} | "
                  f"{ext_det_s:10s} | "
                  f"{delta:8s}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_figure(fig, fname, show_title):
    """
    Save a figure to the right folder, keeping the same filename.

    show_title=True  -> results/extension/figures/
    show_title=False -> results/extension/figures/notitle/
    """
    out_dir = FIGURES_DIR if show_title else FIGURES_DIR_NOTITLE
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fpath}")


def plot_all():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR_NOTITLE, exist_ok=True)
    for show_title in (True, False):
        _plot_detection_comparison(show_title=show_title)
        _plot_burst_comparison(show_title=show_title)
        _plot_episode_lengths(show_title=show_title)
    print(f"\nFigures saved to:")
    print(f"  Titled   -> {os.path.abspath(FIGURES_DIR)}/")
    print(f"  No-title -> {os.path.abspath(FIGURES_DIR_NOTITLE)}/")


def _load_both(agent_type: str, noise: float):
    """Load original and extension results for one agent-noise pair."""
    noise_str = f"{int(round(noise * 100)):02d}"
    orig_path = os.path.join(ORIG_DIR,
                             f"{agent_type}_noise{noise_str}.json")
    ext_path  = os.path.join(RESULTS_DIR,
                             f"randckpt_{agent_type}_noise{noise_str}.json")

    orig = load_result(orig_path) if os.path.exists(orig_path) else None
    ext  = load_result(ext_path)  if os.path.exists(ext_path)  else None
    return orig, ext


def _plot_detection_comparison(show_title=True):
    """
    Side-by-side bar chart: original vs random-checkpoint detection rates.
    One group per noise level, one pair of bars per agent.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    if show_title:
        fig.suptitle(
            "Detection Rate: Fixed vs Random Checkpoints",
            fontsize=14, fontweight="bold",
        )

    bar_width = 0.35
    x         = np.arange(len(AGENT_TYPES))

    for ax, noise in zip(axes, NOISE_LEVELS):
        orig_rates = []
        ext_rates  = []

        for agent_type in AGENT_TYPES:
            orig, ext = _load_both(agent_type, noise)
            orig_rates.append(orig.detection_rate_last_500 * 100
                              if orig else 0)
            ext_rates.append(ext.detection_rate_last_500 * 100
                             if ext else 0)

        bars1 = ax.bar(x - bar_width/2, orig_rates, bar_width,
                       label="Fixed (original)",
                       color=[AGENT_COLORS[a] for a in AGENT_TYPES],
                       alpha=0.9)
        bars2 = ax.bar(x + bar_width/2, ext_rates, bar_width,
                       label="Random [5–15]",
                       color=[AGENT_COLORS[a] for a in AGENT_TYPES],
                       alpha=0.45, hatch="//")

        for bar, val in zip(bars1, orig_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
        for bar, val in zip(bars2, ext_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(f"μ_ch = {NOISE_LABELS[noise]}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS[a].replace(" ", "\n")
                            for a in AGENT_TYPES], fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Detection Rate (%)" if ax == axes[0] else "")
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

    # Legend — solid = fixed, hatched = random
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.9,          label="Fixed checkpoints"),
        Patch(facecolor="gray", alpha=0.45, hatch="//", label="Random [5–15]"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    _save_figure(fig, "ext_fig1_detection_comparison.png", show_title)
    plt.close()


def _plot_burst_comparison(show_title=True):
    """
    For μ_ch = 3%: compare mean attacks per block between
    fixed and random-checkpoint conditions.
    One row per agent (3 rows), two subplots per row (fixed | random).
    This is the key figure — does block 9 still spike?
    """
    noise      = 0.03
    noise_str  = "03"
    n_last     = 500

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    if show_title:
        fig.suptitle(
            "End-Game Burst: Fixed vs Random Checkpoints  (μ_ch = 3%, "
            "last 500 episodes, full episodes only)",
            fontsize=13, fontweight="bold",
        )

    for row, agent_type in enumerate(AGENT_TYPES):
        orig, ext = _load_both(agent_type, noise)
        color     = AGENT_COLORS[agent_type]
        label     = AGENT_LABELS[agent_type]

        for col, (result, title) in enumerate(
            [(orig, "Fixed Checkpoints (original)"),
             (ext,  f"Random Checkpoints [5–15]")]
        ):
            ax = axes[row][col]
            ax.set_title(f"{label} — {title}", fontsize=10)

            if result is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            # Full episodes only (all 10 blocks for fixed;
            # variable length for random — use episodes with ≥10 blocks
            # for comparability, or all full episodes)
            last_logs  = result.logs[-n_last:]
            if col == 0:
                # Fixed: full episode = exactly 10 blocks
                full_eps = [e for e in last_logs
                            if len(e.attacks_per_block) == 10]
                n_blocks_plot = 10
            else:
                # Random: full episode = no detection, variable length
                # Plot up to 15 blocks; pad missing blocks with 0
                full_eps      = [e for e in last_logs
                                 if not e.detected]
                n_blocks_plot = MAX_BLOCKS

            if not full_eps:
                ax.text(0.5, 0.5, "No full episodes", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            # Pad each episode to n_blocks_plot with 0s
            padded = []
            for e in full_eps:
                blocks = list(e.attacks_per_block)
                while len(blocks) < n_blocks_plot:
                    blocks.append(0)
                padded.append(blocks[:n_blocks_plot])

            mean_attacks = np.mean(padded, axis=0)
            b08_avg      = np.mean(mean_attacks[:9]) if col == 0 else \
                           np.mean(mean_attacks[:-1])

            bar_colors = [color] * n_blocks_plot
            if col == 0:
                bar_colors[-1] = "#F44336"   # highlight block 9 in red

            ax.bar(range(n_blocks_plot), mean_attacks, color=bar_colors,
                   alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.axhline(b08_avg, color=color, linestyle="--",
                       linewidth=1.2, alpha=0.7,
                       label=f"B0–B{n_blocks_plot-2} avg: {b08_avg:.1f}")

            if col == 0:
                # Annotate burst ratio
                b9 = mean_attacks[-1]
                ratio = b9 / b08_avg if b08_avg > 0 else float("inf")
                ax.annotate(
                    f"Block 9:\n{b9:.1f} attacks\n({ratio:.1f}× avg)",
                    xy=(9, b9), xytext=(7, b9 * 1.1 + 0.5),
                    fontsize=8, color="#F44336", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="#F44336", alpha=0.8),
                )

            ax.set_xlabel("Block Index")
            ax.set_ylabel("Avg Attacks")
            ax.set_xticks(range(n_blocks_plot))
            ax.set_xticklabels([f"B{i}" for i in range(n_blocks_plot)],
                               fontsize=7)
            ax.legend(fontsize=8)
            ax.text(0.02, 0.96, f"n={len(full_eps)}/{n_last} full eps",
                    transform=ax.transAxes, fontsize=8, va="top")
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "ext_fig2_burst_comparison.png", show_title)
    plt.close()


def _plot_episode_lengths(show_title=True):
    """
    Histogram of episode lengths in the random-checkpoint condition.
    Confirms the Uniform[5,15] block distribution is working correctly.
    One subplot per agent at μ_ch = 3%.
    """
    noise = 0.03

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    if show_title:
        fig.suptitle(
            "Episode Length Distribution — Random Checkpoints  (μ_ch = 3%)",
            fontsize=13, fontweight="bold",
        )

    for ax, agent_type in zip(axes, AGENT_TYPES):
        _, ext = _load_both(agent_type, noise)
        color  = AGENT_COLORS[agent_type]
        label  = AGENT_LABELS[agent_type]

        ax.set_title(label, fontsize=11, color=color, fontweight="bold")

        if ext is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        lengths = [e.episode_length for e in ext.logs[-500:]]
        ax.hist(lengths, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(np.mean(lengths), color="black", linestyle="--",
                   linewidth=1.2, label=f"Mean: {np.mean(lengths):.0f}")
        ax.axvline(500,  color="red",   linestyle=":", linewidth=1,
                   label="Min (5 blocks)")
        ax.axvline(1500, color="green", linestyle=":", linewidth=1,
                   label="Max (15 blocks)")
        ax.set_xlabel("Episode Length (qubits)")
        ax.set_ylabel("Count" if ax == axes[0] else "")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, "ext_fig3_episode_lengths.png", show_title)
    plt.close()


# ---------------------------------------------------------------------------
# Quick verification (sanity check before full run)
# ---------------------------------------------------------------------------

def verify():
    """
    Run a quick smoke test to confirm everything works before
    committing to the full 10,000-episode training run.
    """
    print("=" * 65)
    print("Extension Verification")
    print("=" * 65)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"

    def check(condition, label, observed=None, expected=None):
        status = PASS if condition else FAIL
        msg    = f"  {status} | {label}"
        if observed is not None:
            msg += f"  (observed={observed}, expected={expected})"
        print(msg)

    # V1: Short training run completes without error
    print("\n[V1] Short training run (100 episodes)")
    for agent_type in AGENT_TYPES:
        result = run_training_randckpt(
            agent_type=agent_type,
            noise_level=0.03,
            alpha=0.5,
            gamma=0.5,
            n_episodes=100,
            rng_seed=42,
        )
        ok = (len(result.logs) == 100 and
              all(isinstance(e.total_reward, float) for e in result.logs))
        check(ok, f"{agent_type}: 100 episodes complete")

    # V2: Episode lengths vary within [500, 1500]
    print("\n[V2] Episode length distribution")
    _env = BB84Environment(channel_error_rate=0.0, qber_threshold=1.0,
                           min_blocks=MIN_BLOCKS, max_blocks=MAX_BLOCKS)
    lengths = []
    for _ in range(300):
        _env.reset()
        done  = False
        steps = 0
        while not done:
            _, _, done, _ = _env.step(0)
            steps += 1
        lengths.append(steps)
    check(min(lengths) >= 500,  "Min length >= 500",  min(lengths),  500)
    check(max(lengths) <= 1500, "Max length <= 1500", max(lengths), 1500)
    check(abs(np.mean(lengths) - 1000) < 60,
          "Mean length ≈ 1000", round(np.mean(lengths), 1), 1000)

    # V3: State space is 5040
    print("\n[V3] State space")
    check(N_STATES == 5040, "N_STATES = 5040", N_STATES, 5040)

    # V4: attacks_per_block length matches episode length
    print("\n[V4] attacks_per_block length consistency")
    result_v4 = run_training_randckpt(
        "qlearning", noise_level=0.0, alpha=0.5, gamma=0.5,
        n_episodes=100, rng_seed=1,
    )
    all_ok = True
    for e in result_v4.logs:
        expected_blocks = e.episode_length // 100
        if len(e.attacks_per_block) != expected_blocks:
            all_ok = False
            break
    check(all_ok, "attacks_per_block length == episode_length // 100")

    # V5: Hyperparams file accessible
    print("\n[V5] best_hyperparams.json accessible")
    hp_path = ("results/best_hyperparams.json"
               if os.path.exists("results/best_hyperparams.json")
               else "best_hyperparams.json")
    check(os.path.exists(hp_path),
          f"Found hyperparams at {hp_path}")

    print("\n" + "=" * 65)
    print("Verification complete. Ready for full run.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if mode == "verify":
        verify()

    elif mode == "run":
        run_all(rng_seed=42)

    elif mode == "plot":
        plot_all()

    elif mode == "all":
        verify()
        run_all(rng_seed=42)
        plot_all()

    elif mode == "summary":
        _print_summary()

    else:
        print(f"Unknown mode '{mode}'.")
        print("Usage: python extension_random_checkpoints.py "
              "[verify|run|plot|all|summary]")