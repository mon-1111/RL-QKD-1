"""
Baseline Policies for BB84 Eavesdropping
==========================================
Two non-learning baselines to compare against RL agents:

    1. AlwaysAttackBaseline   — attacks every qubit (reckless Eve)
    2. FixedRateBaseline      — attacks at analytically derived optimal
                                fixed rate (rational but non-adaptive Eve)

Both use the same EpisodeLog format as the training module
so results can be directly compared in plots.
"""

import numpy as np
import json
import os
import time
from typing import Optional

from bb84_environment import BB84Environment, RewardConfig, N_STATES, N_ACTIONS
from training import EpisodeLog, TrainingResult

# Noise levels — must match training.py
NOISE_LEVELS = [0.01, 0.03, 0.05]


# ---------------------------------------------------------------------------
# Save / load (self-contained, no dependency on training._save_result)
# ---------------------------------------------------------------------------

def save_result(result: TrainingResult, filepath: str):
    """Save a TrainingResult to JSON."""
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
    print(f"  Saved → {filepath}  ({os.path.getsize(filepath)//1024} KB)")


# ---------------------------------------------------------------------------
# Fixed rate calculation
# ---------------------------------------------------------------------------

def compute_fixed_rate(noise_level: float, qber_threshold: float = 0.11) -> float:
    """
    Derive the analytically optimal fixed attack rate.

    From Lee et al. eq. 7 (simplified linear approximation):
        Expected QBER ≈ r × 0.25 + μ_ch

    Setting expected QBER just below threshold with safety margin:
        r = (threshold - safety_margin - μ_ch) / 0.25

    Parameters
    ----------
    noise_level     : channel error rate μ_ch
    qber_threshold  : detection threshold (default 0.11)

    Returns
    -------
    float : attack rate r in [0, 1]
    """
    safety_margin = 0.005
    r = (qber_threshold - safety_margin - noise_level) / 0.25
    return float(np.clip(r, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def _run_baseline_episode(
    env         : BB84Environment,
    rng         : np.random.Generator,
    attack_rate : float,
) -> EpisodeLog:
    """
    Run one episode with a fixed attack rate policy.
    Each qubit is attacked independently with probability attack_rate.
    """
    env.reset()
    done              = False
    total_reward      = 0.0
    attacks_per_block = []
    info              = {}

    while not done:
        action = 1 if rng.random() < attack_rate else 0
        _, reward, done, info = env.step(action)
        total_reward += reward
        if info["completed_block_attacks"] is not None:
            attacks_per_block.append(info["completed_block_attacks"])

    return EpisodeLog(
        episode           = 0,
        total_reward      = total_reward,
        detected          = info.get("detected", False),
        correct_bits      = info.get("total_correct_bits", 0),
        total_attacks     = info.get("total_attacks", 0),
        episode_length    = info.get("qubit_idx", 0) + 1,
        attacks_per_block = attacks_per_block,
    )


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def run_baseline(
    policy          : str,
    noise_level     : float,
    n_episodes      : int = 10000,
    reward_config   : RewardConfig = None,
    qber_threshold  : float = 0.11,
    rng_seed        : Optional[int] = None,
    verbose         : bool = False,
    log_interval    : int = 1000,
) -> TrainingResult:
    """
    Run a baseline policy for n_episodes and return TrainingResult.

    Parameters
    ----------
    policy        : 'always_attack' or 'fixed_rate'
    noise_level   : channel error rate
    n_episodes    : number of evaluation episodes
    reward_config : reward values (uses defaults if None)
    qber_threshold: detection threshold
    rng_seed      : for reproducibility
    verbose       : print progress
    log_interval  : how often to print

    Returns
    -------
    TrainingResult (agent_type set to policy name)
    """
    assert policy in ("always_attack", "fixed_rate"), \
        f"Unknown policy '{policy}'. Choose 'always_attack' or 'fixed_rate'."

    env = BB84Environment(
        channel_error_rate=noise_level,
        reward_config=reward_config or RewardConfig(),
        qber_threshold=qber_threshold,
        rng_seed=rng_seed,
    )
    rng = np.random.default_rng(rng_seed)

    attack_rate = 1.0 if policy == "always_attack" else \
                  compute_fixed_rate(noise_level, qber_threshold)

    result = TrainingResult(
        agent_type  = policy,
        noise_level = noise_level,
        alpha       = 0.0,
        gamma       = 0.0,
        n_episodes  = n_episodes,
    )

    for ep in range(n_episodes):
        ep_log         = _run_baseline_episode(env, rng, attack_rate)
        ep_log.episode = ep
        result.logs.append(ep_log)

        if verbose and (ep + 1) % log_interval == 0:
            recent      = result.logs[-log_interval:]
            avg_reward  = np.mean([e.total_reward for e in recent])
            detect_rate = np.mean([e.detected     for e in recent])
            avg_correct = np.mean([e.correct_bits for e in recent])
            print(f"  Ep {ep+1:6d}/{n_episodes} | "
                  f"rate={attack_rate:.3f} | "
                  f"avg_reward={avg_reward:7.2f} | "
                  f"detect_rate={detect_rate:.3f} | "
                  f"avg_correct={avg_correct:.1f}")

    result.compute_summaries()
    return result


# ---------------------------------------------------------------------------
# Run all baselines
# ---------------------------------------------------------------------------

def run_all_baselines(
    save_dir   : str = "results/baselines",
    n_episodes : int = 10000,
    rng_seed   : int = 42,
) -> dict:
    """
    Run both baselines across all noise levels and save results.
    Returns baseline_results[policy][noise] = TrainingResult
    """
    os.makedirs(save_dir, exist_ok=True)

    policies   = ["always_attack", "fixed_rate"]
    total_runs = len(policies) * len(NOISE_LEVELS)

    print("=" * 65)
    print("Phase 5 — Baseline Evaluation")
    print(f"  {len(policies)} baselines × {len(NOISE_LEVELS)} noise levels")
    print(f"  = {total_runs} runs × {n_episodes} episodes each")
    print(f"  Saving to: {os.path.abspath(save_dir)}")
    print()
    print("  Fixed attack rates by noise level:")
    for noise in NOISE_LEVELS:
        r = compute_fixed_rate(noise)
        print(f"    μ_ch={noise:.2f} → r={r:.4f} "
              f"(expected QBER ≈ {r*0.25 + noise:.4f}, threshold=0.11)")
    print("=" * 65)

    results   = {}
    run_count = 0
    t_start   = time.time()

    for policy in policies:
        results[policy] = {}
        for noise in NOISE_LEVELS:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {policy} | noise={noise}")

            result = run_baseline(
                policy=policy,
                noise_level=noise,
                n_episodes=n_episodes,
                rng_seed=rng_seed,
                verbose=True,
                log_interval=2000,
            )
            results[policy][noise] = result

            filename = f"{policy}_noise{int(round(noise * 100)):02d}.json"
            filepath = os.path.join(save_dir, filename)
            save_result(result, filepath)

    print(f"\n{'='*65}")
    print(f"Baselines complete in {(time.time()-t_start)/60:.1f} minutes.")
    _print_summary(results)
    print("=" * 65)

    # List all saved files
    print("\nFiles saved:")
    for f in sorted(os.listdir(save_dir)):
        full = os.path.join(save_dir, f)
        print(f"  {full}  ({os.path.getsize(full)//1024} KB)")

    return results


def _print_summary(results: dict):
    print("\nBaseline Summary:")
    print(f"  {'Policy':20s} | {'Noise':6s} | "
          f"{'Detect%':8s} | {'AvgCorrect':10s} | {'AvgAttacks':10s}")
    print("  " + "-" * 65)
    for policy, noise_dict in results.items():
        for noise, result in noise_dict.items():
            print(f"  {policy:20s} | {noise:.2f}   | "
                  f"{result.detection_rate_last_500*100:6.1f}%   | "
                  f"{result.mean_correct_last_500:9.1f}  | "
                  f"{result.mean_attacks_last_500:9.1f}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_baselines():
    print("=" * 65)
    print("Baseline Verification")
    print("=" * 65)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"

    def check(condition, label, observed=None, expected=None):
        status = PASS if condition else FAIL
        msg = f"  {status} | {label}"
        if observed is not None:
            msg += f"  (observed={observed}, expected={expected})"
        print(msg)

    # V1: Always-attack always gets detected
    print("\n[V1] Always-attack baseline")
    result = run_baseline("always_attack", noise_level=0.03,
                          n_episodes=200, rng_seed=42)
    check(result.detection_rate_last_500 == 1.0,
          "Always-attack detection rate = 100%",
          result.detection_rate_last_500, 1.0)
    avg_length = np.mean([e.episode_length for e in result.logs[-200:]])
    check(avg_length < 200,
          "Always-attack aborts early",
          round(avg_length, 1), "< 200")

    # V2: Fixed rate calculation is correct
    print("\n[V2] Fixed rate calculation")
    for noise in NOISE_LEVELS:
        r = compute_fixed_rate(noise)
        expected_qber = r * 0.25 + noise
        check(expected_qber < 0.11,
              f"Expected QBER below threshold at noise={noise}",
              round(expected_qber, 4), "< 0.11")
        check(0.0 <= r <= 1.0,
              f"Fixed rate in valid range at noise={noise}",
              round(r, 4), "0.0–1.0")

    # V3: Fixed rate has lower detection than always-attack
    print("\n[V3] Fixed rate vs always-attack")
    fixed  = run_baseline("fixed_rate",    noise_level=0.03,
                          n_episodes=500, rng_seed=42)
    always = run_baseline("always_attack", noise_level=0.03,
                          n_episodes=500, rng_seed=42)
    check(fixed.detection_rate_last_500 < always.detection_rate_last_500,
          "Fixed rate has lower detection rate",
          round(fixed.detection_rate_last_500, 3),
          f"< {round(always.detection_rate_last_500, 3)}")

    # V4: Fixed rate gains less info than always-attack
    print("\n[V4] Information gain comparison")
    check(fixed.mean_correct_last_500 < always.mean_correct_last_500,
          "Fixed rate gains fewer correct bits",
          round(fixed.mean_correct_last_500, 1),
          f"< {round(always.mean_correct_last_500, 1)}")

    # V5: Attack rate accuracy
    print("\n[V5] Fixed rate attack rate accuracy")
    for noise in [0.01, 0.03]:
        r = compute_fixed_rate(noise)
        result = run_baseline("fixed_rate", noise_level=noise,
                              n_episodes=500, rng_seed=42)
        avg_length    = np.mean([e.episode_length for e in result.logs[-500:]])
        observed_rate = result.mean_attacks_last_500 / avg_length
        check(abs(observed_rate - r) < 0.03,
              f"Attack rate matches at noise={noise}",
              round(observed_rate, 3), round(r, 3))

    # V6: Save and reload works correctly
    print("\n[V6] Save and reload")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")
        save_result(fixed, filepath)
        check(os.path.exists(filepath), "File was created")
        with open(filepath) as f:
            data = json.load(f)
        check(len(data["logs"]) == 500, "All logs saved",
              len(data["logs"]), 500)
        check(data["agent_type"] == "fixed_rate",
              "Agent type correct", data["agent_type"], "fixed_rate")

    print("\n" + "=" * 65)
    print("Baseline verification complete.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if mode == "verify":
        verify_baselines()

    elif mode == "run":
        run_all_baselines(
            save_dir="results/baselines",
            n_episodes=10000,
            rng_seed=42,
        )

    else:
        print(f"Unknown mode '{mode}'.")
        print("Usage: python baselines.py [verify|run]")