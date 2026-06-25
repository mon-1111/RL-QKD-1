"""
Training Loop for BB84 RL Agents
==================================
Two phases:
    Phase 4a — sensitivity analysis across alpha and gamma combinations
    Phase 4b — main training with best hyperparameters

Logs everything needed for paper figures:
    - Total reward per episode
    - Detection flag
    - Total correct bits gained
    - Total attacks made
    - Episode length
    - Attack rate per checkpoint block (for temporal pacing)
"""

import numpy as np
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from bb84_environment import BB84Environment, RewardConfig, N_STATES, N_ACTIONS
from agents import make_agent, SARSAAgent


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLog:
    """Everything logged for one training episode."""
    episode         : int
    total_reward    : float
    detected        : bool
    correct_bits    : int
    total_attacks   : int
    episode_length  : int
    # Attack count per checkpoint block (list of 10 values, or fewer if detected)
    attacks_per_block: list = field(default_factory=list)


@dataclass
class TrainingResult:
    """Full result of one training run."""
    agent_type      : str
    noise_level     : float
    alpha           : float
    gamma           : float
    n_episodes      : int
    logs            : list = field(default_factory=list)

    # Computed summaries (filled after training)
    mean_reward_last_500    : float = 0.0
    detection_rate_last_500 : float = 0.0
    mean_correct_last_500   : float = 0.0
    mean_attacks_last_500   : float = 0.0

    def compute_summaries(self):
        """Compute summary statistics from the last 500 episodes."""
        last = self.logs[-500:] if len(self.logs) >= 500 else self.logs
        self.mean_reward_last_500    = float(np.mean([e.total_reward for e in last]))
        self.detection_rate_last_500 = float(np.mean([e.detected for e in last]))
        self.mean_correct_last_500   = float(np.mean([e.correct_bits for e in last]))
        self.mean_attacks_last_500   = float(np.mean([e.total_attacks for e in last]))


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def run_training(
    agent_type      : str,
    noise_level     : float,
    alpha           : float,
    gamma           : float,
    n_episodes      : int,
    reward_config   : RewardConfig = None,
    epsilon_start   : float = 1.0,
    epsilon_end     : float = 0.01,
    rng_seed        : Optional[int] = None,
    verbose         : bool = False,
    log_interval    : int = 500,
) -> TrainingResult:
    """
    Train one agent for n_episodes and return full training result.

    Parameters
    ----------
    agent_type   : 'qlearning', 'sarsa', or 'doubleq'
    noise_level  : channel error rate (0.03, 0.05, 0.08)
    alpha        : learning rate
    gamma        : discount factor
    n_episodes   : number of training episodes
    reward_config: reward values (uses defaults if None)
    epsilon_start: initial exploration rate
    epsilon_end  : final exploration rate
    rng_seed     : for reproducibility
    verbose      : print progress every log_interval episodes
    log_interval : how often to print progress

    Returns
    -------
    TrainingResult with all episode logs
    """
    env = BB84Environment(
        channel_error_rate=noise_level,
        reward_config=reward_config or RewardConfig(),
        rng_seed=rng_seed,
    )

    agent = make_agent(
        agent_type,
        n_states=N_STATES,
        n_actions=N_ACTIONS,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
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
        ep_log = _run_episode(env, agent, agent_type)
        ep_log.episode = ep
        result.logs.append(ep_log)
        agent.decay_epsilon()

        if verbose and (ep + 1) % log_interval == 0:
            recent = result.logs[-log_interval:]
            avg_reward    = np.mean([e.total_reward for e in recent])
            detect_rate   = np.mean([e.detected for e in recent])
            avg_correct   = np.mean([e.correct_bits for e in recent])
            print(f"  Ep {ep+1:6d}/{n_episodes} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"avg_reward={avg_reward:7.2f} | "
                  f"detect_rate={detect_rate:.3f} | "
                  f"avg_correct={avg_correct:.1f}")

    result.compute_summaries()
    return result


def _run_episode(env, agent, agent_type: str) -> EpisodeLog:
    """
    Run one episode and return the log.
    Handles SARSA's need for next_action separately.

    attacks_per_block is collected using completed_block_attacks from the
    info dict — this captures the count BEFORE the environment resets it,
    fixing the off-by-one bug where we were always recording 0.
    """
    state = env.reset()
    done  = False
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
            # Capture completed block count before reset
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
# Phase 4a — Sensitivity analysis
# ---------------------------------------------------------------------------

ALPHA_VALUES = [0.01, 0.1, 0.5]
GAMMA_VALUES = [0.50, 0.75, 0.95, 0.99]
NOISE_LEVELS = [0.01, 0.03, 0.05]
AGENT_TYPES  = ["qlearning", "sarsa", "doubleq"]

SENSITIVITY_EPISODES = 2000    # fast sweep
MAIN_EPISODES        = 10000   # full training
MAIN_EPISODES_HARD   = 20000   # for high-noise conditions


def run_sensitivity_analysis(
    save_dir: str = "results/sensitivity",
    n_episodes: int = SENSITIVITY_EPISODES,
    rng_seed: int = 42,
) -> dict:
    """
    Sweep alpha and gamma combinations for all agents and noise levels.
    Returns a nested dict: results[agent][noise][alpha][gamma] = TrainingResult

    Parameters
    ----------
    save_dir   : directory to save results
    n_episodes : episodes per combination (default 2000 for speed)
    rng_seed   : for reproducibility
    """
    os.makedirs(save_dir, exist_ok=True)

    total_runs = len(AGENT_TYPES) * len(NOISE_LEVELS) * len(ALPHA_VALUES) * len(GAMMA_VALUES)
    print("=" * 65)
    print(f"Phase 4a — Sensitivity Analysis")
    print(f"  {len(AGENT_TYPES)} agents × {len(NOISE_LEVELS)} noise levels × "
          f"{len(ALPHA_VALUES)} alphas × {len(GAMMA_VALUES)} gammas")
    print(f"  = {total_runs} runs × {n_episodes} episodes each")
    print("=" * 65)

    results   = {}
    run_count = 0
    t_start   = time.time()

    for agent_type in AGENT_TYPES:
        results[agent_type] = {}
        for noise in NOISE_LEVELS:
            results[agent_type][noise] = {}
            for alpha in ALPHA_VALUES:
                results[agent_type][noise][alpha] = {}
                for gamma in GAMMA_VALUES:
                    run_count += 1
                    elapsed = time.time() - t_start
                    eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 1 else 0

                    print(f"\n[{run_count}/{total_runs}] "
                          f"{agent_type} | noise={noise} | "
                          f"α={alpha} | γ={gamma} | "
                          f"ETA={eta/60:.1f}min")

                    result = run_training(
                        agent_type=agent_type,
                        noise_level=noise,
                        alpha=alpha,
                        gamma=gamma,
                        n_episodes=n_episodes,
                        rng_seed=rng_seed,
                        verbose=True,
                        log_interval=500,
                    )
                    results[agent_type][noise][alpha][gamma] = result

    # Save summary (without full logs to keep file small)
    summary = {}
    for agent_type in AGENT_TYPES:
        summary[agent_type] = {}
        for noise in NOISE_LEVELS:
            summary[agent_type][str(noise)] = {}
            for alpha in ALPHA_VALUES:
                summary[agent_type][str(noise)][str(alpha)] = {}
                for gamma in GAMMA_VALUES:
                    r = results[agent_type][noise][alpha][gamma]
                    summary[agent_type][str(noise)][str(alpha)][str(gamma)] = {
                        "mean_reward"    : r.mean_reward_last_500,
                        "detection_rate" : r.detection_rate_last_500,
                        "mean_correct"   : r.mean_correct_last_500,
                        "mean_attacks"   : r.mean_attacks_last_500,
                    }

    with open(os.path.join(save_dir, "sensitivity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"Sensitivity analysis complete in {(time.time()-t_start)/60:.1f} minutes.")
    print(f"Summary saved to {save_dir}/sensitivity_summary.json")
    print("=" * 65)

    return results


def get_best_hyperparams(sensitivity_results: dict) -> dict:
    """
    Extract best alpha and gamma for each agent and noise level
    based on highest mean_reward_last_500.

    Returns
    -------
    best[agent_type][noise] = {'alpha': ..., 'gamma': ...}
    """
    best = {}
    print("\nBest hyperparameters per agent and noise level:")
    print("-" * 65)

    for agent_type in AGENT_TYPES:
        best[agent_type] = {}
        for noise in NOISE_LEVELS:
            best_reward = -np.inf
            best_alpha  = None
            best_gamma  = None

            for alpha in ALPHA_VALUES:
                for gamma in GAMMA_VALUES:
                    r = sensitivity_results[agent_type][noise][alpha][gamma]
                    if r.mean_reward_last_500 > best_reward:
                        best_reward = r.mean_reward_last_500
                        best_alpha  = alpha
                        best_gamma  = gamma

            best[agent_type][noise] = {
                "alpha": best_alpha,
                "gamma": best_gamma,
                "mean_reward": best_reward,
            }
            print(f"  {agent_type:12s} | noise={noise} | "
                  f"α={best_alpha} | γ={best_gamma} | "
                  f"reward={best_reward:.2f}")

    return best


# ---------------------------------------------------------------------------
# Phase 4b — Main training
# ---------------------------------------------------------------------------

def run_main_training(
    best_hyperparams: dict,
    save_dir: str = "results/main",
    n_episodes: int = MAIN_EPISODES,
    rng_seed: int = 42,
) -> dict:
    """
    Train all 9 agents using best hyperparameters from sensitivity analysis.
    Returns trained_results[agent_type][noise] = TrainingResult

    Parameters
    ----------
    best_hyperparams : output of get_best_hyperparams()
    save_dir         : directory to save results
    n_episodes       : episodes per agent (default 10000)
    rng_seed         : for reproducibility
    """
    os.makedirs(save_dir, exist_ok=True)

    total_runs = len(AGENT_TYPES) * len(NOISE_LEVELS)
    print("=" * 65)
    print(f"Phase 4b — Main Training")
    print(f"  {len(AGENT_TYPES)} agents × {len(NOISE_LEVELS)} noise levels")
    print(f"  = {total_runs} runs × {n_episodes} episodes each")
    print("=" * 65)

    trained = {}
    run_count = 0
    t_start = time.time()

    for agent_type in AGENT_TYPES:
        trained[agent_type] = {}
        for noise in NOISE_LEVELS:
            run_count += 1
            elapsed = time.time() - t_start
            eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 1 else 0

            hp = best_hyperparams[agent_type][noise]
            print(f"\n[{run_count}/{total_runs}] "
                  f"{agent_type} | noise={noise} | "
                  f"α={hp['alpha']} | γ={hp['gamma']} | "
                  f"ETA={eta/60:.1f}min")

            result = run_training(
                agent_type=agent_type,
                noise_level=noise,
                alpha=hp["alpha"],
                gamma=hp["gamma"],
                n_episodes=n_episodes,
                rng_seed=rng_seed,
                verbose=True,
                log_interval=1000,
            )
            trained[agent_type][noise] = result

            # Save full logs for this agent+noise combination
            filename = f"{agent_type}_noise{int(noise*100):02d}.json"
            filepath = os.path.join(save_dir, filename)
            _save_result(result, filepath)
            print(f"  Saved → {filepath}")

    print(f"\n{'='*65}")
    print(f"Main training complete in {(time.time()-t_start)/60:.1f} minutes.")
    print("=" * 65)

    return trained


def _save_result(result: TrainingResult, filepath: str):
    """Save a TrainingResult to JSON."""
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
                "episode"          : e.episode,
                "total_reward"     : e.total_reward,
                "detected"         : e.detected,
                "correct_bits"     : e.correct_bits,
                "total_attacks"    : e.total_attacks,
                "episode_length"   : e.episode_length,
                "attacks_per_block": e.attacks_per_block,
            }
            for e in result.logs
        ]
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_result(filepath: str) -> TrainingResult:
    """Load a saved TrainingResult from JSON."""
    with open(filepath) as f:
        data = json.load(f)

    logs = [
        EpisodeLog(
            episode           = e["episode"],
            total_reward      = e["total_reward"],
            detected          = e["detected"],
            correct_bits      = e["correct_bits"],
            total_attacks     = e["total_attacks"],
            episode_length    = e["episode_length"],
            attacks_per_block = e["attacks_per_block"],
        )
        for e in data["logs"]
    ]

    result = TrainingResult(
        agent_type = data["agent_type"],
        noise_level= data["noise_level"],
        alpha      = data["alpha"],
        gamma      = data["gamma"],
        n_episodes = data["n_episodes"],
        logs       = logs,
    )
    result.mean_reward_last_500    = data["mean_reward_last_500"]
    result.detection_rate_last_500 = data["detection_rate_last_500"]
    result.mean_correct_last_500   = data["mean_correct_last_500"]
    result.mean_attacks_last_500   = data["mean_attacks_last_500"]
    return result


# ---------------------------------------------------------------------------
# Quick smoke test — verifies training runs without crashing
# ---------------------------------------------------------------------------

def smoke_test():
    """
    Quick smoke test — runs 3 episodes per agent to verify
    the training loop works before committing to a full run.
    """
    print("=" * 65)
    print("Training Smoke Test (3 episodes per agent)")
    print("=" * 65)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"

    for agent_type in AGENT_TYPES:
        for noise in [0.03]:
            result = run_training(
                agent_type=agent_type,
                noise_level=noise,
                alpha=0.1,
                gamma=0.95,
                n_episodes=3,
                rng_seed=42,
                verbose=False,
            )
            ok = (
                len(result.logs) == 3 and
                all(isinstance(e.total_reward, float) for e in result.logs) and
                all(isinstance(e.detected, bool) for e in result.logs) and
                all(len(e.attacks_per_block) > 0 for e in result.logs)
            )
            status = PASS if ok else FAIL
            print(f"  {status} | {agent_type:12s} noise={noise} | "
                  f"rewards={[e.total_reward for e in result.logs]}")

    print("\n  Summaries:")
    for agent_type in AGENT_TYPES:
        result = run_training(
            agent_type=agent_type,
            noise_level=0.03,
            alpha=0.1,
            gamma=0.95,
            n_episodes=100,
            rng_seed=42,
        )
        result.compute_summaries()
        print(f"  {agent_type:12s} | "
              f"mean_reward={result.mean_reward_last_500:.2f} | "
              f"detect_rate={result.detection_rate_last_500:.3f} | "
              f"mean_correct={result.mean_correct_last_500:.1f}")

    print("\n" + "=" * 65)
    print("Smoke test complete. Ready for full training.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "smoke":
        smoke_test()

    elif mode == "sensitivity":
        results = run_sensitivity_analysis(
            save_dir="results/sensitivity",
            n_episodes=SENSITIVITY_EPISODES,
            rng_seed=42,
        )
        best = get_best_hyperparams(results)

        # Save best hyperparams
        os.makedirs("results", exist_ok=True)
        with open("results/best_hyperparams.json", "w") as f:
            # Convert noise keys to strings for JSON
            best_json = {
                agent: {str(noise): v for noise, v in noise_dict.items()}
                for agent, noise_dict in best.items()
            }
            json.dump(best_json, f, indent=2)
        print("Best hyperparams saved to results/best_hyperparams.json")

    elif mode == "train":
        # Load best hyperparams from sensitivity analysis
        with open("results/best_hyperparams.json") as f:
            best_json = json.load(f)
        # Convert string keys back to float
        best = {
            agent: {float(noise): v for noise, v in noise_dict.items()}
            for agent, noise_dict in best_json.items()
        }
        run_main_training(
            best_hyperparams=best,
            save_dir="results/main",
            n_episodes=MAIN_EPISODES,
            rng_seed=42,
        )

    elif mode == "all":
        # Run everything in sequence
        print("Running full pipeline: sensitivity → main training")
        results = run_sensitivity_analysis(
            save_dir="results/sensitivity",
            n_episodes=SENSITIVITY_EPISODES,
            rng_seed=42,
        )
        best = get_best_hyperparams(results)
        os.makedirs("results", exist_ok=True)
        with open("results/best_hyperparams.json", "w") as f:
            best_json = {
                agent: {str(noise): v for noise, v in noise_dict.items()}
                for agent, noise_dict in best.items()
            }
            json.dump(best_json, f, indent=2)
        run_main_training(
            best_hyperparams=best,
            save_dir="results/main",
            n_episodes=MAIN_EPISODES,
            rng_seed=42,
        )

    elif mode == "retrain":
        # Retrain only noise=0.05 and noise=0.08 with more episodes
        # Use best hyperparams from existing sensitivity analysis
        with open("results/best_hyperparams.json") as f:
            best_json = json.load(f)
        best = {
            agent: {float(noise): v for noise, v in noise_dict.items()}
            for agent, noise_dict in best_json.items()
        }
        os.makedirs("results/main", exist_ok=True)
        print("Retraining noise=0.05 and noise=0.08 with 20,000 episodes...")
        for agent_type in AGENT_TYPES:
            for noise in [0.05, 0.08]:
                hp = best[agent_type][noise]
                print(f"\n{agent_type} | noise={noise} | α={hp['alpha']} | γ={hp['gamma']}")
                result = run_training(
                    agent_type=agent_type,
                    noise_level=noise,
                    alpha=hp["alpha"],
                    gamma=hp["gamma"],
                    n_episodes=20000,
                    rng_seed=42,
                    verbose=True,
                    log_interval=2000,
                )
                filename = f"{agent_type}_noise{int(noise*100):02d}.json"
                filepath = os.path.join("results/main", filename)
                _save_result(result, filepath)
                print(f"  Saved → {filepath}")

    else:
        print(f"Unknown mode '{mode}'.")
        print("Usage: python training.py [smoke|sensitivity|train|all|retrain]")