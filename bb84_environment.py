"""
BB84 Environment — Gym-style Interface for Eve RL Agent
========================================================
Wraps the BB84Simulator into a step-by-step RL environment.

Eve interacts one qubit at a time:
    state  = (qber_bin, attacks_bin, blocks_survived)
    action = 0 (do not attack) or 1 (attack)
    reward = per-qubit signal based on information gain and detection

State is frozen within a block and only updates at each checkpoint.

Discretization:
    QBER        : 0.00–0.15 in steps of 0.01  → 16 bins (0–15)
    Attacks     : 0–100 in steps of 5          → 21 bins (0–20)
    Blocks      : 0–14                         → 15 bins (0–14)
                  Covers up to max_blocks=15.
                  In fixed mode (original), only bins 0–9 are used.
    Total states: 16 × 21 × 15 = 5,040

v2 changes (extension):
    - Added min_blocks / max_blocks constructor parameters.
      Default min_blocks=max_blocks=10 reproduces original behavior exactly,
      except N_STATES is now 5,040 instead of 3,360 to accommodate up to
      15 blocks. Agents trained on the original should be retrained.
    - reset() now draws a new n_blocks for each episode from the simulator.
    - The third state dimension is renamed "blocks_survived" (was block_idx).
      Semantically: "I have survived this many checkpoints so far."
      In fixed mode this is identical to the original block_idx.
      In random mode, Eve no longer knows which block is last — she only
      knows how many she has survived. This is the key change that removes
      the terminal-block signal and should eliminate the end-game burst.
    - _process_checkpoint() no longer caps checkpoint_idx at n_checkpoints-1
      because n_checkpoints varies per episode.
    - N_STATES and CHECKPOINT_BINS are updated to reflect the new range.
      The original constants are preserved as _ORIGINAL_N_STATES etc.
      for reference.
"""

import numpy as np
from bb84_simulator import BB84Simulator, QubitRecord


# ---------------------------------------------------------------------------
# Discretization constants
# ---------------------------------------------------------------------------

QBER_BINS        = 16    # 0.00 to 0.15, step 0.01
ATTACK_BINS      = 21    # 0 to 100, step 5
CHECKPOINT_BINS  = 15    # 0 to 14 — covers max_blocks=15
                         # In fixed-10-block mode, bins 10–14 are unused

N_STATES  = QBER_BINS * ATTACK_BINS * CHECKPOINT_BINS   # 5,040
N_ACTIONS = 2                                            # 0=pass, 1=attack

# Original constants preserved for reference
_ORIGINAL_CHECKPOINT_BINS = 10
_ORIGINAL_N_STATES        = 3360


def discretize_qber(qber: float) -> int:
    bin_idx = int(qber / 0.01)
    return min(bin_idx, QBER_BINS - 1)


def discretize_attacks(attacks: int) -> int:
    return min(attacks // 5, ATTACK_BINS - 1)


def discretize_blocks(blocks_survived: int) -> int:
    """
    Map blocks survived (0-based) to bin index.
    Capped at CHECKPOINT_BINS - 1 = 14.
    """
    return min(blocks_survived, CHECKPOINT_BINS - 1)


def state_to_index(qber_bin: int, attack_bin: int, blocks_bin: int) -> int:
    return (qber_bin * ATTACK_BINS * CHECKPOINT_BINS
            + attack_bin * CHECKPOINT_BINS
            + blocks_bin)


def index_to_state(index: int) -> tuple:
    blocks_bin = index % CHECKPOINT_BINS
    index     //= CHECKPOINT_BINS
    attack_bin = index % ATTACK_BINS
    qber_bin   = index // ATTACK_BINS
    return (qber_bin, attack_bin, blocks_bin)


# ---------------------------------------------------------------------------
# Reward configuration  (unchanged)
# ---------------------------------------------------------------------------

class RewardConfig:
    def __init__(
        self,
        correct_basis: float       = +1.0,
        wrong_basis: float         = -1.0,
        checkpoint_survived: float = +2.0,
        detected: float            = -50.0,
        no_attack: float           =  0.0,
    ):
        self.correct_basis        = correct_basis
        self.wrong_basis          = wrong_basis
        self.checkpoint_survived  = checkpoint_survived
        self.detected             = detected
        self.no_attack            = no_attack


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class BB84Environment:
    """
    Gym-style environment for the Eve RL agent.

    Usage (unchanged from v1):
        env = BB84Environment(channel_error_rate=0.03)
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state

    Parameters
    ----------
    channel_error_rate : float
        Background channel noise.
    reward_config : RewardConfig
        Reward values.
    qber_threshold : float
        Detection threshold. Default 0.11.
    checkpoint_interval : int
        Qubits per block. Default 100.
    min_blocks : int
        Minimum blocks per episode. Default 10.
    max_blocks : int
        Maximum blocks per episode. Default 10.
        Set min_blocks=5, max_blocks=15 for the random-checkpoint extension.
    rng_seed : optional int
        For reproducibility.
    """

    def __init__(
        self,
        channel_error_rate: float = 0.03,
        reward_config: RewardConfig = None,
        qber_threshold: float = 0.11,
        checkpoint_interval: int = 100,
        min_blocks: int = 10,
        max_blocks: int = 10,
        rng_seed=None,
    ):
        self.channel_error_rate  = channel_error_rate
        self.reward_config       = reward_config or RewardConfig()
        self.qber_threshold      = qber_threshold
        self.checkpoint_interval = checkpoint_interval
        self.min_blocks          = min_blocks
        self.max_blocks          = max_blocks

        self.simulator = BB84Simulator(
            channel_error_rate=channel_error_rate,
            qber_threshold=qber_threshold,
            checkpoint_interval=checkpoint_interval,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            rng_seed=rng_seed,
        )
        self.rng = np.random.default_rng(rng_seed)

        # Episode state — set properly on reset()
        self._n_blocks_this_episode = max_blocks   # will be redrawn on reset
        self._reset_episode_state()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> int:
        """
        Start a new episode.

        Key change from v1: draws a new n_blocks from the simulator
        before resetting episode state. In fixed mode (min==max==10)
        this always returns 10, identical to v1.

        Returns the initial state index (integer).
        """
        self._n_blocks_this_episode = self.simulator.sample_n_blocks()
        self._reset_episode_state()
        return self._get_state()

    def step(self, action: int):
        """
        Eve takes one action for the current qubit.

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool
        info       : dict
        """
        assert not self._done, "Episode is done. Call reset() first."
        assert action in (0, 1), f"Invalid action {action}. Must be 0 or 1."

        # --- Simulate this single qubit ---
        record = self.simulator._alice_prepare(self._qubit_idx)

        if action == 1:
            record = self.simulator._eve_intercept(record)
            self._attacks_this_block += 1
            self._total_attacks      += 1

        record = self.simulator._channel_noise(record)
        record = self.simulator._bob_measure(record)

        self._current_block_records.append(record)
        self._all_records.append(record)
        self._qubit_idx += 1

        # --- Per-qubit reward ---
        reward = self._compute_qubit_reward(action, record)

        # --- Check if block is complete ---
        block_done = (len(self._current_block_records) == self.checkpoint_interval)
        detected   = False
        completed_block_attacks = self._attacks_this_block if block_done else None

        if block_done:
            reward  += self._process_checkpoint()
            detected = self._detected_this_checkpoint

        # --- Build info dict ---
        info = {
            "qubit_idx"               : self._qubit_idx - 1,
            "checkpoint_idx"          : self._blocks_survived,
            "attacks_this_block"      : self._attacks_this_block,
            "completed_block_attacks" : completed_block_attacks,
            "total_attacks"           : self._total_attacks,
            "last_qber"               : self._last_qber,
            "detected"                : detected,
            "total_correct_bits"      : self._total_correct_bits,
            "episode_done"            : self._done,
            "n_blocks_this_episode"   : self._n_blocks_this_episode,  # NEW
        }

        return self._get_state(), reward, self._done, info

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_states(self) -> int:
        return N_STATES

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def is_random_checkpoints(self) -> bool:
        return self.min_blocks != self.max_blocks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_episode_state(self):
        self._qubit_idx             = 0
        self._blocks_survived       = 0        # renamed from _checkpoint_idx
        self._attacks_this_block    = 0
        self._total_attacks         = 0
        self._total_correct_bits    = 0
        self._last_qber             = 0.0
        self._done                  = False
        self._detected_this_checkpoint = False
        self._current_block_records = []
        self._all_records           = []

    def _get_state(self) -> int:
        qber_bin   = discretize_qber(self._last_qber)
        attack_bin = discretize_attacks(self._attacks_this_block)
        blocks_bin = discretize_blocks(self._blocks_survived)
        return state_to_index(qber_bin, attack_bin, blocks_bin)

    def _compute_qubit_reward(self, action: int, record: QubitRecord) -> float:
        cfg = self.reward_config
        if action == 0:
            return cfg.no_attack
        if record.eve_correct_basis:
            self._total_correct_bits += 1
            return cfg.correct_basis
        else:
            return cfg.wrong_basis

    def _process_checkpoint(self) -> float:
        """
        Called when a block is complete.

        Key change from v1: no longer caps _blocks_survived at
        n_checkpoints - 1, because n_checkpoints varies per episode.
        The blocks_survived bin is capped by discretize_blocks() at
        CHECKPOINT_BINS - 1 = 14 instead.
        """
        records = self._current_block_records

        # QBER
        sifted = [r for r in records if r.bases_match]
        if len(sifted) == 0:
            qber = 0.0
        else:
            errors = sum(1 for r in sifted if r.alice_bit != r.bob_measured_bit)
            qber   = errors / len(sifted)

        self._last_qber = qber
        detected = qber >= self.qber_threshold
        self._detected_this_checkpoint = detected

        if detected:
            self._done       = True
            ckpt_reward      = self.reward_config.detected
        else:
            ckpt_reward      = self.reward_config.checkpoint_survived

        # Increment blocks survived (no cap — discretize_blocks handles it)
        self._blocks_survived       += 1
        self._attacks_this_block     = 0
        self._current_block_records  = []

        # Natural episode end: all blocks completed
        n_qubits_this_episode = (self._n_blocks_this_episode
                                 * self.checkpoint_interval)
        if self._qubit_idx >= n_qubits_this_episode and not self._done:
            self._done = True

        return ckpt_reward


# ---------------------------------------------------------------------------
# Quick verification
# ---------------------------------------------------------------------------

def verify_environment():
    print("=" * 65)
    print("BB84 Environment Verification (v2)")
    print("=" * 65)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"

    def check(condition, label, observed=None, expected=None):
        status = PASS if condition else FAIL
        msg = f"  {status} | {label}"
        if observed is not None:
            msg += f"  (observed={observed}, expected={expected})"
        print(msg)

    # V1: Fixed mode is backward compatible (10 blocks, 1000 qubits)
    print("\n[V1] Fixed mode — backward compatibility")
    env = BB84Environment(channel_error_rate=0.0, qber_threshold=1.0)
    env.reset()
    steps = 0
    done  = False
    while not done:
        _, _, done, _ = env.step(0)
        steps += 1
    check(steps == 1000, "Fixed mode: exactly 1000 steps", steps, 1000)

    # V2: Random mode — episode length varies across resets
    print("\n[V2] Random mode — variable episode length")
    env = BB84Environment(
        channel_error_rate=0.0, qber_threshold=1.0,
        min_blocks=5, max_blocks=15,
    )
    lengths = []
    for _ in range(500):
        env.reset()
        steps = 0
        done  = False
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
        lengths.append(steps)
    import numpy as np
    check(min(lengths) == 500,  "Min episode length = 500  (5 blocks)",
          min(lengths), 500)
    check(max(lengths) == 1500, "Max episode length = 1500 (15 blocks)",
          max(lengths), 1500)
    check(abs(np.mean(lengths) - 1000) < 30,
          "Mean episode length ≈ 1000", round(np.mean(lengths), 1), 1000)

    # V3: State space is 5040
    print("\n[V3] State space size")
    check(N_STATES == 5040, "N_STATES = 5040", N_STATES, 5040)

    # V4: blocks_survived increments correctly
    print("\n[V4] blocks_survived increments each block")
    env = BB84Environment(channel_error_rate=0.0, qber_threshold=1.0,
                          min_blocks=10, max_blocks=10)
    env.reset()
    for _ in range(100):   # complete block 0
        env.step(0)
    _, _, _, info = env.step(0)
    check(info["checkpoint_idx"] == 1,
          "blocks_survived = 1 after first block", info["checkpoint_idx"], 1)

    # V5: Detection still works in random mode
    print("\n[V5] Detection works in random mode")
    env = BB84Environment(channel_error_rate=0.0, qber_threshold=0.11,
                          min_blocks=5, max_blocks=15)
    detected_count = 0
    for _ in range(200):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(1)
        if info["detected"]:
            detected_count += 1
    check(detected_count == 200, "Full attack always detected in random mode",
          detected_count, 200)

    # V6: n_blocks_this_episode in info dict
    print("\n[V6] n_blocks_this_episode reported in info")
    env = BB84Environment(channel_error_rate=0.0, qber_threshold=1.0,
                          min_blocks=5, max_blocks=15)
    seen_lengths = set()
    for _ in range(1000):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(0)
        seen_lengths.add(info["n_blocks_this_episode"])
    check(min(seen_lengths) == 5 and max(seen_lengths) == 15,
          "n_blocks_this_episode spans [5, 15]",
          (min(seen_lengths), max(seen_lengths)), (5, 15))

    print("\n" + "=" * 65)
    print("Environment v2 verification complete.")
    print("=" * 65)


if __name__ == "__main__":
    verify_environment()