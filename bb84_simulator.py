"""
BB84 Quantum Key Distribution Simulator
========================================
Simulates the quantum physics of BB84 from Alice → Eve (optional) → Channel → Bob.

Physics conventions:
- Bases:   0 = rectilinear (+),  1 = diagonal (×)
- Bits:    0 or 1
- When Eve measures in wrong basis, output bit is random (50/50) — this is
  the source of the 25% QBER under full intercept-and-resend attack.
- Channel noise flips the bit with probability `channel_error_rate`.

v2 changes (extension):
- Added `min_blocks` / `max_blocks` parameters to BB84Simulator.
  When min_blocks == max_blocks == n_blocks (default), behavior is
  identical to v1. When they differ, n_blocks is drawn uniformly from
  [min_blocks, max_blocks] at the start of each run_episode() call.
- n_qubits is now derived as n_blocks * checkpoint_interval rather
  than being a fixed constructor argument, so attack_decisions no
  longer needs a pre-declared length.
- run_episode() accepts attack_decisions as a callable (the environment
  step function) OR as a list, preserving backward compatibility.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Callable


# ---------------------------------------------------------------------------
# Data containers  (unchanged from v1)
# ---------------------------------------------------------------------------

@dataclass
class QubitRecord:
    """Stores everything that happened to a single qubit."""
    index: int

    # Alice
    alice_bit: int
    alice_basis: int

    # Eve (None if she didn't attack)
    eve_attacked: bool = False
    eve_basis: Optional[int] = None
    eve_measured_bit: Optional[int] = None
    eve_correct_basis: bool = False

    # Channel
    channel_flipped: bool = False

    # Bob
    bob_basis: int = 0
    bob_measured_bit: int = 0

    # Sifting
    bases_match: bool = False
    used_for_qber: bool = False


@dataclass
class CheckpointResult:
    """Result of a QBER check after every 100 qubits."""
    checkpoint_index: int
    qber: float
    n_sifted: int
    n_sampled: int
    detected: bool
    eve_correct_bits: int
    eve_attacks: int


@dataclass
class EpisodeResult:
    """Full result of one BB84 episode."""
    detected: bool
    detection_checkpoint: Optional[int]
    total_qubits_sent: int
    n_blocks: int                              # NEW: actual blocks this episode
    checkpoints: list = field(default_factory=list)
    qubit_records: list = field(default_factory=list)

    @property
    def total_eve_correct_bits(self) -> int:
        return sum(c.eve_correct_bits for c in self.checkpoints)

    @property
    def total_eve_attacks(self) -> int:
        return sum(c.eve_attacks for c in self.checkpoints)


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

class BB84Simulator:
    """
    Simulates one full BB84 episode.

    Parameters
    ----------
    channel_error_rate : float
        Probability that the channel flips a bit. Default 0.03.
    qber_threshold : float
        Abort threshold. Default 0.11.
    checkpoint_interval : int
        Qubits per block. Default 100.
    min_blocks : int
        Minimum number of blocks per episode. Default 10.
    max_blocks : int
        Maximum number of blocks per episode. Default 10.
        When min_blocks == max_blocks, episode length is fixed (v1 behavior).
        When they differ, n_blocks ~ Uniform[min_blocks, max_blocks] each
        episode — this is the random-checkpoint extension.
    qber_sample_size : optional int
        Sifted bits used for QBER estimation. None = use all.
    rng_seed : optional int
        For reproducibility.
    """

    def __init__(
        self,
        channel_error_rate: float = 0.03,
        qber_threshold: float = 0.11,
        checkpoint_interval: int = 100,
        min_blocks: int = 10,
        max_blocks: int = 10,
        qber_sample_size: Optional[int] = None,
        rng_seed: Optional[int] = None,
    ):
        assert min_blocks >= 1, "min_blocks must be >= 1"
        assert max_blocks >= min_blocks, "max_blocks must be >= min_blocks"

        self.channel_error_rate  = channel_error_rate
        self.qber_threshold      = qber_threshold
        self.checkpoint_interval = checkpoint_interval
        self.min_blocks          = min_blocks
        self.max_blocks          = max_blocks
        self.qber_sample_size    = qber_sample_size
        self.rng                 = np.random.default_rng(rng_seed)

        # For backward compatibility: expose n_qubits as max possible
        self.n_qubits = max_blocks * checkpoint_interval

    def sample_n_blocks(self) -> int:
        """
        Draw the number of blocks for one episode.
        Returns a fixed value when min_blocks == max_blocks.
        """
        if self.min_blocks == self.max_blocks:
            return self.min_blocks
        return int(self.rng.integers(self.min_blocks, self.max_blocks + 1))

    # ------------------------------------------------------------------
    # Single-qubit physics  (unchanged from v1)
    # ------------------------------------------------------------------

    def _alice_prepare(self, index: int) -> QubitRecord:
        return QubitRecord(
            index=index,
            alice_bit=int(self.rng.integers(0, 2)),
            alice_basis=int(self.rng.integers(0, 2)),
        )

    def _eve_intercept(self, record: QubitRecord) -> QubitRecord:
        eve_basis = int(self.rng.integers(0, 2))
        record.eve_attacked      = True
        record.eve_basis         = eve_basis
        record.eve_correct_basis = (eve_basis == record.alice_basis)

        if record.eve_correct_basis:
            record.eve_measured_bit = record.alice_bit
        else:
            record.eve_measured_bit = int(self.rng.integers(0, 2))

        return record

    def _channel_noise(self, record: QubitRecord) -> QubitRecord:
        if self.rng.random() < self.channel_error_rate:
            record.channel_flipped = True
        return record

    def _bob_measure(self, record: QubitRecord) -> QubitRecord:
        if record.eve_attacked:
            arriving_bit = record.eve_measured_bit
        else:
            arriving_bit = record.alice_bit

        if record.channel_flipped:
            arriving_bit = 1 - arriving_bit

        bob_basis = int(self.rng.integers(0, 2))
        record.bob_basis = bob_basis

        if bob_basis == record.alice_basis:
            record.bob_measured_bit = arriving_bit
            record.bases_match      = True
        else:
            record.bob_measured_bit = int(self.rng.integers(0, 2))
            record.bases_match      = False

        return record

    # ------------------------------------------------------------------
    # Block-level processing  (unchanged from v1)
    # ------------------------------------------------------------------

    def _process_block(
        self,
        block_records: list,
        checkpoint_index: int,
        attack_decisions: list,
    ) -> CheckpointResult:
        for i, record in enumerate(block_records):
            if attack_decisions[i] == 1:
                record = self._eve_intercept(record)
            record = self._channel_noise(record)
            record = self._bob_measure(record)
            block_records[i] = record

        sifted = [r for r in block_records if r.bases_match]

        if len(sifted) == 0:
            qber = 0.0
            n_sampled = 0
        else:
            if self.qber_sample_size is not None:
                sample = (sifted[:self.qber_sample_size]
                          if len(sifted) >= self.qber_sample_size
                          else sifted)
            else:
                sample = sifted

            n_sampled = len(sample)
            errors    = sum(1 for r in sample if r.alice_bit != r.bob_measured_bit)
            qber      = errors / n_sampled if n_sampled > 0 else 0.0

            for r in sample:
                r.used_for_qber = True

        detected = qber >= self.qber_threshold

        eve_correct = sum(
            1 for r in block_records
            if r.eve_attacked and r.eve_correct_basis
        )
        eve_attacks = sum(1 for r in block_records if r.eve_attacked)

        return CheckpointResult(
            checkpoint_index=checkpoint_index,
            qber=qber,
            n_sifted=len(sifted),
            n_sampled=n_sampled,
            detected=detected,
            eve_correct_bits=eve_correct,
            eve_attacks=eve_attacks,
        )

    # ------------------------------------------------------------------
    # Full episode
    # ------------------------------------------------------------------

    def run_episode(self, attack_decisions: list) -> EpisodeResult:
        """
        Run a complete BB84 episode.

        attack_decisions : list of int
            Length must equal n_blocks * checkpoint_interval for this episode.
            The environment feeds this correctly via its step() loop.

        For the random-checkpoint extension, the environment calls
        sample_n_blocks() first (via reset()), then feeds exactly
        n_blocks * checkpoint_interval decisions.
        """
        n_blocks  = len(attack_decisions) // self.checkpoint_interval
        n_qubits  = n_blocks * self.checkpoint_interval

        assert len(attack_decisions) == n_qubits, (
            f"attack_decisions length {len(attack_decisions)} must equal "
            f"n_blocks*checkpoint_interval = {n_qubits}"
        )

        all_records = []
        checkpoints = []

        for block_idx in range(n_blocks):
            start = block_idx * self.checkpoint_interval
            end   = start + self.checkpoint_interval

            block_records = [self._alice_prepare(i) for i in range(start, end)]

            result = self._process_block(
                block_records,
                checkpoint_index=block_idx,
                attack_decisions=attack_decisions[start:end],
            )

            all_records.extend(block_records)
            checkpoints.append(result)

            if result.detected:
                return EpisodeResult(
                    detected=True,
                    detection_checkpoint=block_idx,
                    total_qubits_sent=end,
                    n_blocks=n_blocks,
                    checkpoints=checkpoints,
                    qubit_records=all_records,
                )

        return EpisodeResult(
            detected=False,
            detection_checkpoint=None,
            total_qubits_sent=n_qubits,
            n_blocks=n_blocks,
            checkpoints=checkpoints,
            qubit_records=all_records,
        )


# ---------------------------------------------------------------------------
# Sanity checks  (unchanged from v1, plus one new check for variable blocks)
# ---------------------------------------------------------------------------

def sanity_check():
    print("=" * 60)
    print("BB84 Simulator Sanity Checks")
    print("=" * 60)

    N_EPISODES = 5000

    # Check 1: No Eve, no noise → QBER ~0%
    sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=1.0)
    qbers = []
    for _ in range(N_EPISODES):
        result = sim.run_episode([0] * 1000)
        for cp in result.checkpoints:
            qbers.append(cp.qber)
    print(f"\nCheck 1 — No Eve, no noise")
    print(f"  Expected QBER: ~0.00%")
    print(f"  Observed QBER: {np.mean(qbers)*100:.3f}%")

    # Check 2: Full attack, no noise → QBER ~25%
    sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=1.0)
    qbers = []
    for _ in range(N_EPISODES):
        result = sim.run_episode([1] * 1000)
        for cp in result.checkpoints:
            qbers.append(cp.qber)
    print(f"\nCheck 2 — Full attack, no noise")
    print(f"  Expected QBER: ~25.00%")
    print(f"  Observed QBER: {np.mean(qbers)*100:.2f}%")

    # Check 3: No Eve, 5% noise → QBER ~5%
    sim = BB84Simulator(channel_error_rate=0.05, qber_threshold=1.0)
    qbers = []
    for _ in range(N_EPISODES):
        result = sim.run_episode([0] * 1000)
        for cp in result.checkpoints:
            qbers.append(cp.qber)
    print(f"\nCheck 3 — No Eve, 5% noise")
    print(f"  Expected QBER: ~5.00%")
    print(f"  Observed QBER: {np.mean(qbers)*100:.2f}%")

    # Check 4: Full attack, 3% noise → matches Lee et al. eq. 7
    mu_ch    = 0.03
    expected = 0.25 + mu_ch - mu_ch**2
    sim = BB84Simulator(channel_error_rate=mu_ch, qber_threshold=1.0)
    qbers = []
    for _ in range(N_EPISODES):
        result = sim.run_episode([1] * 1000)
        for cp in result.checkpoints:
            qbers.append(cp.qber)
    print(f"\nCheck 4 — Full attack, 3% noise (Lee et al. eq. 7)")
    print(f"  Expected QBER: ~{expected*100:.2f}%")
    print(f"  Observed QBER: {np.mean(qbers)*100:.2f}%")

    # Check 5 (NEW): Variable blocks — episode length distribution is correct
    print(f"\nCheck 5 — Variable blocks (min=5, max=15)")
    sim = BB84Simulator(
        channel_error_rate=0.0,
        qber_threshold=1.0,
        min_blocks=5,
        max_blocks=15,
    )
    n_blocks_seen = []
    for _ in range(N_EPISODES):
        n_b = sim.sample_n_blocks()
        result = sim.run_episode([0] * (n_b * 100))
        n_blocks_seen.append(result.n_blocks)
    print(f"  Expected mean blocks: ~10.0")
    print(f"  Observed mean blocks: {np.mean(n_blocks_seen):.2f}")
    print(f"  Min observed: {min(n_blocks_seen)}, Max observed: {max(n_blocks_seen)}")
    assert min(n_blocks_seen) == 5 and max(n_blocks_seen) == 15, \
        "Block range outside [5, 15]"
    print(f"  Range check: PASS")

    print("\n" + "=" * 60)
    print("Checks complete.")
    print("=" * 60)


if __name__ == "__main__":
    sanity_check()