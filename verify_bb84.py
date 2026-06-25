"""
Extended verification suite for bb84_simulator.py
Run this to confirm all physics are correct before moving to Phase 2.
"""

import numpy as np
from bb84_simulator import BB84Simulator

N_EPISODES = 3000
N_QUBITS = 1000
PASS = "✓ PASS"
FAIL = "✗ FAIL"

def check(condition, label, observed, expected, tolerance):
    status = PASS if condition else FAIL
    print(f"  {status} | {label}")
    print(f"         Observed: {observed:.4f}  |  Expected: {expected:.4f}  |  Tolerance: ±{tolerance:.4f}")

print("=" * 65)
print("Extended BB84 Verification Suite")
print("=" * 65)


# ------------------------------------------------------------------
# Verification 1: Sifting rate ~50%
# ------------------------------------------------------------------
print("\n[V1] Sifting Rate")
print("     Alice and Bob each pick a random basis independently.")
print("     They should agree ~50% of the time.")

sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=1.0)
sift_rates = []
for _ in range(N_EPISODES):
    result = sim.run_episode([0] * N_QUBITS)
    for cp in result.checkpoints:
        sift_rates.append(cp.n_sifted / 100)

mean_sift = np.mean(sift_rates)
check(abs(mean_sift - 0.50) < 0.02, "Sift rate", mean_sift, 0.50, 0.02)


# ------------------------------------------------------------------
# Verification 2: Eve's correct basis rate ~50%
# ------------------------------------------------------------------
print("\n[V2] Eve's Information Gain Rate")
print("     Eve randomly guesses basis. She should be correct ~50% of attacks.")

sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=1.0)
correct_rates = []
for _ in range(N_EPISODES):
    result = sim.run_episode([1] * N_QUBITS)
    for cp in result.checkpoints:
        if cp.eve_attacks > 0:
            correct_rates.append(cp.eve_correct_bits / cp.eve_attacks)

mean_correct = np.mean(correct_rates)
check(abs(mean_correct - 0.50) < 0.02, "Correct basis rate", mean_correct, 0.50, 0.02)


# ------------------------------------------------------------------
# Verification 3: Partial attack QBER scales linearly
# ------------------------------------------------------------------
print("\n[V3] Partial Attack QBER Scaling")
print("     At attack rate r, QBER ≈ r × 0.25 (no channel noise).")
print("     Testing r = 0.25, 0.50, 0.75, 1.00")

sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=1.0)
all_pass = True
for r in [0.25, 0.50, 0.75, 1.00]:
    qbers = []
    for _ in range(N_EPISODES):
        decisions = [1 if np.random.random() < r else 0 for _ in range(N_QUBITS)]
        result = sim.run_episode(decisions)
        for cp in result.checkpoints:
            qbers.append(cp.qber)
    observed = np.mean(qbers)
    expected = r * 0.25
    ok = abs(observed - expected) < 0.02
    if not ok:
        all_pass = False
    status = PASS if ok else FAIL
    print(f"  {status} | r={r:.2f} → observed={observed:.4f}, expected≈{expected:.4f}")


# ------------------------------------------------------------------
# Verification 4: Detection triggers correctly (revised)
# ------------------------------------------------------------------
print("\n[V4] Detection Checkpoint Logic (Revised)")
print("     Full attack (no noise) → QBER ~25% → should always detect.")
print("     Detection can happen at any checkpoint due to statistical variance.")

sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=0.11)
detected_count = 0
for _ in range(N_EPISODES):
    result = sim.run_episode([1] * N_QUBITS)
    if result.detected:
        detected_count += 1

detection_rate = detected_count / N_EPISODES
check(detection_rate > 0.99, "Always detected under full attack", detection_rate, 1.0, 0.01)


# ------------------------------------------------------------------
# Verification 5: False detection rate is stable and consistent
# ------------------------------------------------------------------
print("\n[V5] False Detection Rate — Stability Check")
print("     No attack, 3% noise, threshold 11%.")
print("     FPR should be stable across runs (~4-6% due to small block size).")
print("     Note: theoretical binomial prediction assumes fixed n=50,")
print("     but actual sifted count varies (29-68), reducing true FPR.")

sim = BB84Simulator(channel_error_rate=0.03, qber_threshold=0.11)
detections = 0
for _ in range(N_EPISODES):
    result = sim.run_episode([0] * N_QUBITS)
    if result.detected:
        detections += 1

fpr = detections / N_EPISODES
check(0.03 <= fpr <= 0.07,
      "Episode FPR in expected range (3-7%)", fpr, 0.05, 0.02)
print(f"         Interpretation: ~{fpr*100:.1f}% of episodes see a noise spike")
print(f"         above threshold by chance. RL agent must learn to account for this.")


# ------------------------------------------------------------------
# Verification 6: Episode stops after detection (no extra qubits)
# ------------------------------------------------------------------
print("\n[V6] Early Termination — No Qubits Sent After Detection")
print("     If detected at checkpoint k, total qubits sent = (k+1) × 100.")

sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=0.11)
termination_correct = True
for _ in range(200):
    # Random attack pattern that might trigger detection mid-episode
    decisions = [1 if np.random.random() < 0.5 else 0 for _ in range(N_QUBITS)]
    result = sim.run_episode(decisions)
    if result.detected:
        expected_qubits = (result.detection_checkpoint + 1) * 100
        if result.total_qubits_sent != expected_qubits:
            termination_correct = False
            break

status = PASS if termination_correct else FAIL
print(f"  {status} | Early termination count is always (checkpoint+1) × 100")


# ------------------------------------------------------------------
# Verification 7: QBER distribution is approximately normal
# ------------------------------------------------------------------
print("\n[V7] QBER Distribution Normality Check")
print("     By CLT, QBER over 50 sifted bits should be approximately normal.")
print("     Checking skewness is close to 0.")

sim = BB84Simulator(channel_error_rate=0.0, qber_threshold=1.0)
qbers = []
for _ in range(N_EPISODES):
    result = sim.run_episode([1] * N_QUBITS)
    for cp in result.checkpoints:
        qbers.append(cp.qber)

qbers = np.array(qbers)
mean = np.mean(qbers)
std = np.std(qbers)
skewness = np.mean(((qbers - mean) / std) ** 3)
check(abs(skewness) < 0.3, "Skewness near 0", skewness, 0.0, 0.3)
print(f"         Mean={mean:.4f}  Std={std:.4f}")


# ------------------------------------------------------------------
# Verification 8: Episode independence
# ------------------------------------------------------------------
print("\n[V8] Episode Independence")
print("     QBER in episode N should be uncorrelated with episode N+1.")

sim = BB84Simulator(channel_error_rate=0.03, qber_threshold=1.0)
first_qbers = []
for _ in range(N_EPISODES):
    result = sim.run_episode([1] * N_QUBITS)
    first_qbers.append(result.checkpoints[0].qber)

correlation = np.corrcoef(first_qbers[:-1], first_qbers[1:])[0, 1]
check(abs(correlation) < 0.05, "Inter-episode correlation", correlation, 0.0, 0.05)


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 65)
print("Verification complete.")
print("All checks passing = simulator is ready for Phase 2.")
print("=" * 65)