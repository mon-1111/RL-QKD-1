"""
RL Agents for BB84 Eavesdropping
==================================
Three tabular agents sharing the same interface:
    - QLearningAgent      (off-policy, aggressive)
    - SARSAAgent          (on-policy, cautious)
    - DoubleQLearningAgent(reduced overestimation, balanced)

All agents share:
    - Biased epsilon-greedy exploration
    - Configurable alpha (learning rate) and gamma (discount)
    - Q-table shape: (n_states, n_actions)
    - select_action(state)  → action
    - update(...)           → updates Q-table
    - decay_epsilon()       → called once per episode

The ONLY difference between them is the update rule.

Key fix from v1: biased exploration
    Original: random exploration attacked 50% of qubits
              → QBER ~25% → detected every episode
              → agent never survived past checkpoint 0
              → never learned restraint

    Fixed:    random exploration attacks explore_attack_bias % of qubits
              Default 0.20 → QBER ~8% → agent sometimes survives
              → can learn the value of not attacking
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Shared infrastructure for all three tabular RL agents.

    Parameters
    ----------
    n_states : int
        Total number of discrete states. For BB84: 3360.
    n_actions : int
        Number of actions. For BB84: 2 (attack or not).
    alpha : float
        Learning rate. Range 0-1. Default 0.1.
    gamma : float
        Discount factor. Range 0-1. Default 0.95.
        Low  (0.5)  -> shortsighted Eve, maximizes immediate gain
        High (0.99) -> farsighted Eve, plans across whole episode
    epsilon_start : float
        Initial exploration rate. 1.0 = fully random at start.
    epsilon_end : float
        Minimum exploration rate. 0.01 = 1% random at end.
    epsilon_decay_episodes : int
        Number of episodes over which epsilon decays linearly.
    explore_attack_bias : float
        During random exploration, probability of choosing attack.
        Default 0.20 — biased toward no-attack so agent survives
        early checkpoints and can learn restraint.
        Set to 0.5 for unbiased exploration.
    rng_seed : optional int
        For reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_episodes: int = 10000,
        explore_attack_bias: float = 0.20,
        rng_seed=None,
    ):
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.alpha      = alpha
        self.gamma      = gamma
        self.epsilon    = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes
        self.explore_attack_bias = explore_attack_bias
        self.rng        = np.random.default_rng(rng_seed)

        # Q-table: rows = states, columns = actions
        self.Q = np.zeros((n_states, n_actions))

        # Tracking
        self.episode_count = 0

    def select_action(self, state: int) -> int:
        """
        Biased epsilon-greedy action selection.

        During exploration, bias toward no-attack (action 0) to allow
        the agent to survive early checkpoints and learn restraint.
        Without this bias, 50% random attack rate causes constant
        detection and the agent never learns to avoid it.
        """
        if self.rng.random() < self.epsilon:
            return 1 if self.rng.random() < self.explore_attack_bias else 0
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self):
        """Linear epsilon decay — call once per episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.episode_count += 1

    def greedy_action(self, state: int) -> int:
        """Pure greedy action — no exploration. Used during evaluation."""
        return int(np.argmax(self.Q[state]))

    def get_q_values(self, state: int) -> np.ndarray:
        """Return Q-values for both actions at a given state."""
        return self.Q[state].copy()

    def reset_qtable(self):
        """Reset Q-table to zeros."""
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.epsilon = self.epsilon_start
        self.episode_count = 0


# ---------------------------------------------------------------------------
# Agent 1: Q-Learning (off-policy)
# ---------------------------------------------------------------------------

class QLearningAgent(BaseAgent):
    """
    Q-Learning — off-policy temporal difference learning.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Characteristics:
        - Learns optimal policy regardless of exploration behavior
        - Tends to overestimate Q-values (optimism bias)
        - Expected: aggressive Eve, higher attack rate
    """

    def update(self, state, action, reward, next_state, done, **kwargs):
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error


# ---------------------------------------------------------------------------
# Agent 2: SARSA (on-policy)
# ---------------------------------------------------------------------------

class SARSAAgent(BaseAgent):
    """
    SARSA — on-policy temporal difference learning.

    Update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s', a') - Q(s,a)]
        where a' is the ACTUAL next action taken (not greedy)

    Characteristics:
        - Learns the policy it actually follows including exploration
        - More conservative than Q-Learning
        - Expected: cautious Eve, lower attack rate
    """

    def update(self, state, action, reward, next_state, done,
               next_action=None, **kwargs):
        assert next_action is not None or done, \
            "SARSA requires next_action when not done"
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.Q[next_state, next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error


# ---------------------------------------------------------------------------
# Agent 3: Double Q-Learning
# ---------------------------------------------------------------------------

class DoubleQLearningAgent(BaseAgent):
    """
    Double Q-Learning — reduces overestimation bias.

    Uses two Q-tables updated alternately:
        50%: a* = argmax Q1(s',a);  Q1(s,a) += alpha*[r + gamma*Q2(s',a*) - Q1(s,a)]
        50%: a* = argmax Q2(s',a);  Q2(s,a) += alpha*[r + gamma*Q1(s',a*) - Q2(s,a)]

    Action selection uses average of Q1 and Q2.

    Characteristics:
        - Most accurate Q-value estimates
        - Expected: balanced Eve, best trade-off frontier
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q2 = np.zeros((self.n_states, self.n_actions))

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return 1 if self.rng.random() < self.explore_attack_bias else 0
        return int(np.argmax(self.Q[state] + self.Q2[state]))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.Q[state] + self.Q2[state]))

    def get_q_values(self, state: int) -> np.ndarray:
        return ((self.Q[state] + self.Q2[state]) / 2).copy()

    def update(self, state, action, reward, next_state, done, **kwargs):
        if done:
            td_target = reward
            self.Q[state, action]  += self.alpha * (td_target - self.Q[state, action])
            self.Q2[state, action] += self.alpha * (td_target - self.Q2[state, action])
            return
        if self.rng.random() < 0.5:
            best_action = int(np.argmax(self.Q[next_state]))
            td_target   = reward + self.gamma * self.Q2[next_state, best_action]
            self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])
        else:
            best_action = int(np.argmax(self.Q2[next_state]))
            td_target   = reward + self.gamma * self.Q[next_state, best_action]
            self.Q2[state, action] += self.alpha * (td_target - self.Q2[state, action])

    def reset_qtable(self):
        super().reset_qtable()
        self.Q2 = np.zeros((self.n_states, self.n_actions))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_agent(
    agent_type: str,
    n_states: int,
    n_actions: int,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay_episodes: int = 10000,
    explore_attack_bias: float = 0.20,
    rng_seed=None,
) -> BaseAgent:
    """Create an agent by name: 'qlearning', 'sarsa', 'doubleq'"""
    kwargs = dict(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        explore_attack_bias=explore_attack_bias,
        rng_seed=rng_seed,
    )
    agents = {
        "qlearning" : QLearningAgent,
        "sarsa"     : SARSAAgent,
        "doubleq"   : DoubleQLearningAgent,
    }
    agent_type = agent_type.lower()
    if agent_type not in agents:
        raise ValueError(f"Unknown agent '{agent_type}'. "
                         f"Choose from: {list(agents.keys())}")
    return agents[agent_type](**kwargs)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_agents():
    from bb84_environment import BB84Environment, N_STATES, N_ACTIONS

    print("=" * 65)
    print("Agent Verification")
    print("=" * 65)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"

    def check(condition, label, observed=None, expected=None):
        status = PASS if condition else FAIL
        msg = f"  {status} | {label}"
        if observed is not None:
            msg += f"  (observed={observed}, expected={expected})"
        print(msg)

    env = BB84Environment(channel_error_rate=0.03)

    for agent_name in ["qlearning", "sarsa", "doubleq"]:
        print(f"\n[{agent_name.upper()}]")
        agent = make_agent(agent_name, N_STATES, N_ACTIONS,
                           alpha=0.1, gamma=0.95,
                           epsilon_decay_episodes=1000,
                           explore_attack_bias=0.20,
                           rng_seed=42)

        check(agent.epsilon == 1.0, "Initial epsilon = 1.0", agent.epsilon, 1.0)
        check(np.all(agent.Q == 0), "Q-table initialized to zeros")

        for _ in range(1000):
            agent.decay_epsilon()
        check(abs(agent.epsilon - 0.01) < 0.001,
              "Epsilon decays to 0.01", round(agent.epsilon, 4), 0.01)

        agent.reset_qtable()
        agent.Q[0, 1] = 1.0
        agent.epsilon = 0.0
        action = agent.select_action(0)
        check(action == 1, "Greedy selects highest Q-value", action, 1)

        agent.reset_qtable()
        if agent_name == "sarsa":
            agent.update(0, 1, 1.0, 5, done=False, next_action=0)
        else:
            agent.update(0, 1, 1.0, 5, done=False)
        q_val = agent.Q[0, 1] if agent_name != "doubleq" else \
                (agent.Q[0, 1] + agent.Q2[0, 1])
        check(q_val != 0.0, "Q-value updated after one step",
              round(q_val, 4), "!= 0.0")

        agent.reset_qtable()
        agent.Q[0, 1] = 99.0
        if agent_name == "sarsa":
            agent.update(0, 0, 5.0, 0, done=True, next_action=1)
        else:
            agent.update(0, 0, 5.0, 0, done=True)
        expected_q = 0.1 * 5.0
        check(abs(agent.Q[0, 0] - expected_q) < 0.01,
              "Terminal update ignores future Q-values",
              round(agent.Q[0, 0], 4), round(expected_q, 4))

    # Biased exploration check
    print("\n[BIASED EXPLORATION CHECK]")
    agent = make_agent("qlearning", N_STATES, N_ACTIONS,
                       explore_attack_bias=0.20, rng_seed=0)
    agent.epsilon = 1.0   # force full exploration
    actions = [agent.select_action(0) for _ in range(5000)]
    attack_rate = np.mean(actions)
    check(abs(attack_rate - 0.20) < 0.03,
          "Biased exploration attacks ~20% of the time",
          round(attack_rate, 3), 0.20)

    # Confirm biased exploration allows survival
    print("\n[SURVIVAL CHECK WITH BIASED EXPLORATION]")
    env2 = BB84Environment(channel_error_rate=0.03, qber_threshold=0.11)
    survival_count = 0
    N = 500
    agent2 = make_agent("qlearning", N_STATES, N_ACTIONS,
                        explore_attack_bias=0.20, rng_seed=1)
    agent2.epsilon = 1.0
    for _ in range(N):
        state = env2.reset()
        done = False
        while not done:
            action = agent2.select_action(state)
            state, _, done, info = env2.step(action)
        if not info["detected"]:
            survival_count += 1
    survival_rate = survival_count / N
    check(survival_rate > 0.10,
          "Biased exploration survives >10% of episodes (was 0% before fix)",
          round(survival_rate, 3), "> 0.10")

    # SARSA vs Q-Learning difference
    print("\n[SARSA vs Q-LEARNING UPDATE DIFFERENCE]")
    sarsa  = SARSAAgent(N_STATES, N_ACTIONS, alpha=1.0, gamma=1.0)
    qlearn = QLearningAgent(N_STATES, N_ACTIONS, alpha=1.0, gamma=1.0)
    sarsa.Q[5, 0]  = 10.0;  sarsa.Q[5, 1]  = 1.0
    qlearn.Q[5, 0] = 10.0;  qlearn.Q[5, 1] = 1.0
    qlearn.update(0, 0, 0.0, next_state=5, done=False)
    sarsa.update(0, 0, 0.0, next_state=5, next_action=1, done=False)
    check(qlearn.Q[0, 0] > sarsa.Q[0, 0],
          "Q-Learning > SARSA when exploratory action is suboptimal",
          round(qlearn.Q[0, 0], 2), f"> {round(sarsa.Q[0, 0], 2)}")

    print("\n" + "=" * 65)
    print("Agent verification complete.")
    print("=" * 65)


if __name__ == "__main__":
    verify_agents()