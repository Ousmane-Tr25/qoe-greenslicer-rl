from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

from .config import ACTIONS, RANDOM_SEED
from .environment import NetworkSlicingEnv

State = Tuple[int, int, int]


class QLearningAgent:
    """Tabular Q-learning agent for a small discrete slicing environment."""

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.12,
        discount: float = 0.95,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.03,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table: Dict[State, np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=float))

    def choose_action(self, state: State, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        values = self.q_table[state]
        return int(np.flatnonzero(values == values.max())[0])

    def update(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        current = self.q_table[state][action]
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.discount * next_best
        self.q_table[state][action] = current + self.learning_rate * (target - current)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def random_action(env: NetworkSlicingEnv) -> int:
    return int(env.rng.integers(0, env.n_actions))


def heuristic_action(env: NetworkSlicingEnv) -> int:
    """Move capacity from the least-loaded slice to the most-loaded slice."""
    utilization = env.last_utilization
    most_loaded = int(np.argmax(utilization))
    least_loaded = int(np.argmin(utilization))
    if utilization[most_loaded] < 0.85 or most_loaded == least_loaded:
        return 0
    for action_id, (source, target) in ACTIONS.items():
        if source == least_loaded and target == most_loaded:
            return action_id
    return 0
