from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .agents import QLearningAgent, heuristic_action, random_action
from .config import RANDOM_SEED
from .environment import NetworkSlicingEnv


def train_agent(episodes: int = 450, seed: int = RANDOM_SEED) -> QLearningAgent:
    env = NetworkSlicingEnv(seed=seed)
    agent = QLearningAgent(n_actions=env.n_actions, seed=seed)
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
    return agent


def evaluate_policy(
    policy_name: str,
    policy: Callable[[NetworkSlicingEnv, Tuple[int, int, int]], int],
    episodes: int = 60,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    env = NetworkSlicingEnv(seed=seed)
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        total_qoe = 0.0
        total_energy = 0.0
        total_overloaded = 0
        steps = 0
        while not done:
            action = policy(env, state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            total_qoe += float(info["qoe"])
            total_energy += float(info["energy"])
            total_overloaded += int(info["overloaded_slices"])
            steps += 1
        rows.append(
            {
                "policy": policy_name,
                "episode": episode,
                "mean_reward": total_reward / steps,
                "mean_qoe": total_qoe / steps,
                "mean_energy": total_energy / steps,
                "overload_events": total_overloaded,
            }
        )
    return pd.DataFrame(rows)


def run_experiment(output_dir: str | Path = "results", seed: int = RANDOM_SEED) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    agent = train_agent(seed=seed)

    def q_learning_policy(_: NetworkSlicingEnv, state: Tuple[int, int, int]) -> int:
        return agent.choose_action(state, training=False)

    policies = {
        "random": lambda env, state: random_action(env),
        "heuristic": lambda env, state: heuristic_action(env),
        "q_learning": q_learning_policy,
    }
    results = pd.concat(
        [evaluate_policy(name, policy, seed=seed) for name, policy in policies.items()],
        ignore_index=True,
    )
    summary = (
        results.groupby("policy")[["mean_reward", "mean_qoe", "mean_energy", "overload_events"]]
        .mean()
        .reset_index()
        .sort_values("mean_reward", ascending=False)
    )
    results.to_csv(output_path / "episode_results.csv", index=False)
    summary.to_csv(output_path / "policy_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["policy"], summary["mean_reward"])
    ax.set_title("Policy comparison for QoE-GreenSlicer")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Mean reward")
    for i, value in enumerate(summary["mean_reward"]):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path / "policy_comparison.png", dpi=300)
    plt.close(fig)
    return summary
