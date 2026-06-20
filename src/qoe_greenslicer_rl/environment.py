from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import ACTIONS, DEFAULT_INITIAL_SHARES, EnvConfig, RANDOM_SEED, SLICE_NAMES


@dataclass
class StepInfo:
    qoe: float
    energy: float
    reward: float
    overloaded_slices: int
    demand: Tuple[float, float, float]
    shares: Tuple[float, float, float]
    utilization: Tuple[float, float, float]


class NetworkSlicingEnv:
    """Small reinforcement-learning environment for 5G resource slicing.

    The environment is intentionally lightweight: it does not require Gymnasium,
    PyTorch or TensorFlow. It is a portfolio-friendly baseline that can later be
    upgraded to SAC/PPO with Stable-Baselines3.
    """

    def __init__(self, config: EnvConfig | None = None, seed: int = RANDOM_SEED) -> None:
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.t = 0
        self.shares = np.array(DEFAULT_INITIAL_SHARES, dtype=float)
        self.last_demand = np.zeros(3, dtype=float)
        self.last_utilization = np.zeros(3, dtype=float)

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def reset(self) -> Tuple[int, int, int]:
        self.t = 0
        self.shares = np.array(DEFAULT_INITIAL_SHARES, dtype=float)
        self.last_demand = self._generate_demand(self.t)
        self.last_utilization = self._utilization(self.last_demand)
        return self._state_from_utilization(self.last_utilization)

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, object]]:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action {action}. Expected one of {sorted(ACTIONS)}")

        self._apply_action(action)
        demand = self._generate_demand(self.t)
        utilization = self._utilization(demand)
        qoe = self._qoe(utilization)
        energy = self._energy(utilization)
        overloaded = int(np.sum(utilization > 1.0))
        reward = qoe - self.config.gamma_energy * energy - self.config.overload_penalty * overloaded

        self.last_demand = demand
        self.last_utilization = utilization
        self.t += 1
        done = self.t >= self.config.episode_length
        info = StepInfo(
            qoe=float(qoe),
            energy=float(energy),
            reward=float(reward),
            overloaded_slices=overloaded,
            demand=tuple(np.round(demand, 4)),
            shares=tuple(np.round(self.shares, 4)),
            utilization=tuple(np.round(utilization, 4)),
        ).__dict__
        return self._state_from_utilization(utilization), float(reward), done, info

    def _apply_action(self, action: int) -> None:
        source, target = ACTIONS[action]
        if source is None or target is None:
            return
        step = self.config.resource_step
        if self.shares[source] - step >= self.config.min_share:
            self.shares[source] -= step
            self.shares[target] += step
        self.shares = self.shares / self.shares.sum()

    def _generate_demand(self, t: int) -> np.ndarray:
        """Generate time-varying demand for URLLC, eMBB and mMTC."""
        day_phase = 2 * np.pi * (t % 96) / 96
        business_phase = 2 * np.pi * ((t + 24) % 96) / 96

        urllc = 22 + 6 * np.sin(business_phase) + self.rng.normal(0, 2.5)
        embb = 42 + 16 * np.sin(day_phase - 0.8) + self.rng.normal(0, 5.0)
        mmtc = 26 + 5 * np.sin(day_phase + 1.4) + self.rng.normal(0, 3.0)

        # Occasional bursts imitate events or sudden IoT activity.
        if self.rng.random() < 0.08:
            embb += self.rng.uniform(10, 24)
        if self.rng.random() < 0.06:
            mmtc += self.rng.uniform(8, 18)
        if self.rng.random() < 0.04:
            urllc += self.rng.uniform(6, 14)

        return np.maximum(np.array([urllc, embb, mmtc], dtype=float), 1.0)

    def _utilization(self, demand: np.ndarray) -> np.ndarray:
        allocated_capacity = self.shares * self.config.total_capacity
        return demand / allocated_capacity

    def _qoe(self, utilization: np.ndarray) -> float:
        # URLLC drops sharply after 70% utilization because latency is critical.
        urllc_qoe = np.exp(-2.4 * np.maximum(utilization[0] - 0.70, 0))
        # eMBB is close to served-throughput ratio.
        embb_qoe = min(1.0, 1.0 / max(utilization[1], 1e-9))
        # mMTC focuses on successful delivery; overload has a smoother penalty.
        mmtc_qoe = 1.0 / (1.0 + np.exp(5.0 * (utilization[2] - 1.05)))
        weights = np.array([
            self.config.qoe_weight_urllc,
            self.config.qoe_weight_embb,
            self.config.qoe_weight_mmtc,
        ])
        return float(np.dot(weights, np.array([urllc_qoe, embb_qoe, mmtc_qoe])))

    def _energy(self, utilization: np.ndarray) -> float:
        # Normalized energy: static component + dynamic component from active load.
        static_energy = 0.55
        dynamic_energy = 0.45 * float(np.dot(self.shares, np.minimum(utilization, 1.25)))
        return static_energy + dynamic_energy

    @staticmethod
    def _state_from_utilization(utilization: np.ndarray) -> Tuple[int, int, int]:
        # 0 = low load, 1 = normal, 2 = high, 3 = overloaded.
        bins = np.array([0.55, 0.85, 1.00])
        return tuple(int(np.digitize(value, bins)) for value in utilization)

    def current_load_summary(self) -> Dict[str, float]:
        return {
            f"{name}_utilization": float(value)
            for name, value in zip(SLICE_NAMES, self.last_utilization)
        }
