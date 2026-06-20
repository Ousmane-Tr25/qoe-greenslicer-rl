from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

RANDOM_SEED = 42
SLICE_NAMES: Tuple[str, str, str] = ("URLLC", "eMBB", "mMTC")


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for the simplified 5G slicing environment."""

    total_capacity: float = 120.0
    min_share: float = 0.15
    resource_step: float = 0.05
    episode_length: int = 96
    gamma_energy: float = 0.25
    overload_penalty: float = 1.5
    qoe_weight_urllc: float = 0.40
    qoe_weight_embb: float = 0.35
    qoe_weight_mmtc: float = 0.25


DEFAULT_INITIAL_SHARES = (0.34, 0.40, 0.26)

# Discrete actions: move a small part of total capacity between slices.
# Action 0 is no-op; other actions are (source_slice, target_slice).
ACTIONS: Dict[int, Tuple[int | None, int | None]] = {
    0: (None, None),
    1: (0, 1),
    2: (0, 2),
    3: (1, 0),
    4: (1, 2),
    5: (2, 0),
    6: (2, 1),
}
