"""Feature constants, reward shaping, and PPO sample packing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from PPO.conf.conf import Config


RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}

RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


def reward_process(end_dist: float, history_dist: float, reward_info: dict | None = None) -> list[float]:
    """Return the nine reward components consumed by the workflow.

    The shaping is intentionally conservative: the terminal workflow still adds
    a final outcome reward, while this function rewards progress, collection,
    and discourages idle moves.
    """

    info = reward_info or {}
    if info.get("is_exploit"):
        return [0.0] * 9

    step_reward = -0.001
    dist_reward = min(0.001, 0.005 * float(history_dist or 0.0))

    last_end_dist = info.get("last_end_pos_dist")
    if last_end_dist is None:
        end_reward = 0.0
    else:
        delta = float(last_end_dist) - float(end_dist)
        end_reward = 0.1 * delta if delta > 0 else 0.08 * delta

    treasure_view_reward = 0.0005 * float(info.get("visible_treasure_count", 0))
    treasure_get_reward = 5.0 if info.get("get_treasure") else 0.0

    talent_reward = -0.5 if info.get("use_talent") else 0.0
    if info.get("use_talent") and (info.get("get_treasure") or info.get("flash_over_wall")):
        talent_reward += 0.2

    stagnated_reward = -0.0055 if info.get("stagnated") else 0.0
    treasure_dist_reward = float(info.get("treasure_dist_delta", 0.0) or 0.0)

    buff_reward = 0.0
    if info.get("get_buff"):
        buff_reward += 0.5
    if info.get("speed_up"):
        buff_reward += 0.001

    return [
        step_reward,
        dist_reward,
        end_reward,
        treasure_view_reward,
        treasure_get_reward,
        talent_reward,
        stagnated_reward,
        treasure_dist_reward,
        buff_reward,
    ]


@dataclass
class SampleData:
    npdata: np.ndarray | None = None


class SampleManager:
    """Collects one episode and packs it into flat PPO samples."""

    def __init__(self, gamma: float | None = None, tdlambda: float | None = None) -> None:
        self.gamma = Config.GAMMA if gamma is None else gamma
        self.tdlambda = Config.TDLAMBDA if tdlambda is None else tdlambda
        self.feature: list[np.ndarray] = []
        self.probs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.reward: list[np.ndarray] = []
        self.value: list[np.ndarray] = []
        self.adv: list[np.ndarray] = []
        self.tdlamret: list[np.ndarray] = []
        self.legal_action: list[np.ndarray] = []
        self.count = 0
        self.samples: list[SampleData] = []

    def add(self, feature, legal_action, prob, action, value, reward) -> None:
        self.feature.append(np.asarray(feature, dtype=np.float32))
        self.legal_action.append(np.asarray(legal_action, dtype=np.float32))
        self.probs.append(np.asarray(prob, dtype=np.float32).reshape(Config.ACTION_LEN))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(Config.ACTION_LEN))
        self.value.append(np.asarray(value, dtype=np.float32).reshape(Config.VALUE_NUM))
        self.reward.append(np.asarray(reward, dtype=np.float32).reshape(Config.VALUE_NUM))
        self.adv.append(np.zeros(Config.VALUE_NUM, dtype=np.float32))
        self.tdlamret.append(np.zeros(Config.VALUE_NUM, dtype=np.float32))
        self.count += 1

    def add_last_reward(self, reward) -> None:
        self.reward.append(np.asarray(reward, dtype=np.float32).reshape(Config.VALUE_NUM))
        self.value.append(np.zeros(Config.VALUE_NUM, dtype=np.float32))

    def update_sample_info(self) -> None:
        last_gae = np.zeros(Config.VALUE_NUM, dtype=np.float32)
        for i in range(self.count - 1, -1, -1):
            reward = self.reward[i + 1]
            next_val = self.value[i + 1]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.tdlambda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val

    def sample_process(self, feature, legal_action, prob, action, value, reward) -> None:
        self.add(feature, legal_action, prob, action, value, reward)

    def process_last_frame(self, reward) -> None:
        self.add_last_reward(reward)
        self.update_sample_info()
        self.samples = self._get_game_data()

    def get_game_data(self) -> list[SampleData]:
        ret = self.samples
        self.samples = []
        return ret

    def _stack(self, values: Iterable[np.ndarray]) -> np.ndarray:
        return np.asarray(list(values), dtype=np.float32)

    def _get_game_data(self) -> list[SampleData]:
        if self.count == 0:
            return []

        data = np.concatenate(
            [
                self._stack(self.feature),
                self._stack(self.reward[:-1]),
                self._stack(self.value[:-1]),
                self._stack(self.tdlamret),
                self._stack(self.adv),
                self._stack(self.actions),
                self._stack(self.probs),
                self._stack(self.legal_action),
            ],
            axis=1,
        )

        if data.shape[1] != Config.SAMPLE_DIM:
            raise ValueError(f"Sample width {data.shape[1]} != expected {Config.SAMPLE_DIM}")

        return [SampleData(npdata=row.astype(np.float32, copy=False)) for row in data]
