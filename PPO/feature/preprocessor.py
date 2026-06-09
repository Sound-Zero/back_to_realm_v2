"""Observation preprocessing for the treasure-hunt PPO agent."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from PPO.conf.conf import Config
from PPO.feature.definition import DirectionAngles, RelativeDirection, RelativeDistance, reward_process


MAP_SIZE = 128
LOCAL_VIEW_SIZE = 11


def norm(v, max_v, min_v=0):
    """Clamp and normalize a scalar or numpy array into [0, 1]."""

    value = np.asarray(v, dtype=np.float32)
    value = np.maximum(np.minimum(max_v, value), min_v)
    return (value - min_v) / (max_v - min_v)


class Preprocessor:
    """Converts environment observations into the configured flat feature vector."""

    def __init__(self) -> None:
        self.move_action_num = Config.ACTION_SPACE_SIZE
        self.reset()

    def reset(self) -> None:
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.zeros(2, dtype=np.float32)
        self.last_pos_norm = np.zeros(2, dtype=np.float32)
        self.last_pos: tuple[int, int] | None = None
        self.history_pos: list[tuple[int, int]] = []
        self.bad_move_ids: set[int] = set()
        self.move_usable = True
        self.last_action = -1

        self.end_pos: tuple[int, int] | None = None
        self.is_end_pos_found = False
        self.last_end_pos_dist: float | None = None

        self.local_map = np.zeros((LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), dtype=np.int32)
        self.last_local_map = np.zeros_like(self.local_map)
        self.global_memory_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        self.local_memory_map = np.zeros((LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), dtype=np.float32)
        self.memory_flag = self.local_memory_map.flatten()

        self.treasures_info: dict[int, dict[str, Any]] = {}
        self.treasures_state = np.zeros(13, dtype=np.float32)
        self.total_treasure_count = -1
        self.last_score_info: dict[str, Any] | None = None
        self.minest_treasure = [-1, 999.0]
        self.last_minest_treasure = [-1, 999.0]

        self.buff_info: dict[int, dict[str, Any]] = {}
        self.buff_remain_time = 0

        self.flash_info = {
            "flash_usable": True,
            "last_flash_usable": True,
            "flash_count": 0,
            "flash_step_no": 0,
            "max_flash_count": 999,
        }

    def _get_pos_feature(self, found: int | bool, cur_pos, target_pos) -> np.ndarray:
        if target_pos is None:
            target_pos = cur_pos

        cur = np.asarray(cur_pos, dtype=np.float32)
        target = np.asarray(target_pos, dtype=np.float32)
        relative_pos = target - cur
        dist = float(np.linalg.norm(relative_pos))
        direction = relative_pos / max(dist, 1e-4)
        target_pos_norm = norm(target, MAP_SIZE, 0)

        return np.asarray(
            [
                float(found),
                float(norm(direction[0], 1, -1)),
                float(norm(direction[1], 1, -1)),
                float(target_pos_norm[0]),
                float(target_pos_norm[1]),
                float(norm(dist, 1.4142 * MAP_SIZE)),
            ],
            dtype=np.float32,
        )

    def pb2struct(self, frame_state, last_action: int) -> None:
        obs, _extra_info = frame_state
        self.step_no = int(obs["frame_state"]["step_no"])
        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        self._update_end_position(obs)
        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, MAP_SIZE, 0).astype(np.float32)
        self.feature_end_pos = self._get_pos_feature(1 if self.is_end_pos_found else 0, self.cur_pos, self.end_pos)
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])
        self.move_usable = True
        self.last_action = int(last_action)

    def _update_end_position(self, obs: dict[str, Any]) -> None:
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] != 4:
                continue

            if organ["status"] != -1:
                self.end_pos = (int(organ["pos"]["x"]), int(organ["pos"]["z"]))
                self.is_end_pos_found = True
                return

            if self.end_pos is None:
                rel = organ["relative_pos"]
                dist_level = RelativeDistance[rel["l2_distance"]]
                dir_level = RelativeDirection[rel["direction"]]
                distance = dist_level * 20
                theta = DirectionAngles[dir_level]
                delta_x = distance * math.cos(math.radians(theta))
                delta_z = distance * math.sin(math.radians(theta))
                self.end_pos = (
                    max(0, min(MAP_SIZE, round(self.cur_pos[0] + delta_x))),
                    max(0, min(MAP_SIZE, round(self.cur_pos[1] + delta_z))),
                )
                self.is_end_pos_found = False

    def _get_one_hot_map(self, obs) -> np.ndarray:
        self.local_map = np.asarray([row["values"] for row in obs["map_info"]], dtype=np.int32)
        onehot = np.zeros((5, LOCAL_VIEW_SIZE, LOCAL_VIEW_SIZE), dtype=np.float32)
        value_to_channel = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4}
        for value, channel in value_to_channel.items():
            onehot[channel] = (self.local_map == value).astype(np.float32)
        return onehot.flatten()

    def memory_update(self, cur_pos) -> None:
        x, z = int(cur_pos[0]), int(cur_pos[1])
        x = int(np.clip(x, 0, MAP_SIZE - 1))
        z = int(np.clip(MAP_SIZE - 1 - z, 0, MAP_SIZE - 1))
        self.global_memory_map[z, x] = min(1.0, self.global_memory_map[z, x] + 0.1)

        src_top = max(0, z - 5)
        src_bottom = min(MAP_SIZE, z + 6)
        src_left = max(0, x - 5)
        src_right = min(MAP_SIZE, x + 6)
        dst_top = src_top - (z - 5)
        dst_bottom = dst_top + (src_bottom - src_top)
        dst_left = src_left - (x - 5)
        dst_right = dst_left + (src_right - src_left)

        self.local_memory_map.fill(0.0)
        self.local_memory_map[dst_top:dst_bottom, dst_left:dst_right] = self.global_memory_map[
            src_top:src_bottom,
            src_left:src_right,
        ]
        self.memory_flag = self.local_memory_map.flatten()

    def _get_treasure_state(self, obs) -> np.ndarray:
        state = np.zeros(13, dtype=np.float32)
        count = 0
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] != 1:
                continue
            count += 1
            idx = int(organ["config_id"]) - 1
            if 0 <= idx < 13 and organ["status"] != 0:
                state[idx] = 1.0

        if self.total_treasure_count == -1:
            self.total_treasure_count = count

        self.treasures_state = state
        return state

    def _estimate_organ_pos(self, organ) -> tuple[int, int] | None:
        if organ["status"] == 1:
            return int(organ["pos"]["x"]), int(organ["pos"]["z"])

        rel = organ.get("relative_pos") or {}
        if not rel:
            return None

        dist_level = RelativeDistance.get(rel.get("l2_distance"), 0)
        dir_level = RelativeDirection.get(rel.get("direction"), 1)
        distance = dist_level * 20
        theta = DirectionAngles[dir_level]
        return (
            max(0, min(MAP_SIZE, round(self.cur_pos[0] + distance * math.cos(math.radians(theta))))),
            max(0, min(MAP_SIZE, round(self.cur_pos[1] + distance * math.sin(math.radians(theta))))),
        )

    def _get_treasure_feature(self, obs) -> np.ndarray:
        features = np.zeros((13, 6), dtype=np.float32)
        self.last_minest_treasure = list(self.minest_treasure)
        self.minest_treasure = [-1, 999.0]

        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] != 1:
                continue
            idx = int(organ["config_id"]) - 1
            if not 0 <= idx < 13:
                continue
            target_pos = self._estimate_organ_pos(organ)
            found = 1 if organ["status"] == 1 else 0
            features[idx] = self._get_pos_feature(found, self.cur_pos, target_pos)
            dist = float(features[idx][-1])
            if organ["status"] != 0 and dist < self.minest_treasure[1]:
                self.minest_treasure = [int(organ["config_id"]), dist]

        return features.flatten()

    def _get_buff_feature(self, obs) -> np.ndarray:
        feature = np.zeros(8, dtype=np.float32)
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] != 2:
                continue
            pos_feature = self._get_pos_feature(1 if organ["status"] == 1 else 0, self.cur_pos, self._estimate_organ_pos(organ))
            feature[:6] = pos_feature
            feature[6] = float(np.clip(self.buff_remain_time / 51, 0, 1))
            feature[7] = float(obs["score_info"].get("buff_count", 0))
            break
        return feature

    def _get_talent_feature(self, obs) -> np.ndarray:
        hero = obs["frame_state"]["heroes"][0]
        usable = bool(hero["talent"]["status"])
        cooldown = float(hero["talent"].get("cooldown", 0))

        if not usable and self.flash_info["last_flash_usable"]:
            self.flash_info["flash_count"] += 1
            self.flash_info["flash_step_no"] = self.step_no

        self.flash_info["flash_usable"] = usable
        self.flash_info["last_flash_usable"] = usable
        return np.asarray(
            [
                float(self.flash_info["flash_count"]),
                1.0 if usable else 0.0,
                float(np.clip(cooldown / 100.0, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def _get_reward_info(self, obs, last_action, extra_info, is_exploit=False) -> dict[str, Any]:
        score_info = obs["score_info"]
        last_score = self.last_score_info or score_info
        cur_pos = self.cur_pos
        stagnated = self.last_pos is not None and cur_pos == self.last_pos
        visible_treasure_count = sum(
            1 for organ in obs["frame_state"]["organs"] if organ["sub_type"] == 1 and organ["status"] == 1
        )

        treasure_dist_delta = 0.0
        if self.last_minest_treasure[0] == self.minest_treasure[0] and self.last_minest_treasure[0] != -1:
            treasure_dist_delta = max(0.0, float(self.last_minest_treasure[1]) - float(self.minest_treasure[1]))

        return {
            "is_exploit": is_exploit,
            "get_treasure": score_info.get("treasure_collected_count", 0) > last_score.get("treasure_collected_count", 0),
            "get_buff": score_info.get("buff_count", 0) > last_score.get("buff_count", 0),
            "use_talent": score_info.get("talent_count", 0) > last_score.get("talent_count", 0),
            "stagnated": stagnated,
            "flash_over_wall": False,
            "speed_up": bool(obs["frame_state"]["heroes"][0].get("speed_up")),
            "visible_treasure_count": visible_treasure_count,
            "treasure_dist_delta": treasure_dist_delta,
            "last_end_pos_dist": self.last_end_pos_dist,
        }

    def process(self, frame_state, last_action, is_exploit=False):
        self.pb2struct(frame_state, last_action)
        obs, extra_info = frame_state

        one_hot_map = self._get_one_hot_map(obs)
        local_map = self.local_map.astype(np.float32).flatten()
        treasure_state = self._get_treasure_state(obs)
        feature_treasures_pos = self._get_treasure_feature(obs)
        feature_buff = self._get_buff_feature(obs)
        self.memory_update(self.cur_pos)
        talent_feature = self._get_talent_feature(obs)
        time_feat = np.asarray([obs["frame_state"]["step_no"] / 2000], dtype=np.float32)
        legal_action = np.asarray(self.get_legal_action(), dtype=np.float32)

        feature = np.concatenate(
            [
                self.cur_pos_norm,
                one_hot_map,
                local_map,
                self.memory_flag,
                self.feature_end_pos,
                self.feature_history_pos,
                treasure_state,
                feature_treasures_pos,
                feature_buff,
                talent_feature,
                time_feat,
            ]
        ).astype(np.float32)

        if feature.shape[0] != Config.FEATURE_LEN:
            raise ValueError(f"Feature length {feature.shape[0]} != expected {Config.FEATURE_LEN}")

        reward_info = self._get_reward_info(obs, last_action, extra_info, is_exploit)
        self.last_end_pos_dist = reward_info.get("end_pos_dist", self.feature_end_pos[-1])
        self.last_local_map = self.local_map.copy()
        self.last_score_info = dict(obs["score_info"])

        return feature, legal_action, reward_process(self.feature_end_pos[-1], self.feature_history_pos[-1], reward_info)

    def get_legal_action(self):
        if (
            abs(float(self.cur_pos_norm[0]) - float(self.last_pos_norm[0])) < 0.001
            and abs(float(self.cur_pos_norm[1]) - float(self.last_pos_norm[1])) < 0.001
            and 0 <= self.last_action < Config.ACTION_NUM
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        if not self.flash_info["flash_usable"] or self.flash_info["flash_count"] > self.flash_info["max_flash_count"]:
            self.bad_move_ids.update(range(Config.ACTION_NUM, Config.ACTION_SPACE_SIZE))

        legal_action = [1.0] * self.move_action_num
        for move_id in self.bad_move_ids:
            if 0 <= move_id < len(legal_action):
                legal_action[move_id] = 0.0

        if not any(legal_action):
            self.bad_move_ids = set()
            legal_action = [1.0] * self.move_action_num

        return legal_action
