"""Local treasure-hunt environment used by the PPO training workflow."""

from __future__ import annotations

import logging
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import toml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Env_v2:
    def __init__(self):
        self.image_path = PROJECT_ROOT / "map" / "map.png"
        self.toml_path = PROJECT_ROOT / "env" / "config.toml"
        self.log_file_path = PROJECT_ROOT / "env" / "log.txt"
        self.map = None
        self._map_cache = {}
        self._config_cache = None
        self.logger = self.setup_logger()

        self.sqrt_2 = 1.41421356237
        self.move_deltas = [
            (1, 0),
            (self.sqrt_2 / 2, self.sqrt_2 / 2),
            (0, 1),
            (-self.sqrt_2 / 2, self.sqrt_2 / 2),
            (-1, 0),
            (-self.sqrt_2 / 2, -self.sqrt_2 / 2),
            (0, -1),
            (self.sqrt_2 / 2, -self.sqrt_2 / 2),
        ]
        self.flash_deltas = [
            (16, 0),
            (16 / self.sqrt_2, 16 / self.sqrt_2),
            (0, 16),
            (-16 / self.sqrt_2, 16 / self.sqrt_2),
            (-16, 0),
            (-16 / self.sqrt_2, -16 / self.sqrt_2),
            (0, -16),
            (16 / self.sqrt_2, -16 / self.sqrt_2),
        ]

    def reset(self):
        config = self.read_usr_conf()["conf"]
        self._init_config(config)
        self.make_map()
        self.set_game_info()
        self.get_obs()
        return self.obs, self.get_extra_info()

    def _init_config(self, config):
        self.start_pos = config["start"].copy()
        self.end_pos = config["end"].copy()
        self.buff_pos = config["buff"].copy()
        self.treasure_pos = [pos.copy() for pos in config["treasure_pos"]]
        self.start_random = config["start_random"]
        self.end_random = config["end_random"]
        self.buff_random = config["buff_random"]
        self.treasure_random = config["treasure_random"]
        self.obstacle_random = config["obstacle_random"]
        self.buff_cooldown = config["buff_cooldown"]
        self.buff_duration = 510
        self.talent_total_cooldown = config["talent_cooldown"]
        self.treasure_count = config["treasure_count"] if self.treasure_random else len(self.treasure_pos)
        self.obstacle_id = config["obstacle_id"]
        self.max_step = config["max_step"]
        self.unavailable_positions = {(111, 70), (20, 55), (57, 68)}

    def set_game_info(self):
        self.use_talent = False
        self.talent_cooldown = 0
        self.use_talent_cnt = 0
        self.collect_buff = False
        self.last_get_buff_step_no = 0
        self.buff_remain_time = 0
        self.collect_buff_cnt = 0
        self.speed_up = False
        self.collect_treasure = False
        self.treasure_status = [1] * len(self.treasure_pos)
        self.cur_pos = self.start_pos.copy()
        self.last_pos = self.cur_pos.copy()
        self.cur_location = [float(i) for i in self.cur_pos]
        self.game_win = False
        self.game_over = False
        self.obs = None
        self.cur_step_no = 0
        self.total_score = 0
        self.collect_treasure_config_id = None

    def step(self, act):
        if self.game_over or self.game_win:
            self.logger.error("Game already ended; reset before stepping again.")
            return self.cur_step_no, self.obs, self.game_win, self.game_over, self.get_extra_info()

        self.last_pos = self.cur_pos.copy()
        if 0 <= act < 8:
            self.move(act)
        elif 8 <= act < 16:
            self.flash_move(act)
        else:
            self.logger.error("Invalid action: %s", act)

        self.status_update()
        self.get_obs()
        return self.cur_step_no, self.obs, self.game_win, self.game_over, self.get_extra_info()

    def move(self, act):
        dx, dz = self.move_deltas[act]
        if self.speed_up:
            dx, dz = dx * 1.5, dz * 1.5
        self.move_once(dx, dz)

    def move_once(self, dx, dz):
        start_x, start_z = self.cur_location
        target_x, target_z = start_x + dx, start_z + dz
        path_points = self.get_path_points(start_x, start_z, target_x, target_z)
        final_valid = (start_x, start_z)
        map_height, map_width = len(self.map), len(self.map[0])

        for px, pz in path_points:
            gx, gz = int(px), int(pz)
            if not (0 <= gx < map_height and 0 <= gz < map_width) or self.map[gx][gz] == 0:
                break
            final_valid = (px, pz)

        if final_valid == (start_x, start_z):
            return

        final_path = [pt for pt in path_points if abs(pt[0] - final_valid[0]) >= 1e-6 or abs(pt[1] - final_valid[1]) >= 1e-6]
        final_path.append(final_valid)
        collected_items = set()
        for px, pz in final_path:
            gx, gz = int(px), int(pz)
            if self.map[gx][gz] in {3, 4, 6} and (gx, gz) not in collected_items:
                self.collect_item_at(gx, gz, self.map[gx][gz])
                collected_items.add((gx, gz))

        self.cur_pos = [int(final_valid[0]), int(final_valid[1])]
        self.cur_location = list(final_valid)

    def get_path_points(self, start_x, start_z, end_x, end_z, num_samples=None):
        path_length = max(abs(end_x - start_x), abs(end_z - start_z))
        num_samples = max(int(path_length * 2), 10) if num_samples is None else num_samples
        return [
            (start_x + t * (end_x - start_x) / num_samples, start_z + t * (end_z - start_z) / num_samples)
            for t in range(num_samples + 1)
        ]

    def flash_move(self, act):
        if self.talent_cooldown:
            self.move(act - 8)
            return
        dx, dz = self.flash_deltas[act - 8]
        self.flash_once(dx, dz)
        self.use_talent = True

    def flash_once(self, dx, dz):
        target_x = self.cur_location[0] + dx
        target_z = self.cur_location[1] + dz
        tx, tz = int(round(target_x)), int(round(target_z))
        map_height, map_width = len(self.map), len(self.map[0])

        if not (0 <= tx < map_height and 0 <= tz < map_width):
            return

        if self.map[tx][tz] == 0:
            length = max(abs(dx), abs(dz)) or 1
            for k in range(int(length * 10), -1, -10):
                ratio = k / (length * 10)
                px = self.cur_location[0] + dx * ratio
                pz = self.cur_location[1] + dz * ratio
                ix, iz = int(round(px)), int(round(pz))
                if 0 <= ix < map_height and 0 <= iz < map_width and self.map[ix][iz] != 0:
                    self.cur_location = [px, pz]
                    self.cur_pos = [ix, iz]
                    return
            return

        self.cur_location = [target_x, target_z]
        self.cur_pos = [tx, tz]
        if self.map[tx][tz] in {3, 4, 6}:
            self.collect_item_at(tx, tz, self.map[tx][tz])

    def collect_item_at(self, grid_x, grid_z, cell_value):
        if cell_value == 4:
            self.collect_treasure = True
            try:
                self.collect_treasure_config_id = self.treasure_pos.index([grid_x, grid_z]) + 1
            except ValueError:
                self.collect_treasure_config_id = None
        elif cell_value == 6:
            self.collect_buff = True
        elif cell_value == 3:
            self.game_win = True
        self.map[grid_x][grid_z] = 1

    def status_update(self):
        if self.talent_cooldown > 0:
            self.talent_cooldown -= 1
        self.speed_up = self.buff_remain_time > 0
        if self.buff_remain_time > 0:
            self.buff_remain_time -= 1
        self.cur_step_no += 1

        if self.use_talent:
            self.use_talent_cnt += 1
            self.talent_cooldown = self.talent_total_cooldown
            self.use_talent = False

        if self.collect_treasure:
            self.collect_treasure = False
            self.total_score += 100
            if self.collect_treasure_config_id is not None:
                self.treasure_status[self.collect_treasure_config_id - 1] = 0

        if self.collect_buff:
            self.collect_buff = False
            self.buff_remain_time = self.buff_duration
            self.collect_buff_cnt += 1
            self.last_get_buff_step_no = self.cur_step_no

        if self.cur_step_no - self.last_get_buff_step_no >= self.buff_cooldown:
            self.map[self.buff_pos[0]][self.buff_pos[1]] = 6

        if self.game_win:
            self.total_score += 150 + (self.max_step - self.cur_step_no) * 0.2
        if self.cur_step_no >= self.max_step:
            self.game_over = True

    def get_obs(self):
        obs = {
            "frame_state": {
                "heroes": {
                    0: {
                        "pos": {"x": self.cur_pos[0], "z": self.cur_pos[1]},
                        "talent": {"status": 0 if self.talent_cooldown else 1, "cooldown": self.talent_cooldown},
                        "speed_up": self.speed_up,
                        "buff_remain_time": self.buff_remain_time,
                    }
                },
                "organs": [],
                "step_no": self.cur_step_no,
            },
            "result_message": {},
            "score_info": self.get_score_info(),
            "map_info": self.get_map_info(),
            "legal_act": [1, 0 if self.talent_cooldown else 1],
        }

        organs = [
            (3, 21, self.start_pos, ""),
            (4, 22, self.end_pos, "end"),
            (2, 0, self.buff_pos, "buff"),
        ] + [(1, i + 1, pos, "treasure") for i, pos in enumerate(self.treasure_pos)]

        for sub_type, config_id, pos, organ_type in organs:
            obs["frame_state"]["organs"].append(
                {
                    "sub_type": sub_type,
                    "config_id": config_id,
                    "pos": self.get_pos(self.cur_pos, pos),
                    "status": self.get_status(self.cur_pos, pos, organ_type),
                    "relative_pos": self.get_relative_pos(self.cur_pos, pos),
                    "cool_down": self.buff_cooldown if sub_type == 2 else 0,
                }
            )
        self.obs = obs

    def get_score_info(self):
        treasure_collected_count = len(self.treasure_status) - sum(self.treasure_status)
        return {
            "score": 150 if self.game_win else 0,
            "total_score": self.total_score,
            "step_no": self.cur_step_no,
            "treasure_collected_count": treasure_collected_count,
            "treasure_score": treasure_collected_count * 100,
            "buff_count": self.collect_buff_cnt,
            "talent_count": self.use_talent_cnt,
        }

    def get_extra_info(self):
        return {
            "result_code": 0,
            "result_message": "",
            "frame_state": self.get_extra_info_frame_state(),
            "game_info": self.get_game_info(),
        }

    def get_extra_info_frame_state(self):
        frame_state = {
            "heroes": {
                0: {
                    "pos": {"x": self.cur_pos[0], "z": self.cur_pos[1]},
                    "talent": {"status": 0 if self.talent_cooldown else 1, "cooldown": self.talent_cooldown},
                    "speed_up": self.speed_up,
                    "buff_remain_time": self.buff_remain_time,
                }
            },
            "organs": [],
            "step_no": self.cur_step_no,
        }
        organs = [
            (3, 21, self.start_pos, ""),
            (4, 22, self.end_pos, "end"),
            (2, 0, self.buff_pos, "buff_extra_info"),
        ] + [(1, i + 1, pos, "treasure_extra_info") for i, pos in enumerate(self.treasure_pos)]

        for sub_type, config_id, pos, organ_type in organs:
            visible = abs(self.cur_pos[0] - pos[0]) <= 5 and abs(self.cur_pos[1] - pos[1]) <= 5
            frame_state["organs"].append(
                {
                    "sub_type": sub_type,
                    "config_id": config_id,
                    "pos": {"x": pos[0], "z": pos[1]} if visible else {"x": -1, "z": -1},
                    "status": self.get_status(self.cur_pos, pos, organ_type),
                    "relative_pos": self.get_relative_pos(self.cur_pos, pos),
                    "cool_down": self.buff_cooldown if sub_type == 2 else 0,
                }
            )
        return frame_state

    def get_game_info(self):
        treasure_collected_count = len(self.treasure_status) - sum(self.treasure_status)
        return {
            "score": self.obs["score_info"]["score"] if self.obs else 0,
            "total_score": self.total_score,
            "step_no": self.cur_step_no,
            "pos": self.cur_pos,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "treasure_collected_count": treasure_collected_count,
            "treasure_score": treasure_collected_count * 100,
            "treasure_count": len(self.treasure_pos),
            "buff_count": self.collect_buff_cnt,
            "talent_count": self.use_talent_cnt,
            "buff_remain_time": self.buff_remain_time,
            "buff_duration": self.buff_duration,
            "map_info": self.get_map_info(),
            "obstacle_id": self.obstacle_id,
        }

    def get_map_info(self):
        x, z = self.cur_pos
        map_array = np.asarray(self.map)
        x_start, x_end = max(0, x - 5), min(128, x + 6)
        z_start, z_end = max(0, z - 5), min(128, z + 6)
        local = np.zeros((11, 11), dtype=int)
        local_x_start = max(0, 5 - (x - x_start))
        local_z_start = max(0, 5 - (z - z_start))
        local[
            local_x_start : local_x_start + x_end - x_start,
            local_z_start : local_z_start + z_end - z_start,
        ] = map_array[x_start:x_end, z_start:z_end]
        return [{"values": row.tolist()} for row in local]

    @lru_cache(maxsize=512)
    def _get_relative_pos_cached(self, dx, dz):
        angle = math.atan2(dz, dx)
        degrees = math.degrees(angle) % 360
        direction_map = [
            (22.5, "East"),
            (67.5, "NorthEast"),
            (112.5, "North"),
            (157.5, "NorthWest"),
            (202.5, "West"),
            (247.5, "SouthWest"),
            (292.5, "South"),
            (337.5, "SouthEast"),
        ]
        direction = next((name for threshold, name in direction_map if degrees < threshold), "East")
        dist = math.sqrt(dx * dx + dz * dz)
        l2_distance = (
            "RELATIVE_DISTANCE_NONE"
            if dist == 0
            else "VerySmall"
            if dist <= 20
            else "Small"
            if dist <= 40
            else "Medium"
            if dist <= 60
            else "Large"
            if dist <= 80
            else "VeryLarge"
        )
        return {"direction": direction, "l2_distance": l2_distance}

    def get_relative_pos(self, pos, target_pos):
        return self._get_relative_pos_cached(target_pos[0] - pos[0], target_pos[1] - pos[1])

    def get_pos(self, pos, target_pos):
        visible = abs(pos[0] - target_pos[0]) <= 5 and abs(pos[1] - target_pos[1]) <= 5
        return {"x": target_pos[0], "z": target_pos[1]} if visible else {"x": -1, "z": -1}

    def get_status(self, pos, target_pos, organ_type=""):
        if abs(pos[0] - target_pos[0]) > 5 or abs(pos[1] - target_pos[1]) > 5:
            return -1
        if organ_type in ("treasure", "treasure_extra_info"):
            return self.treasure_status[self.treasure_pos.index(target_pos)]
        if organ_type in ("buff", "buff_extra_info"):
            return 1 if self.map[self.buff_pos[0]][self.buff_pos[1]] == 6 else 0
        if organ_type == "end":
            return 1
        return 1

    def random_pos(self, num):
        if num <= 0:
            return []
        map_array = np.asarray(self.map)
        valid_positions = [
            (int(x), int(z))
            for x, z in zip(*np.where(map_array == 1))
            if (int(x), int(z)) not in self.unavailable_positions
        ]
        if len(valid_positions) < num:
            raise RuntimeError(f"Not enough valid positions: available {len(valid_positions)}, required {num}")
        return random.sample(valid_positions, num)

    def make_map(self):
        if self.obstacle_random:
            self.obstacle_id = random.randint(1, 6)
            self.image_path = PROJECT_ROOT / "map" / f"obstacle_{self.obstacle_id}.png"
        elif self.obstacle_id:
            self.image_path = PROJECT_ROOT / "map" / f"obstacle_{self.obstacle_id[0]}.png"
        else:
            self.image_path = PROJECT_ROOT / "map" / "map.png"
        self.map = self.read_image_to_map(self.image_path)
        self.make_organs()

    def make_organs(self):
        random_count = sum([self.start_random, self.end_random, self.buff_random])
        if random_count:
            positions = self.random_pos(random_count)
            pos_index = 0
            for is_random, pos_attr, map_value in [
                (self.start_random, "start_pos", 2),
                (self.end_random, "end_pos", 3),
                (self.buff_random, "buff_pos", 6),
            ]:
                if is_random:
                    pos = list(positions[pos_index])
                    setattr(self, pos_attr, pos)
                    self.map[pos[0]][pos[1]] = map_value
                    self.unavailable_positions.add(tuple(pos))
                    pos_index += 1

        for pos_attr, random_attr, map_value, default_pos in [
            ("start_pos", "start_random", 2, [111, 70]),
            ("end_pos", "end_random", 3, [20, 55]),
            ("buff_pos", "buff_random", 6, [57, 68]),
        ]:
            if not getattr(self, random_attr):
                pos = getattr(self, pos_attr)
                if self.map[pos[0]][pos[1]] != 0:
                    self.map[pos[0]][pos[1]] = map_value
                    self.unavailable_positions.add(tuple(pos))
                else:
                    self.logger.warning("Invalid %s %s, using default %s", pos_attr, pos, default_pos)
                    setattr(self, pos_attr, default_pos)
                    self.map[default_pos[0]][default_pos[1]] = map_value
                    self.unavailable_positions.add(tuple(default_pos))

        if self.treasure_random and self.treasure_count > 0:
            random_positions = self.random_pos(self.treasure_count)
            self.treasure_pos = [list(pos) for pos in random_positions]
            for pos in random_positions:
                self.map[pos[0]][pos[1]] = 4
                self.unavailable_positions.add(tuple(pos))
        elif self.treasure_pos:
            unique_pos = list(dict.fromkeys(map(tuple, self.treasure_pos)))
            self.treasure_pos = [list(pos) for pos in unique_pos]
            for i, pos in enumerate(self.treasure_pos):
                if tuple(pos) in self.unavailable_positions or self.map[pos[0]][pos[1]] == 0:
                    replacement = self.random_pos(1)[0]
                    pos = list(replacement)
                    self.treasure_pos[i] = pos
                self.map[pos[0]][pos[1]] = 4
                self.unavailable_positions.add(tuple(pos))

    def read_image_to_map(self, png_path="", white_threshold=200):
        png_path = Path(png_path or self.image_path)
        cache_key = str(png_path)
        if cache_key in self._map_cache:
            return [row[:] for row in self._map_cache[cache_key]]
        image = Image.open(png_path).convert("L")
        binary_map = (np.asarray(image) > white_threshold).astype(int).tolist()
        self._map_cache[cache_key] = [row[:] for row in binary_map]
        return binary_map

    def read_usr_conf(self, file_path=""):
        file_path = Path(file_path or self.toml_path)
        if self._config_cache is None:
            self._config_cache = toml.load(file_path)
        return self._config_cache

    def setup_logger(self, log_file_path=""):
        log_file_path = Path(log_file_path or self.log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("back_to_realm.env")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger


def start_pygame_game():
    """Run a small manual-play loop if pygame is installed."""

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Install the optional 'game' extra to use the pygame demo.") from exc

    env = Env_v2()
    env.reset()
    pygame.init()
    screen = pygame.display.set_mode((700, 760))
    pygame.display.set_caption("Back To Realm - Local Environment")
    clock = pygame.time.Clock()
    action_map = {
        pygame.K_d: 0,
        pygame.K_e: 1,
        pygame.K_s: 2,
        pygame.K_z: 3,
        pygame.K_a: 4,
        pygame.K_q: 5,
        pygame.K_w: 6,
        pygame.K_c: 7,
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key in action_map:
                    shift = pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]
                    env.step(action_map[event.key] + (8 if shift else 0))

        screen.fill((245, 245, 245))
        local = env.get_map_info()
        cell = 50
        colors = {0: (40, 40, 40), 1: (255, 255, 255), 2: (20, 180, 80), 3: (220, 60, 60), 4: (240, 210, 60), 6: (60, 140, 230)}
        for y, row in enumerate(local):
            for x, value in enumerate(row["values"]):
                pygame.draw.rect(screen, colors.get(value, (255, 255, 255)), (x * cell, y * cell, cell, cell))
                pygame.draw.rect(screen, (210, 210, 210), (x * cell, y * cell, cell, cell), 1)
        pygame.draw.rect(screen, (120, 40, 170), (5 * cell, 5 * cell, cell, cell))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    start_pygame_game()
