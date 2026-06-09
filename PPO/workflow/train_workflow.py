"""Single-process training workflow."""

from __future__ import annotations

import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PPO.feature.definition import SampleManager


def setup_logger(log_dir="log"):
    """Create an idempotent logger for training runs."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("PPO")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_path / f"ppo_training_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def workflow(envs, agents, logger=None, max_epochs=None):
    """Run training until interrupted or until `max_epochs` is reached."""

    logger = logger or setup_logger()
    env, agent = envs[0], agents[0]
    save_path = Path("./ckpt/dump_model")
    save_path.mkdir(parents=True, exist_ok=True)
    episodes = 0
    last_save_model_time = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Models will be saved to %s", save_path)

    while max_epochs is None or episodes < max_epochs:
        for game_data in run_episodes(1, env, agent, logger):
            agent.learn(game_data)
            game_data.clear()

        episodes += 1
        logger.info("Episode %s completed", episodes)

        now = time.time()
        if now - last_save_model_time >= 1800:
            agent.save_model(path=save_path, id=str(episodes))
            episode_model_path = save_path / f"model.ckpt-{episodes}.pkl"
            latest_model_path = save_path / "model.ckpt-latest.pkl"
            try:
                shutil.copy(episode_model_path, latest_model_path)
                logger.info("Updated latest model checkpoint")
            except Exception as exc:
                logger.warning("Failed to update latest checkpoint: %s", exc)
            last_save_model_time = now


def run_episodes(n_episode, env, agent, logger):
    """Yield packed PPO samples for `n_episode` environment rollouts."""

    for _episode in range(n_episode):
        collector = SampleManager()
        win_rate = 0.0

        obs, extra_info = env.reset()
        if extra_info["result_code"] < 0:
            logger.error("env.reset failed: %s", extra_info["result_message"])
            raise RuntimeError(extra_info["result_message"])
        if extra_info["result_code"] > 0:
            logger.warning("Skipping invalid episode: %s", extra_info["result_message"])
            continue

        agent.reset()
        done = False
        step = 0
        max_step_no = 2000
        total_reward = 0.0
        reward_totals = np.zeros(9, dtype=np.float32)

        while not done:
            feature, legal_action, reward_list = agent.observation_process(obs, extra_info)
            prob, value, action = agent.predict(feature, legal_action)
            agent.last_action = action

            hero = obs["frame_state"]["heroes"][0]
            agent.preprocessor.last_pos = (hero["pos"]["x"], hero["pos"]["z"])

            step_no, next_obs, terminated, truncated, next_extra_info = env.step(action)
            if next_extra_info["result_code"] != 0:
                logger.warning("env.step returned %s: %s", next_extra_info["result_code"], next_extra_info["result_message"])
                break

            step += 1
            reward = np.array([sum(reward_list)], dtype=np.float32)
            reward_totals += np.asarray(reward_list, dtype=np.float32)
            total_reward += float(reward[0])

            collector.sample_process(
                feature=feature,
                legal_action=legal_action,
                prob=prob,
                action=[action],
                value=value,
                reward=reward,
            )

            final_reward = 0.0
            if truncated or terminated:
                game_info = next_extra_info["game_info"]
                total_treasure_count = agent.preprocessor.total_treasure_count
                treasure_collected_count = next_obs["score_info"]["treasure_collected_count"]
                time_ratio = step_no / 2000
                missed_treasure_count = total_treasure_count - treasure_collected_count
                if missed_treasure_count:
                    shaped_finish_reward = (-2 / 5) * missed_treasure_count + (-10 / 7) * time_ratio + (3 / 7)
                    timeout_reward = shaped_finish_reward - 10
                else:
                    shaped_finish_reward = 10
                    timeout_reward = 0

                if truncated:
                    win_rate = agent.update_win_rate(False)
                    final_reward += timeout_reward
                    logger.info(
                        "Episode timed out: step=%s score=%.2f win_rate=%.4f treasures=%s/%s skills=%s reward=%.2f",
                        step_no,
                        game_info["total_score"],
                        win_rate,
                        treasure_collected_count,
                        total_treasure_count,
                        next_obs["score_info"]["talent_count"],
                        total_reward,
                    )
                elif terminated:
                    win_rate = agent.update_win_rate(True)
                    final_reward += shaped_finish_reward
                    logger.info(
                        "Episode completed: step=%s score=%.2f win_rate=%.4f treasures=%s/%s skills=%s reward=%.2f",
                        step_no,
                        game_info["total_score"],
                        win_rate,
                        treasure_collected_count,
                        total_treasure_count,
                        next_obs["score_info"]["talent_count"],
                        total_reward,
                    )
                logger.info("Reward components: %s", np.round(reward_totals, 2).tolist())

            done = terminated or truncated or (max_step_no > 0 and step >= max_step_no)
            if done:
                collector.process_last_frame(np.array([final_reward], dtype=np.float32))
                if collector.samples:
                    yield collector.get_game_data()
                break

            obs = next_obs
            extra_info = next_extra_info


def main():
    logger = setup_logger()
    from env.env_v2 import Env_v2 as Env
    from PPO.agent import Agent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env()
    agent = Agent(device=device, logger=logger)
    latest_model = Path("./ckpt/dump_model/model.ckpt-latest.pkl")
    if latest_model.exists():
        try:
            agent.load_model(path="./ckpt/dump_model", id="latest")
        except Exception as exc:
            logger.warning("Failed to preload latest model: %s", exc)
    workflow([env], [agent], logger)


if __name__ == "__main__":
    raise SystemExit(main())
