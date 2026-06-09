"""Multi-actor / single-learner launcher."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
import webbrowser
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def actor_process_main(actor_id: int, queue: mp.Queue, device_str: str):
    import torch

    from PPO.agent import Agent
    from PPO.conf.conf import Config
    from PPO.workflow.train_workflow import run_episodes, setup_logger
    from env.env_v2 import Env_v2

    logger = setup_logger()
    logger.info("Actor %s starting", actor_id)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    env = Env_v2()
    agent = Agent(device=device, logger=logger)

    episode = 1
    last_save_model_time = time.time()
    try:
        while True:
            for game_data in run_episodes(1, env, agent, logger):
                queue.put((actor_id, [sample.npdata for sample in game_data]))
                time.sleep(0)
            episode += 1

            if episode % Config.LOAD_FREQ == 0 and Path("./ckpt/dump_model/model.ckpt-latest.pkl").exists():
                try:
                    agent.load_model(path="./ckpt/dump_model", id="latest")
                    logger.info("Actor %s reloaded latest model", actor_id)
                except Exception as exc:
                    logger.warning("Actor %s failed to reload latest model: %s", actor_id, exc)

            now = time.time()
            if now - last_save_model_time >= Config.SAVE_FREQ:
                agent.save_model(path="./ckpt/dump_model", id=f"{episode}-{actor_id}")
                last_save_model_time = now
    except KeyboardInterrupt:
        logger.info("Actor %s interrupted", actor_id)
    except Exception as exc:
        logger.exception("Actor %s failed: %s", actor_id, exc)
    finally:
        logger.info("Actor %s stopping", actor_id)


def learner_process_main(queue: mp.Queue, device_str: str, batch_size: int, batch_timeout: float):
    import logging

    import torch

    from PPO.algorithm.algorithm import Algorithm
    from PPO.conf.conf import Config
    from PPO.feature.definition import SampleData

    logger = logging.getLogger("PPO_learner")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    algorithm = Algorithm(device=device, logger=logger)
    if Config.PRELOAD_MODEL_ID and Path("./ckpt/dump_model/model.ckpt-latest.pkl").exists():
        algorithm.load_model(path="./ckpt/dump_model", id=Config.PRELOAD_MODEL_ID)

    logger.info("Learner starting on %s", device)
    buffer: deque[SampleData] = deque(maxlen=Config.CAPACITY)
    last_get_time = time.time()

    try:
        while True:
            try:
                _actor_id, raw_list = queue.get(timeout=1.0)
            except Exception:
                raw_list = None

            now = time.time()
            if raw_list:
                buffer.extend(SampleData(npdata=arr) for arr in raw_list)

            if len(buffer) >= batch_size or (buffer and now - last_get_time >= batch_timeout):
                take = min(batch_size, len(buffer))
                batch = [buffer.popleft() for _ in range(take)]
                last_get_time = now
                algorithm.learn(batch)
    except KeyboardInterrupt:
        logger.info("Learner interrupted")
    except Exception as exc:
        logger.exception("Learner failed: %s", exc)
    finally:
        logger.info("Learner stopping")


def parse_args():
    parser = argparse.ArgumentParser(description="Run distributed local PPO training.")
    parser.add_argument("--actors", type=int, default=4, help="Number of actor processes.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Preferred torch device.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Samples per learner update.")
    parser.add_argument("--batch-timeout", type=float, default=5.0, help="Max seconds before training on a partial batch.")
    parser.add_argument("--dashboard", action="store_true", help="Also launch the Streamlit dashboard.")
    return parser.parse_args()


def main():
    args = parse_args()
    mp_start = "spawn" if sys.platform == "win32" else "fork"
    try:
        mp.set_start_method(mp_start)
    except RuntimeError:
        pass

    queue = mp.Queue(maxsize=1000)
    learner = mp.Process(target=learner_process_main, args=(queue, args.device, args.batch_size, args.batch_timeout), name="learner")
    learner.start()

    dashboard_process = None
    if args.dashboard:
        dashboard_process = subprocess.Popen(["streamlit", "run", "dashboard.py"], cwd=os.getcwd())
        time.sleep(2)
        webbrowser.open("http://localhost:8501")

    actors = []
    for i in range(args.actors):
        process = mp.Process(target=actor_process_main, args=(i, queue, args.device), name=f"actor-{i}")
        process.start()
        actors.append(process)

    try:
        while True:
            time.sleep(1)
            if not learner.is_alive():
                print("Learner process stopped; terminating actors")
                break
            for i, process in enumerate(actors):
                if not process.is_alive():
                    print(f"Actor {i} stopped; respawning")
                    replacement = mp.Process(target=actor_process_main, args=(i, queue, args.device), name=f"actor-{i}")
                    replacement.start()
                    actors[i] = replacement
    except KeyboardInterrupt:
        print("Launcher interrupted; shutting down")
    finally:
        for process in actors:
            process.terminate()
        learner.terminate()
        if dashboard_process:
            dashboard_process.terminate()


if __name__ == "__main__":
    main()
