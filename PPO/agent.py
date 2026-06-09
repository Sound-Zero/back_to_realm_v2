"""Agent wrapper combining preprocessing, action selection, and learning."""

from __future__ import annotations

from pathlib import Path

import torch

from PPO.algorithm.algorithm import Algorithm
from PPO.conf.conf import Config
from PPO.feature.preprocessor import Preprocessor
from PPO.model.model import NetworkModelActor


class Agent:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.monitor = None
        self.algorithm = Algorithm(device=self.device, logger=self.logger)
        self.preprocessor = Preprocessor()
        self.win_history: list[bool] = []
        self.actor = NetworkModelActor().to(self.device)
        self.reset()

        if Config.PRELOAD_MODEL_ID:
            model_path = Path("./ckpt/dump_model") / f"model.ckpt-{Config.PRELOAD_MODEL_ID}.pkl"
            if model_path.exists():
                self.load_model(path="./ckpt/dump_model", id=Config.PRELOAD_MODEL_ID)

    def observation_process(self, obs, extra_info=None):
        return self.preprocessor.process([obs, extra_info], self.last_action)

    def reset(self):
        self.preprocessor.reset()
        self.last_prob = 0
        self.last_action = -1

    def select_action(self, prob):
        with torch.no_grad():
            dist = torch.distributions.Categorical(prob)
            action = dist.sample()
            action_prob = prob[action]
        return action, action_prob

    def predict(self, feature, legal_action):
        feature, legal_action = self.actor.format_data(feature, legal_action)
        feature = feature.to(self.device)
        legal_action = legal_action.to(self.device)
        with torch.no_grad():
            probs, value = self.actor(feature.unsqueeze(0), legal_action.unsqueeze(0))
            action, action_prob = self.select_action(probs.squeeze(0))
            return action_prob.cpu().numpy().reshape(-1), value.cpu().numpy().reshape(-1), action.item()

    def learn(self, list_sample_data):
        if self.algorithm.learn_cnt % Config.UPDATE_FREQ == 0 and self.algorithm.learn_cnt > 0:
            latest_model = Path("./ckpt/dump_model/model.ckpt-latest.pkl")
            if latest_model.exists():
                try:
                    self.load_model(path="./ckpt/dump_model", id="latest")
                except Exception as exc:
                    self.logger.warning("Failed to reload latest actor model: %s", exc)
        return self.algorithm.learn(list_sample_data)

    def update_win_rate(self, is_win):
        self.win_history.append(bool(is_win))
        if len(self.win_history) > 100:
            self.win_history.pop(0)
        return sum(self.win_history) / len(self.win_history) if len(self.win_history) > 10 else 0

    def save_model(self, path=None, id="1"):
        self.algorithm.save_model(path=path, id=id)

    def load_model(self, path=None, id="1"):
        model_file_path = Path(path or "./ckpt/dump_model") / f"model.ckpt-{id}.pkl"
        checkpoint = torch.load(model_file_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Loaded actor model %s from %s", id, model_file_path)
