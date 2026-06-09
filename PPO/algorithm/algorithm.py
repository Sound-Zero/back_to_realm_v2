"""PPO learner implementation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from PPO.conf.conf import Config
from PPO.model.model import NetworkModelLearner


class Algorithm:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.model = NetworkModelLearner().to(self.device)
        self.lr = Config.START_LR
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        self.label_size = Config.ACTION_SPACE_SIZE
        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.99,
            patience=2500,
            min_lr=Config.END_LR,
        )
        self.learn_cnt = 0

    def learn(self, list_sample_data):
        if not list_sample_data:
            return None

        self.learn_cnt += 1
        if self.learn_cnt % Config.UPDATE_FREQ == 0:
            self.save_model("./ckpt/dump_model", "latest")

        self.model.train()
        self.optimizer.zero_grad()

        list_npdata = [torch.as_tensor(sample_data.npdata, device=self.device) for sample_data in list_sample_data]
        input_datas = torch.stack(list_npdata, dim=0)
        data_list = self.model.format_data(input_datas)
        rst_list = self.model(data_list)
        total_loss, info_list = self.compute_loss(data_list, rst_list)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 0.5)
        self.optimizer.step()
        self.scheduler.step(total_loss.detach())

        metrics = self._build_metrics(total_loss, info_list)
        self._append_metrics(metrics)
        return metrics

    def _build_metrics(self, total_loss, info_list):
        clean_info = []
        for info in info_list:
            clean_info.append(info.detach().mean().cpu().item() if torch.is_tensor(info) else info)

        return {
            "timestamp": time.time(),
            "learn_cnt": self.learn_cnt,
            "total_loss": total_loss.detach().cpu().item(),
            "tdret_mean": clean_info[0],
            "value_loss": clean_info[1],
            "policy_loss": clean_info[2],
            "entropy_loss": clean_info[3],
            "clip_frac": clean_info[4] if len(clean_info) > 4 else 0.0,
            "adv_mean": clean_info[5],
            "adv_std": clean_info[6],
            "reward_mean": clean_info[7],
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _append_metrics(self, metrics, metrics_file="metrics.json") -> None:
        path = Path(metrics_file)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = []
        else:
            data = []
        data.append(metrics)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def compute_loss(self, data_list, rst_list):
        (
            _feature,
            reward,
            old_value,
            tdret,
            adv,
            old_action,
            old_prob,
            _legal_action,
        ) = data_list

        value = rst_list[1].squeeze(1)
        old_value = old_value.squeeze(1)
        tdret = tdret.squeeze(1)
        adv = adv.squeeze(1)
        old_prob = old_prob.clamp_min(1e-9)

        value_clip = old_value + (value - old_value).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum(torch.square(tdret - value_clip), torch.square(tdret - value)).mean()

        prob = rst_list[0].clamp(1e-9, 1.0)
        entropy_loss = (-prob * torch.log(prob)).sum(1).mean()

        one_hot_action = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size)
        new_prob = (one_hot_action * prob).sum(1, keepdim=True)
        ratio = (new_prob / old_prob).squeeze(1)
        clip_frac = (ratio - 1.0).abs().gt(self.clip_param).float().mean()
        policy_loss = torch.maximum(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        total_loss = value_loss * self.vf_coef + policy_loss - self.var_beta * entropy_loss
        self.logger.info(
            "loss total=%.6f value=%.6f policy=%.6f entropy=%.6f",
            total_loss.detach().cpu().item(),
            value_loss.detach().cpu().item(),
            policy_loss.detach().cpu().item(),
            entropy_loss.detach().cpu().item(),
        )
        info_list = [tdret.mean(), value_loss, policy_loss, entropy_loss, clip_frac, adv.mean(), adv.std(), reward.mean()]
        return total_loss, info_list

    def load_model(self, path=None, id="1"):
        model_file_path = Path(path or "./ckpt/dump_model") / f"model.ckpt-{id}.pkl"
        checkpoint = torch.load(model_file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Loaded model %s from %s", id, model_file_path)

    def save_model(self, path=None, id="1"):
        save_dir = Path(path or "./ckpt/dump_model")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = save_dir / f"model.ckpt-{id}.pkl"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            model_file_path,
        )
        self.logger.info("Saved model %s to %s", id, model_file_path)
