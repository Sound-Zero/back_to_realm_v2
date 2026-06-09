"""Neural network models used by the actor and learner."""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

from PPO.conf.conf import Config


try:
    if os.path.basename(sys.argv[0]) == "learner.py":
        torch.set_num_interop_threads(2)
        torch.set_num_threads(2)
    else:
        torch.set_num_interop_threads(4)
        torch.set_num_threads(4)
except RuntimeError:
    # PyTorch only allows thread settings before parallel work starts.
    pass


def split_raw_feat(x: torch.Tensor):
    """Split the configured flat feature tensor into semantic groups."""

    batch = x.size(0)
    ptr = 0

    cur_pos_norm = x[:, ptr : ptr + 2]
    ptr += 2
    one_hot_map = x[:, ptr : ptr + 605].view(batch, 5, 11, 11)
    ptr += 605
    local_map = x[:, ptr : ptr + 121].view(batch, 1, 11, 11)
    ptr += 121
    memory_flag = x[:, ptr : ptr + 121].view(batch, 1, 11, 11)
    ptr += 121
    end_pos_feat = x[:, ptr : ptr + 6]
    ptr += 6
    hist_pos_feat = x[:, ptr : ptr + 6]
    ptr += 6
    treasure_state = x[:, ptr : ptr + 13]
    ptr += 13
    treasure_pos = x[:, ptr : ptr + 78].view(batch, 13, 6)
    ptr += 78
    buff_feat = x[:, ptr : ptr + 8]
    ptr += 8
    talent_feat = x[:, ptr : ptr + 3]
    ptr += 3
    time_ratio = x[:, ptr : ptr + 1]
    ptr += 1

    if ptr != Config.FEATURE_LEN:
        raise ValueError(f"Consumed {ptr} feature values, expected {Config.FEATURE_LEN}")

    spatial_features = {
        "cur_pos_norm": cur_pos_norm,
        "local_maps": torch.cat([one_hot_map, local_map, memory_flag], dim=1),
        "end_pos_feat": end_pos_feat,
        "hist_pos_feat": hist_pos_feat,
    }
    temporal_features = {
        "memory_map": memory_flag,
        "talent_cooldown": talent_feat[:, 2:3],
        "buff_time": buff_feat[:, 6:7],
        "time_ratio": time_ratio,
    }
    entity_features = {
        "treasures": torch.cat([treasure_pos, treasure_state.unsqueeze(-1)], dim=-1),
        "buff": buff_feat,
        "talent": talent_feat,
    }
    return spatial_features, temporal_features, entity_features


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, height: int = 11, width: int = 11):
        super().__init__()
        pe = torch.zeros(d_model, height, width)
        y_pos = torch.arange(height).unsqueeze(1).float()
        x_pos = torch.arange(width).unsqueeze(0).float()
        div_term = torch.exp(torch.arange(0, d_model, 4).float() * -(math.log(10000.0) / d_model))
        pe[0::4] = torch.sin(x_pos * div_term.view(-1, 1, 1))
        pe[1::4] = torch.cos(x_pos * div_term.view(-1, 1, 1))
        pe[2::4] = torch.sin(y_pos * div_term.view(-1, 1, 1))
        pe[3::4] = torch.cos(y_pos * div_term.view(-1, 1, 1))
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe


class SpatialAttentionCNN(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.terrain_conv = nn.Conv2d(5, d_model // 2, 3, padding=1)
        self.local_conv = nn.Conv2d(1, d_model // 4, 3, padding=1)
        self.memory_conv = nn.Conv2d(1, d_model // 4, 3, padding=1)
        self.fusion_conv = nn.Conv2d(d_model, d_model, 1)
        self.pos_encoding = PositionalEncoding2D(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, local_maps):
        batch = local_maps.shape[0]
        terrain_feat = F.relu(self.terrain_conv(local_maps[:, :5]))
        local_feat = F.relu(self.local_conv(local_maps[:, 5:6]))
        memory_feat = F.relu(self.memory_conv(local_maps[:, 6:7]))
        fused = F.relu(self.fusion_conv(torch.cat([terrain_feat, local_feat, memory_feat], dim=1)))
        fused = self.pos_encoding(fused)
        fused_flat = fused.view(batch, fused.size(1), -1).transpose(1, 2)
        attn_out, _ = self.self_attn(fused_flat, fused_flat, fused_flat)
        return self.norm(attn_out + fused_flat).mean(dim=1)


class PositionEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.found_embed = nn.Linear(1, d_model // 4)
        self.direction_embed = nn.Linear(2, d_model // 2)
        self.target_pos_embed = nn.Linear(2, d_model // 2)
        self.distance_embed = nn.Linear(1, d_model // 4)
        self.fusion = nn.Sequential(nn.Linear(d_model + d_model // 2, d_model), nn.LayerNorm(d_model), nn.ReLU())

    def forward(self, pos_feat):
        combined = torch.cat(
            [
                self.found_embed(pos_feat[:, 0:1]),
                self.direction_embed(pos_feat[:, 1:3]),
                self.target_pos_embed(pos_feat[:, 3:5]),
                self.distance_embed(pos_feat[:, 5:6]),
            ],
            dim=-1,
        )
        return self.fusion(combined)


class TreasureEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.pos_encoder = PositionEncoder(d_model)
        self.state_embed = nn.Embedding(2, d_model)
        self.treasure_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, treasure_feat):
        batch, count, _ = treasure_feat.shape
        pos_encoded = self.pos_encoder(treasure_feat[:, :, :6].reshape(-1, 6)).view(batch, count, -1)
        state_feat = treasure_feat[:, :, 6].clamp(0, 1).long()
        treasure_encoded = pos_encoded + self.state_embed(state_feat)
        attn_out, _ = self.treasure_attn(treasure_encoded, treasure_encoded, treasure_encoded)
        return self.norm(attn_out + treasure_encoded)


class BuffEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.pos_encoder = PositionEncoder(d_model // 2)
        self.time_embed = nn.Linear(1, d_model // 4)
        self.count_embed = nn.Linear(1, d_model // 4)
        self.fusion = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())

    def forward(self, buff_feat):
        return self.fusion(
            torch.cat(
                [
                    self.pos_encoder(buff_feat[:, :6]),
                    self.time_embed(buff_feat[:, 6:7]),
                    self.count_embed(buff_feat[:, 7:8]),
                ],
                dim=-1,
            )
        )


class TalentEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.use_count_embed = nn.Linear(1, d_model // 4)
        self.available_embed = nn.Linear(1, d_model // 4)
        self.cooldown_embed = nn.Linear(1, d_model // 2)
        self.fusion = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())

    def forward(self, talent_feat):
        return self.fusion(
            torch.cat(
                [
                    self.use_count_embed(talent_feat[:, 0:1]),
                    self.available_embed(talent_feat[:, 1:2]),
                    self.cooldown_embed(talent_feat[:, 2:3]),
                ],
                dim=-1,
            )
        )


class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, temporal_features):
        memory_avg = temporal_features["memory_map"].mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        time_feat = torch.cat(
            [
                temporal_features["talent_cooldown"],
                temporal_features["buff_time"],
                memory_avg,
                temporal_features["time_ratio"],
            ],
            dim=-1,
        )
        return self.time_mlp(time_feat)


class ImprovedPolicyNet(nn.Module):
    def __init__(self, action_dim: int = Config.ACTION_SPACE_SIZE, d_model: int = 64):
        super().__init__()
        self.spatial_encoder = SpatialAttentionCNN(d_model)
        self.position_encoder = PositionEncoder(d_model)
        self.treasure_encoder = TreasureEncoder(d_model)
        self.temporal_encoder = TemporalEncoder(d_model)
        self.buff_encoder = BuffEncoder(d_model)
        self.talent_encoder = TalentEncoder(d_model)
        self.cur_pos_encoder = nn.Sequential(nn.Linear(2, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 8, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(d_model * 2, action_dim)
        self.value_head = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        spatial_feat, temporal_feat, entity_feat = split_raw_feat(x)
        feature_sequence = torch.stack(
            [
                self.cur_pos_encoder(spatial_feat["cur_pos_norm"]),
                self.spatial_encoder(spatial_feat["local_maps"]),
                self.position_encoder(spatial_feat["end_pos_feat"]),
                self.position_encoder(spatial_feat["hist_pos_feat"]),
                self.treasure_encoder(entity_feat["treasures"]).mean(1),
                self.buff_encoder(entity_feat["buff"]),
                self.talent_encoder(entity_feat["talent"]),
                self.temporal_encoder(temporal_feat),
            ],
            dim=1,
        )
        decoded = self.decoder(feature_sequence.reshape(feature_sequence.size(0), -1))
        return self.policy_head(decoded), self.value_head(decoded)


class NetworkModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE
        self.label_size = Config.ACTION_SPACE_SIZE
        self.feature_len = Config.MLP_FEATURE_LEN
        self.value_num = Config.VALUE_NUM
        self.data_len = Config.data_len
        self.label_net = ImprovedPolicyNet(action_dim=self.label_size, d_model=64)

    def process_legal_action(self, label, legal_action):
        legal_action = legal_action.float()
        if legal_action.sum(dim=1).eq(0).any():
            legal_action = torch.where(legal_action.sum(dim=1, keepdim=True).eq(0), torch.ones_like(legal_action), legal_action)
        return label.masked_fill(legal_action <= 0, -1e9)

    def forward(self, feature, legal_action):
        label_net_out, value = self.label_net(feature)
        label_out = self.process_legal_action(label_net_out, legal_action)
        prob = torch.softmax(label_out, dim=1)
        return prob, value


class NetworkModelActor(NetworkModelBase):
    def format_data(self, obs, legal_action):
        return torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(legal_action, dtype=torch.float32)


class NetworkModelLearner(NetworkModelBase):
    def format_data(self, datas):
        return datas.view(-1, self.data_len).float().split(self.data_split_shape, dim=1)

    def forward(self, data_list, inference: bool = False):
        feature = data_list[0]
        legal_action = data_list[-1]
        return super().forward(feature, legal_action)


def make_fc_layer(in_features: int, out_features: int):
    fc_layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)
    return fc_layer
