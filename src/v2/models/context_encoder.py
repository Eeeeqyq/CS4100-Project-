"""
Context encoder for the V2 rebuild.

Encodes wrist sequence, environment, and optional self-report into a shared
context embedding plus auxiliary predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.v2.data.schema import CTX_EMB_DIM, ENV_DIM, SELF_DIM, WRIST_DIM


class ContextEncoder(nn.Module):
    def __init__(
        self,
        wrist_dim: int = WRIST_DIM,
        env_dim: int = ENV_DIM,
        self_dim: int = SELF_DIM,
        emb_dim: int = CTX_EMB_DIM,
        hidden: int = 128,
        rnn_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.wrist_gru = nn.GRU(
            input_size=wrist_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.wrist_attn = nn.Sequential(
            nn.Linear(rnn_hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.self_proj = nn.Sequential(
            nn.Linear(self_dim, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
        )
        self.sensor_fuse = nn.Sequential(
            nn.Linear(rnn_hidden * 2 + hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
        )
        self.full_fuse = nn.Sequential(
            nn.Linear(emb_dim + hidden // 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.pre_affect_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.movement_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.unc_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x_wrist: torch.Tensor,
        x_env: torch.Tensor,
        x_self: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        wrist_hidden, _ = self.wrist_gru(x_wrist)
        attn_logits = self.wrist_attn(wrist_hidden)
        attn_weights = torch.softmax(attn_logits, dim=1)
        wrist_pooled = torch.sum(attn_weights * wrist_hidden, dim=1)

        env_feat = self.env_proj(x_env)
        self_feat = self.self_proj(x_self)

        sensor_input = torch.cat([wrist_pooled, env_feat], dim=-1)
        z_sensor = self.sensor_fuse(sensor_input)
        z_ctx = self.full_fuse(torch.cat([z_sensor, self_feat], dim=-1))

        pre_affect_hat = self.pre_affect_head(z_sensor)
        movement_logits = self.movement_head(z_sensor)
        ctx_unc = self.unc_head(z_sensor)

        return {
            "z_ctx": z_ctx,
            "z_sensor": z_sensor,
            "pre_affect_hat": pre_affect_hat,
            "movement_logits": movement_logits,
            "ctx_unc": ctx_unc,
            "attn_weights": attn_weights,
        }
