"""
Goal-conditioned query tower for the V2 rebuild.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.v2.data.schema import CTX_EMB_DIM, GOAL_DIM, SONG_EMB_DIM, TAU_DIM, USER_EMB_DIM


class QueryTower(nn.Module):
    def __init__(
        self,
        ctx_dim: int = CTX_EMB_DIM,
        user_dim: int = USER_EMB_DIM,
        goal_dim: int = GOAL_DIM,
        tau_dim: int = TAU_DIM,
        out_dim: int = SONG_EMB_DIM,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + user_dim + goal_dim + tau_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        z_ctx: torch.Tensor,
        u_user: torch.Tensor,
        goal_onehot: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([z_ctx, u_user, goal_onehot, tau], dim=-1))
