"""
Utility reranker for the V2 rebuild.

Scores candidate songs given context, user, and goal-conditioned target state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UtilityReranker(nn.Module):
    def __init__(
        self,
        feature_dim: int = 346,
        hidden: int = 256,
        dropout: float = 0.15,
        alpha: float = 0.7,
        beta: float = 0.3,
        benefit_range: float = 1.5,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.benefit_range = float(benefit_range)
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.benefit_head = nn.Linear(hidden, 1)
        self.accept_pref_head = nn.Linear(hidden, 1)
        self.accept_rating_head = nn.Linear(hidden, 1)
        self.relevance_head = nn.Linear(hidden, 1)
        self.register_buffer("benefit_cal_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("benefit_cal_bias", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("accept_pref_cal_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("accept_pref_cal_bias", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("accept_rating_cal_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("accept_rating_cal_bias", torch.tensor(0.0, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_pair: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, cand, feat = x_pair.shape
        flat = x_pair.view(batch * cand, feat)
        hidden = self.backbone(flat)
        benefit_raw = self.benefit_range * torch.tanh(self.benefit_head(hidden))
        accept_pref_raw = torch.tanh(self.accept_pref_head(hidden))
        accept_rating_raw = torch.tanh(self.accept_rating_head(hidden))
        relevance = self.relevance_head(hidden).view(batch, cand)
        benefit = torch.clamp(
            benefit_raw * self.benefit_cal_scale + self.benefit_cal_bias,
            -self.benefit_range,
            self.benefit_range,
        ).view(batch, cand)
        accept_pref = torch.clamp(
            accept_pref_raw * self.accept_pref_cal_scale + self.accept_pref_cal_bias,
            -1.0,
            1.0,
        ).view(batch, cand)
        accept_rating = torch.clamp(
            accept_rating_raw * self.accept_rating_cal_scale + self.accept_rating_cal_bias,
            -1.0,
            1.0,
        ).view(batch, cand)
        acceptance = 0.5 * (accept_pref + accept_rating)
        utility = self.alpha * torch.clamp(benefit / self.benefit_range, -1.0, 1.0) + self.beta * acceptance
        return {
            "benefit_hat": benefit,
            "accept_pref_hat": accept_pref,
            "accept_rating_hat": accept_rating,
            "accept_hat": acceptance,
            "utility_hat": utility,
            "relevance_logit": relevance,
        }

    def set_calibration(
        self,
        benefit_scale: float,
        benefit_bias: float,
        accept_pref_scale: float,
        accept_pref_bias: float,
        accept_rating_scale: float,
        accept_rating_bias: float,
    ) -> None:
        self.benefit_cal_scale.fill_(float(benefit_scale))
        self.benefit_cal_bias.fill_(float(benefit_bias))
        self.accept_pref_cal_scale.fill_(float(accept_pref_scale))
        self.accept_pref_cal_bias.fill_(float(accept_pref_bias))
        self.accept_rating_cal_scale.fill_(float(accept_rating_scale))
        self.accept_rating_cal_bias.fill_(float(accept_rating_bias))
