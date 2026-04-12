"""
Anchor encoder for mapping supervised SiTunes interventions into a retrieval space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.v2.data.anchor_features import ANCHOR_FEATURE_DIM
from src.v2.data.schema import SONG_EMB_DIM


class AnchorEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = ANCHOR_FEATURE_DIM,
        out_dim: int = SONG_EMB_DIM,
        hidden: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, anchor_features: torch.Tensor) -> torch.Tensor:
        return self.net(anchor_features)
