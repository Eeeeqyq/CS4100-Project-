"""
Song encoder for the V2 rebuild.

Learns a shared embedding from static song metadata plus optional dynamic
valence/arousal trajectories.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.v2.data.schema import SONG_DYN_DIM, SONG_EMB_DIM, SONG_STATIC_DIM


class SongEncoder(nn.Module):
    def __init__(
        self,
        static_dim: int = SONG_STATIC_DIM,
        dyn_dim: int = SONG_DYN_DIM,
        emb_dim: int = SONG_EMB_DIM,
        hidden: int = 128,
        dyn_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.static_tower = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, emb_dim),
            nn.GELU(),
            nn.LayerNorm(emb_dim),
        )

        self.dynamic_encoder = nn.Sequential(
            nn.Conv1d(dyn_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, dyn_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dyn_hidden, dyn_hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.dynamic_proj = nn.Sequential(
            nn.Linear(dyn_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.dynamic_recon = nn.Sequential(
            nn.Conv1d(dyn_hidden, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, dyn_dim, kernel_size=1),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(emb_dim)

        self.static_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_song_static: torch.Tensor,
        x_song_dyn: torch.Tensor,
        x_song_dyn_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        static_emb = self.static_tower(x_song_static)

        dyn_input = x_song_dyn.transpose(1, 2)
        dyn_mask = x_song_dyn_mask.transpose(1, 2)
        dyn_hidden = self.dynamic_encoder(dyn_input * dyn_mask)
        dyn_hidden = dyn_hidden * dyn_mask

        denom = dyn_mask.sum(dim=2).clamp(min=1.0)
        dyn_pooled = dyn_hidden.sum(dim=2) / denom
        dyn_emb = self.dynamic_proj(dyn_pooled)

        dyn_presence = (x_song_dyn_mask.mean(dim=(1, 2), keepdim=False) > 0).float().unsqueeze(1)
        gate = self.fusion_gate(torch.cat([static_emb, dyn_emb, dyn_presence], dim=1))
        embedding = self.fusion_norm(static_emb + gate * dyn_emb)

        static_affect_hat = self.static_head(embedding)
        song_quality = self.quality_head(embedding)
        song_dyn_hat = self.dynamic_recon(dyn_hidden).transpose(1, 2)

        return {
            "embedding": embedding,
            "song_affect_hat": static_affect_hat,
            "song_quality": song_quality,
            "song_dyn_hat": song_dyn_hat,
            "dyn_gate": gate,
        }
