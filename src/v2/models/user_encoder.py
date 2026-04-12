"""
User encoder for the V2 rebuild.

Consumes Stage 1 listening survey histories and produces a compact user/taste
representation plus a rating score for candidate songs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.data.schema import SONG_EMB_DIM, USER_EMB_DIM


class UserEncoder(nn.Module):
    def __init__(
        self,
        song_emb_dim: int = SONG_EMB_DIM,
        user_dim: int = USER_EMB_DIM,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.item_proj = nn.Sequential(
            nn.Linear(song_emb_dim + 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.user_head = nn.Sequential(
            nn.Linear(hidden + song_emb_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, user_dim),
        )
        self.pref_affect_head = nn.Sequential(
            nn.Linear(user_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(user_dim, hidden),
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
        hist_song_emb: torch.Tensor,
        hist_rating: torch.Tensor,
        hist_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rating_feat = torch.cat([hist_rating, torch.abs(hist_rating)], dim=-1)
        item_input = torch.cat([hist_song_emb, rating_feat], dim=-1)
        item_hidden = self.item_proj(item_input)

        attn_logits = self.attn(item_hidden).squeeze(-1)
        attn_logits = attn_logits.masked_fill(hist_mask.squeeze(-1) <= 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        attn_weights = attn_weights * hist_mask
        attn_weights = attn_weights / attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        pooled_hidden = torch.sum(attn_weights * item_hidden, dim=1)
        pooled_song = torch.sum(attn_weights * hist_song_emb, dim=1)

        user_input = torch.cat([pooled_hidden, pooled_song], dim=-1)
        u_user = self.user_head(user_input)
        taste_affect = self.pref_affect_head(u_user)
        user_conf = self.conf_head(u_user)

        return {
            "u_user": u_user,
            "taste_affect": taste_affect,
            "user_conf": user_conf,
            "attn_weights": attn_weights,
        }


class UserPreferenceModel(nn.Module):
    def __init__(
        self,
        song_emb_dim: int = SONG_EMB_DIM,
        user_dim: int = USER_EMB_DIM,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = UserEncoder(song_emb_dim=song_emb_dim, user_dim=user_dim, hidden=hidden)
        self.rating_head = nn.Sequential(
            nn.Linear(user_dim + song_emb_dim + 2 + 3, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.rating_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hist_song_emb: torch.Tensor,
        hist_rating: torch.Tensor,
        hist_mask: torch.Tensor,
        candidate_song_emb: torch.Tensor,
        candidate_song_affect: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        user_outputs = self.encoder(hist_song_emb, hist_rating, hist_mask)
        user_vec = user_outputs["u_user"]
        user_conf = user_outputs["user_conf"]

        cosine = F.cosine_similarity(user_vec, candidate_song_emb[:, : user_vec.shape[1]], dim=-1).unsqueeze(-1)
        dot = torch.sum(user_vec * candidate_song_emb[:, : user_vec.shape[1]], dim=-1, keepdim=True) / max(user_vec.shape[1], 1)
        rating_input = torch.cat(
            [
                user_vec,
                candidate_song_emb,
                candidate_song_affect,
                cosine,
                dot,
                user_conf,
            ],
            dim=-1,
        )
        pred_rating = self.rating_head(rating_input)
        return {
            **user_outputs,
            "pred_rating": pred_rating,
        }
