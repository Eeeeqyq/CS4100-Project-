"""
Experimental end-to-end inference for V2.2.

Contract:
- explicit goal is the primary inference mode
- retrieve train-only SiTunes anchors
- rerank anchors by benefit, acceptance, and uncertainty
- expand into public songs only when anchor-conditioned transfer support is strong
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v2.data.anchor_features import (
    build_row_user_views,
    build_stage1_acceptance_features,
    latent_acceptance,
    normalize_rows,
)
from src.v2.data.schema import GOAL_NAMES, Goal
from src.v2.data.targets import GOAL_SPECS
from src.v2.inference.anchor_pipeline import build_anchor_pair_features, build_train_anchor_views
from src.v2.inference.final_rank import build_anchor_final_score, pmemo_dynamic_bonus, pmemo_dynamic_reason
from src.v2.models.anchor_encoder import AnchorEncoder
from src.v2.models.query_tower import QueryTower
from src.v2.models.reranker import UtilityReranker


REBUILD_DIR = PROJECT_ROOT / "data" / "processed" / "rebuild"
MODELS_DIR = PROJECT_ROOT / "models" / "rebuild"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


class AmbientRecommenderV2:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.decision_df = pd.read_parquet(REBUILD_DIR / "decision_table.parquet")
        self.anchor_df = pd.read_parquet(REBUILD_DIR / "anchor_table.parquet")
        self.song_catalog = pd.read_parquet(REBUILD_DIR / "song_catalog.parquet")
        self.stage1_df = pd.read_parquet(REBUILD_DIR / "stage1_history_table.parquet")
        self.context_embeddings = np.load(REBUILD_DIR / "context_embeddings.npy").astype(np.float32)
        self.song_embeddings = np.load(REBUILD_DIR / "song_embeddings.npy").astype(np.float32)
        self.query_embeddings_bootstrap = np.load(REBUILD_DIR / "query_embeddings.npy").astype(np.float32)
        self.user_out = np.load(REBUILD_DIR / "user_encoder_outputs.npz")
        self.song_pred = pd.read_parquet(REBUILD_DIR / "song_encoder_predictions.parquet")
        self.anchor_index_artifact = np.load(REBUILD_DIR / "situnes_anchor_index.npz", allow_pickle=True)

        self.row_user_embeddings, self.row_user_conf, _ = build_row_user_views(self.decision_df, self.user_out)
        self.stage1_exact_rating, self.stage1_user_prior, self.stage1_global_prior = build_stage1_acceptance_features(self.stage1_df)

        self.song_sources = self.song_catalog["source"].astype(str).to_numpy()
        self.song_ids = self.song_catalog["song_id"].astype(str).to_numpy()
        self.song_titles = self.song_catalog["title"].astype(str).to_numpy()
        self.song_artists = self.song_catalog["artist"].astype(str).to_numpy()
        self.song_quality = self.song_catalog["song_quality"].to_numpy(dtype=np.float32)
        self.song_affect = self.song_pred[["pred_valence", "pred_arousal"]].to_numpy(dtype=np.float32)
        self.song_dyn_summary = self.song_catalog[
            [
                "dyn_valence_delta",
                "dyn_arousal_delta",
                "dyn_valence_vol",
                "dyn_arousal_vol",
                "dyn_arousal_peak",
                "eda_impact_norm",
            ]
        ].to_numpy(dtype=np.float32)
        self.song_dyn_quality = self.song_catalog["song_quality"].to_numpy(dtype=np.float32)

        self.query_model, self.anchor_model = self._load_query_stack()
        self.reranker = self._load_reranker()

        self.train_anchor_idx = self.anchor_index_artifact["anchor_global_idx"].astype(np.int64)
        self.anchor_embeddings = np.load(REBUILD_DIR / "anchor_query_embeddings.npy").astype(np.float32)
        self.train_anchor_views = build_train_anchor_views(
            anchor_df=self.anchor_df,
            train_anchor_idx=self.train_anchor_idx,
            context_embeddings=self.context_embeddings,
            row_user_embeddings=self.row_user_embeddings,
            song_embeddings=self.song_embeddings,
            song_affect=self.song_affect,
            song_dyn_summary=self.song_dyn_summary,
        )

        self.public_song_idx = np.where(self.song_sources != "situnes")[0].astype(np.int64)
        self.public_norm_song_emb = normalize_rows(self.song_embeddings[self.public_song_idx])
        self.public_song_affect = self.song_affect[self.public_song_idx]
        self.public_song_dyn = self.song_dyn_summary[self.public_song_idx]
        self.public_song_quality = self.song_quality[self.public_song_idx]
        self.public_song_sources = self.song_sources[self.public_song_idx]

        # Calibrated on the validation split to prevent post-retrain public-source collapse.
        self.transfer_support_threshold = 0.72
        self.transfer_anchor_k = 3
        self.public_candidate_k = 80

    def _load_query_stack(self) -> tuple[QueryTower, AnchorEncoder]:
        checkpoint = torch.load(MODELS_DIR / "query_tower.pt", map_location=self.device)
        query_model = QueryTower().to(self.device)
        query_model.load_state_dict(checkpoint["query_model"])
        query_model.eval()
        anchor_model = AnchorEncoder().to(self.device)
        anchor_model.load_state_dict(checkpoint["anchor_model"])
        anchor_model.eval()
        return query_model, anchor_model

    def _load_reranker(self) -> UtilityReranker:
        checkpoint = torch.load(MODELS_DIR / "reranker.pt", map_location=self.device)
        config = checkpoint.get("config", {})
        model = UtilityReranker(
            feature_dim=int(checkpoint.get("feature_dim", 346)),
            alpha=float(config.get("alpha", 0.7)),
            beta=float(config.get("beta", 0.3)),
            benefit_range=float(config.get("benefit_range", 1.5)),
        ).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model

    @staticmethod
    def goal_name(goal_idx: int) -> str:
        return GOAL_NAMES.get(int(goal_idx), f"goal_{int(goal_idx)}")

    @staticmethod
    def _generic_public_reason(goal_idx: int) -> str:
        goal_name = GOAL_NAMES.get(int(goal_idx), "focus")
        if goal_name == "focus":
            return "supported by similar successful focus anchors"
        if goal_name == "wind_down":
            return "supported by similar successful wind-down anchors"
        if goal_name == "uplift":
            return "supported by similar successful uplift anchors"
        if goal_name == "movement":
            return "supported by similar successful movement anchors"
        return "supported by similar successful anchors"

    def _tau_for_goal(self, goal_idx: int, default_tau: np.ndarray) -> np.ndarray:
        if goal_idx < 0:
            return np.asarray(default_tau, dtype=np.float32)
        spec = GOAL_SPECS[Goal(int(goal_idx))]
        return np.asarray([spec.tau_valence, spec.tau_arousal], dtype=np.float32)

    def _compute_queries(
        self,
        row_indices: np.ndarray,
        goal_idx: np.ndarray,
        tau: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        goal_onehot = np.eye(4, dtype=np.float32)[goal_idx.astype(np.int64)]
        outs = []
        with torch.no_grad():
            for start in range(0, len(row_indices), batch_size):
                end = min(start + batch_size, len(row_indices))
                query = self.query_model(
                    torch.from_numpy(self.context_embeddings[row_indices[start:end]]).to(self.device),
                    torch.from_numpy(self.row_user_embeddings[row_indices[start:end]]).to(self.device),
                    torch.from_numpy(goal_onehot[start:end]).to(self.device),
                    torch.from_numpy(tau[start:end]).to(self.device),
                )
                outs.append(F.normalize(query, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def _anchor_candidates(
        self,
        query_embeddings: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sims = np.asarray(query_embeddings, dtype=np.float32) @ np.asarray(self.anchor_embeddings, dtype=np.float32).T
        k = min(int(k), sims.shape[1])
        top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        top_score = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_score, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        top_score = np.take_along_axis(top_score, order, axis=1)
        return top_idx.astype(np.int32), top_score.astype(np.float32)

    def _score_anchor_rows(
        self,
        row_indices: np.ndarray,
        goal_idx: np.ndarray,
        tau: np.ndarray,
        candidate_k: int,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        query_embeddings = self._compute_queries(row_indices=row_indices, goal_idx=goal_idx, tau=tau, batch_size=batch_size)
        candidate_local_idx, candidate_score = self._anchor_candidates(query_embeddings=query_embeddings, k=candidate_k)
        pair_features, diagnostics = build_anchor_pair_features(
            row_indices=row_indices,
            query_embeddings=query_embeddings,
            candidate_local_idx=candidate_local_idx,
            candidate_score=candidate_score,
            decision_df=self.decision_df,
            train_anchor_views=self.train_anchor_views,
            row_user_embeddings=self.row_user_embeddings,
            stage1_exact_rating=self.stage1_exact_rating,
            stage1_user_prior=self.stage1_user_prior,
            stage1_global_prior=self.stage1_global_prior,
            anchor_embeddings=self.anchor_embeddings,
        )

        benefit_hat = []
        pref_hat = []
        rating_hat = []
        relevance = []
        with torch.no_grad():
            pair_tensor = torch.from_numpy(pair_features)
            for start in range(0, len(pair_tensor), batch_size):
                end = min(start + batch_size, len(pair_tensor))
                outputs = self.reranker(pair_tensor[start:end].to(self.device))
                benefit_hat.append(outputs["benefit_hat"].cpu().numpy())
                pref_hat.append(outputs["accept_pref_hat"].cpu().numpy())
                rating_hat.append(outputs["accept_rating_hat"].cpu().numpy())
                relevance.append(outputs["relevance_logit"].cpu().numpy())

        benefit_hat_arr = np.concatenate(benefit_hat, axis=0)
        pref_hat_arr = np.concatenate(pref_hat, axis=0)
        rating_hat_arr = np.concatenate(rating_hat, axis=0)
        relevance_arr = np.concatenate(relevance, axis=0)
        latent_accept = latent_acceptance(
            accept_pref_hat=pref_hat_arr,
            accept_rating_hat=rating_hat_arr,
            user_prior=pair_features[:, :, -3],
            exact_rating=pair_features[:, :, -2],
            exact_mask=pair_features[:, :, -1],
        )
        anchor_final = build_anchor_final_score(
            relevance_logit=relevance_arr,
            benefit_hat=benefit_hat_arr,
            latent_accept_hat=latent_accept,
            uncertainty=diagnostics["uncertainty"],
        )
        anchor_strength = _sigmoid(anchor_final)
        order = np.argsort(-anchor_final, axis=1)

        return {
            "row_indices": np.asarray(row_indices, dtype=np.int64),
            "goal_idx": goal_idx.astype(np.int64),
            "tau": tau.astype(np.float32),
            "query_embeddings": query_embeddings.astype(np.float32),
            "candidate_local_idx": candidate_local_idx.astype(np.int32),
            "candidate_score": candidate_score.astype(np.float32),
            "benefit_hat": benefit_hat_arr.astype(np.float32),
            "accept_pref_hat": pref_hat_arr.astype(np.float32),
            "accept_rating_hat": rating_hat_arr.astype(np.float32),
            "latent_accept_hat": latent_accept.astype(np.float32),
            "relevance_logit": relevance_arr.astype(np.float32),
            "uncertainty": diagnostics["uncertainty"].astype(np.float32),
            "anchor_final_score": anchor_final.astype(np.float32),
            "anchor_strength": anchor_strength.astype(np.float32),
            "anchor_order": order.astype(np.int32),
        }

    def _public_transfer_for_row(
        self,
        row_global_idx: int,
        goal_idx: int,
        tau: np.ndarray,
        ranked_anchor_local: np.ndarray,
        ranked_anchor_strength: np.ndarray,
    ) -> tuple[list[dict[str, object]], float, float]:
        top_local = ranked_anchor_local[: self.transfer_anchor_k]
        top_strength = np.asarray(ranked_anchor_strength[: self.transfer_anchor_k], dtype=np.float32)
        if len(top_local) == 0:
            return [], 0.0, 0.0

        weights = np.exp(top_strength - np.max(top_strength))
        weights = weights / np.clip(weights.sum(), 1e-6, None)
        anchor_song_emb = self.train_anchor_views["anchor_song_emb"][top_local]
        centroid = normalize_rows(np.sum(anchor_song_emb * weights[:, None], axis=0, keepdims=True))[0]
        emb_support = self.public_norm_song_emb @ centroid
        top_pub_local = np.argpartition(-emb_support, kth=min(self.public_candidate_k, len(emb_support)) - 1)[: self.public_candidate_k]
        top_pub_scores = emb_support[top_pub_local]
        order = np.argsort(-top_pub_scores)
        top_pub_local = top_pub_local[order]
        top_pub_scores = top_pub_scores[order]

        anchor_support_strength = float(np.sum(weights * top_strength))
        user_emb = normalize_rows(self.row_user_embeddings[row_global_idx][None, :])[0]
        tau = np.asarray(tau, dtype=np.float32)

        public_items: list[dict[str, object]] = []
        max_support = 0.0
        for local_idx, emb_score in zip(top_pub_local.tolist(), top_pub_scores.tolist()):
            song_idx = int(self.public_song_idx[int(local_idx)])
            song_emb = self.song_embeddings[song_idx]
            song_affect = 2.0 * self.song_affect[song_idx] - 1.0
            affect_dist = 0.5 * float((song_affect[0] - tau[0]) ** 2 + (song_affect[1] - tau[1]) ** 2)
            affect_fit = float(1.0 - np.clip(affect_dist / 3.0, 0.0, 1.0))
            taste_fit = float(np.clip((np.dot(user_emb, normalize_rows(song_emb[: user_emb.shape[0]][None, :])[0]) + 1.0) / 2.0, 0.0, 1.0))
            dyn = self.song_dyn_summary[song_idx]
            dyn_bonus = float(
                pmemo_dynamic_bonus(
                    goal_idx=goal_idx,
                    dyn_valence_delta=np.asarray([dyn[0]], dtype=np.float32),
                    dyn_arousal_delta=np.asarray([dyn[1]], dtype=np.float32),
                    dyn_arousal_vol=np.asarray([dyn[3]], dtype=np.float32),
                    dyn_arousal_peak=np.asarray([dyn[4]], dtype=np.float32),
                    dyn_quality=np.asarray([self.song_dyn_quality[song_idx]], dtype=np.float32),
                )[0]
            )
            transfer_support = float(
                0.45 * np.clip((float(emb_score) + 1.0) / 2.0, 0.0, 1.0)
                + 0.20 * affect_fit
                + 0.15 * taste_fit
                + 0.10 * float(np.clip(self.song_quality[song_idx], 0.0, 1.0))
                + 0.10 * dyn_bonus
            )
            public_uncertainty = float(
                0.45 * (1.0 - np.clip((float(emb_score) + 1.0) / 2.0, 0.0, 1.0))
                + 0.25 * (1.0 - taste_fit)
                + 0.30 * (1.0 - anchor_support_strength)
            )
            final_score = float(
                0.44 * anchor_support_strength
                + 0.65 * transfer_support
                + 0.10 * float(np.clip(self.song_quality[song_idx], 0.0, 1.0))
                + 0.05 * dyn_bonus
                - 0.09 * public_uncertainty
            )
            supported = bool(transfer_support >= self.transfer_support_threshold and anchor_support_strength >= 0.55)
            if supported:
                final_score += 0.05
            if str(self.song_sources[song_idx]) == "pmemo" and dyn_bonus > 0.0:
                final_score += 0.02
            max_support = max(max_support, transfer_support)
            reason = pmemo_dynamic_reason(
                goal_idx=goal_idx,
                dyn_valence_delta=float(dyn[0]),
                dyn_arousal_delta=float(dyn[1]),
                dyn_arousal_vol=float(dyn[3]),
                dyn_arousal_peak=float(dyn[4]),
                dyn_quality=float(self.song_dyn_quality[song_idx]),
            )
            if not reason:
                reason = self._generic_public_reason(goal_idx)

            public_items.append(
                {
                    "kind": "public_transfer",
                    "song_idx": song_idx,
                    "song_id": str(self.song_ids[song_idx]),
                    "source": str(self.song_sources[song_idx]),
                    "title": str(self.song_titles[song_idx]),
                    "artist": str(self.song_artists[song_idx]),
                    "display_score": final_score,
                    "benefit_hat": float(anchor_support_strength),
                    "accept_hat": float(taste_fit),
                    "anchor_support_strength": float(anchor_support_strength),
                    "transfer_support": float(transfer_support),
                    "supported": supported,
                    "reason": reason,
                }
            )
        supported_items = [item for item in public_items if bool(item["supported"])]
        supported_items.sort(key=lambda item: float(item["display_score"]), reverse=True)
        return supported_items, float(anchor_support_strength), float(max_support)

    def score_rows(
        self,
        row_indices: np.ndarray,
        explicit_goal: str | None = None,
        top_k: int = 10,
        candidate_k: int = 50,
        batch_size: int = 128,
    ) -> dict[str, object]:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        rows = self.decision_df.iloc[row_indices]
        if explicit_goal is None:
            goal_idx = rows["goal_idx"].to_numpy(dtype=np.int64)
            goal_source = np.asarray(["router_fallback"] * len(row_indices))
            tau = rows[["tau_valence", "tau_arousal"]].to_numpy(dtype=np.float32)
        else:
            explicit_goal = str(explicit_goal).strip().lower()
            goal_map = {name: idx for idx, name in GOAL_NAMES.items()}
            if explicit_goal not in goal_map:
                raise ValueError(f"Unknown goal: {explicit_goal}")
            forced_goal = int(goal_map[explicit_goal])
            goal_idx = np.full(len(row_indices), forced_goal, dtype=np.int64)
            goal_source = np.asarray(["explicit"] * len(row_indices))
            tau = np.stack(
                [self._tau_for_goal(forced_goal, rows.iloc[pos][["tau_valence", "tau_arousal"]].to_numpy(dtype=np.float32)) for pos in range(len(rows))],
                axis=0,
            ).astype(np.float32)

        anchor_scores = self._score_anchor_rows(
            row_indices=row_indices,
            goal_idx=goal_idx,
            tau=tau,
            candidate_k=candidate_k,
            batch_size=batch_size,
        )

        final_recommendations: list[list[dict[str, object]]] = []
        top1_source = []
        top1_song_id = []
        top1_score = []
        top1_accept = []
        top1_transfer_supported = []
        top1_kind = []
        max_public_support = []
        anchor_support_strength = []
        legacy_query_song_ids = []
        legacy_final_song_ids = []

        for row_pos, global_idx in enumerate(row_indices.tolist()):
            order = anchor_scores["anchor_order"][row_pos]
            cand_local = anchor_scores["candidate_local_idx"][row_pos]
            ranked_anchor_local = cand_local[order]
            ranked_anchor_strength = anchor_scores["anchor_strength"][row_pos][order]
            ranked_anchor_benefit = anchor_scores["benefit_hat"][row_pos][order]
            ranked_anchor_accept = anchor_scores["latent_accept_hat"][row_pos][order]
            ranked_anchor_score = anchor_scores["anchor_final_score"][row_pos][order]

            anchor_song_best: dict[str, dict[str, object]] = {}
            for rank_pos, (anchor_local, strength, benefit_hat, accept_hat, raw_score) in enumerate(
                zip(
                    ranked_anchor_local.tolist(),
                    ranked_anchor_strength.tolist(),
                    ranked_anchor_benefit.tolist(),
                    ranked_anchor_accept.tolist(),
                    ranked_anchor_score.tolist(),
                )
            ):
                if anchor_local < 0:
                    continue
                song_idx = int(self.train_anchor_views["anchor_song_idx"][int(anchor_local)])
                song_id = str(self.song_ids[song_idx])
                payload = {
                    "kind": "anchor",
                    "song_idx": song_idx,
                    "song_id": song_id,
                    "source": "situnes",
                    "title": str(self.song_titles[song_idx]),
                    "artist": str(self.song_artists[song_idx]),
                    "display_score": float(strength),
                    "anchor_raw_score": float(raw_score),
                    "benefit_hat": float(benefit_hat),
                    "accept_hat": float(accept_hat),
                    "anchor_support_strength": float(strength),
                    "transfer_support": 0.0,
                    "supported": True,
                    "reason": f"supported by similar successful {self.goal_name(int(goal_idx[row_pos]))} anchors",
                }
                if song_id not in anchor_song_best or float(payload["display_score"]) > float(anchor_song_best[song_id]["display_score"]):
                    anchor_song_best[song_id] = payload
                if len(anchor_song_best) >= max(candidate_k, top_k):
                    break

            public_items, support_strength, public_support = self._public_transfer_for_row(
                row_global_idx=int(global_idx),
                goal_idx=int(goal_idx[row_pos]),
                tau=tau[row_pos],
                ranked_anchor_local=ranked_anchor_local,
                ranked_anchor_strength=ranked_anchor_strength,
            )

            combined = list(anchor_song_best.values()) + public_items
            if not public_items:
                for item in anchor_song_best.values():
                    item["reason"] = f"{item['reason']}; public transfer support stayed below threshold"
            combined.sort(key=lambda item: float(item["display_score"]), reverse=True)
            combined = combined[:top_k]
            final_recommendations.append(combined)

            top1 = combined[0] if combined else {
                "source": "situnes",
                "song_id": "",
                "display_score": 0.0,
                "accept_hat": 0.0,
                "supported": False,
                "kind": "anchor",
            }
            top1_source.append(str(top1["source"]))
            top1_song_id.append(str(top1["song_id"]))
            top1_score.append(float(top1["display_score"]))
            top1_accept.append(float(top1.get("accept_hat", 0.0)))
            top1_transfer_supported.append(int(bool(top1.get("supported", False)) and str(top1.get("kind")) == "public_transfer"))
            top1_kind.append(str(top1.get("kind", "anchor")))
            max_public_support.append(float(public_support))
            anchor_support_strength.append(float(support_strength))

            legacy_query_song_ids.append([str(self.song_ids[int(self.train_anchor_views["anchor_song_idx"][int(local)])]) for local in ranked_anchor_local[:candidate_k].tolist() if int(local) >= 0])
            legacy_final_song_ids.append([str(item["song_id"]) for item in combined])

        return {
            **anchor_scores,
            "goal_source": goal_source,
            "final_recommendations": final_recommendations,
            "top1_source": np.asarray(top1_source),
            "top1_song_id": np.asarray(top1_song_id),
            "top1_score": np.asarray(top1_score, dtype=np.float32),
            "top1_accept": np.asarray(top1_accept, dtype=np.float32),
            "top1_transfer_supported": np.asarray(top1_transfer_supported, dtype=np.int32),
            "top1_kind": np.asarray(top1_kind),
            "max_public_support": np.asarray(max_public_support, dtype=np.float32),
            "anchor_support_strength": np.asarray(anchor_support_strength, dtype=np.float32),
            "legacy_query_song_ids": legacy_query_song_ids,
            "legacy_final_song_ids": legacy_final_song_ids,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2.2 anchor retrieval + public transfer recommendations.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--explicit-goal", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    recommender = AmbientRecommenderV2(device=args.device)
    split_rows = recommender.decision_df.index[recommender.decision_df["split"] == args.split].to_numpy(dtype=np.int64)
    if len(split_rows) == 0:
        raise RuntimeError(f"No rows found for split={args.split}")
    row_indices = split_rows[: max(1, int(args.limit))]
    explicit_goal = str(args.explicit_goal).strip().lower() or None
    result = recommender.score_rows(
        row_indices=row_indices,
        explicit_goal=explicit_goal,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    payload = []
    for pos, global_idx in enumerate(row_indices.tolist()):
        row = recommender.decision_df.iloc[int(global_idx)]
        payload.append(
            {
                "row_idx": int(global_idx),
                "decision_id": str(row["decision_id"]),
                "goal": recommender.goal_name(int(result["goal_idx"][pos])),
                "goal_source": str(result["goal_source"][pos]),
                "top1_source": str(result["top1_source"][pos]),
                "top1_song_id": str(result["top1_song_id"][pos]),
                "top1_score": float(result["top1_score"][pos]),
                "top1_transfer_supported": int(result["top1_transfer_supported"][pos]),
                "max_public_support": float(result["max_public_support"][pos]),
                "anchor_support_strength": float(result["anchor_support_strength"][pos]),
                "recommendations": result["final_recommendations"][pos],
            }
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
