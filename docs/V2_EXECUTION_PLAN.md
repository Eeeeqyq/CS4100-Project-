# V2.2 Execution Plan

Last updated: 2026-04-17

This file is the execution contract for the rebuilt `v2.2` path under `src/v2/`.

## Objective

`v2.2` is trying to solve this problem:

> Given an explicit goal, the current biometric and environmental context, and the user's baseline taste profile, retrieve strong historical SiTunes intervention anchors and only transfer to public songs when that transfer is well-supported.

This is intentionally different from the old mixed-catalog exact-song objective.

Primary claim:

- anchor retrieval
- anchor reranking
- support-aware public transfer

Secondary diagnostic only:

- exact-song recovery

## Stable Design

`v2.2` now assumes all of these are true:

- explicit goal is the main inference interface
- `goal_router_v1` remains only as a fallback baseline
- SiTunes is the only intervention-outcome supervision source
- Spotify and PMEmo are public transfer catalogs
- acceptance is split into:
  - `accept_pref_target`
  - `accept_rating_target`
  - `accept_pref_mask`
  - `accept_rating_mask`
- val/test deployment is split-safe:
  - only train anchors are allowed into the live anchor index

## Ordered Phases

### Phase 1. Data Contract

Status:

- complete

Key outputs:

- `decision_table.parquet`
- `anchor_table.parquet`
- `anchor_positive_sets.npz`
- `anchor_negative_pools.npz`

### Phase 2. Anchor Retrieval

Status:

- complete

Key outputs:

- `query_tower.pt`
- `query_embeddings.npy`
- `anchor_query_embeddings.npy`
- `situnes_anchor_index.npz`

### Phase 3. Anchor Reranking

Status:

- complete

Key outputs:

- `reranker.pt`
- `reranker_metrics.json`

### Phase 4. Public Transfer

Status:

- complete

Key outputs:

- anchor-conditioned public transfer inside `src/v2/inference/recommend.py`
- calibrated public-transfer threshold inside `src/v2/inference/recommend.py`

### Phase 5. Offline Eval And Readiness

Status:

- complete on the primary contract

Key outputs:

- `offline_eval_v2.json`
- `offline_eval_v2_predictions.parquet`
- `v2_readiness.json`

## Primary Hard Gates

These are the gates that define `v2.2` readiness.

- anchor retrieval `recall@20 >= 0.20`
- anchor retrieval `recall@50 >= 0.35`
- anchor retrieval weighted `recall@20 >= 0.15`
- anchor retrieval weighted `recall@50 >= 0.25`
- anchor rerank `hit@10 >= 0.25`
- anchor rerank conditional mean positive rank `<= 6.0`
- anchor rerank weighted NDCG@10 `>= 0.20`
- benefit MAE `<= 0.20`
- blended acceptance MAE `<= 0.35`
- top-1 predicted acceptance mean `>= -0.10`
- public-transfer-supported share `>= 0.05`
- no single top-1 source share `> 0.85` (equivalently `top-1 source max share <= 0.85`)
- per supported goal weighted anchor `recall@20 >= 0.08`

## Current Verified Status

Last verified commands:

```bash
python train_v2.py
python eval_v2.py --no-rerun
python demo_v2.py
```

Current verified primary test metrics:

- anchor query `recall@20 = 0.7460`
- anchor query `recall@50 = 0.8810`
- anchor query weighted `recall@20 = 0.3011`
- anchor query weighted `recall@50 = 0.4358`
- anchor query weighted NDCG@10 `= 0.1688`
- anchor rerank `hit@10 = 0.7170`
- anchor rerank conditional mean rank `= 5.9270`
- anchor rerank weighted NDCG@10 `= 0.2306`
- benefit MAE `= 0.1262`
- blended acceptance MAE `= 0.2947`
- top-1 predicted acceptance mean `= 0.5101`
- public-transfer-supported share `= 0.6399`
- top-1 source max share `= 0.6399`

Current readiness result:

- `ready = true`

## Legacy Diagnostics

These remain useful, but they no longer define success:

- exact-song query `recall@50 = 0.3698`
- exact-song rerank `hit@10 = 0.0418`
- exact-song conditional rank `= 37.58`

The exact-song rerank `hit@10` diagnostic is still slightly below the old `0.05` sanity threshold.

## What To Do Next

If work resumes on `v2.2`, the next highest-value steps are:

1. improve public-transfer quality with a learned transfer scorer instead of relying only on heuristic support-weighting
2. decide whether DEAM should be added for public-song affect representation after the stricter contract is stable
3. only revisit anchor positive-set tuning if a future pass shows one goal losing support under the current cap-5 supervision
