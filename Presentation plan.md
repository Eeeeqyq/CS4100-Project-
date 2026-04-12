# Presentation plan — CS4100 Ambient Music (6 minutes)

Use this file in two ways: (1) paste the **Slide generator prompt** into your slide tool or chat assistant, and (2) follow the **Timing outline** while speaking.

---

## Slide generator prompt (copy everything below the line)

---

**Role:** You are designing a short academic slide deck.

**Constraints:**

- Total length: **6 minutes** of speaking (~8–10 slides including title and thank-you).
- Visual style: **minimal** — lots of whitespace, one idea per slide, no clip art, no busy gradients. Prefer a **single sans-serif font**, **black or dark gray text on off-white**, and **at most one simple diagram** (flowchart or boxes-and-arrows).
- Audience: CS / AI classmates and instructors who know ML basics.
- Tone: clear, honest, not hype.

**Project summary (facts to use):**

- **Course / title:** CS4100 Artificial Intelligence, Northeastern University — *Ambient Music Recommendation Agent*.
- **Problem:** Standard music apps personalize from past behavior; they under-use **right-now** context: biometrics, environment, and current mood, plus baseline taste.
- **Core question:** Given current **wrist-sensor context**, **environment**, optional **emotion check-in**, and **baseline preferences**, what **music intervention** is appropriate?
- **Two pipelines in the repo (say this clearly):**
  1. **Legacy graded path:** 3-state **HMM** (NumPy, Baum–Welch) on wrist observations → **Double DQN** picks one of **8 mood buckets** → deterministic **retrieval** across catalogs. Core AI is from-scratch (no hmmlearn / stable-baselines3).
  2. **Experimental v2.2 rebuild (what we ran for this demo):** **Explicit goals** — `focus`, `wind_down`, `uplift`, `movement`. The system retrieves **SiTunes intervention anchors** supervised from past successful sessions, **reranks** by predicted **benefit** and **acceptance**; **Spotify / PMEmo** are optional **transfer** candidates only when support is strong.
- **Data:** **SiTunes** field study (wrist, env, emotions); optional **PMEmo** and **Spotify** for larger catalogs in retrieval/transfer.
- **Commands we actually ran:** `python train_v2.py` → `python eval_v2.py` → `python demo_v2.py`.

**Offline evaluation results we obtained (v2.2 test split, candidate pool size 50, 311 rows):**

- Readiness flag: **NOT READY** (system reports blockers; present honestly).
- **Anchor query recall@20:** 0.7910 — **Anchor query recall@50:** 0.9196  
- **Weighted query recall@20:** 0.3364  
- **Anchor rerank hit@10:** 0.7556 — **Anchor rerank mean rank:** 5.96  
- **Weighted rerank NDCG@10:** 0.2522  
- **Benefit MAE:** 0.1424 — **Blended acceptance MAE:** 0.3056  
- **Top-1 predicted acceptance mean:** 0.2354  
- **Public-transfer-supported share:** **0.0000** — **Top-1 source max share (situnes):** **1.0000**  
- **Blocker called out by eval:** public transfer is weak or collapsing; **legacy** exact-song metrics are much lower (exact-song query recall@50 ~0.36, hit@10 ~0.14).

**Per-goal snapshot (from same eval):**

| Goal       | Rows | WQ@20  | Hit@10 | Mean rank |
|-----------|------|--------|--------|-----------|
| focus     | 140  | 0.3281 | 0.8214 | 5.37      |
| wind_down | 46   | 0.4606 | 0.9130 | 3.80      |
| uplift    | 107  | 0.2387 | 0.6075 | 7.63      |
| movement  | 18   | 0.6646 | 0.7222 | 7.56      |

**Demo snapshot (one test user row; illustrative):**

- User in **test** split; example pre-state valence **+0.73**, arousal **+0.28**; context: time bucket 1, clear weather, speed 0.
- For each **goal**, top recommendations were **anchor-supported** SiTunes tracks (e.g. *Thriller*, *Get Up*, *Losing My Mind*); repeated line in reasons: **“public transfer support stayed below threshold.”**

**Slides to generate:**

1. Title — project name, course, team (placeholder names).
2. Problem — why “ambient” recommendation matters (one short paragraph).
3. Data — SiTunes + optional PMEmo/Spotify (bullets only).
4. Two pipelines — small diagram: Legacy HMM+DQN vs v2.2 goal+anchors (side-by-side or two rows).
5. v2.2 idea — explicit goals; anchor-first; transfer only when supported.
6. What we ran — `train_v2` / `eval_v2` / `demo_v2` as a horizontal three-step strip.
7. Results — **key metrics table** (anchor recall, rerank hit@10, MAEs); **one line** on NOT READY and 0% public transfer.
8. Demo — one screenshot or bullet list of example top-3 tracks per goal (optional: say “same small SiTunes pool visible across goals”).
9. Limitations & honesty — small data, transfer not firing, goal-specific spread (e.g. uplift harder).
10. Takeaway — what we learned + future work (stronger transfer, more data, optional legacy HMM+DQN comparison).

**Do not:** invent better numbers, claim readiness is green, or hide the NOT READY / 0% transfer facts.

---

## Timing outline (6 minutes)

| Time   | Slide focus |
|--------|-------------|
| 0:00–0:45 | Title + hook (problem in one sentence). |
| 0:45–1:45 | Data + why SiTunes wrist + context. |
| 1:45–3:00 | Architecture: legacy vs v2.2 (keep v2.2 as “what we demo”). |
| 3:00–4:30 | Results table: anchor retrieval strong; weighted / transfer weak; NOT READY. |
| 4:30–5:30 | Demo: same user, different goals, anchor-based SiTunes picks. |
| 5:30–6:00 | Limitations + conclusion + questions. |

---

## Speaker notes (one-liners you can read)

- **Problem:** Past taste ≠ how you feel **now**; we use sensors + context + goals.
- **v2.2:** Pick a **goal**, find **similar successful past sessions** (anchors), **rerank** by predicted mood benefit and acceptance.
- **Results:** Good at finding **relevant anchors** in top-20/50; **weighted** retrieval and **public transfer** are weak right now (0% transfer support in top-1).
- **Honest:** Eval says **NOT READY** — fine for a class project if you explain **why** (transfer threshold, catalog imbalance).
- **Demo:** Shows the system **behaving consistently** with anchor reasoning; not yet diverse catalog at top-1.

---

## Files in this repo to cite if asked

- `README.md` — problem, both pipelines, v2.2 commands and metrics definitions.
- `docs/PRESENTATION_REPORT.md` — deeper v2.2 walkthrough (if present).
