# V2.2 Presentation Script

Last updated: 2026-04-12

This file is the short talk track for the class presentation.

Use this instead of reading `docs/PRESENTATION_REPORT.md` out loud. The report is the preparation document. This script is the 5-6 minute version.

## How To Use This Script

- Keep the presentation to 7-8 slides.
- Do not show code.
- Show only:
  - the problem
  - the method logic
  - the live results summary
  - the live demo output
  - the conclusion
- If all team members must speak, split the script by speaker blocks below.
- If your team size is not exactly 3, keep the same sections and reassign them.

## Recommended Slide Order

1. Motivation
2. Problem formulation
3. Why the old approach was not enough
4. Rebuilt `v2.2` pipeline
5. Why these AI methods make sense
6. Demo
7. Results
8. Conclusion

## Commands To Run During Presentation

Open the terminal in the project root and run:

```bash
python eval_v2.py --no-rerun
python demo_v2.py
```

Do not run `train_v2.py` live.

## 5-6 Minute Talk Track

### Speaker 1 - Motivation And Problem Setup (~1 minute)

"Our project asks a simple question that standard music recommenders usually ignore: not just what music a person likes in general, but what music is likely to help them right now.

Most recommendation systems focus on past listening history. But if someone is stressed, trying to focus, winding down, or trying to feel more energized, the same person may need a very different intervention even if their long-term taste stays the same.

So our problem is: given the user's current biometric and environmental context, an explicit goal, and their baseline taste profile, recommend music that moves them toward the right emotional state while still being acceptable enough that they would actually follow it.

That wording matters, because we are not trying to predict the exact historical song. We are trying to recommend a useful and realistic music intervention."

### Speaker 1 - Transition To Method (~20 seconds)

"The repo contains an older HMM plus DQN pipeline, but for the presentation we focus on our rebuilt `v2.2` system, because it gives a cleaner and more defensible AI formulation."

### Speaker 2 - Key Methods And Logic (~1.5 minutes)

"The key methodological decision in `v2.2` is to treat the problem as anchor retrieval and reranking, instead of exact-song classification.

Why? Because only SiTunes contains the full intervention story: user context, pre-state, post-state, and feedback. So SiTunes is the only dataset we trust for intervention supervision.

Spotify and PMEmo are still used, but only as public transfer catalogs and music representation sources. We do not pretend they contain the same kind of intervention labels.

The pipeline works in five stages.

First, we build a context representation from the wrist sequence, environment, and optional self-report. We use a bidirectional GRU with attention because wrist data is sequential. A single average heart rate misses the temporal pattern, but a sequence model can capture whether the user is settling down, ramping up, or staying volatile.

Second, we build a user representation from Stage 1 rating history. This gives the model a compact baseline taste profile instead of treating every user the same.

Third, we build a song representation from the merged music catalog. This uses static song features for all songs, and PMEmo dynamic emotion curves where they exist. That helps us represent how a song behaves emotionally over time, not just its average metadata.

Fourth, we retrieve SiTunes anchors. These are real historical intervention rows from SiTunes. Instead of saying there is only one correct song, we define a tiered positive set: the factual intervention, other successful same-song examples, and nearby successful context neighbors. That makes the supervision much closer to the real recommendation problem.

Fifth, we rerank those anchors by predicted benefit and predicted acceptance. We model benefit and acceptance separately, because a song can be helpful but disliked, or liked but not helpful. Only after we have strong anchor support do we allow transfer to public songs from Spotify or PMEmo."

### Speaker 2 - Theoretical Grounding (~40 seconds)

"The theoretical grounding is really about matching the method to the data.

We use explicit goals because the data is not strong enough to support a strong claim about autonomous goal inference.

We use sequence modeling because wrist data is temporal.

We use anchor retrieval because SiTunes provides intervention examples, not a complete labeled public music universe.

And we separate benefit from acceptance because those are different parts of the decision problem.

So the method is not just more complex. It is more faithful to what the data can actually justify."

### Speaker 3 - Demo Setup (~20 seconds)

"Now we will show the rebuilt system's output. First we show the offline evaluation summary, then we show the live demo on one held-out test context under different goals."

### Speaker 3 - Demo Commentary While Running `eval_v2.py --no-rerun` (~30 seconds)

"This summary shows that the rebuilt system is currently marked `READY` on its primary contract.

The important numbers are that anchor retrieval and reranking are strong, benefit and blended acceptance are within the target range, and the system is using public transfer without collapsing to a single source."

### Speaker 3 - Demo Commentary While Running `demo_v2.py` (~1.5 minutes)

"This demo keeps the same held-out test row and changes only the explicit goal.

That matters because it isolates the decision logic. If the model is doing what we claim, changing the goal should change the recommendation behavior in a goal-consistent way.

For `focus`, the system stays with SiTunes anchors. The explanation says public transfer support stayed below threshold, so it falls back to strong historical anchors. This is exactly what we want: if the evidence for transfer is weak, the system should not force a public song.

For `wind_down`, the system promotes Spotify songs. Here public transfer support is strong enough, and the recommended songs have a calmer dynamic contour, which matches the goal.

For `uplift` and `movement`, the system again prefers anchors when transfer support is weaker. That shows the system is not simply biased toward the larger Spotify catalog. It is using an anchor-first logic and only transferring when the support is good enough.

So the demo shows three things:

one, the model is goal-conditional;

two, it is support-aware rather than blindly expanding into public catalogs;

and three, its explanations line up with the retrieval and transfer logic we described."

### Speaker 1 - Results And Interpretation (~1 minute)

"Our main results are these:

- anchor query recall@20 is `0.7460`
- anchor rerank hit@10 is `0.7170`
- anchor rerank mean rank is `5.9270`
- benefit MAE is `0.1262`
- blended acceptance MAE is `0.2947`
- public-transfer-supported share is `0.6399`
- top-1 source max share is `0.6399`

The interpretation is that the system is reliably finding useful SiTunes anchors, reranking them well, and extending into public songs without collapsing into one catalog.

We also report exact-song recovery as a legacy diagnostic, and that metric is still weaker. But that is no longer the main claim of `v2.2`, because the rebuilt task is intervention recommendation, not historical song imitation."

### Speaker 2 - Honest Limitations And Conclusion (~1 minute)

"The main takeaway is that `v2.2` is a more theoretically sound formulation than the original bucket-first setup.

It uses the right supervision source for intervention quality, it models temporal context with a sequence encoder, it personalizes with Stage 1 taste history, and it keeps public transfer constrained by evidence instead of pretending those public songs are directly labeled.

At the same time, we should be honest about the limitations.

SiTunes is still a small dataset, especially for some goals like movement.

Public transfer is still engineered rather than fully learned.

And exact-song imitation remains weak, which is why we explicitly frame it as a diagnostic rather than the main objective.

So our conclusion is not that we solved music recommendation in general. Our conclusion is that we built a more defensible, goal-conditioned, anchor-first music intervention system that matches the structure of the available data much better than the original approach."

## Shorter Emergency Version

If you are running out of time, keep only these points:

1. We are solving situational music recommendation, not just history matching.
2. SiTunes is the only intervention-outcome dataset, so it supervises anchor quality.
3. `v2.2` uses:
   - sequence context encoding
   - user taste encoding
   - SiTunes anchor retrieval
   - reranking by benefit and acceptance
   - cautious public transfer to Spotify and PMEmo
4. The rebuilt system is currently `READY`.
5. The demo shows that changing the goal changes the recommendation behavior in a consistent way.
6. Main takeaway: the rebuilt approach is more theoretically grounded because it matches what the data can actually support.

## Fallback If Live Demo Fails

If the terminal demo fails during recording, do not improvise. Use the already verified outputs:

- `python eval_v2.py --no-rerun`
- `python demo_v2.py`

You can paste screenshots or read from the saved verified outputs:

- `models/rebuild/offline_eval_v2.json`
- `models/rebuild/v2_readiness.json`

The key thing is that the demo section should still show:

- `Readiness: READY`
- goal-dependent recommendation changes
- anchor fallback when transfer support is weak
- Spotify transfer when transfer support is strong enough

## Questions You Should Be Ready For

### Why not train directly on the mixed catalog?

Because Spotify and PMEmo do not contain intervention-outcome labels. Treating unlabeled public songs as negatives would be methodologically wrong.

### Why use a sequence model for wrist data?

Because current state is not just a static average. The temporal pattern of wrist signals matters for distinguishing contexts.

### Why separate benefit and acceptance?

Because usefulness and likability are different objectives, and collapsing them too early hides that tradeoff.

### Why is exact-song hit@10 still low?

Because the rebuilt task is not optimized for exact historical song imitation. It is optimized for intervention-quality anchors and support-aware transfer.

## Final One-Sentence Takeaway

"Our main contribution is not just a better model, but a better problem formulation: use SiTunes to learn which interventions help in context, then transfer to public songs only when that transfer is actually supported by evidence." 
