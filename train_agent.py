"""
train_agent.py
Train the DQN agent on SiTunes.

Run from project root:
    python train_agent.py

Prerequisites (run first):
    python src/hmm/hmm_train.py

Reads:  data/processed/stage2_clean.csv
        data/processed/stage3_clean.csv
        data/processed/wrist2_encoded.npy
        data/processed/wrist3_encoded.npy
        models/hmm.npz

Writes: models/agent.pt
        models/training_log.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from src.hmm.hmm_model        import HMM
from src.rl_agent.environment  import MusicEnv
from src.rl_agent.dqn_agent    import DQNAgent

PROCESSED  = Path("data/processed")
MODELS     = Path("models")

# ── Hyperparameters ───────────────────────────────────────────────────────
N_EPISODES    = 2000
BURN_IN       = 200    # random episodes before training starts
LOG_EVERY     = 100
EVAL_EVERY    = 200
EVAL_EPS      = 50

LR            = 1e-3
GAMMA         = 0.9
EPS_START     = 1.0
EPS_DECAY     = 0.995
EPS_MIN       = 0.05
BATCH_SIZE    = 64
BUFFER_CAP    = 10_000
TARGET_SYNC   = 10
HIDDEN        = 64
SEED          = 42


def load_data():
    df2    = pd.read_csv(PROCESSED / "stage2_clean.csv")
    df3    = pd.read_csv(PROCESSED / "stage3_clean.csv")
    wrist2 = np.load(PROCESSED / "wrist2_encoded.npy")
    wrist3 = np.load(PROCESSED / "wrist3_encoded.npy")

    # Stage3 obs_idx must be offset by len(df2) so it indexes into stacked wrist
    df3         = df3.copy()
    df3["obs_idx"] = df3["obs_idx"] + len(df2)

    df    = pd.concat([df2, df3], ignore_index=True)
    wrist = np.vstack([wrist2, wrist3])

    print(f"Interactions: {len(df)}  users: {df.user_id.nunique()}")
    return df, wrist


def evaluate(agent, env, n=50):
    total, lifts, drops, steps = 0, 0, 0, 0
    for _ in range(n):
        state = env.reset()
        while True:
            action              = agent.greedy_action(state)
            state, r, done, _   = env.step(action)
            total  += r
            steps  += 1
            if r > 0: lifts += 1
            if r < 0: drops += 1
            if done: break
    mean_r = total / n
    lift_r = lifts / steps if steps else 0
    drop_r = drops / steps if steps else 0
    return mean_r, lift_r, drop_r


def main():
    # Load
    df, wrist = load_data()
    hmm = HMM.load(str(MODELS / "hmm.npz"))

    # Environments
    env      = MusicEnv(df, wrist, hmm, seed=SEED)
    eval_env = MusicEnv(df, wrist, hmm, seed=SEED+1)
    print(env)

    # Agent
    agent = DQNAgent(
        state_dim=MusicEnv.STATE_DIM, action_dim=MusicEnv.ACTION_DIM,
        lr=LR, gamma=GAMMA, epsilon=EPS_START, epsilon_decay=EPS_DECAY,
        epsilon_min=EPS_MIN, buffer_cap=BUFFER_CAP, batch_size=BATCH_SIZE,
        target_sync=TARGET_SYNC, hidden=HIDDEN, seed=SEED,
    )

    # Burn-in: fill replay buffer with random transitions
    print(f"\nBurn-in ({BURN_IN} episodes)...")
    for _ in range(BURN_IN):
        state = env.reset()
        while True:
            a              = env.sample_action()
            ns, r, done, _ = env.step(a)
            agent.replay.push(state, a, r, ns, done)
            state = ns
            if done: break
    print(f"Buffer size: {len(agent.replay)}")

    # Training
    print(f"\nTraining {N_EPISODES} episodes...")
    print(f"{'Ep':>6}  {'MeanR':>7}  {'Lift%':>6}  {'Drop%':>6}  "
          f"{'ε':>5}  {'Loss':>7}")
    print("─" * 50)

    ep_rewards, ep_losses, log = [], [], []

    for ep in range(1, N_EPISODES+1):
        state      = env.reset()
        ep_reward  = 0
        ep_loss    = []

        while True:
            a              = agent.select_action(state)
            ns, r, done, _ = env.step(a)
            agent.replay.push(state, a, r, ns, done)
            loss = agent.update()
            if loss: ep_loss.append(loss)
            ep_reward += r
            state      = ns
            if done: break

        agent.end_episode()
        ep_rewards.append(ep_reward)
        if ep_loss: ep_losses.append(np.mean(ep_loss))

        if ep % LOG_EVERY == 0:
            mr   = np.mean(ep_rewards[-LOG_EVERY:])
            ml   = np.mean(ep_losses[-LOG_EVERY:]) if ep_losses else 0

            if ep % EVAL_EVERY == 0:
                er, lift, drop = evaluate(agent, eval_env, EVAL_EPS)
                print(f"{ep:>6}  {er:>7.3f}  {lift*100:>5.1f}%  "
                      f"{drop*100:>5.1f}%  {agent.epsilon:>5.3f}  {ml:>7.4f}  ←eval")
                log.append({"ep": ep, "eval_reward": er,
                             "lift": lift, "drop": drop,
                             "epsilon": agent.epsilon, "loss": ml})
            else:
                print(f"{ep:>6}  {mr:>7.3f}  {'—':>6}  {'—':>6}  "
                      f"{agent.epsilon:>5.3f}  {ml:>7.4f}")
                log.append({"ep": ep, "eval_reward": None,
                             "lift": None, "drop": None,
                             "epsilon": agent.epsilon, "loss": ml})

    # Save
    MODELS.mkdir(exist_ok=True)
    agent.save(str(MODELS / "agent.pt"))
    pd.DataFrame(log).to_csv(MODELS / "training_log.csv", index=False)

    # Final eval
    print("\n" + "="*50)
    print("FINAL EVALUATION (greedy, 100 episodes)")
    er, lift, drop = evaluate(agent, eval_env, 100)
    print(f"  Mean reward: {er:.3f}")
    print(f"  Lift rate:   {lift*100:.1f}%")
    print(f"  Drop rate:   {drop*100:.1f}%")

    # Random baseline
    rand = DQNAgent(epsilon=1.0, epsilon_min=1.0)
    rr, rl, rd = evaluate(rand, eval_env, 100)
    print(f"\nRandom baseline:")
    print(f"  Mean reward: {rr:.3f}")
    print(f"  Lift rate:   {rl*100:.1f}%")
    print(f"\nImproved over random: {'YES ✓' if lift > rl else 'NO — tune hyperparameters'}")
    print("\n✓ Done. Next: python demo.py")


if __name__ == "__main__":
    main()