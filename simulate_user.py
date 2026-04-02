"""
Deterministic multi-session simulation using the shared reward model.
"""

from __future__ import annotations

import numpy as np

from src.data.common import ACTIVITY_LABELS, ACTIVITY_REMAP, BUCKET_LABELS, TIME_LABELS
from src.hmm.hmm_inference import corrected_belief, encode_obs_seq
from src.hmm.hmm_model import HMM
from src.music.music_library import MusicLibrary
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.reward_model import HierarchicalRewardModel


USER_PROFILES = [
    {
        "name": "Stressed CS Student",
        "mode": "focus",
        "sessions": [
            {"intensity": 4, "activity_raw": 0, "time_bucket": 2, "valence": -0.45, "arousal": 0.55},
            {"intensity": 6, "activity_raw": 0, "time_bucket": 1, "valence": -0.25, "arousal": 0.30},
            {"intensity": 8, "activity_raw": 0, "time_bucket": 1, "valence": -0.10, "arousal": 0.20},
            {"intensity": 15, "activity_raw": 2, "time_bucket": 0, "valence": 0.10, "arousal": 0.10},
        ],
    },
    {
        "name": "Morning Runner",
        "mode": "exercise",
        "sessions": [
            {"intensity": 110, "activity_raw": 5, "time_bucket": 0, "valence": 0.10, "arousal": 0.55},
            {"intensity": 95, "activity_raw": 5, "time_bucket": 0, "valence": 0.15, "arousal": 0.60},
            {"intensity": 5, "activity_raw": 4, "time_bucket": 2, "valence": 0.25, "arousal": -0.10},
            {"intensity": 20, "activity_raw": 2, "time_bucket": 2, "valence": 0.20, "arousal": -0.05},
        ],
    },
    {
        "name": "Desk-Work Professional",
        "mode": "wind_down",
        "sessions": [
            {"intensity": 5, "activity_raw": 0, "time_bucket": 0, "valence": -0.15, "arousal": -0.10},
            {"intensity": 6, "activity_raw": 0, "time_bucket": 1, "valence": -0.05, "arousal": 0.05},
            {"intensity": 4, "activity_raw": 0, "time_bucket": 1, "valence": -0.20, "arousal": -0.20},
            {"intensity": 3, "activity_raw": 4, "time_bucket": 2, "valence": 0.00, "arousal": -0.15},
        ],
    },
]


def mood_bar(value: float, width: int = 12) -> str:
    normalized = (max(-1.0, min(1.0, value)) + 1.0) / 2.0
    filled = int(round(normalized * width))
    return "#" * filled + "." * (width - filled)


def reward_label(reward: int) -> str:
    return {1: "lift", 0: "hold", -1: "drop"}[int(reward)]


def main() -> None:
    hmm = HMM.load("models/hmm.npz")
    reward_model = HierarchicalRewardModel.load("models/reward_model.json")
    agent = DQNAgent(state_dim=5, action_dim=8)
    agent.load("models/agent.pt")
    agent.epsilon = 0.0
    library = MusicLibrary.build()

    print("=" * 76)
    print("AMBIENT MUSIC AGENT - MULTI-SESSION SIMULATION")
    print("=" * 76)

    for profile in USER_PROFILES:
        print("\n" + "=" * 76)
        print(profile["name"])
        print("=" * 76)
        print(f"{'Session':<8} {'Context':<28} {'Bucket':<15} {'Reward':<8} {'Valence':<14}")
        print("-" * 76)

        valence_trace = []
        drops = 0
        lifts = 0
        last_action = 0

        for idx, session in enumerate(profile["sessions"]):
            activity_remapped = ACTIVITY_REMAP[session["activity_raw"]]
            belief = corrected_belief(hmm, encode_obs_seq(session["intensity"], session["activity_raw"]), activity_remapped)
            state = np.asarray(
                [belief[0], belief[1], belief[2], session["time_bucket"] / 2.0, activity_remapped / 4.0],
                dtype=np.float32,
            )
            action = int(agent.greedy_action(state))
            reward = reward_model.sample_reward(int(np.argmax(belief)), session["time_bucket"], activity_remapped, action)
            dv, da = reward_model.sample_mood_delta(reward)

            new_valence = float(np.clip(session["valence"] + dv, -0.99, 0.99))
            new_arousal = float(np.clip(session["arousal"] + da, -0.99, 0.99))

            valence_trace.append(session["valence"])
            lifts += int(reward > 0)
            drops += int(reward < 0)
            last_action = action

            context_text = f"{TIME_LABELS[session['time_bucket']]} / {ACTIVITY_LABELS[activity_remapped]}"
            print(
                f"{idx + 1:<8} {context_text:<28} {BUCKET_LABELS[action]:<15} "
                f"{reward_label(reward):<8} {mood_bar(session['valence']):<14}"
            )

            if idx + 1 < len(profile["sessions"]):
                profile["sessions"][idx + 1]["valence"] = new_valence
                profile["sessions"][idx + 1]["arousal"] = new_arousal

        tracks = library.get_tracks(last_action, n=2, context={"mode": profile["mode"]})
        print("\nSummary")
        print(f"  Starting valence: {valence_trace[0]:+.2f}")
        print(f"  Ending valence:   {profile['sessions'][-1]['valence']:+.2f}")
        print(f"  Lifts: {lifts}  Drops: {drops}")
        print(f"  Final bucket: {last_action} ({BUCKET_LABELS[last_action]})")
        print("  Example tracks:")
        for _, track in tracks.iterrows():
            print(f"    - {track['track_name']} / {track['artist']} [{track['source']}]")

    print("\n" + "=" * 76)
    print("Simulation complete.")
    print("=" * 76)


if __name__ == "__main__":
    main()
