"""
Deterministic multi-session simulation using the shared reward model.
"""

from __future__ import annotations

import copy

import numpy as np

from src.data.common import ACTIVITY_LABELS, ACTIVITY_REMAP, BUCKET_LABELS, STATE_DIM, TIME_LABELS, state_vector_from_components
from src.hmm.hmm_inference import corrected_belief, encode_obs_seq
from src.hmm.hmm_model import HMM
from src.music.music_library import MusicLibrary
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.reward_model import HierarchicalRewardModel


USER_PROFILES = [
    {
        "name": "Stressed CS Student",
        "mode": "focus",
        "user_profile": {
            "user_valence_pref": 0.10,
            "user_energy_pref": -0.20,
            "top_genres": ["indie", "ambient", "piano"],
        },
        "sessions": [
            {"intensity": 4, "hr_mean": 20.0, "activity_raw": 0, "step_mean": 0.0, "step_nonzero_frac": 0.0, "step_active": 0, "time_bucket": 2, "weather_bucket": 2, "gps_speed": 0.0, "valence": -0.45, "arousal": 0.55, "mask": 1.0},
            {"intensity": 6, "hr_mean": 12.0, "activity_raw": 0, "step_mean": 0.0, "step_nonzero_frac": 0.0, "step_active": 0, "time_bucket": 1, "weather_bucket": 1, "gps_speed": 0.0, "valence": -0.25, "arousal": 0.30, "mask": 0.0},
            {"intensity": 8, "hr_mean": 8.0, "activity_raw": 0, "step_mean": 0.5, "step_nonzero_frac": 0.03, "step_active": 0, "time_bucket": 1, "weather_bucket": 1, "gps_speed": 0.0, "valence": -0.10, "arousal": 0.20, "mask": 0.0},
            {"intensity": 15, "hr_mean": 10.0, "activity_raw": 2, "step_mean": 8.0, "step_nonzero_frac": 0.35, "step_active": 1, "time_bucket": 0, "weather_bucket": 0, "gps_speed": 1.8, "valence": 0.10, "arousal": 0.10, "mask": 1.0},
        ],
    },
    {
        "name": "Morning Runner",
        "mode": "exercise-lite",
        "user_profile": {
            "user_valence_pref": 0.20,
            "user_energy_pref": 0.45,
            "top_genres": ["rock", "pop", "electro"],
        },
        "sessions": [
            {"intensity": 110, "hr_mean": 26.0, "activity_raw": 5, "step_mean": 42.0, "step_nonzero_frac": 0.85, "step_active": 1, "time_bucket": 0, "weather_bucket": 0, "gps_speed": 4.8, "valence": 0.10, "arousal": 0.55, "mask": 0.0},
            {"intensity": 95, "hr_mean": 22.0, "activity_raw": 5, "step_mean": 33.0, "step_nonzero_frac": 0.70, "step_active": 1, "time_bucket": 0, "weather_bucket": 0, "gps_speed": 4.4, "valence": 0.15, "arousal": 0.60, "mask": 0.0},
            {"intensity": 5, "hr_mean": 2.0, "activity_raw": 4, "step_mean": 0.0, "step_nonzero_frac": 0.0, "step_active": 0, "time_bucket": 2, "weather_bucket": 1, "gps_speed": 0.0, "valence": 0.25, "arousal": -0.10, "mask": 1.0},
            {"intensity": 20, "hr_mean": 8.0, "activity_raw": 2, "step_mean": 10.0, "step_nonzero_frac": 0.40, "step_active": 1, "time_bucket": 2, "weather_bucket": 1, "gps_speed": 1.5, "valence": 0.20, "arousal": -0.05, "mask": 0.0},
        ],
    },
    {
        "name": "Desk-Work Professional",
        "mode": "wind_down",
        "user_profile": {
            "user_valence_pref": 0.05,
            "user_energy_pref": -0.30,
            "top_genres": ["acoustic", "folk", "ambient"],
        },
        "sessions": [
            {"intensity": 5, "hr_mean": 6.0, "activity_raw": 0, "step_mean": 0.0, "step_nonzero_frac": 0.0, "step_active": 0, "time_bucket": 0, "weather_bucket": 1, "gps_speed": 0.0, "valence": -0.15, "arousal": -0.10, "mask": 1.0},
            {"intensity": 6, "hr_mean": 7.0, "activity_raw": 0, "step_mean": 0.0, "step_nonzero_frac": 0.0, "step_active": 0, "time_bucket": 1, "weather_bucket": 1, "gps_speed": 0.0, "valence": -0.05, "arousal": 0.05, "mask": 0.0},
            {"intensity": 4, "hr_mean": 3.0, "activity_raw": 0, "step_mean": 0.0, "step_nonzero_frac": 0.0, "step_active": 0, "time_bucket": 1, "weather_bucket": 2, "gps_speed": 0.0, "valence": -0.20, "arousal": -0.20, "mask": 1.0},
            {"intensity": 3, "hr_mean": 1.0, "activity_raw": 4, "step_mean": 0.0, "step_nonzero_frac": 0.0, "step_active": 0, "time_bucket": 2, "weather_bucket": 1, "gps_speed": 0.0, "valence": 0.00, "arousal": -0.15, "mask": 1.0},
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
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=8, hidden=128)
    agent.load("models/agent.pt")
    agent.epsilon = 0.0
    library = MusicLibrary.build()

    print("=" * 112)
    print("AMBIENT MUSIC AGENT - MULTI-SESSION SIMULATION")
    print("=" * 112)

    for base_profile in USER_PROFILES:
        profile = copy.deepcopy(base_profile)
        print("\n" + "=" * 112)
        print(profile["name"])
        print("=" * 112)
        print(
            f"{'Session':<8} {'Context':<28} {'Mask':<6} {'Step':<6} {'Bucket':<15} "
            f"{'Combined':<10} {'Emotion':<10} {'Accept':<10} {'Valence':<14}"
        )
        print("-" * 112)

        valence_trace = []
        drops = 0
        lifts = 0
        last_action = 0

        for idx, session in enumerate(profile["sessions"]):
            activity_remapped = ACTIVITY_REMAP[session["activity_raw"]]
            belief = corrected_belief(
                hmm,
                encode_obs_seq(session["intensity"], session["activity_raw"], hr_mean=session["hr_mean"]),
                activity_remapped,
            )
            state = state_vector_from_components(
                belief,
                session["time_bucket"],
                activity_remapped,
                weather_bucket=session["weather_bucket"],
                gps_speed=session["gps_speed"],
                hr_mean_rel_user=session["hr_mean"],
                hr_std=6.0,
                pre_valence=session["valence"],
                pre_arousal=session["arousal"],
                pre_emotion_mask=session["mask"],
                user_valence_pref=profile["user_profile"]["user_valence_pref"],
                user_energy_pref=profile["user_profile"]["user_energy_pref"],
                step_mean=session.get("step_mean", 0.0),
                step_nonzero_frac=session.get("step_nonzero_frac", 0.0),
            )
            action = int(agent.greedy_action(state))
            components = reward_model.expected_components(
                int(np.argmax(belief)),
                session["time_bucket"],
                activity_remapped,
                int(session.get("step_active", 0)),
                action,
                pre_valence=float(session["valence"]),
                pre_arousal=float(session["arousal"]),
                user_valence_pref=float(profile["user_profile"]["user_valence_pref"]),
                user_energy_pref=float(profile["user_profile"]["user_energy_pref"]),
            )
            reward = reward_model.sample_reward(
                int(np.argmax(belief)),
                session["time_bucket"],
                activity_remapped,
                int(session.get("step_active", 0)),
                action,
                pre_valence=float(session["valence"]),
                pre_arousal=float(session["arousal"]),
            )
            dv, da = reward_model.sample_mood_delta(reward)

            new_valence = float(np.clip(session["valence"] + dv, -0.99, 0.99))
            new_arousal = float(np.clip(session["arousal"] + da, -0.99, 0.99))

            valence_trace.append(session["valence"])
            lifts += int(reward > 0)
            drops += int(reward < 0)
            last_action = action

            context_text = f"{TIME_LABELS[session['time_bucket']]} / {ACTIVITY_LABELS[activity_remapped]}"
            print(
                f"{idx + 1:<8} {context_text:<28} {int(session['mask']):<6} {int(session.get('step_active', 0)):<6} {BUCKET_LABELS[action]:<15} "
                f"{components['combined_reward']:+.3f}     {components['emotion_benefit']:+.3f}     "
                f"{components['acceptance']:+.3f}     {mood_bar(session['valence']):<14}"
            )

            if idx + 1 < len(profile["sessions"]):
                profile["sessions"][idx + 1]["valence"] = new_valence
                profile["sessions"][idx + 1]["arousal"] = new_arousal

        tracks = library.get_tracks(last_action, n=2, context={"mode": profile["mode"], "user_profile": profile["user_profile"]})
        print("\nSummary")
        print(f"  Taste profile: valence_pref={profile['user_profile']['user_valence_pref']:+.2f} energy_pref={profile['user_profile']['user_energy_pref']:+.2f}")
        print(f"  Starting valence: {valence_trace[0]:+.2f}")
        print(f"  Ending valence:   {profile['sessions'][-1]['valence']:+.2f}")
        print(f"  Lifts: {lifts}  Drops: {drops}")
        print(f"  Final bucket: {last_action} ({BUCKET_LABELS[last_action]})")
        print("  Example tracks:")
        for _, track in tracks.iterrows():
            explanation = f" | {track['dynamic_reason']}" if str(track.get("dynamic_reason", "")) else ""
            print(f"    - {track['track_name']} / {track['artist']} [{track['source']}]{explanation}")

    print("\n" + "=" * 112)
    print("Simulation complete.")
    print("=" * 112)


if __name__ == "__main__":
    main()
