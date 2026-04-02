"""
Deterministic presentation demo for the ambient-first music recommendation agent.
"""

from __future__ import annotations

import numpy as np

from src.data.common import ACTIVITY_LABELS, ACTIVITY_REMAP, BUCKET_LABELS, STATE_DIM, TIME_LABELS, state_vector_from_components
from src.hmm.hmm_inference import corrected_belief, encode_obs_seq
from src.hmm.hmm_model import HMM
from src.music.music_library import MusicLibrary
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.reward_model import HierarchicalRewardModel


COMPARISON_GROUPS = [
    {
        "title": "Same Physical State, Different Emotion",
        "cases": [
            {
                "name": "Stressed Afternoon Student",
                "intensity": 5,
                "hr_mean": 18.0,
                "activity_raw": 0,
                "time_bucket": 1,
                "weather_bucket": 2,
                "gps_speed": 0.1,
                "pre_valence": -0.80,
                "pre_arousal": -0.60,
                "pre_emotion_mask": 1.0,
                "mode": "wind_down",
                "goal": "Recover a low mood without flattening the listener further.",
                "user_profile": {
                    "user_valence_pref": 0.20,
                    "user_energy_pref": -0.10,
                    "top_genres": ["ambient", "indie", "piano"],
                },
            },
            {
                "name": "Balanced Afternoon Student",
                "intensity": 5,
                "hr_mean": 18.0,
                "activity_raw": 0,
                "time_bucket": 1,
                "weather_bucket": 2,
                "gps_speed": 0.1,
                "pre_valence": 0.00,
                "pre_arousal": 0.20,
                "pre_emotion_mask": 1.0,
                "mode": "focus",
                "goal": "Maintain balance while staying gently focused.",
                "user_profile": {
                    "user_valence_pref": 0.20,
                    "user_energy_pref": -0.10,
                    "top_genres": ["indie", "classical", "acoustic"],
                },
            },
        ],
    },
    {
        "title": "Same Context, Different Preference",
        "cases": [
            {
                "name": "Prefers Dark / High-Energy Music",
                "intensity": 5,
                "hr_mean": 18.0,
                "activity_raw": 0,
                "time_bucket": 1,
                "weather_bucket": 2,
                "gps_speed": 0.1,
                "pre_valence": 0.20,
                "pre_arousal": 0.00,
                "pre_emotion_mask": 1.0,
                "mode": "uplift",
                "goal": "Stay emotionally steady, but honor a darker personal taste.",
                "user_profile": {
                    "user_valence_pref": -0.80,
                    "user_energy_pref": -0.50,
                    "top_genres": ["rock", "hip-hop", "electro"],
                },
            },
            {
                "name": "Prefers Calm / Bright Music",
                "intensity": 5,
                "hr_mean": 18.0,
                "activity_raw": 0,
                "time_bucket": 1,
                "weather_bucket": 2,
                "gps_speed": 0.1,
                "pre_valence": 0.20,
                "pre_arousal": 0.00,
                "pre_emotion_mask": 1.0,
                "mode": "uplift",
                "goal": "Stay emotionally steady with a brighter, calmer taste profile.",
                "user_profile": {
                    "user_valence_pref": 0.20,
                    "user_energy_pref": -0.10,
                    "top_genres": ["indie", "folk", "acoustic"],
                },
            },
        ],
    },
    {
        "title": "Same User, Different Scenarios",
        "cases": [
            {
                "name": "Deep Focus Block",
                "intensity": 6,
                "hr_mean": 6.0,
                "activity_raw": 0,
                "time_bucket": 1,
                "weather_bucket": 1,
                "gps_speed": 0.0,
                "pre_valence": -0.05,
                "pre_arousal": 0.10,
                "pre_emotion_mask": 1.0,
                "mode": "focus",
                "goal": "Support sustained concentration.",
                "user_profile": {
                    "user_valence_pref": 0.15,
                    "user_energy_pref": -0.15,
                    "top_genres": ["indie", "ambient", "classical"],
                },
            },
            {
                "name": "Evening Recovery",
                "intensity": 4,
                "hr_mean": 2.0,
                "activity_raw": 4,
                "time_bucket": 2,
                "weather_bucket": 1,
                "gps_speed": 0.0,
                "pre_valence": -0.20,
                "pre_arousal": -0.10,
                "pre_emotion_mask": 1.0,
                "mode": "wind_down",
                "goal": "Relax and recover without overstimulation.",
                "user_profile": {
                    "user_valence_pref": 0.15,
                    "user_energy_pref": -0.15,
                    "top_genres": ["indie", "ambient", "classical"],
                },
            },
            {
                "name": "Commute Walk",
                "intensity": 16,
                "hr_mean": 10.0,
                "activity_raw": 2,
                "time_bucket": 0,
                "weather_bucket": 0,
                "gps_speed": 2.3,
                "pre_valence": 0.00,
                "pre_arousal": 0.20,
                "pre_emotion_mask": 1.0,
                "mode": "uplift",
                "goal": "Add light energy and positive momentum.",
                "user_profile": {
                    "user_valence_pref": 0.15,
                    "user_energy_pref": -0.15,
                    "top_genres": ["indie", "ambient", "classical"],
                },
            },
        ],
    },
]


def bar(value: float, width: int = 18) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def recommend_case(hmm, agent, reward_model, library, case: dict) -> tuple[np.ndarray, int, dict, np.ndarray]:
    activity_remapped = ACTIVITY_REMAP[case["activity_raw"]]
    obs_seq = encode_obs_seq(case["intensity"], case["activity_raw"], hr_mean=case["hr_mean"])
    belief = corrected_belief(hmm, obs_seq, activity_remapped)
    state = state_vector_from_components(
        belief,
        case["time_bucket"],
        activity_remapped,
        weather_bucket=case["weather_bucket"],
        gps_speed=case["gps_speed"],
        hr_mean_rel_user=case["hr_mean"],
        hr_std=6.0,
        pre_valence=case["pre_valence"],
        pre_arousal=case["pre_arousal"],
        pre_emotion_mask=case["pre_emotion_mask"],
        user_valence_pref=case["user_profile"]["user_valence_pref"],
        user_energy_pref=case["user_profile"]["user_energy_pref"],
    )
    action = int(agent.greedy_action(state))
    components = reward_model.expected_components(
        int(np.argmax(belief)),
        int(case["time_bucket"]),
        int(activity_remapped),
        action,
        pre_valence=float(case["pre_valence"]),
        pre_arousal=float(case["pre_arousal"]),
        user_valence_pref=float(case["user_profile"]["user_valence_pref"]),
        user_energy_pref=float(case["user_profile"]["user_energy_pref"]),
    )
    tracks = library.get_tracks(action, n=3, context={"mode": case["mode"], "user_profile": case["user_profile"]})
    return belief, action, components, tracks


def main() -> None:
    hmm = HMM.load("models/hmm.npz")
    reward_model = HierarchicalRewardModel.load("models/reward_model.json")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=8, hidden=128)
    agent.load("models/agent.pt")
    agent.epsilon = 0.0
    library = MusicLibrary.build()

    print("=" * 92)
    print("AMBIENT-FIRST PERSONALIZED MUSIC INTERVENTION DEMO")
    print("=" * 92)
    print("Pipeline: biometrics + context + optional check-in + taste profile -> HMM belief -> DQN bucket -> ranked tracks")

    for group in COMPARISON_GROUPS:
        print("\n" + "=" * 92)
        print(group["title"])
        print("=" * 92)
        for case in group["cases"]:
            belief, action, components, tracks = recommend_case(hmm, agent, reward_model, library, case)
            activity_remapped = ACTIVITY_REMAP[case["activity_raw"]]

            print("\n" + "-" * 92)
            print(case["name"])
            print("-" * 92)
            print(
                f"Context: hr={case['hr_mean']:<5.1f} intensity={case['intensity']:<4} "
                f"activity={ACTIVITY_LABELS[activity_remapped]:<13} "
                f"time={TIME_LABELS[case['time_bucket']]:<9} weather={case['weather_bucket']} speed={case['gps_speed']:.1f}"
            )
            print(
                f"Mood:    pre_valence={case['pre_valence']:+.2f} "
                f"pre_arousal={case['pre_arousal']:+.2f} "
                f"mask={case['pre_emotion_mask']:.0f}"
            )
            print(
                f"Taste:   valence_pref={case['user_profile']['user_valence_pref']:+.2f} "
                f"energy_pref={case['user_profile']['user_energy_pref']:+.2f} "
                f"genres={', '.join(case['user_profile']['top_genres'])}"
            )
            print(f"Goal:    {case['goal']}")
            print("Belief:")
            for idx, prob in enumerate(belief):
                marker = " <" if idx == int(np.argmax(belief)) else ""
                print(f"  S{idx} {hmm.metadata.get('state_names', hmm.STATE_NAMES)[idx]:<12} {bar(float(prob))} {prob:>5.1%}{marker}")
            print(
                f"Action:  bucket {action} ({BUCKET_LABELS[action]}) | "
                f"combined={components['combined_reward']:+.3f} "
                f"emotion={components['emotion_benefit']:+.3f} "
                f"accept={components['acceptance']:+.3f}"
            )
            print("Tracks:")
            for _, track in tracks.iterrows():
                soft_tag = " soft" if bool(track["bucket_is_soft"]) else ""
                print(
                    f"  - {track['track_name']} / {track['artist']} "
                    f"[{track['source']}{soft_tag}, score={track['score']:.2f}]"
                )

    print("\n" + "=" * 92)
    print("Use this output directly for presentation capture.")
    print("=" * 92)


if __name__ == "__main__":
    main()
