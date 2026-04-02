"""
Deterministic presentation demo for the rebuilt ambient music recommendation agent.
"""

from __future__ import annotations

import numpy as np

from src.data.common import ACTIVITY_LABELS, ACTIVITY_REMAP, BUCKET_LABELS, TIME_LABELS
from src.hmm.hmm_inference import corrected_belief, encode_obs_seq
from src.hmm.hmm_model import HMM
from src.music.music_library import MusicLibrary
from src.rl_agent.dqn_agent import DQNAgent
from src.rl_agent.reward_model import HierarchicalRewardModel


SCENARIOS = [
    {
        "name": "Late-Night Stressed Study Session",
        "intensity": 4,
        "activity_raw": 0,
        "time_bucket": 2,
        "mode": "wind_down",
        "goal": "Reduce stress and avoid over-activating the listener.",
    },
    {
        "name": "Afternoon Focus Block",
        "intensity": 7,
        "activity_raw": 0,
        "time_bucket": 1,
        "mode": "focus",
        "goal": "Support concentration without becoming sleepy.",
    },
    {
        "name": "Morning Run",
        "intensity": 118,
        "activity_raw": 5,
        "time_bucket": 0,
        "mode": "exercise",
        "goal": "Maintain energy and match sustained physical activity.",
    },
]


def bar(value: float, width: int = 18) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def main() -> None:
    hmm = HMM.load("models/hmm.npz")
    reward_model = HierarchicalRewardModel.load("models/reward_model.json")
    agent = DQNAgent(state_dim=5, action_dim=8)
    agent.load("models/agent.pt")
    agent.epsilon = 0.0
    library = MusicLibrary.build()

    print("=" * 72)
    print("AMBIENT MUSIC RECOMMENDATION AGENT - DEMO")
    print("=" * 72)
    print("Pipeline: wrist context -> HMM belief -> DQN bucket -> ranked tracks")

    for scenario in SCENARIOS:
        activity_remapped = ACTIVITY_REMAP[scenario["activity_raw"]]
        obs_seq = encode_obs_seq(scenario["intensity"], scenario["activity_raw"])
        belief = corrected_belief(hmm, obs_seq, activity_remapped)
        state = np.asarray(
            [belief[0], belief[1], belief[2], scenario["time_bucket"] / 2.0, activity_remapped / 4.0],
            dtype=np.float32,
        )
        action = int(agent.greedy_action(state))
        expected_reward = reward_model.expected_reward(
            int(np.argmax(belief)),
            int(scenario["time_bucket"]),
            int(activity_remapped),
            action,
        )
        tracks = library.get_tracks(action, n=3, context={"mode": scenario["mode"]})

        print("\n" + "-" * 72)
        print(scenario["name"])
        print("-" * 72)
        print(
            f"Context: intensity={scenario['intensity']:<4} "
            f"activity={ACTIVITY_LABELS[activity_remapped]:<13} "
            f"time={TIME_LABELS[scenario['time_bucket']]}"
        )
        print(f"Goal:    {scenario['goal']}")
        print("Belief:")
        for idx, prob in enumerate(belief):
            marker = " <" if idx == int(np.argmax(belief)) else ""
            print(f"  S{idx} {hmm.metadata.get('state_names', hmm.STATE_NAMES)[idx]:<12} {bar(float(prob))} {prob:>5.1%}{marker}")
        print(f"Action:  bucket {action} ({BUCKET_LABELS[action]})")
        print(f"Value:   expected reward {expected_reward:+.3f}")
        print("Tracks:")
        for _, track in tracks.iterrows():
            print(
                f"  - {track['track_name']} / {track['artist']} "
                f"[{track['source']}, score={track['score']:.2f}]"
            )

    print("\n" + "=" * 72)
    print("Use this output directly for presentation capture.")
    print("=" * 72)


if __name__ == "__main__":
    main()
