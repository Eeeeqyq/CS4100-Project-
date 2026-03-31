"""
demo.py
Runs 3 pre-scripted sessions showing the full pipeline:
    wrist signals → HMM belief → DQN action → music recommendations

Run from project root:
    python demo.py

Screen-record this output for your presentation video.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from src.hmm.hmm_model        import HMM
from src.rl_agent.dqn_agent    import DQNAgent
from src.music.music_library   import MusicLibrary, BUCKET_LABELS

MODELS = Path("models")

# ── Wrist encoding (must match clean_situnes exactly) ─────────────────────
ACTIVITY_REMAP = {0:0, 1:1, 2:2, 3:0, 4:3, 5:4}

def encode_timestep(intensity, activity_type, time_bucket, weather_bucket):
    ib  = 0 if intensity<10 else (1 if intensity<30 else (2 if intensity<80 else 3))
    act = ACTIVITY_REMAP.get(int(activity_type), 0)
    return int((ib*5 + act)*9 + time_bucket*3 + weather_bucket)

def make_obs_seq(intensity, activity_type, time_bucket, weather_bucket, length=30):
    ts = encode_timestep(intensity, activity_type, time_bucket, weather_bucket)
    return np.full(length, ts, dtype=np.int32)

# ── Display helpers ────────────────────────────────────────────────────────

def bar(v, w=15):
    f = int(round(max(0, min(1, v)) * w))
    return "█"*f + "░"*(w-f)

def print_belief(belief):
    top = int(np.argmax(belief))
    for s, p in enumerate(belief):
        mark = " ◀" if s == top else ""
        print(f"    S{s} {HMM.STATE_NAMES[s]:<20} {bar(p)} {p*100:5.1f}%{mark}")

def print_tracks(tracks, action):
    label = BUCKET_LABELS.get(action, "?")
    print(f"\n  🎵 Bucket {action} — {label.upper()}")
    if tracks.empty:
        print("     (no tracks found)")
        return
    for _, r in tracks.head(3).iterrows():
        print(f"     • {r['track_name']}  —  {r['artist']}  [{r['source']}]")
        print(f"       valence={r['valence']:.2f}  "
              f"energy={r['energy']:.2f}  "
              f"tempo={r['tempo']:.0f}bpm")

# ── Demo sessions ──────────────────────────────────────────────────────────

SESSIONS = [
    {
        "name":         "Session A — Stressed Student, 11pm",
        "intensity":    5,       # very low physical activity
        "activity":     0,       # Still
        "time_bucket":  2,       # Evening/night
        "weather":      0,       # Sunny (indoors, doesn't matter)
        "pre_valence":  -0.4,
        "pre_arousal":   0.5,
    },
    {
        "name":         "Session B — Morning Run, 7am",
        "intensity":    120,     # vigorous
        "activity":     5,       # Running
        "time_bucket":  0,       # Morning
        "weather":      0,       # Sunny
        "pre_valence":   0.3,
        "pre_arousal":   0.6,
    },
    {
        "name":         "Session C — Afternoon Study, 2pm",
        "intensity":    8,       # minimal
        "activity":     0,       # Still
        "time_bucket":  1,       # Afternoon
        "weather":      1,       # Cloudy
        "pre_valence":   0.1,
        "pre_arousal":  -0.1,
    },
]


def main():
    print("=" * 60)
    print("  AMBIENT MUSIC RECOMMENDATION AGENT — DEMO")
    print("  HMM + Deep Q-Network  |  SiTunes Dataset")
    print("=" * 60)

    # Load models
    hmm   = HMM.load(str(MODELS / "hmm.npz"))
    agent = DQNAgent(state_dim=8, action_dim=8)
    pt    = MODELS / "agent.pt"
    if pt.exists():
        agent.load(str(pt))
        agent.epsilon = 0.0   # no exploration at demo time
    else:
        print("WARNING: agent.pt not found — using untrained agent")

    # Load music
    print("\nLoading music library...")
    lib = MusicLibrary.build()

    # Run sessions
    for s in SESSIONS:
        print(f"\n{'─'*60}")
        print(f"  {s['name']}")
        print(f"{'─'*60}")

        obs_seq = make_obs_seq(s["intensity"], s["activity"],
                               s["time_bucket"], s["weather"])

        belief    = hmm.belief_state(obs_seq)
        top_state = int(np.argmax(belief))

        print(f"\n  Signals: intensity={s['intensity']}  "
              f"activity={'Still Running Walking'.split()[min(s['activity'],2)]}  "
              f"time={['Morning','Afternoon','Evening'][s['time_bucket']]}")
        print(f"\n  HMM Belief State:")
        print_belief(belief)
        print(f"\n  → Inferred mood: {HMM.STATE_NAMES[top_state].upper()}")

        # Build state vector for DQN
        time_norm = s["time_bucket"] / 2.0
        act_norm  = ACTIVITY_REMAP.get(s["activity"], 0) / 4.0
        state_vec = np.concatenate([belief, [time_norm, act_norm]]).astype(np.float32)

        action = agent.greedy_action(state_vec)
        tracks = lib.get_tracks(bucket=action, n=5)
        print_tracks(tracks, action)

    print(f"\n{'='*60}")
    print("  Record this output for your presentation video.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()