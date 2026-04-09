Framework: Physio-Aware Music Mood Prediction

The pipeline has three stages, each mapping to a dataset:

Stage 1 — Learn physio → affective state (CASE). Train a 3-state HMM on CASE's physiological signals (focus on ECG-derived heart rate and GSR, the channels most likely to transfer to SiTunes' wristband). The hidden states represent latent affective regions. The continuous V/A joystick annotations serve as ground truth for validating that the HMM states correspond to meaningful emotional clusters. Output: a pretrained HMM that converts a window of physiological observations into a 3-dimensional belief distribution.

Stage 2 — Learn V/A → music mood bucket (MuSe). MuSe's 90K songs have valence, arousal, and dominance scores. Cluster these into discrete mood buckets (start with 4 quadrants from Russell's circumplex: high-V/high-A = happy-energetic, high-V/low-A = calm-relaxed, low-V/high-A = tense-angry, low-V/low-A = sad-melancholic). This gives you the classification target: given a user's predicted V/A state, which bucket should the system recommend? This stage is mostly data engineering — defining the buckets and verifying they correspond to musically distinct groups.

Stage 3 — End-to-end prediction on SiTunes. The 14D-ish state vector feeds into a feedforward neural network that predicts the mood bucket. The state vector combines: HMM belief state (3D, from Stage 1's pretrained model applied to SiTunes wrist data), contextual features (time of day, weather, activity type), optional self-reported pre-emotion (valence, arousal + mask bit), and user preference profile (from SiTunes Stage 1 survey). Evaluation: zero-shot (CASE-only HMM, no SiTunes training) vs. fine-tuned (HMM adapted on 20 SiTunes users, tested on 10 held-out).

The ablation study that makes this a strong project: train the classifier with and without the HMM belief features. If belief features improve prediction, you've demonstrated that latent physiological state inference adds value beyond raw context — that's the core claim.
