"""
CS4100 — Ambient Music Mood Prediction
Hidden Markov Model: Unsupervised Latent State Discovery from Wrist Physio

This module implements a discrete-observation HMM from scratch using NumPy.
No hmmlearn, no sklearn — just Baum-Welch (EM) and Viterbi.

Architecture:
  Input:  30-step wrist sequences (heart_rate, activity_intensity, activity_type)
  Output: 3-dimensional belief state per interaction (P(state=0), P(state=1), P(state=2))

The HMM discovers latent physiological states unsupervised.
The belief state feeds into a downstream neural classifier (separate module).

Usage:
  from hmm import HMM, discretize_wrist_sequence
  
  # Discretize raw wrist data into observation IDs
  obs_sequences = discretize_wrist_sequences(wrist_data)  # (897, 30) array of ints
  
  # Train
  hmm = HMM(n_states=3, n_observations=27)
  hmm.fit(obs_sequences, n_iter=50)
  
  # Extract belief states for classifier
  beliefs = hmm.get_belief_states(obs_sequences)  # (897, 3) array of floats

LifeSnaps upgrade path:
  If you later pretrain on LifeSnaps, you'd:
  1. Discretize LifeSnaps physio into the same 27-observation space
  2. Train hmm.fit(lifesnaps_sequences, n_iter=50)
  3. Apply to SiTunes: beliefs = hmm.get_belief_states(situnes_sequences)
  The HMM parameters (A, B, pi) transfer directly — no architecture changes needed.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict


# ============================================================
# OBSERVATION DISCRETIZATION
# ============================================================

def discretize_wrist_sequences(
    wrist_data: np.ndarray,
    hr_bins: Optional[List[float]] = None,
    intensity_bins: Optional[List[float]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Convert raw 4-channel wrist data into discrete observation IDs for the HMM.
    
    Each timestep's 3 features (HR, intensity, activity_type) are discretized
    and combined into a single integer observation ID.
    
    Parameters
    ----------
    wrist_data : np.ndarray, shape (n_interactions, 30, 4)
        Raw wrist data. Channels: [heart_rate, activity_intensity, 
        activity_step, activity_type]
    hr_bins : list of float, optional
        Bin edges for heart rate terciles. If None, computed from data.
    intensity_bins : list of float, optional
        Bin edges for activity intensity terciles. If None, computed from data.
    
    Returns
    -------
    obs_sequences : np.ndarray, shape (n_interactions, 30)
        Integer observation IDs in range [0, 26] (3 × 3 × 3 = 27 possible)
    discretization_info : dict
        Bin edges and mappings used, needed to discretize new data consistently
    
    Observation encoding:
        obs_id = hr_bin * 9 + intensity_bin * 3 + activity_bin
        
        hr_bin:        0=low, 1=mid, 2=high  (tercile split on per-interaction means)
        intensity_bin: 0=low, 1=mid, 2=high  (tercile split on per-interaction means)
        activity_bin:  0=still/lying, 1=walking, 2=running/active
    """
    n_interactions, n_timesteps, n_channels = wrist_data.shape
    
    # --- Heart rate discretization (3 bins via terciles) ---
    hr_all = wrist_data[:, :, 0].flatten()
    if hr_bins is None:
        hr_bins = [np.percentile(hr_all, 33.3), np.percentile(hr_all, 66.7)]
    hr_discrete = np.digitize(wrist_data[:, :, 0], hr_bins)  # 0, 1, or 2
    
    # --- Activity intensity discretization (3 bins via terciles) ---
    intensity_all = wrist_data[:, :, 1].flatten()
    if intensity_bins is None:
        intensity_bins = [np.percentile(intensity_all, 33.3), np.percentile(intensity_all, 66.7)]
    intensity_discrete = np.digitize(wrist_data[:, :, 1], intensity_bins)  # 0, 1, or 2
    
    # --- Activity type discretization (collapse 6 types → 3 categories) ---
    # Raw types: 0=Still, 1=Act→Still, 2=Walking, 3=Missing, 4=Lying, 5=Running
    # Collapsed: 0=sedentary (still/lying/act→still/missing), 1=walking, 2=running
    activity_raw = wrist_data[:, :, 3].astype(int)
    activity_discrete = np.where(activity_raw == 2, 1,           # walking → 1
                        np.where(activity_raw == 5, 2,           # running → 2
                                 0))                             # everything else → 0
    
    # --- Combine into single observation ID ---
    # obs_id = hr_bin * 9 + intensity_bin * 3 + activity_bin
    # Range: [0, 2*9 + 2*3 + 2] = [0, 26]
    obs_sequences = hr_discrete * 9 + intensity_discrete * 3 + activity_discrete
    
    discretization_info = {
        'hr_bins': hr_bins,
        'intensity_bins': intensity_bins,
        'activity_mapping': {0: 'sedentary', 1: 'walking', 2: 'running'},
        'n_observations': 27,
        'encoding': 'hr_bin * 9 + intensity_bin * 3 + activity_bin',
    }
    
    return obs_sequences, discretization_info


# ============================================================
# HMM IMPLEMENTATION
# ============================================================

class HMM:
    """
    Discrete-observation Hidden Markov Model trained with Baum-Welch (EM).
    
    This is a from-scratch NumPy implementation — no external HMM libraries.
    
    Parameters
    ----------
    n_states : int
        Number of hidden states. Default 3 (intended to capture coarse
        physiological regimes like rest/moderate/active).
    n_observations : int
        Number of possible discrete observations. Default 27 (3×3×3).
    random_state : int
        Seed for reproducible initialization.
    
    Attributes (learned)
    ----------
    pi : np.ndarray, shape (n_states,)
        Initial state distribution. pi[i] = P(state_0 = i)
    A : np.ndarray, shape (n_states, n_states)
        Transition matrix. A[i,j] = P(state_t = j | state_{t-1} = i)
    B : np.ndarray, shape (n_states, n_observations)
        Emission matrix. B[i,k] = P(obs_t = k | state_t = i)
    
    Key methods
    -----------
    fit(sequences, n_iter) : Train via Baum-Welch
    get_belief_states(sequences) : Extract belief distributions for classifier
    decode(sequence) : Viterbi decoding of most likely state sequence
    """
    
    def __init__(self, n_states: int = 3, n_observations: int = 27, 
                 random_state: int = 42):
        self.n_states = n_states
        self.n_obs = n_observations
        self.rng = np.random.RandomState(random_state)
        
        # Initialize parameters with slight randomness + uniform base
        # This avoids symmetric initialization (which prevents states from specializing)
        self.pi = self._random_stochastic_vector(n_states)
        self.A = self._random_stochastic_matrix(n_states, n_states)
        self.B = self._random_stochastic_matrix(n_states, n_observations)
        
        # Training history
        self.log_likelihoods = []
    
    def _random_stochastic_vector(self, size: int) -> np.ndarray:
        """Generate a random probability vector that sums to 1."""
        vec = self.rng.dirichlet(np.ones(size))
        return vec
    
    def _random_stochastic_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Generate a random row-stochastic matrix (each row sums to 1)."""
        mat = np.zeros((rows, cols))
        for i in range(rows):
            mat[i] = self.rng.dirichlet(np.ones(cols))
        return mat
    
    # --------------------------------------------------------
    # FORWARD ALGORITHM (alpha pass)
    # --------------------------------------------------------
    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward probabilities with scaling to prevent underflow.
        
        Parameters
        ----------
        obs : np.ndarray, shape (T,)
            Sequence of observation IDs.
        
        Returns
        -------
        alpha : np.ndarray, shape (T, n_states)
            Scaled forward probabilities. alpha[t, i] ∝ P(o_1..o_t, state_t=i)
        scale : np.ndarray, shape (T,)
            Scaling factors. Product of all scales = P(obs sequence).
        """
        T = len(obs)
        N = self.n_states
        alpha = np.zeros((T, N))
        scale = np.zeros(T)
        
        # --- Base case: t = 0 ---
        alpha[0] = self.pi * self.B[:, obs[0]]
        scale[0] = alpha[0].sum()
        if scale[0] > 0:
            alpha[0] /= scale[0]
        else:
            # Fallback: uniform if all probabilities are 0
            alpha[0] = 1.0 / N
            scale[0] = 1e-300
        
        # --- Recursive case: t = 1..T-1 ---
        for t in range(1, T):
            # alpha[t, j] = sum_i(alpha[t-1, i] * A[i,j]) * B[j, obs[t]]
            alpha[t] = (alpha[t-1] @ self.A) * self.B[:, obs[t]]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
            else:
                alpha[t] = 1.0 / N
                scale[t] = 1e-300
        
        return alpha, scale
    
    # --------------------------------------------------------
    # BACKWARD ALGORITHM (beta pass)
    # --------------------------------------------------------
    def _backward(self, obs: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Compute backward probabilities using the same scaling factors as forward.
        
        Parameters
        ----------
        obs : np.ndarray, shape (T,)
            Sequence of observation IDs.
        scale : np.ndarray, shape (T,)
            Scaling factors from the forward pass.
        
        Returns
        -------
        beta : np.ndarray, shape (T, n_states)
            Scaled backward probabilities.
        """
        T = len(obs)
        N = self.n_states
        beta = np.zeros((T, N))
        
        # --- Base case: t = T-1 ---
        beta[T-1] = 1.0  # Scaled: beta[T-1] = 1/scale[T-1], but we apply scale below
        
        # --- Recursive case: t = T-2..0 ---
        for t in range(T-2, -1, -1):
            # beta[t, i] = sum_j(A[i,j] * B[j, obs[t+1]] * beta[t+1, j])
            beta[t] = (self.A * self.B[:, obs[t+1]]) @ beta[t+1]
            if scale[t+1] > 0:
                beta[t] /= scale[t+1]
        
        return beta
    
    # --------------------------------------------------------
    # E-STEP: Compute gamma and xi from alpha/beta
    # --------------------------------------------------------
    def _e_step(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        E-step of Baum-Welch: compute posterior state probabilities.
        
        Parameters
        ----------
        obs : np.ndarray, shape (T,)
            Single observation sequence.
        
        Returns
        -------
        gamma : np.ndarray, shape (T, n_states)
            gamma[t, i] = P(state_t = i | full observation sequence)
        xi : np.ndarray, shape (T-1, n_states, n_states)
            xi[t, i, j] = P(state_t=i, state_{t+1}=j | full observation sequence)
        log_likelihood : float
            Log P(observation sequence | current parameters)
        """
        T = len(obs)
        N = self.n_states
        
        alpha, scale = self._forward(obs)
        beta = self._backward(obs, scale)
        
        # --- Gamma: P(state_t = i | observations) ---
        # With scaling: gamma[t,i] = alpha[t,i] * beta[t,i]
        # (The scaling factors cancel appropriately)
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)  # Prevent division by zero
        gamma = gamma / gamma_sum
        
        # --- Xi: P(state_t=i, state_{t+1}=j | observations) ---
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            # xi[t, i, j] ∝ alpha[t,i] * A[i,j] * B[j, obs[t+1]] * beta[t+1, j]
            numerator = (alpha[t, :, np.newaxis] * 
                        self.A * 
                        self.B[:, obs[t+1]][np.newaxis, :] * 
                        beta[t+1, :][np.newaxis, :])
            denom = numerator.sum()
            if denom > 0:
                xi[t] = numerator / denom
            else:
                xi[t] = 1.0 / (N * N)
        
        # --- Log-likelihood from scaling factors ---
        # log P(obs) = sum(log(scale[t]))
        log_likelihood = np.sum(np.log(np.maximum(scale, 1e-300)))
        
        return gamma, xi, log_likelihood
    
    # --------------------------------------------------------
    # M-STEP: Update parameters from accumulated statistics
    # --------------------------------------------------------
    def _m_step(self, all_gamma: List[np.ndarray], 
                all_xi: List[np.ndarray],
                all_obs: List[np.ndarray]):
        """
        M-step of Baum-Welch: re-estimate pi, A, B from posterior statistics.
        
        Parameters
        ----------
        all_gamma : list of np.ndarray
            Gamma arrays from all training sequences.
        all_xi : list of np.ndarray
            Xi arrays from all training sequences.
        all_obs : list of np.ndarray
            All training observation sequences.
        """
        N = self.n_states
        M = self.n_obs
        
        # Smoothing constant to prevent zero probabilities
        # (Laplace smoothing / additive smoothing)
        SMOOTH = 1e-6
        
        # --- Update pi: average of gamma[0] across all sequences ---
        pi_new = np.zeros(N)
        for gamma in all_gamma:
            pi_new += gamma[0]
        pi_new += SMOOTH
        pi_new /= pi_new.sum()
        
        # --- Update A: transition counts from xi ---
        A_num = np.zeros((N, N)) + SMOOTH
        A_den = np.zeros(N) + SMOOTH * N
        for gamma, xi in zip(all_gamma, all_xi):
            A_num += xi.sum(axis=0)          # Sum over time
            A_den += gamma[:-1].sum(axis=0)  # Sum over time (excluding last)
        A_new = A_num / A_den[:, np.newaxis]
        
        # --- Update B: emission counts from gamma ---
        B_num = np.zeros((N, M)) + SMOOTH
        B_den = np.zeros(N) + SMOOTH * M
        for gamma, obs in zip(all_gamma, all_obs):
            for t in range(len(obs)):
                B_num[:, obs[t]] += gamma[t]
            B_den += gamma.sum(axis=0)
        B_new = B_num / B_den[:, np.newaxis]
        
        self.pi = pi_new
        self.A = A_new
        self.B = B_new
    
    # --------------------------------------------------------
    # FIT: Full Baum-Welch training loop
    # --------------------------------------------------------
    def fit(self, sequences: np.ndarray, n_iter: int = 50, 
            tol: float = 1e-4, verbose: bool = True) -> 'HMM':
        """
        Train the HMM via Baum-Welch (Expectation-Maximization).
        
        Parameters
        ----------
        sequences : np.ndarray, shape (n_sequences, T)
            Array of observation sequences. Each row is one sequence of
            discrete observation IDs.
        n_iter : int
            Maximum number of EM iterations.
        tol : float
            Convergence threshold on log-likelihood improvement.
            Training stops if improvement < tol for 3 consecutive iterations.
        verbose : bool
            Print progress each iteration.
        
        Returns
        -------
        self : HMM
            Fitted model (for method chaining).
        """
        n_seq = len(sequences)
        self.log_likelihoods = []
        stall_count = 0
        
        if verbose:
            print(f"Training HMM: {self.n_states} states, {self.n_obs} observations, "
                  f"{n_seq} sequences")
            print(f"{'Iter':>4s}  {'Log-Lik':>12s}  {'Δ':>10s}  {'Status':>8s}")
            print("-" * 42)
        
        for iteration in range(n_iter):
            # --- E-step: collect statistics from all sequences ---
            all_gamma = []
            all_xi = []
            all_obs = []
            total_ll = 0.0
            
            for seq in sequences:
                gamma, xi, ll = self._e_step(seq)
                all_gamma.append(gamma)
                all_xi.append(xi)
                all_obs.append(seq)
                total_ll += ll
            
            avg_ll = total_ll / n_seq
            self.log_likelihoods.append(avg_ll)
            
            # --- Check convergence ---
            if len(self.log_likelihoods) > 1:
                delta = avg_ll - self.log_likelihoods[-2]
                if delta < tol:
                    stall_count += 1
                else:
                    stall_count = 0
                status = "STALL" if delta < tol else "OK"
            else:
                delta = float('inf')
                status = "INIT"
            
            if verbose:
                print(f"{iteration+1:4d}  {avg_ll:12.4f}  {delta:+10.4f}  {status:>8s}")
            
            # Early stopping: 3 consecutive stalls
            if stall_count >= 3:
                if verbose:
                    print(f"Converged after {iteration+1} iterations (3 consecutive stalls)")
                break
            
            # --- M-step: update parameters ---
            self._m_step(all_gamma, all_xi, all_obs)
        
        if verbose and stall_count < 3:
            print(f"Reached max iterations ({n_iter})")
        
        return self
    
    # --------------------------------------------------------
    # BELIEF STATE EXTRACTION (for downstream classifier)
    # --------------------------------------------------------
    def get_belief_states(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract belief state distributions for each sequence.
        
        The "belief state" is the posterior probability distribution over
        hidden states at the LAST timestep of each sequence, computed via
        the forward algorithm. This is the key output that feeds into the
        downstream neural classifier.
        
        Parameters
        ----------
        sequences : np.ndarray, shape (n_sequences, T)
            Observation sequences.
        
        Returns
        -------
        beliefs : np.ndarray, shape (n_sequences, n_states)
            beliefs[i, j] = P(state_T = j | obs_1..obs_T) for sequence i.
            Each row sums to 1.
        """
        n_seq = len(sequences)
        beliefs = np.zeros((n_seq, self.n_states))
        
        for i, seq in enumerate(sequences):
            alpha, scale = self._forward(seq)
            # The last row of scaled alpha IS the belief state
            # (because scaling normalizes each timestep to sum to 1)
            beliefs[i] = alpha[-1]
        
        return beliefs
    
    # --------------------------------------------------------
    # VITERBI DECODING
    # --------------------------------------------------------
    def decode(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find the most likely hidden state sequence via the Viterbi algorithm.
        
        Parameters
        ----------
        obs : np.ndarray, shape (T,)
            Single observation sequence.
        
        Returns
        -------
        best_path : np.ndarray, shape (T,)
            Most likely state at each timestep.
        log_prob : float
            Log probability of the best path.
        """
        T = len(obs)
        N = self.n_states
        
        # Use log probabilities to avoid underflow
        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_A = np.log(np.maximum(self.A, 1e-300))
        log_B = np.log(np.maximum(self.B, 1e-300))
        
        # Viterbi tables
        V = np.zeros((T, N))       # V[t,i] = log P(best path ending in state i at time t)
        backptr = np.zeros((T, N), dtype=int)  # backpointer for traceback
        
        # --- Base case ---
        V[0] = log_pi + log_B[:, obs[0]]
        
        # --- Recursive case ---
        for t in range(1, T):
            for j in range(N):
                # V[t,j] = max_i(V[t-1,i] + log A[i,j]) + log B[j, obs[t]]
                scores = V[t-1] + log_A[:, j]
                backptr[t, j] = np.argmax(scores)
                V[t, j] = scores[backptr[t, j]] + log_B[j, obs[t]]
        
        # --- Traceback ---
        best_path = np.zeros(T, dtype=int)
        best_path[T-1] = np.argmax(V[T-1])
        log_prob = V[T-1, best_path[T-1]]
        
        for t in range(T-2, -1, -1):
            best_path[t] = backptr[t+1, best_path[t+1]]
        
        return best_path, log_prob
    
    # --------------------------------------------------------
    # ANALYSIS & INTERPRETATION HELPERS
    # --------------------------------------------------------
    def print_parameters(self):
        """Print learned HMM parameters in a readable format."""
        print("\n--- HMM Parameters ---")
        print(f"States: {self.n_states}, Observations: {self.n_obs}")
        
        print(f"\nInitial distribution (pi):")
        for i in range(self.n_states):
            print(f"  State {i}: {self.pi[i]:.4f}")
        
        print(f"\nTransition matrix (A):")
        print(f"  {'':>8s}", end="")
        for j in range(self.n_states):
            print(f"  → S{j:d}   ", end="")
        print()
        for i in range(self.n_states):
            print(f"  S{i}: ", end="")
            for j in range(self.n_states):
                print(f"  {self.A[i,j]:.4f}", end="")
            print()
        
        print(f"\nEmission matrix (B) — top 5 observations per state:")
        for i in range(self.n_states):
            top_obs = np.argsort(self.B[i])[::-1][:5]
            print(f"  State {i}: ", end="")
            for k in top_obs:
                print(f"obs={k}({self.B[i,k]:.3f}) ", end="")
            print()
    
    def interpret_states(self, sequences: np.ndarray, 
                        wrist_data: np.ndarray,
                        pre_valence: np.ndarray,
                        pre_arousal: np.ndarray) -> Dict:
        """
        Analyze what each hidden state represents by examining the physio
        and emotion characteristics of interactions assigned to each state.
        
        Parameters
        ----------
        sequences : np.ndarray, shape (n_seq, T)
            Discretized observation sequences.
        wrist_data : np.ndarray, shape (n_seq, 30, 4)
            Raw wrist data for computing summary statistics.
        pre_valence : np.ndarray, shape (n_seq,)
            Pre-listening valence values.
        pre_arousal : np.ndarray, shape (n_seq,)
            Pre-listening arousal values.
        
        Returns
        -------
        state_profiles : dict
            Per-state summary statistics.
        """
        # Get dominant state per interaction via Viterbi
        dominant_states = np.zeros(len(sequences), dtype=int)
        for i, seq in enumerate(sequences):
            path, _ = self.decode(seq)
            # Use the most common state in the sequence
            vals, counts = np.unique(path, return_counts=True)
            dominant_states[i] = vals[np.argmax(counts)]
        
        state_profiles = {}
        for s in range(self.n_states):
            mask = dominant_states == s
            n = mask.sum()
            
            if n == 0:
                state_profiles[s] = {'count': 0, 'label': 'EMPTY'}
                continue
            
            # Physio characteristics
            hr_vals = np.nanmean(wrist_data[mask, :, 0], axis=1)
            intensity_vals = np.nanmean(wrist_data[mask, :, 1], axis=1)
            
            # Activity type distribution
            activity_all = wrist_data[mask, :, 3].flatten().astype(int)
            activity_map = {0: 'Still', 1: 'Act→Still', 2: 'Walking', 
                          3: 'Missing', 4: 'Lying', 5: 'Running'}
            activity_counts = {}
            for val in np.unique(activity_all):
                activity_counts[activity_map.get(int(val), f'Unk({val})')] = \
                    (activity_all == val).sum()
            
            # Emotion characteristics
            v_vals = pre_valence[mask]
            a_vals = pre_arousal[mask]
            
            profile = {
                'count': int(n),
                'pct': f"{n/len(sequences)*100:.1f}%",
                'hr_mean': float(np.mean(hr_vals)),
                'hr_std': float(np.std(hr_vals)),
                'intensity_mean': float(np.mean(intensity_vals)),
                'activity_distribution': activity_counts,
                'pre_valence_mean': float(np.mean(v_vals)),
                'pre_arousal_mean': float(np.mean(a_vals)),
            }
            
            # Auto-label based on physio characteristics
            if profile['hr_mean'] < -5 and profile['intensity_mean'] < 15:
                profile['label'] = 'LOW-ENERGY (rest/lying)'
            elif profile['hr_mean'] > 5 and profile['intensity_mean'] > 25:
                profile['label'] = 'HIGH-ENERGY (active/walking)'
            else:
                profile['label'] = 'MODERATE (transitional)'
            
            state_profiles[s] = profile
        
        return state_profiles, dominant_states
    
    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------
    def save(self, filepath: str):
        """Save HMM parameters to a .npz file."""
        np.savez(filepath,
                 pi=self.pi, A=self.A, B=self.B,
                 n_states=self.n_states, n_obs=self.n_obs,
                 log_likelihoods=np.array(self.log_likelihoods))
        print(f"HMM saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HMM':
        """Load HMM parameters from a .npz file."""
        data = np.load(filepath)
        hmm = cls(n_states=int(data['n_states']), 
                  n_observations=int(data['n_obs']))
        hmm.pi = data['pi']
        hmm.A = data['A']
        hmm.B = data['B']
        hmm.log_likelihoods = data['log_likelihoods'].tolist()
        print(f"HMM loaded from {filepath} ({hmm.n_states} states, {hmm.n_obs} obs)")
        return hmm


# ============================================================
# TRAINING SCRIPT (run as main)
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("HMM TRAINING — SiTunes Wrist Data")
    print("=" * 60)
    
    # --- Load data ---
    wrist_data = np.load("situnes_wrist_data.npy")  # (897, 30, 4)
    df = __import__('pandas').read_csv("situnes_exploration_merged.csv")
    
    print(f"Loaded wrist data: {wrist_data.shape}")
    print(f"Loaded merged data: {df.shape}")
    
    # --- Discretize observations ---
    obs_sequences, disc_info = discretize_wrist_sequences(wrist_data)
    print(f"\nDiscretized observations: {obs_sequences.shape}")
    print(f"Observation range: [{obs_sequences.min()}, {obs_sequences.max()}]")
    print(f"Unique observations: {len(np.unique(obs_sequences))}")
    print(f"Discretization info: {disc_info}")
    
    # --- Train HMM ---
    hmm = HMM(n_states=3, n_observations=27, random_state=42)
    hmm.fit(obs_sequences, n_iter=50, verbose=True)
    
    # --- Print learned parameters ---
    hmm.print_parameters()
    
    # --- Extract belief states ---
    beliefs = hmm.get_belief_states(obs_sequences)
    print(f"\nBelief states shape: {beliefs.shape}")
    print(f"Belief state statistics:")
    for s in range(hmm.n_states):
        print(f"  State {s}: mean={beliefs[:, s].mean():.3f}, "
              f"std={beliefs[:, s].std():.3f}, "
              f"range=[{beliefs[:, s].min():.3f}, {beliefs[:, s].max():.3f}]")
    
    # --- Interpret states ---
    pre_v = df['emo_pre_valence'].values
    pre_a = df['emo_pre_arousal'].values
    state_profiles, dominant_states = hmm.interpret_states(
        obs_sequences, wrist_data, pre_v, pre_a)
    
    print("\n--- State Interpretations ---")
    for s, profile in state_profiles.items():
        print(f"\nState {s} ({profile.get('label', 'N/A')}):")
        for k, v in profile.items():
            if k != 'activity_distribution':
                print(f"  {k}: {v}")
            else:
                print(f"  activity_distribution:")
                for act, cnt in v.items():
                    print(f"    {act}: {cnt}")
    
    # --- Plot convergence ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Convergence curve
    axes[0].plot(hmm.log_likelihoods, marker='o', markersize=3)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Average Log-Likelihood")
    axes[0].set_title("Baum-Welch Convergence")
    axes[0].grid(True, alpha=0.3)
    
    # Belief state distributions
    for s in range(hmm.n_states):
        axes[1].hist(beliefs[:, s], bins=30, alpha=0.6, label=f"State {s}")
    axes[1].set_xlabel("Belief Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Belief State Distributions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # State assignments
    state_counts = np.bincount(dominant_states, minlength=hmm.n_states)
    axes[2].bar(range(hmm.n_states), state_counts)
    axes[2].set_xlabel("Dominant State")
    axes[2].set_ylabel("Interactions")
    axes[2].set_title("Dominant State Distribution")
    axes[2].set_xticks(range(hmm.n_states))
    
    plt.tight_layout()
    plt.savefig("hmm_training_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: hmm_training_results.png")
    
    # --- Save model and outputs ---
    hmm.save("hmm_model.npz")
    np.save("hmm_belief_states.npy", beliefs)
    np.save("hmm_obs_sequences.npy", obs_sequences)
    np.save("hmm_dominant_states.npy", dominant_states)
    
    print("\n✅ HMM training complete.")
    print(f"   Belief states saved: hmm_belief_states.npy ({beliefs.shape})")
    print(f"   Model saved: hmm_model.npz")
    print(f"\n   Next step: Build classifier using beliefs + context → mood bucket")