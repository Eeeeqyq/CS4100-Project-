"""
src/hmm/hmm_model.py
Hidden Markov Model built from scratch using numpy only.

Hidden states (6) represent latent psychological mood:
    0: stressed-high     low valence, high arousal
    1: anxious-neutral   low valence, medium arousal
    2: calm-focused      medium valence, medium arousal
    3: relaxed-low       high valence, low arousal
    4: happy-energetic   high valence, high arousal
    5: fatigued          low valence, low arousal

Observations: integers 0-179 encoding wrist signals + environment.
Encoding: wrist_obs(0-19) * 9 + time_bucket(0-2) * 3 + weather(0-2)
"""

import numpy as np


class HMM:

    STATE_NAMES = [
        "stressed-high",
        "anxious-neutral",
        "calm-focused",
        "relaxed-low",
        "happy-energetic",
        "fatigued",
    ]

    def __init__(self, n_states=6, n_obs=180, seed=42):
        self.n_states = n_states
        self.n_obs    = n_obs
        self.rng      = np.random.default_rng(seed)
        self._init_params()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_params(self):
        n, k = self.n_states, self.n_obs
        # Slight self-loop bias so states persist across timesteps
        A_alpha = np.ones((n, n)) + np.eye(n)
        self.A  = np.array([self.rng.dirichlet(A_alpha[i]) for i in range(n)])
        self.B  = np.array([self.rng.dirichlet(np.ones(k)) for _ in range(n)])
        self.pi = self.rng.dirichlet(np.ones(n))
        self._update_log_params()

    def _update_log_params(self):
        self.log_A  = np.log(self.A  + 1e-300)
        self.log_B  = np.log(self.B  + 1e-300)
        self.log_pi = np.log(self.pi + 1e-300)

    def set_params(self, A, B, pi):
        self.A, self.B, self.pi = A, B, pi
        self._update_log_params()

    # ── Forward algorithm ─────────────────────────────────────────────────

    def forward(self, obs_seq):
        """
        Returns log_alpha (T x n_states) and log_likelihood.
        log_alpha[t, s] = log P(o_1..o_t, state_t=s | model)
        """
        obs_seq   = np.asarray(obs_seq, dtype=int)
        T         = len(obs_seq)
        log_alpha = np.full((T, self.n_states), -np.inf)

        log_alpha[0] = self.log_pi + self.log_B[:, obs_seq[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = (
                    self._logsumexp(log_alpha[t-1] + self.log_A[:, j])
                    + self.log_B[j, obs_seq[t]]
                )

        return log_alpha, self._logsumexp(log_alpha[-1])

    # ── Backward algorithm ────────────────────────────────────────────────

    def backward(self, obs_seq):
        """Returns log_beta (T x n_states)."""
        obs_seq  = np.asarray(obs_seq, dtype=int)
        T        = len(obs_seq)
        log_beta = np.full((T, self.n_states), -np.inf)
        log_beta[-1] = 0.0

        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = self._logsumexp(
                    self.log_A[i, :]
                    + self.log_B[:, obs_seq[t+1]]
                    + log_beta[t+1]
                )
        return log_beta

    # ── Viterbi decoding ──────────────────────────────────────────────────

    def viterbi(self, obs_seq):
        """
        Returns most likely state sequence (T,) and its log probability.
        """
        obs_seq = np.asarray(obs_seq, dtype=int)
        T       = len(obs_seq)
        delta   = np.full((T, self.n_states), -np.inf)
        psi     = np.zeros((T, self.n_states), dtype=int)

        delta[0] = self.log_pi + self.log_B[:, obs_seq[0]]

        for t in range(1, T):
            for j in range(self.n_states):
                scores    = delta[t-1] + self.log_A[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + self.log_B[j, obs_seq[t]]

        states       = np.zeros(T, dtype=int)
        states[-1]   = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states, delta[-1, states[-1]]

    # ── Belief state ──────────────────────────────────────────────────────

    def belief_state(self, obs_seq):
        """
        Posterior over hidden states at the final timestep.
        This is the vector fed to the Q-learning agent.
        Returns shape (n_states,), sums to 1.
        """
        log_alpha, _ = self.forward(obs_seq)
        log_b        = log_alpha[-1] - self._logsumexp(log_alpha[-1])
        return np.exp(log_b)

    # ── Baum-Welch EM training ────────────────────────────────────────────

    def baum_welch(self, sequences, n_iter=100, tol=1e-4, verbose=True):
        """
        Train A, B, pi from observation sequences.

        Parameters
        ----------
        sequences : list of array-like  (each is a sequence of ints 0-179)
        n_iter    : max EM iterations
        tol       : stop early if |delta log-likelihood| < tol
        verbose   : print log-likelihood each iteration

        Returns list of log-likelihood values per iteration.
        """
        sequences = [np.asarray(s, dtype=int) for s in sequences]
        log_likelihoods = []
        prev_ll = -np.inf

        for it in range(n_iter):
            # E-step accumulators
            exp_trans  = np.zeros((self.n_states, self.n_states))
            exp_emis   = np.zeros((self.n_states, self.n_obs))
            exp_init   = np.zeros(self.n_states)
            total_ll   = 0.0

            for obs_seq in sequences:
                T = len(obs_seq)
                if T < 2:
                    continue

                log_alpha, ll = self.forward(obs_seq)
                log_beta      = self.backward(obs_seq)
                total_ll     += ll

                # gamma[t,s] = P(state_t=s | obs)
                log_gamma = log_alpha + log_beta
                log_gamma -= self._logsumexp(log_gamma, axis=1, keepdims=True)
                gamma      = np.exp(log_gamma)

                # xi[t,i,j] = P(state_t=i, state_{t+1}=j | obs)
                for t in range(T-1):
                    log_xi  = (log_alpha[t, :, None]
                               + self.log_A
                               + self.log_B[:, obs_seq[t+1]][None, :]
                               + log_beta[t+1, None, :])
                    log_xi -= self._logsumexp(log_xi.ravel())
                    exp_trans += np.exp(log_xi)

                for t in range(T):
                    exp_emis[:, obs_seq[t]] += gamma[t]

                exp_init += gamma[0]

            # M-step
            self.pi = exp_init / (exp_init.sum() + 1e-300)
            self.A  = exp_trans / (exp_trans.sum(axis=1, keepdims=True) + 1e-300)
            self.B  = exp_emis  / (exp_emis.sum(axis=1,  keepdims=True) + 1e-300)
            self._update_log_params()

            log_likelihoods.append(total_ll)
            delta = total_ll - prev_ll
            if verbose:
                print(f"  iter {it+1:3d}  ll={total_ll:.2f}  delta={delta:+.4f}")

            if it > 0 and abs(delta) < tol:
                if verbose:
                    print(f"  Converged at iteration {it+1}")
                break
            prev_ll = total_ll

        return log_likelihoods

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path):
        np.savez(path, A=self.A, B=self.B, pi=self.pi,
                 n_states=self.n_states, n_obs=self.n_obs)
        print(f"HMM saved → {path}.npz")

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(n_states=int(d["n_states"]), n_obs=int(d["n_obs"]))
        m.set_params(d["A"], d["B"], d["pi"])
        print(f"HMM loaded ← {path}")
        return m

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _logsumexp(a, axis=None, keepdims=False):
        a     = np.asarray(a, dtype=float)
        a_max = np.max(a, axis=axis, keepdims=True)
        a_max = np.where(np.isfinite(a_max), a_max, 0.0)
        out   = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims) + 1e-300)
        if keepdims:
            out += a_max
        else:
            out += np.squeeze(a_max, axis=axis) if axis is not None else a_max.ravel()[0]
        return out