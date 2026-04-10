"""
Deep Q-Network agent built from scratch using PyTorch.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.common import ACTION_DIM, STATE_DIM


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buf.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states = np.stack([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.stack([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        lr: float = 1e-3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        buffer_cap: int = 10_000,
        batch_size: int = 64,
        target_sync: int = 10,
        hidden: int = 128,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_sync = target_sync

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cpu")
        self.q_net = QNetwork(state_dim, action_dim, hidden)
        self.target_net = QNetwork(state_dim, action_dim, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(buffer_cap)

        self.episode = 0
        self.train_step = 0

    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return self._greedy(state)

    def greedy_action(self, state) -> int:
        return self._greedy(state)

    def _greedy(self, state) -> int:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    def update(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        s = torch.FloatTensor(states)
        a = torch.LongTensor(actions)
        r = torch.FloatTensor(rewards)
        ns = torch.FloatTensor(next_states)
        d = torch.FloatTensor(dones)

        q_current = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_a = self.q_net(ns).argmax(dim=1)
            next_q = self.target_net(ns).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * next_q * (1.0 - d)

        loss = self.loss_fn(q_current, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step += 1
        return float(loss.item())

    def end_episode(self) -> None:
        self.episode += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if self.episode % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode": self.episode,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden": self.hidden,
            },
            path,
        )
        print(f"Agent saved -> {path}")

    def load(self, path) -> None:
        ck = torch.load(path, map_location="cpu")
        self.q_net.load_state_dict(ck["q_net"])
        self.target_net.load_state_dict(ck["target_net"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.epsilon = ck["epsilon"]
        self.episode = ck["episode"]
        self.state_dim = int(ck.get("state_dim", self.state_dim))
        self.action_dim = int(ck.get("action_dim", self.action_dim))
        self.hidden = int(ck.get("hidden", self.hidden))
        print(f"Agent loaded <- {path}  (episode={self.episode})")
