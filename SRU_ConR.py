# SRU_ConR.py
# ----------------------------------------------------
# SRU_ConR: Smart Reputation Update + Renew decision via DDQN
# - reputation(): Rep = w4 * N_plus - w5 * N_minus
# - reward():     paper-style reward for action {0: renew, 1: execute}
# - DDQNAgent:    GPU-aware Double DQN for the renew decision
# ----------------------------------------------------
"""
SRU_ConR: Reputation & RL Components for Oh-Trust
-------------------------------------------------
This module provides:
  1) A simple, execution-driven reputation function:
       Rep = w4 * N_plus - w5 * N_minus
  2) A paper-style reward function supporting two actions:
       a=0 (renew):   r = Rew1    (typically negative)
       a=1 (execute): r = omega6*(ESP_Utility + MU_Utility) + omega7*Rep
  3) A minimal Double DQN (DDQN) agent used to decide whether to renew or execute.

Notes
-----
- The DDQN implementation is intentionally lightweight for reproducibility.
- If CUDA is available, tensors are moved to GPU automatically.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------- reputation & reward -------------
def reputation(N_plus: int, N_minus: int, w4: float = 1.0, w5: float = 1.0) -> float:
    """
    Compute the execution-driven reputation score.

    Args:
        N_plus (int): Number of fulfilled (positive) contract units.
        N_minus (int): Number of defaulted (negative) contract units.
        w4 (float): Weight for positive executions.
        w5 (float): Weight for negative executions.

    Returns:
        float: Reputation value: w4 * N_plus - w5 * N_minus.
    """
    return float(w4 * N_plus - w5 * N_minus)


def reward(
    action: int,
    ESP_Utility: float,
    MU_Utility: float,
    Rep: float,
    Rew1: float = -10.0,
    omega6: float = 1e-3,
    omega7: float = 1.0,
) -> float:
    """
    Paper-style reward function for the renew/execute decision.

    Policy:
        - If action == 0 (renew):    r = Rew1
        - If action == 1 (execute):  r = omega6 * (ESP_Utility + MU_Utility) + omega7 * Rep

    Args:
        action (int): 0 for renew, 1 for execute.
        ESP_Utility (float): ESP utility realized in this step.
        MU_Utility (float): MU utility realized in this step.
        Rep (float): Reputation value from `reputation(...)`.
        Rew1 (float): Base reward for renewing (typically negative).
        omega6 (float): Weight for total utility.
        omega7 (float): Weight for reputation.

    Returns:
        float: Scalar reward.
    """
    if action == 0:
        return float(Rew1)
    return float(omega6 * (ESP_Utility + MU_Utility) + omega7 * Rep)


# ------------- DDQN Agent -------------
class DQN(nn.Module):
    """
    A small MLP Q-network used by the DDQN agent.

    Architecture:
        input(state_size) -> 64 -> 64 -> output(action_size)

    Notes:
        - Forward returns a tensor of Q-values of shape (action_size,)
          when a single state is provided, or (batch, action_size)
          when a batch is provided.
    """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """
        Forward pass through the Q-network.

        Args:
            x (Tensor): State tensor with shape (state_size,) or (batch, state_size).

        Returns:
            Tensor: Q-values. Shape is (action_size,) for single state,
                    or (batch, action_size) for batches.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.fc3(x)
        return q.squeeze(0) if q.shape[0] == 1 else q


class DDQNAgent:
    """
    Minimal Double DQN agent for the renew/execute decision.

    Key features:
        - Experience replay buffer (FIFO with capacity).
        - Epsilon-greedy exploration with exponential decay.
        - Target network updated via hard copy (update_target()).
        - Double-DQN target: greedy action from online network,
          Q-value from target network.

    Args:
        state_size (int): Dimension of the environment state.
        action_size (int): Number of discrete actions (2 in Oh-Trust).
        capacity (int): Replay buffer capacity.
    """
    def __init__(self, state_size: int, action_size: int, capacity: int = 50000):
        self.action_size = action_size
        self.memory: List[tuple] = []
        self.capacity = capacity

        # RL hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.lr = 1e-3

        # Networks
        self.model = DQN(state_size, action_size).to(DEVICE)
        self.target = DQN(state_size, action_size).to(DEVICE)
        self._sync()

        # Optimizer & loss
        self.crit = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)

    def _sync(self):
        """Hard update: copy online network weights to target network."""
        self.target.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, d):
        """
        Append a transition to the replay buffer (FIFO if full).

        Args:
            s:    state (np.ndarray)
            a:    action (int)
            r:    reward (float)
            s2:   next state (np.ndarray)
            d:    done (bool)
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append((s, a, r, s2, d))

    def act(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Chosen action index.
        """
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(self.action_size))
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
            q = self.model(s)
            return int(torch.argmax(q).item())

    def replay(self):
        """
        Sample a mini-batch and perform one DDQN update step.
        If the buffer has fewer samples than batch_size, no update is performed.
        """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        states = torch.as_tensor([b[0] for b in batch], dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.int64, device=DEVICE)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
        next_states = torch.as_tensor([b[3] for b in batch], dtype=torch.float32, device=DEVICE)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.bool, device=DEVICE)

        # Q(s, a)
        q = self.model(states)
        q_sa = q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: a*(s') from online, Q_target(s', a*)
        with torch.no_grad():
            q_next_online = self.model(next_states)
            a_next = torch.argmax(q_next_online, dim=1)
            q_next_target = self.target(next_states)
            q_next = q_next_target.gather(1, a_next.unsqueeze(1)).squeeze(1)
            target_sa = rewards + (~dones).float() * self.gamma * q_next

        loss = self.crit(q_sa, target_sa)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        """Synchronize target network with online network."""
        self._sync()

    def save(self, path: str):
        """Save the online network weights to disk."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """
        Load weights from disk into the online network and
        immediately synchronize the target network.
        """
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.update_target()
