"""Paper Eqs.21/22 and the original 4 -> 64 -> 64 -> 2 Double DQN."""
from collections import deque
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def reputation(N_plus, N_minus, w4=1.0, w5=1.0):
    return float(w4 * N_plus - w5 * N_minus)


def reward(action, ESP_Utility, MU_Utility, Rep, Rew1=-10.0, omega6=1e-3, omega7=1.0):
    if action not in (0, 1):
        raise ValueError("action must be Renew=0 or Execute=1")
    return float(Rew1 if action == 0 else omega6 * (ESP_Utility + MU_Utility) + omega7 * Rep)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        # Preserve the batch axis even when batch size == 1.
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class DDQNAgent:
    def __init__(self, state_size=4, action_size=2, capacity=50000, *, seed=0,
                 training=True, state_scale=None, batch_size=32, device="cpu",
                 target_tau=0.05):
        self.state_size, self.action_size = state_size, action_size
        self.memory = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
        self.replay_rng = random.Random(seed)
        self.device = torch.device(device)
        self.training = training
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.95, 1.0, 0.01, 0.995
        self.batch_size, self.lr, self.target_tau = batch_size, 1e-3, target_tau
        if not 0 < target_tau <= 1 or batch_size < 1:
            raise ValueError("invalid training configuration")
        self.state_scale = np.ones(state_size) if state_scale is None else np.asarray(state_scale, dtype=float)
        if self.state_scale.shape != (state_size,) or not np.all(np.isfinite(self.state_scale)) or np.any(self.state_scale <= 0):
            raise ValueError("state_scale must be finite and positive")
        # Network initialization is independent of other agents' RNG consumption.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = DQN(state_size, action_size).to(self.device)
            self.target = DQN(state_size, action_size).to(self.device)
        self.update_target(tau=1.0)
        self.target.requires_grad_(False)
        self.target.eval()
        self.model.train(training)
        self.crit = nn.MSELoss()
        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.updates = 0

    def set_training(self, training):
        self.training = bool(training)
        self.model.train(self.training)

    def normalize(self, state):
        x = np.asarray(state, dtype=np.float32)
        if x.shape != (self.state_size,) or not np.all(np.isfinite(x)):
            raise ValueError("invalid state")
        x = x / self.state_scale
        return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)

    def act(self, state, *, evaluate=None, valid_actions=(0, 1)):
        actions = np.asarray(valid_actions, dtype=int)
        if len(actions) == 0 or np.any(actions < 0) or np.any(actions >= self.action_size):
            raise ValueError("invalid action mask")
        explore = self.training if evaluate is None else not evaluate
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(actions))
        with torch.no_grad():
            q = self.model(torch.as_tensor(self.normalize(state), device=self.device)).cpu().numpy()
        return int(actions[np.argmax(q[actions])])

    def remember(self, s, a, r, s2, d, next_valid_actions=(0, 1)):
        mask = np.zeros(self.action_size, dtype=bool)
        mask[list(next_valid_actions)] = True
        if not d and not mask.any():
            raise ValueError("nonterminal state needs a valid next action")
        self.memory.append((self.normalize(s), int(a), float(r), self.normalize(s2), bool(d), mask))

    def bellman_targets(self, rewards, next_states, dones, masks):
        with torch.no_grad():
            # Terminal masks may be empty; bootstrap is removed below.
            masks = masks.clone()
            masks[dones] = True
            q_online = self.model(next_states).masked_fill(~masks, -torch.inf)
            actions = q_online.argmax(dim=1)
            q_target = self.target(next_states).gather(1, actions[:, None]).squeeze(1)
            return rewards + (~dones).float() * self.gamma * q_target

    def replay(self):
        if not self.training or len(self.memory) < self.batch_size:
            return None
        batch = self.replay_rng.sample(list(self.memory), self.batch_size)
        def tensor(column, dtype):
            return torch.as_tensor(np.stack([b[column] for b in batch]), dtype=dtype, device=self.device)
        states, actions = tensor(0, torch.float32), tensor(1, torch.int64)
        rewards, next_states = tensor(2, torch.float32), tensor(3, torch.float32)
        dones, masks = tensor(4, torch.bool), tensor(5, torch.bool)
        q_sa = self.model(states).gather(1, actions[:, None]).squeeze(1)
        target = self.bellman_targets(rewards, next_states, dones, masks)
        loss = self.crit(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.opt.step()
        self.update_target()
        self.updates += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.detach().cpu())

    def update_target(self, tau=None):
        # Algorithm 3 line 17: theta_hat <- mu theta + (1-mu) theta_hat.
        tau = self.target_tau if tau is None else tau
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.model.parameters()):
                target.mul_(1 - tau).add_(online, alpha=tau)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "state_scale": self.state_scale.tolist(),
                    "state_size": self.state_size, "action_size": self.action_size,
                    "format_version": 1}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if ckpt["state_size"] != self.state_size or ckpt["action_size"] != self.action_size:
            raise ValueError("checkpoint dimensions differ")
        self.model.load_state_dict(ckpt["model"])
        self.state_scale = np.asarray(ckpt["state_scale"], dtype=float)
        self.update_target(tau=1.0)
