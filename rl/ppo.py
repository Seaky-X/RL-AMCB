# -*- coding: utf-8 -*-
"""
rl/ppo.py  (Scheme B)

Masked PPO simplified to *pure continuous* 6-dim actions in [0,1] using Beta distribution.

Action:
  a = (mS, nS, mU, nU, mN, nN)  each in [0,1]
Env decodes them to integer m/n ranges.

No extra third-party libs beyond torch + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    vf_coef: float = 0.50
    ent_coef: float = 0.01
    lr: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 256
    max_grad_norm: float = 0.5
    device: str = "cpu"


class RolloutBuffer:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logps: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            logp: float,
            reward: float,
            done: bool,
            value: float):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.logps.append(float(logp))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logps.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.states)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 6):
        super().__init__()
        # --- Scheme-B: explicit conditioning on the last 6 "cond" dims ---
        # Default env state (state_dim=18):
        #   main = [pool stats ...] (12 dims)
        #   cond = [alpha,beta,gamma,thr_priv,thr_urg,round_norm] (6 dims)
        # If you change the env state layout, update MAIN_DIM accordingly.
        self.MAIN_DIM = max(1, int(state_dim) - 6)
        self.COND_DIM = int(state_dim) - int(self.MAIN_DIM)

        h = 256
        # main path
        self.main_net = nn.Sequential(
            nn.Linear(self.MAIN_DIM, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
        )
        # cond path
        self.cond_net = nn.Sequential(
            nn.Linear(self.COND_DIM, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
        )
        # FiLM modulation: h_main -> h_main * (1 + g) + b
        self.film_g = nn.Linear(h, h)
        self.film_b = nn.Linear(h, h)

        self.post = nn.Sequential(
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
        )

        # Beta params (a,b) for each action dim
        self.ab_head = nn.Linear(h, action_dim * 2)
        self.v_head = nn.Linear(h, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        main = x[..., : self.MAIN_DIM]
        cond = x[..., self.MAIN_DIM :]
        h_main = self.main_net(main)
        h_cond = self.cond_net(cond)
        g = torch.tanh(self.film_g(h_cond))
        b = self.film_b(h_cond)
        z = self.post(h_main * (1.0 + 0.5 * g) + b)
        ab = self.ab_head(z)  # (B, 12)
        v = self.v_head(z).squeeze(-1)  # (B,)
        a_raw, b_raw = torch.chunk(ab, 2, dim=-1)
        # ensure > 1.0 (stable entropy)
        a = F.softplus(a_raw) + 1.0
        b = F.softplus(b_raw) + 1.0
        return a, b, v

    @staticmethod
    def dist(a: torch.Tensor, b: torch.Tensor) -> Beta:
        return Beta(a, b)


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int = 6, cfg: Optional[PPOConfig] = None):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.net = ActorCritic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg.lr))

    @torch.no_grad()
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, b, v = self.net(s)
        dist = self.net.dist(a, b)
        action = dist.sample()  # (1, action_dim) in (0,1)
        logp = dist.log_prob(action).sum(dim=-1)  # (1,)
        return action.squeeze(0).cpu().numpy(), float(logp.item()), float(v.item())

    @torch.no_grad()
    def act_mean(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deterministic action for analysis/probing (Beta mean = a/(a+b))."""
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, b, _ = self.net(s)
        mean = (a / (a + b)).clamp(0.0, 1.0)
        return mean.squeeze(0).cpu().numpy(), float(mean.mean().item())

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b, v = self.net(states)
        dist = self.net.dist(a, b)
        logp = dist.log_prob(actions).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return logp, ent, v

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        if len(buf) < 8:
            return {"loss": 0.0, "loss_pi": 0.0, "loss_vf": 0.0, "entropy": 0.0}

        states = torch.tensor(np.stack(buf.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack(buf.actions), dtype=torch.float32, device=self.device)
        old_logps = torch.tensor(np.asarray(buf.logps, dtype=np.float32), device=self.device)
        rewards = np.asarray(buf.rewards, dtype=np.float32)
        dones = np.asarray(buf.dones, dtype=np.bool_)
        values = np.asarray(buf.values, dtype=np.float32)

        # GAE advantages
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        last_value = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 0.0 if dones[t] else 1.0
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + float(self.cfg.gamma) * next_value * nonterminal - values[t]
            lastgaelam = delta + float(self.cfg.gamma) * float(self.cfg.gae_lambda) * nonterminal * lastgaelam
            adv[t] = lastgaelam
            last_value = values[t]

        returns = adv + values
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # normalize advantage
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(buf)
        bs = int(min(self.cfg.batch_size, n))

        loss_pi_sum = 0.0
        loss_vf_sum = 0.0
        ent_sum = 0.0
        loss_sum = 0.0
        steps = 0

        for _ in range(int(self.cfg.update_epochs)):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, bs):
                batch_idx = idx[start:start + bs]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logp = old_logps[batch_idx]
                b_adv = adv_t[batch_idx]
                b_ret = ret_t[batch_idx]

                logp, ent, v = self.evaluate(b_states, b_actions)
                ratio = torch.exp(logp - b_old_logp)

                # clipped objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * b_adv
                loss_pi = -torch.min(surr1, surr2).mean()

                # value loss
                loss_vf = F.mse_loss(v, b_ret)

                # entropy bonus
                entropy = ent.mean()

                loss = loss_pi + self.cfg.vf_coef * loss_vf - self.cfg.ent_coef * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), float(self.cfg.max_grad_norm))
                self.opt.step()

                loss_pi_sum += float(loss_pi.item())
                loss_vf_sum += float(loss_vf.item())
                ent_sum += float(entropy.item())
                loss_sum += float(loss.item())
                steps += 1

        if steps <= 0:
            steps = 1
        return {
            "loss": loss_sum / steps,
            "loss_pi": loss_pi_sum / steps,
            "loss_vf": loss_vf_sum / steps,
            "entropy": ent_sum / steps,
        }

    def save(self, path: str) -> None:
        torch.save({"model": self.net.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model", ckpt)
        try:
            self.net.load_state_dict(state, strict=strict)
        except RuntimeError:
            # When architecture changes across experiments, allow a best-effort partial load.
            if strict:
                self.net.load_state_dict(state, strict=False)
            else:
                raise
