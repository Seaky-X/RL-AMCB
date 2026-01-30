# -*- coding: utf-8 -*-
"""
rl/train_rl_ppo.py  (Scheme B)

Train PPO on BlockchainEnv Scheme-B (continuous 6-dim action):
  a = (m_SEC, n_SEC, m_URG, n_URG, m_NOR, n_NOR) in [0,1]^6

Run:
  python rl/train_rl_ppo.py --no_load

Logs:
  logs/train_metrics.csv
  python logs/plot_metrics.py --csv logs/train_metrics.csv
"""

from __future__ import annotations

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import sys

import numpy as np

# Ensure project root is on sys.path when running: python rl/train_rl_ppo.py
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from rl.ppo import PPOAgent, PPOConfig, RolloutBuffer
from env.blockchain_env import BlockchainEnv, BlockchainEnvConfig
from env.physionet2012 import PhysioNet2012Stream


CSV_FIELDS = [
    "episode", "step", "reward", "episode_return",
    "req", "alpha", "beta", "gamma", "thr_priv", "thr_urg", "rho",
    "pool_SEC", "pool_URG", "pool_NOR",
    "miss_rate_SEC", "miss_rate_URG", "miss_rate_NOR",
    "m_raw_SEC", "m_raw_URG", "m_raw_NOR",
    "m_SEC", "n_SEC", "n_used_SEC", "m_req_SEC", "m_req_cont_SEC", "n_req_SEC",
    "m_URG", "n_URG", "n_used_URG", "m_req_URG", "m_req_cont_URG", "n_req_URG",
    "m_NOR", "n_NOR", "n_used_NOR", "m_req_NOR", "m_req_cont_NOR", "n_req_NOR", "p_req_SEC", "p_req_URG", "p_req_NOR",
    "n_poolcap_SEC", "n_poolcap_URG", "n_poolcap_NOR",
    "budget_total", "budget_used", "budget_left",
    "S_SEC", "D_SEC", "E_SEC",
    "S_URG", "D_URG", "E_URG",
    "S_NOR", "D_NOR", "E_NOR",
    "phi_SEC", "phi_URG", "phi_NOR", "phi",
    "backlog", "backlog_w", "m_order_violation", "pen_gap", "pen_m_over", "pen_m_under", "pen_secure_fail", "pen_miss",
    "pen_soft_deadline", "pen_alloc_align", "slack_press_SEC", "slack_press_URG", "slack_press_NOR",
        "w_focus_SEC",
    "w_focus_URG",
    "w_focus_NOR",
    "phi_focus",
    "delay_cost_SEC",
    "delay_cost_URG",
    "delay_cost_NOR",
    "age_norm_mean_SEC",
    "age_norm_mean_URG",
    "age_norm_mean_NOR",
    "overdue_frac_SEC",
    "overdue_frac_URG",
    "overdue_frac_NOR",
    "pen_delay",
    "pen_q_age",
    "pen_q_overdue",
    "pen_m_softcap",
    "pen_n_softcap",
    "pen_smooth",
    "queue_cost",
    "cost_total",
    "reward_unscaled",
    "reward_SEC", "reward_URG", "reward_NOR",
    "reward_unscaled_SEC", "reward_unscaled_URG", "reward_unscaled_NOR",
    "cost_SEC", "cost_URG", "cost_NOR",
    "train_reward_SEC", "train_reward_URG", "train_reward_NOR",
    "pen_n_softcap_global",
# demand debug
    "share_sec", "share_urg", "share_nor",
    "priv_sec", "urg_urg", "comp_nor",
    "slack_sec", "slack_urg", "slack_nor",
    "p_sec", "p_urg", "p_nor",
    # --- EXTRA: raw S/D/E score breakdown exported by env (optional) ---
    "T_conf_URG", "T_conf_NOR", "T_conf_SEC",
    "lat_score_URG", "lat_score_NOR", "lat_score_SEC",
    "eff_score_URG", "eff_score_NOR", "eff_score_SEC",
    "T_prop_URG", "T_vote_URG", "T_exec_URG",
    "T_prop_NOR", "T_vote_NOR", "T_exec_NOR",
    "T_prop_SEC", "T_vote_SEC", "T_exec_SEC",
    "E_comm_URG", "E_comp_URG", "E_store_URG",
    "E_comm_NOR", "E_comp_NOR", "E_store_NOR",
    "E_comm_SEC", "E_comp_SEC", "E_store_SEC",

]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # training loop
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--steps_per_ep", type=int, default=50)
    p.add_argument("--update_every_ep", type=int, default=4)

    # data
    p.add_argument("--data_mode", type=str, default="physionet", choices=["physionet", "synthetic"])
    p.add_argument("--set_dir", type=str, default="rl/data/set-a")
    p.add_argument("--outcomes", type=str, default="rl/data/Outcomes-a.txt")

    # env basics
    p.add_argument("--warmup_rounds", type=int, default=5)
    p.add_argument("--tx_gen_rate", type=int, default=280)
    p.add_argument("--max_txpool_capacity", type=int, default=10000)

    # m/n ranges
    p.add_argument("--m_min", type=int, default=8)
    p.add_argument("--m_max", type=int, default=24)
    p.add_argument("--n_min", type=int, default=40)
    p.add_argument("--n_max", type=int, default=200)

    # global budget / n-allocation
    p.add_argument("--budget_mode", type=str, default="global", choices=["global", "per_channel"])
    p.add_argument("--n_total_max", type=int, default=300)
    p.add_argument("--n_queue_gain", type=float, default=1.0)
    p.add_argument("--lambda_n_waste", type=float, default=2.0)

    # deadlines
    p.add_argument("--sec_deadline_lo", type=int, default=10)
    p.add_argument("--sec_deadline_hi", type=int, default=20)
    p.add_argument("--urg_deadline_lo", type=int, default=2)
    p.add_argument("--urg_deadline_hi", type=int, default=6)
    p.add_argument("--nor_deadline_lo", type=int, default=6)
    p.add_argument("--nor_deadline_hi", type=int, default=12)

    # thresholds
    p.add_argument("--q_priv", type=float, default=0.80)
    p.add_argument("--q_urg", type=float, default=0.80)
    p.add_argument("--min_thr", type=float, default=0.55)
    p.add_argument("--override_channel", action="store_true")
    p.add_argument("--no_override_channel", action="store_true")

    # demand shaping knobs
    p.add_argument("--w_sec_pressure", type=float, default=1.6)
    p.add_argument("--w_priv_mean", type=float, default=0.6)
    p.add_argument("--w_slack_sec", type=float, default=0.35)

    p.add_argument("--w_urg_pressure", type=float, default=1.2)
    p.add_argument("--w_urg_mean", type=float, default=0.3)
    p.add_argument("--w_slack_urg", type=float, default=0.6)
    p.add_argument("--slack_quantile", type=float, default=0.10)

    p.add_argument("--w_nor_pressure", type=float, default=1.0)
    p.add_argument("--w_comp_mean", type=float, default=0.8)
    p.add_argument("--w_slack_nor", type=float, default=0.2)

    # reward
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--throughput_bonus", type=float, default=0.05)
    p.add_argument("--queue_penalty", type=float, default=0.20)
    p.add_argument("--miss_penalty", type=float, default=2.0)
    p.add_argument("--lambda_gap", type=float, default=0.60)
    # NOTE(v5): stronger regularizer to avoid always-saturating m at m_max.
    p.add_argument("--lambda_m_over", type=float, default=10.0)
    p.add_argument("--lambda_m_under_sec", type=float, default=3.0)
    p.add_argument("--lambda_m_under_other", type=float, default=0.25)
    p.add_argument("--lambda_secure_fail", type=float, default=5.0)

    # v6 focus + anti-explosion
    p.add_argument("--focus_other_w", type=float, default=0.05)
    p.add_argument("--lambda_delay", type=float, default=1.0)
    p.add_argument("--lambda_age", type=float, default=0.6)
    p.add_argument("--lambda_overdue", type=float, default=1.2)
    p.add_argument("--lambda_m_softcap", type=float, default=6.0)
    p.add_argument("--lambda_n_softcap", type=float, default=2.0)
    p.add_argument("--lambda_smooth", type=float, default=0.5)
    p.add_argument("--m_softcap_margin_sec", type=float, default=6.0)
    p.add_argument("--m_softcap_margin_other", type=float, default=4.0)
    p.add_argument("--risk_max_scan", type=int, default=2000)

    # training: avoid sparse gradients from focus-onehot; always give each channel a floor weight
    p.add_argument("--train_w_floor", type=float, default=0.35)

    # D/E semantics
    # By default, we assume ConsensusModule returns D/E as normalized scores in (0,1]
    # (higher is better). If your ConsensusModule returns raw costs (seconds/joules),
    # enable --de_is_cost.
    p.add_argument("--de_is_cost", action="store_true")

    # v5 shaping
    p.add_argument("--lambda_soft_deadline", type=float, default=0.80)
    p.add_argument("--soft_deadline_k", type=float, default=0.25)
    p.add_argument("--lambda_alloc_align", type=float, default=0.30)

    # curriculum generalization (Dirichlet around the scenario corner)
    p.add_argument("--curriculum_dirichlet_kappa", type=float, default=40.0)
    p.add_argument("--curriculum_uniform_prob", type=float, default=0.10)
    p.add_argument("--curriculum_mix_max", type=float, default=0.50)

    p.add_argument("--p_secure_req_sec", type=float, default=0.95)
    p.add_argument("--p_secure_req_urg", type=float, default=0.75)
    p.add_argument("--p_secure_req_nor", type=float, default=0.85)
    p.add_argument("--p_secure_req_sec_min", type=float, default=0.80)
    p.add_argument("--p_secure_req_urg_min", type=float, default=0.55)
    p.add_argument("--p_secure_req_nor_min", type=float, default=0.65)
    p.add_argument("--p_secure_req_alpha_pow", type=float, default=1.0)
    p.add_argument("--p_secure_req_order_gap", type=float, default=0.005)

    # network/consensus
    p.add_argument("--total_nodes", type=int, default=50)
    p.add_argument("--base_malicious_ratio", type=float, default=0.20)
    p.add_argument("--offline_prob", type=float, default=0.05)
    p.add_argument("--net_bw_mbps", type=float, default=50.0)
    p.add_argument("--base_net_lat", type=float, default=0.10)

    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--clip_eps", type=float, default=0.20)
    p.add_argument("--vf_coef", type=float, default=0.50)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_load", action="store_true")
    p.add_argument("--model_path", type=str, default="logs/ppo_schemeB.pt")
    return p


def _resolve_to_root(root: Path, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return str((root / p).resolve())


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def main() -> None:
    args = build_argparser().parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    LOG_DIR = ROOT / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    set_dir = _resolve_to_root(ROOT, str(args.set_dir))
    outcomes = _resolve_to_root(ROOT, str(args.outcomes))

    physio_stream: Optional[PhysioNet2012Stream] = None
    data_mode = str(args.data_mode)
    if data_mode == "physionet":
        if (not set_dir) or (not outcomes) or (not os.path.isdir(set_dir)) or (not os.path.isfile(outcomes)):
            print("[WARN] data_mode=physionet but set_dir/outcomes invalid -> fallback synthetic.")
            data_mode = "synthetic"
        else:
            physio_stream = PhysioNet2012Stream(
                set_dir=set_dir,
                outcomes_path=outcomes,
                seed=int(args.seed),
                shuffle=True,
                cache=True,
            )
            print(f"[DATA] mode=physionet set_dir={set_dir} outcomes={outcomes}")

    override_channel = True
    if bool(args.no_override_channel):
        override_channel = False
    if bool(args.override_channel):
        override_channel = True

    env_cfg = BlockchainEnvConfig(
        max_steps=int(args.steps_per_ep),
        warmup_rounds=int(args.warmup_rounds),

        m_min=int(args.m_min), m_max=int(args.m_max),
        n_min=int(args.n_min), n_max=int(args.n_max),
        budget_mode=str(getattr(args, "budget_mode", "global")),
        n_total_max=int(getattr(args, "n_total_max", 300)),
        n_queue_gain=float(getattr(args, "n_queue_gain", 1.0)),

        tx_gen_rate=int(args.tx_gen_rate),
        max_txpool_capacity=int(args.max_txpool_capacity),

        sec_deadline_range=(int(args.sec_deadline_lo), int(args.sec_deadline_hi)),
        urg_deadline_range=(int(args.urg_deadline_lo), int(args.urg_deadline_hi)),
        nor_deadline_range=(int(args.nor_deadline_lo), int(args.nor_deadline_hi)),

        q_priv=float(args.q_priv),
        q_urg=float(args.q_urg),
        min_thr=float(args.min_thr),
        override_channel=bool(override_channel),

        total_nodes=int(args.total_nodes),
        base_malicious_ratio=float(args.base_malicious_ratio),
        offline_prob=float(args.offline_prob),
        net_bw_mbps=float(args.net_bw_mbps),
        base_net_lat=float(args.base_net_lat),

        reward_scale=float(args.reward_scale),
        throughput_bonus=float(args.throughput_bonus),
        queue_penalty=float(args.queue_penalty),
        miss_penalty=float(args.miss_penalty),

        lambda_gap=float(args.lambda_gap),
        lambda_m_over=float(args.lambda_m_over),
        lambda_m_under_sec=float(args.lambda_m_under_sec),
        lambda_m_under_other=float(args.lambda_m_under_other),
        lambda_secure_fail=float(args.lambda_secure_fail),

        focus_other_w=float(args.focus_other_w),
        lambda_delay=float(args.lambda_delay),
        lambda_age=float(args.lambda_age),
        lambda_overdue=float(args.lambda_overdue),
        lambda_m_softcap=float(args.lambda_m_softcap),
        lambda_n_softcap=float(args.lambda_n_softcap),
            lambda_n_waste=float(args.lambda_n_waste),
        lambda_smooth=float(args.lambda_smooth),
        m_softcap_margin_sec=float(args.m_softcap_margin_sec),
        m_softcap_margin_other=float(args.m_softcap_margin_other),
        risk_max_scan=int(args.risk_max_scan),

        de_is_score=bool(not args.de_is_cost),

        lambda_soft_deadline=float(args.lambda_soft_deadline),
        soft_deadline_k=float(args.soft_deadline_k),
        lambda_alloc_align=float(args.lambda_alloc_align),
        curriculum_dirichlet_kappa=float(args.curriculum_dirichlet_kappa),
        curriculum_uniform_prob=float(args.curriculum_uniform_prob),
        curriculum_mix_max=float(args.curriculum_mix_max),

        p_secure_req_sec=float(args.p_secure_req_sec),
        p_secure_req_urg=float(args.p_secure_req_urg),
        p_secure_req_nor=float(args.p_secure_req_nor),
        p_secure_req_sec_min=float(args.p_secure_req_sec_min),
        p_secure_req_urg_min=float(args.p_secure_req_urg_min),
        p_secure_req_nor_min=float(args.p_secure_req_nor_min),
        p_secure_req_alpha_pow=float(args.p_secure_req_alpha_pow),
        p_secure_req_order_gap=float(args.p_secure_req_order_gap),

        # demand shaping
        w_sec_pressure=float(args.w_sec_pressure),
        w_priv_mean=float(args.w_priv_mean),
        w_slack_sec=float(args.w_slack_sec),

        w_urg_pressure=float(args.w_urg_pressure),
        w_urg_mean=float(args.w_urg_mean),
        w_slack_urg=float(args.w_slack_urg),
        slack_quantile=float(args.slack_quantile),

        w_nor_pressure=float(args.w_nor_pressure),
        w_comp_mean=float(args.w_comp_mean),
        w_slack_nor=float(args.w_slack_nor),

        seed=int(args.seed),
    )

    env = BlockchainEnv(cfg=env_cfg, data_mode=data_mode, physio_stream=physio_stream, seed=int(args.seed))

    ppo_cfg = PPOConfig(
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_eps=float(args.clip_eps),
        vf_coef=float(args.vf_coef),
        ent_coef=float(args.ent_coef),
        lr=float(args.lr),
        update_epochs=int(args.update_epochs),
        batch_size=int(args.batch_size),
        device="cpu",
    )
    agent_sec = PPOAgent(state_dim=int(env.state_dim), action_dim=2, cfg=ppo_cfg)
    agent_urg = PPOAgent(state_dim=int(env.state_dim), action_dim=2, cfg=ppo_cfg)
    agent_nor = PPOAgent(state_dim=int(env.state_dim), action_dim=2, cfg=ppo_cfg)

    # Save/load three independent policies (no shared weights across channels)
    model_path = _resolve_to_root(ROOT, str(args.model_path))
    mp_root, mp_ext = os.path.splitext(str(model_path)) if model_path else ('', '.pt')
    if not mp_ext:
        mp_ext = '.pt'
    model_paths = {
        'SEC': f"{mp_root}_SEC{mp_ext}" if mp_root else '',
        'URG': f"{mp_root}_URG{mp_ext}" if mp_root else '',
        'NOR': f"{mp_root}_NOR{mp_ext}" if mp_root else '',
    }

    def _try_load(agent, path: str) -> None:
        if path and (not args.no_load) and os.path.isfile(path):
            try:
                agent.load(path)
                print(f"[MODEL] loaded: {path}")
            except Exception as e:
                print(f"[WARN] failed to load model ({path}): {e}")

    _try_load(agent_sec, model_paths['SEC'])
    _try_load(agent_urg, model_paths['URG'])
    _try_load(agent_nor, model_paths['NOR'])

    print(f"[ENV] state_dim={env.state_dim} action_dim={env.action_dim} (3x2) device={ppo_cfg.device}")

    csv_path = LOG_DIR / "train_metrics.csv"
    print(f"[LOG] csv -> {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        def _csv_safe(v):
            """Convert values to csv-writable scalars; strip NaN/Inf/null-bytes to avoid Windows OSError 22."""
            try:
                import math
                import numpy as _np
            except Exception:
                math = None
                _np = None
            if v is None:
                return ""
            # numpy scalars
            try:
                if _np is not None and isinstance(v, (_np.floating, _np.integer)):
                    v = v.item()
            except Exception:
                pass
            if isinstance(v, (bool, int)):
                return int(v)
            if isinstance(v, float):
                if math is not None and (math.isnan(v) or math.isinf(v)):
                    return ""
                return float(v)
            s = str(v)
            if "\x00" in s:
                s = s.replace("\x00", "")
            return s

        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        buf_sec = RolloutBuffer()
        buf_urg = RolloutBuffer()
        buf_nor = RolloutBuffer()

        for ep in range(int(args.episodes)):
            s, _ = env.reset()
            ep_ret = 0.0
            last_info: Dict[str, Any] = {}

            # episode-level diagnostics (avoid只看最后一步导致误判)
            req_counts = {"SEC": 0, "URG": 0, "NOR": 0}
            req_act_sum = {k: [0.0] * 6 for k in req_counts}
            req_act_n = {k: 0 for k in req_counts}
            req_m_sum = {k: [0.0, 0.0, 0.0] for k in req_counts}
            req_n_sum = {k: [0.0, 0.0, 0.0] for k in req_counts}
            req_mn_n = {k: 0 for k in req_counts}

            upd_sec = None
            upd_urg = None
            upd_nor = None

            for t in range(int(args.steps_per_ep)):
                aS, logpS, vS = agent_sec.act(s)
                aU, logpU, vU = agent_urg.act(s)
                aN, logpN, vN = agent_nor.act(s)

                # combine to 6-D action: (SEC_m,SEC_n, URG_m,URG_n, NOR_m,NOR_n)
                a = [float(aS[0]), float(aS[1]), float(aU[0]), float(aU[1]), float(aN[0]), float(aN[1])]
                ns, _, r, done, info = env.step(a)

                # per-step diagnostics by req
                req = str(info.get("req", "NOR"))
                if req in req_counts:
                    req_counts[req] += 1
                    req_act_n[req] += 1
                    for i in range(6):
                        req_act_sum[req][i] += float(a[i])
                    req_m_sum[req][0] += float(info.get("m_SEC", 0.0))
                    req_m_sum[req][1] += float(info.get("m_URG", 0.0))
                    req_m_sum[req][2] += float(info.get("m_NOR", 0.0))
                    req_n_sum[req][0] += float(info.get("n_used_SEC", 0.0))
                    req_n_sum[req][1] += float(info.get("n_used_URG", 0.0))
                    req_n_sum[req][2] += float(info.get("n_used_NOR", 0.0))
                    req_mn_n[req] += 1

                # train each policy with per-channel reward every step (avoid drift when not focus)
                wS = float(info.get('w_focus_SEC', 0.0))
                wU = float(info.get('w_focus_URG', 0.0))
                wN = float(info.get('w_focus_NOR', 0.0))
                rS = float(info.get('reward_SEC', float(r)))
                rU = float(info.get('reward_URG', float(r)))
                rN = float(info.get('reward_NOR', float(r)))
                # focus-onehot still acts as a *bias* (stronger gradient when focus), but never shuts off learning
                w_floor = float(np.clip(getattr(args, 'train_w_floor', 0.35), 0.0, 1.0))
                twS = float(w_floor + (1.0 - w_floor) * wS)
                twU = float(w_floor + (1.0 - w_floor) * wU)
                twN = float(w_floor + (1.0 - w_floor) * wN)
                trS = float(rS) * float(twS)
                trU = float(rU) * float(twU)
                trN = float(rN) * float(twN)
                info['train_reward_SEC'] = float(trS)
                info['train_reward_URG'] = float(trU)
                info['train_reward_NOR'] = float(trN)
                buf_sec.add(s, aS, logpS, float(trS), done, vS)
                buf_urg.add(s, aU, logpU, float(trU), done, vU)
                buf_nor.add(s, aN, logpN, float(trN), done, vN)

                last_info = dict(info)
                ep_ret += float(r)

                # log CSV row
                row = {k: "" for k in CSV_FIELDS}
                row.update({"episode": int(ep), "step": int(t), "reward": _csv_safe(float(r)), "episode_return": _csv_safe(float(last_info.get("episode_return", ep_ret)))})
                for k in CSV_FIELDS:
                    if k in info:
                        row[k] = _csv_safe(info[k])
                writer.writerow(row)

                s = ns
                if done:
                    break

            # PPO update every N episodes
            if int(args.update_every_ep) > 0 and ((ep + 1) % int(args.update_every_ep) == 0):
                upd_sec = agent_sec.update(buf_sec)
                upd_urg = agent_urg.update(buf_urg)
                upd_nor = agent_nor.update(buf_nor)
                buf_sec.clear(); buf_urg.clear(); buf_nor.clear()

                for ch, ag, path in [('SEC', agent_sec, model_paths.get('SEC','')), ('URG', agent_urg, model_paths.get('URG','')), ('NOR', agent_nor, model_paths.get('NOR',''))]:
                    if path:
                        try:
                            ag.save(path)
                        except Exception:
                            pass

            # episode print
            li = dict(last_info)
            if not li:
                li = {"episode_return": ep_ret}
            li["episode_return"] = float(li.get("episode_return", ep_ret))

            # req fraction summary
            denom = float(max(1, req_counts['SEC'] + req_counts['URG'] + req_counts['NOR']))
            req_frac = {k: float(req_counts[k]) / denom for k in req_counts}

            def _f(x, d=0.0):
                try:
                    return float(x)
                except Exception:
                    return float(d)

            def _mean3(arr_sum, n):
                n = float(max(1, n))
                return (arr_sum[0]/n, arr_sum[1]/n, arr_sum[2]/n)

            m_sec = _mean3(req_m_sum['SEC'], req_mn_n['SEC'])
            m_urg = _mean3(req_m_sum['URG'], req_mn_n['URG'])
            m_nor = _mean3(req_m_sum['NOR'], req_mn_n['NOR'])
            n_sec = _mean3(req_n_sum['SEC'], req_mn_n['SEC'])
            n_urg = _mean3(req_n_sum['URG'], req_mn_n['URG'])
            n_nor = _mean3(req_n_sum['NOR'], req_mn_n['NOR'])

            msg = (
                f"[EP {ep:4d}] R={ep_ret:8.2f} req={li.get('req','?')} "
                f"aβγ=({_f(li.get('alpha')):.3f},{_f(li.get('beta')):.3f},{_f(li.get('gamma')):.3f}) "
                f"rho={_f(li.get('rho')):.3f} "
                f"m_raw=(S:{li.get('m_raw_SEC','?')},U:{li.get('m_raw_URG','?')},N:{li.get('m_raw_NOR','?')}) "
                f"m=(S:{li.get('m_SEC','?')},U:{li.get('m_URG','?')},N:{li.get('m_NOR','?')}) "
                f"n_used=(S:{li.get('n_used_SEC','?')},U:{li.get('n_used_URG','?')},N:{li.get('n_used_NOR','?')}) "
                f"pool=(S:{li.get('pool_SEC','?')},U:{li.get('pool_URG','?')},N:{li.get('pool_NOR','?')}) "
                f"miss=(S:{_f(li.get('miss_rate_SEC')):.3f},U:{_f(li.get('miss_rate_URG')):.3f},N:{_f(li.get('miss_rate_NOR')):.3f}) "
                f"phi={_f(li.get('phi')):.3f} gap={_f(li.get('pen_gap')):.3f} "
                f"thr=({_f(li.get('thr_priv')):.2f},{_f(li.get('thr_urg')):.2f})"
            )
            msg += f" req_frac=(S:{req_frac['SEC']:.2f},U:{req_frac['URG']:.2f},N:{req_frac['NOR']:.2f})"
            msg += f" m_req=(S:{_f(li.get('m_req_SEC')):.0f},U:{_f(li.get('m_req_URG')):.0f},N:{_f(li.get('m_req_NOR')):.0f})"
            msg += f" m_req_cont=(S:{_f(li.get('m_req_cont_SEC')):.2f},U:{_f(li.get('m_req_cont_URG')):.2f},N:{_f(li.get('m_req_cont_NOR')):.2f})"
            msg += f" p_req=(S:{_f(li.get('p_req_SEC')):.3f},U:{_f(li.get('p_req_URG')):.3f},N:{_f(li.get('p_req_NOR')):.3f})"
            msg += f" diag(m/n)=(S:{m_sec[0]:.1f}/{n_sec[0]:.1f},U:{m_urg[1]:.1f}/{n_urg[1]:.1f},N:{m_nor[2]:.1f}/{n_nor[2]:.1f})"

            if upd_sec or upd_urg or upd_nor:
                def _u(u):
                    if not u:
                        return '(none)'
                    return f"loss={u.get('loss',0.0):.4f} pi={u.get('loss_pi',0.0):.4f} vf={u.get('loss_vf',0.0):.4f} ent={u.get('entropy',0.0):.4f}"
                msg += f" | UPDATE SEC[{_u(upd_sec)}] URG[{_u(upd_urg)}] NOR[{_u(upd_nor)}]"
            print(msg)

    print(f"[DONE] CSV saved to: {csv_path}")
    print("[PLOT] python logs/plot_metrics.py --csv logs/train_metrics.csv")


if __name__ == "__main__":
    main()
