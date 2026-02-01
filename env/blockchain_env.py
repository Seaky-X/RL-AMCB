# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

import math
import numpy as np

from env.txpool import TxPool, TxPoolConfig, Tx
from env.consensus import ConsensusModule
from env.physionet2012 import PhysioNet2012Stream, RecordDemand


# ---------------------------
# helpers
# ---------------------------

def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _binom_tail_prob(m: int, p: float, k_min: int) -> float:
    """P(X >= k_min), X ~ Binomial(m, p)."""
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    s = 0.0
    for k in range(int(k_min), int(m) + 1):
        s += math.comb(int(m), int(k)) * (p ** k) * ((1.0 - p) ** (m - k))
    return float(np.clip(s, 0.0, 1.0))


def _p_pbft_secure(m: int, rho: float) -> float:
    """
    PBFT-ish success probability:
    succeed if #malicious <= floor((m-1)/3)
    X ~ Binomial(m, rho)
    """
    m = int(max(1, m))
    f = int((m - 1) // 3)
    # P(X <= f) = 1 - P(X >= f+1)
    return float(np.clip(1.0 - _binom_tail_prob(m, rho, f + 1), 0.0, 1.0))


def _m_req_from_rho(rho: float, p_req: float, m_min: int, m_max: int) -> int:
    """Minimal m such that PBFT-ish secure prob >= p_req."""
    rho = float(np.clip(float(rho), 0.0, 0.49))
    p_req = float(np.clip(float(p_req), 0.5, 0.999))
    for m in range(int(m_min), int(m_max) + 1):
        if _p_pbft_secure(m, rho) >= p_req:
            return int(m)
    return int(m_max)


def _m_req_cont_from_rho(rho: float, p_req: float, m_min: int, m_max: int) -> float:
    """Continuous (fractional) committee requirement.

    Returns a real-valued m_req in [m_min, m_max] by linearly interpolating
    between the two nearest integers around the target security requirement.
    This provides smoother shaping for PPO while keeping execution m integer.
    """
    rho = float(np.clip(float(rho), 0.0, 0.49))
    p_req = float(np.clip(float(p_req), 0.5, 0.999))
    m_min = int(m_min); m_max = int(m_max)
    if m_max <= m_min:
        return float(m_min)
    p_min = _p_pbft_secure(m_min, rho)
    if p_req <= p_min:
        return float(m_min)
    p_max = _p_pbft_secure(m_max, rho)
    if p_req >= p_max:
        return float(m_max)
    for m in range(m_min, m_max):
        p1 = _p_pbft_secure(m, rho)
        p2 = _p_pbft_secure(m + 1, rho)
        if p1 <= p_req <= p2:
            if p2 <= p1 + 1e-12:
                return float(m + 1)
            t = (p_req - p1) / (p2 - p1)
            return float(m) + float(np.clip(t, 0.0, 1.0))
    return float(m_max)


# ---------------------------
# Config (robust kwargs)
# ---------------------------

@dataclass(init=False)
class BlockchainEnvConfig:
    # episode
    max_steps: int = 50
    warmup_rounds: int = 5

    # m/n ranges
    m_min: int = 4
    m_max: int = 100
    n_min: int = 10
    n_max: int = 300

    # tx generation / pool
    tx_gen_rate: int = 40
    max_txpool_capacity: int = 10000

    # deadlines (offset rounds)
    sec_deadline_range: Tuple[int, int] = (10, 20)
    urg_deadline_range: Tuple[int, int] = (2, 6)
    nor_deadline_range: Tuple[int, int] = (6, 12)

    # channel trigger thresholds (quantile with floor)
    q_priv: float = 0.80
    q_urg: float = 0.80
    min_thr: float = 0.55
    override_channel: bool = True  # use stream-provided channel assignment

    # curriculum / diversity (training aid). Default ON to avoid single-channel collapse.
    use_curriculum: bool = True
    # probabilities for sampling a per-episode scenario (SEC/URG/NOR)
    curriculum_mix: Tuple[float, float, float] = (0.30, 0.35, 0.35)
    # If True, deterministically rotate scenarios across episodes to guarantee diversity
    curriculum_force_cycle: bool = True
    curriculum_cycle_order: Tuple[str, str, str] = ('SEC', 'URG', 'NOR')
    # with probability force_req_prob (annealed), override (alpha,beta,gamma) by scenario
    force_req_prob: float = 0.70
    force_req_prob_min: float = 0.15
    force_req_anneal_episodes: int = 800
    # make SEC/URG easier to appear by lowering quantile thresholds in that scenario
    curriculum_q_priv: float = 0.75
    curriculum_q_urg: float = 0.85
    # deadline range shrinking factors to increase queue pressure in a scenario
    curriculum_deadline_shrink_sec: float = 0.90
    curriculum_deadline_shrink_urg: float = 0.65


    # consensus/network
    total_nodes: int = 50
    base_malicious_ratio: float = 0.20
    offline_prob: float = 0.05
    net_bw_mbps: float = 50.0
    base_net_lat: float = 0.10

    # reward weights
    reward_scale: float = 1.0
    throughput_bonus: float = 0.05  # per tx
    queue_penalty: float = 0.20     # backlog pressure
    miss_penalty: float = 2.0       # per missed ratio unit
    lambda_gap: float = 0.60
    # NOTE(v5): Oversized committees were frequently saturating at m_max because
    # the (m - m_req) regularizer was too weak compared to reward_scale*phi.
    # We strengthen it so PPO has a learnable trade-off instead of "always max".
    lambda_m_over: float = 10.0
    lambda_m_under_sec: float = 3.0
    lambda_m_under_other: float = 0.25
    lambda_secure_fail: float = 5.0

    # --- v6: focus-onehot reward + anti-explosion shaping ---
    # Reward aggregation uses req-focused one-hot weights instead of directly
    # using alpha/beta/gamma to mix 3 channels.
    focus_other_w: float = 0.05

    # sharpen focus weights (w_focus^focus_pow then renormalize).
    # Larger -> more one-hot, improves channel separation.
    focus_pow: float = 2.0

    # Mix a small demand prior into *penalty* aggregation weights (NOT into reward mixing).
    # w_pen = normalize((1-eta)*w_focus + eta*w_dem), where w_dem ~ (alpha,beta,gamma).
    demand_reward_mix: float = 0.25

    # nonlinearity for demand-conditioned targets (e.g., (1-beta)^target_pow).
    target_pow: float = 2.0

    # demand-conditioned targets (use *normalized cost* units: delay_cost=D/lat_ref, energy_cost=E/eng_ref).
    delay_target_urg_lo: float = 0.8
    delay_target_urg_hi: float = 1.6
    energy_target_nor_lo: float = 0.6
    energy_target_nor_hi: float = 1.4
    lambda_delay_target: float = 1.8
    lambda_energy_target: float = 1.4

    # soft-deadline early-trigger (round-based; uses txpool slack_q/min_slack).
    slack_thr: float = 8.0
    slack_thr_gain_alpha: float = 6.0
    slack_thr_gain_beta: float = 6.0
    slack_thr_gain_gamma: float = 6.0
    slack_press_pow: float = 2.0
    lambda_soft_deadline: float = 1.2

    # enforce ordering as a *constraint* (projection) rather than as a reward penalty
    order_as_constraint: bool = False
    m_order_margin: int = 0

    # delay / queue risk penalties
    lambda_delay: float = 1.0
    lambda_age: float = 0.6
    lambda_overdue: float = 1.2

    # soft caps to prevent m/n explosions
    lambda_m_softcap: float = 0.2
    lambda_n_softcap: float = 2.0

    # system-level ordering / monotonicity constraint
    # target: m_SEC >= m_NOR >= m_URG
    lambda_m_order: float = 0.3

    # smooth actions to avoid oscillation
    lambda_smooth: float = 0.5

    # margins for m softcap around m_req_cont
    m_softcap_margin_sec: float = 6.0
    m_softcap_margin_other: float = 4.0

    m_softcap_ratio: float = 1.0  # allow (1+ratio)*m_req before softcap penalty
    # queue risk scanning
    risk_max_scan: int = 2000

    # --- D/E semantics ---
    # In this project, ConsensusModule.consensus_round() often returns D/E as
    # normalized *scores* in (0,1] where higher is better (e.g., exp(-latency),
    # exp(-energy)). If you instead use raw cost semantics (seconds/joules), set
    # this to False.
    de_is_score: bool = False

    # When de_is_score=False, the consensus module returns raw COSTS: D=delay (smaller is better), E=energy (smaller is better).
    # We convert costs to scores in (0,1] via 1/(1 + cost/scale).
    lat_scale: float = 50.0
    lat_ref: float = 0.0  # <=0 means fallback to lat_scale
    energy_scale: float = 50.0

    eng_ref: float = 0.0  # <=0 means fallback to energy_scale
    # How to interpret per-channel n proposals:
    # - "per_channel": each channel independently uses n_used[ch]=min(n_raw[ch], pool[ch]) (no cross-channel filling).
    # - "global": share a total throughput budget across channels by demand weights.
    budget_mode: str = "per_channel"

    # Reward magnitude stabilization (applied to BOTH per-channel rewards and global reward).
    # - "log": sign(x)*log1p(|x|)
    # - "tanh": tanh(x)
    # - "none": no transform
    # - "sqrt": sign(x)*sqrt(|x|)
    reward_transform: str = "none"
    reward_clip: float = 500.0

    # --- v5: shaping for generalization ---
    # Soft deadline: penalize getting close to deadlines even before expiry.
    lambda_soft_deadline: float = 0.8
    soft_deadline_k: float = 0.25  # exp(-k*slack)
    # Encourage allocation shares to follow demand (alpha,beta,gamma).
    lambda_alloc_align: float = 0.15

    # Curriculum weights sampling (avoid fixed corners; improve interpolation/generalization)
    curriculum_dirichlet_kappa: float = 40.0

    # security requirement per channel (for m_req)
    p_secure_req_sec: float = 0.95
    p_secure_req_urg: float = 0.75
    p_secure_req_nor: float = 0.85

    # --- dynamic committee requirement (m_req) ---
    # Lower bounds; upper bounds are p_secure_req_* above.
    # These make m_req respond continuously to (alpha,beta,gamma).
    p_secure_req_sec_min: float = 0.80
    p_secure_req_urg_min: float = 0.55
    p_secure_req_nor_min: float = 0.65
    p_secure_req_alpha_pow: float = 1.0
    p_secure_req_beta_pow: float = 1.0
    p_secure_req_gamma_pow: float = 1.0
    p_secure_req_order_gap: float = 0.005
    p_secure_req_req_boost: float = 0.00
    p_secure_req_other_relax: float = 0.00

    # demand vector shaping (to make SEC appear)
    w_sec_pressure: float = 1.6
    w_priv_mean: float = 0.6
    w_slack_sec: float = 0.35

    w_urg_pressure: float = 1.2
    w_urg_mean: float = 0.3
    w_slack_urg: float = 0.6
    slack_quantile: float = 0.10  # >0 reduces URG domination by using slack quantile instead of min

    w_nor_pressure: float = 1.0
    w_comp_mean: float = 0.8
    w_slack_nor: float = 0.2

    # misc
    seed: int = 0

    # channel names (kept in config for backward compatibility with older code
    # paths that iterate over cfg.channels)
    channels: tuple = ("SEC", "URG", "NOR")

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # clamp some
        self.max_steps = int(max(1, getattr(self, "max_steps", 50)))
        self.warmup_rounds = int(max(0, getattr(self, "warmup_rounds", 5)))
        self.m_min = int(max(1, getattr(self, "m_min", 4)))
        self.m_max = int(max(self.m_min, getattr(self, "m_max", 100)))
        self.n_min = int(max(0, getattr(self, "n_min", 10)))
        self.n_max = int(max(self.n_min, getattr(self, "n_max", 300)))
        self.tx_gen_rate = int(max(0, getattr(self, "tx_gen_rate", 40)))
        self.max_txpool_capacity = int(max(100, getattr(self, "max_txpool_capacity", 10000)))
        self.total_nodes = int(max(4, getattr(self, "total_nodes", 50)))

        # normalize channels
        chs = getattr(self, "channels", ("SEC", "URG", "NOR"))
        if isinstance(chs, (list, tuple)) and len(chs) > 0:
            self.channels = tuple(str(x) for x in chs)
        else:
            self.channels = ("SEC", "URG", "NOR")


# ---------------------------
# Env
# ---------------------------

class BlockchainEnv:
    """
    3-channel blockchain env with Scheme-B action space.

    Action (continuous, 6 dims in [0,1]):
      a = (mS, nS, mU, nU, mN, nN)
    """

    CHANNELS = ("SEC", "URG", "NOR")

    def __init__(self,
                 cfg: BlockchainEnvConfig,
                 data_mode: str = "physionet",
                 physio_stream: Optional[PhysioNet2012Stream] = None,
                 seed: int = 0):
        self.cfg = cfg
        self.data_mode = str(data_mode)
        self.stream = physio_stream
        self.rng = np.random.RandomState(int(seed))

        self.txpool = TxPool(cfg=TxPoolConfig(max_capacity=int(cfg.max_txpool_capacity)), seed=int(seed))
        self.consensus = ConsensusModule(
            total_nodes=int(cfg.total_nodes),
            base_malicious_ratio=float(cfg.base_malicious_ratio),
            offline_prob=float(cfg.offline_prob),
        )

        self.n_channels = 3
        self.action_dim = 6

        # running stats
        self.round_id = 0
        self.step_id = 0
        self.ep_return = 0.0

        # episode-level curriculum scenario (SEC/URG/NOR) to ensure training diversity
        self.episode_id = 0
        self._scenario: Optional[str] = None
        self._req_src: str = 'pool'

        self._thr_priv = float(cfg.min_thr)
        self._thr_urg = float(cfg.min_thr)
        self._ema_thr = 0.2  # smoothing
        self._last_alpha = 1/3
        self._last_beta = 1/3
        self._last_gamma = 1/3
        self._last_req = "NOR"

        self._last_miss_rate = {"SEC": 0.0, "URG": 0.0, "NOR": 0.0}

        # state dim computed by _get_state
        s0, _ = self.reset()
        self.state_dim = int(len(s0))


    def _txpool_stats_all(self, now_round: int) -> Dict[str, Dict[str, float]]:
        """
        Unify TxPool.stats() differences across versions.

        - In your current TxPool (txpool.py), stats(now_round=..) returns a dict of all channels:
            { "SEC": {...}, "URG": {...}, "NOR": {...} }
        - Some older env versions used stats(ch, now_round=..) returning a single-channel dict.

        This helper always returns ALL-channel stats.
        """
        try:
            st_all = self.txpool.stats(now_round=int(now_round), slack_quantile=float(getattr(self.cfg, 'slack_quantile', 0.0)))
            if isinstance(st_all, dict) and ("SEC" in st_all or "URG" in st_all or "NOR" in st_all):
                # normalize missing keys
                return {
                    "SEC": dict(st_all.get("SEC", {})),
                    "URG": dict(st_all.get("URG", {})),
                    "NOR": dict(st_all.get("NOR", {})),
                }
        except TypeError:
            pass
        except Exception:
            # fall through to legacy probing
            pass

        out: Dict[str, Dict[str, float]] = {}
        for ch in self.CHANNELS:
            try:
                st = self.txpool.stats(ch, now_round=int(now_round), slack_quantile=float(getattr(self.cfg, 'slack_quantile', 0.0)))
                out[ch] = dict(st) if isinstance(st, dict) else {}
            except Exception:
                out[ch] = {}
        return out

    # -----------------------
    # core API
    # -----------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        # TxPool compatibility: some versions expose reset(), some only clear().
        if hasattr(self.txpool, "reset") and callable(getattr(self.txpool, "reset")):
            self.txpool.reset()
        elif hasattr(self.txpool, "clear") and callable(getattr(self.txpool, "clear")):
            self.txpool.clear()
        else:
            # last-resort: reinitialize internal queues if present
            if hasattr(self.txpool, "qs") and isinstance(getattr(self.txpool, "qs"), dict):
                for k in list(self.txpool.qs.keys()):
                    try:
                        self.txpool.qs[k].clear()
                    except Exception:
                        self.txpool.qs[k] = []
        self.round_id = 0
        self.step_id = 0
        self.ep_return = 0.0
        self._thr_priv = float(self.cfg.min_thr)
        self._thr_urg = float(self.cfg.min_thr)

        # Scenario sampling (curriculum): per-episode diversity to prevent req collapse
        self.episode_id += 1
        self._scenario = None
        if bool(getattr(self.cfg, 'use_curriculum', False)):
            if bool(getattr(self.cfg, 'curriculum_force_cycle', True)):
                order = list(getattr(self.cfg, 'curriculum_cycle_order', ('SEC','URG','NOR')))
                if not order:
                    order = ['SEC','URG','NOR']
                idx = int((self.episode_id - 1) % len(order))
                self._scenario = str(order[idx]).upper()
            else:
                mix = getattr(self.cfg, 'curriculum_mix', (0.33, 0.33, 0.34))
                try:
                    p_sec, p_urg, p_nor = float(mix[0]), float(mix[1]), float(mix[2])
                except Exception:
                    p_sec, p_urg, p_nor = 0.33, 0.33, 0.34
                psum = max(1e-9, p_sec + p_urg + p_nor)
                p = np.asarray([p_sec/psum, p_urg/psum, p_nor/psum], dtype=np.float64)
                self._scenario = str(self.rng.choice(['SEC','URG','NOR'], p=p)).upper()


        # warmup fills pool so policy sees meaningful state
        # NOTE: warmup is only for filling; we still keep a sane time order:
        #   generate -> (round end) drop_expired
        for _ in range(int(self.cfg.warmup_rounds)):
            self.round_id += 1
            self._generate_transactions(now_round=int(self.round_id))
            self.txpool.drop_expired(now_round=int(self.round_id))

        # reset per-episode miss rates

        self._last_miss_rate = {"SEC": 0.0, "URG": 0.0, "NOR": 0.0}

        s = self._get_state()
        mask = np.ones((3,), dtype=np.float32)  # kept for compatibility; Scheme B doesn't need it
        return s, mask

    def step(self, action: Any) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict[str, Any]]:

        # 0) snapshot current round and pool state (decision is based on CURRENT pool)
        now_round = int(self.round_id)
        q0_sec = int(self.txpool.q_len("SEC"))
        q0_urg = int(self.txpool.q_len("URG"))
        q0_nor = int(self.txpool.q_len("NOR"))
        q0_sum = float(max(1.0, float(q0_sec + q0_urg + q0_nor)))
        q0 = {"SEC": int(q0_sec), "URG": int(q0_urg), "NOR": int(q0_nor)}

        # 1) demand vector from CURRENT pool + thresholds
        alpha, beta, gamma, req, dbg_dem = self._compute_demand_vector()

        

        # demand fractions used for per-round global n allocation
        req_frac = {
            "SEC": float(dbg_dem.get("share_sec", 0.0)),
            "URG": float(dbg_dem.get("share_urg", 0.0)),
            "NOR": float(dbg_dem.get("share_nor", 0.0)),
        }
        _s = req_frac["SEC"] + req_frac["URG"] + req_frac["NOR"]
        if _s > 0.0:
            req_frac["SEC"] /= _s
            req_frac["URG"] /= _s
            req_frac["NOR"] /= _s
        else:
            req_frac = {"SEC": 1.0 / 3.0, "URG": 1.0 / 3.0, "NOR": 1.0 / 3.0}
        # v14: req-focused aggregation weights (one-hot + small other weight),
        # then sharpen with focus_pow to increase separation.
        w_other = float(np.clip(getattr(self.cfg, 'focus_other_w', 0.05), 0.0, 0.49))
        w_main = float(max(0.0, 1.0 - 2.0 * w_other))
        w_focus = {'SEC': float(w_other), 'URG': float(w_other), 'NOR': float(w_other)}
        w_focus[str(req)] = float(w_main)
        focus_pow = float(max(1e-6, float(getattr(self.cfg, 'focus_pow', 1.0))))
        if abs(focus_pow - 1.0) > 1e-6:
            wf_tmp = {ch: float(w_focus[ch]) ** focus_pow for ch in self.CHANNELS}
            s_tmp = float(sum(float(v) for v in wf_tmp.values()))
            if s_tmp > 0.0:
                w_focus = {ch: float(wf_tmp[ch]) / s_tmp for ch in self.CHANNELS}

        # Demand prior used ONLY for penalty aggregation (NOT for reward mixing)
        w_dem = {'SEC': float(alpha), 'URG': float(beta), 'NOR': float(gamma)}
        s_dem = float(w_dem['SEC'] + w_dem['URG'] + w_dem['NOR'])
        if s_dem > 0.0:
            w_dem = {ch: float(w_dem[ch]) / s_dem for ch in self.CHANNELS}
        else:
            w_dem = {ch: 1.0 / 3.0 for ch in self.CHANNELS}
        eta = float(np.clip(getattr(self.cfg, 'demand_reward_mix', 0.0), 0.0, 1.0))
        w_pen = {ch: float((1.0 - eta) * float(w_focus[ch]) + eta * float(w_dem[ch])) for ch in self.CHANNELS}
        s_pen = float(sum(float(v) for v in w_pen.values()))
        if s_pen > 0.0:
            w_pen = {ch: float(w_pen[ch]) / s_pen for ch in self.CHANNELS}

        # 2) decode action (6 dims -> per channel m,n)
        a = self._normalize_action(action)
        mS = self._decode_int(a[0], self.cfg.m_min, self.cfg.m_max)
        nS = self._decode_int(a[1], self.cfg.n_min, self.cfg.n_max)
        mU = self._decode_int(a[2], self.cfg.m_min, self.cfg.m_max)
        nU = self._decode_int(a[3], self.cfg.n_min, self.cfg.n_max)
        mN = self._decode_int(a[4], self.cfg.m_min, self.cfg.m_max)
        nN = self._decode_int(a[5], self.cfg.n_min, self.cfg.n_max)

        m_map_raw = {"SEC": int(mS), "URG": int(mU), "NOR": int(mN)}
        n_map = {"SEC": int(nS), "URG": int(nU), "NOR": int(nN)}

        # keep a raw copy for diagnostics / legacy references
        n_map_raw = dict(n_map)

        # 3) sample rho ONCE (important to align)
        rho = float(np.clip(self.consensus.sample_dynamic_malicious_ratio(), 0.0, 0.49))

        # 4) dynamic m_req: continuous in (alpha,beta,gamma)
        #    - alpha/beta/gamma come from CURRENT demand vector (SEC/URG/NOR).
        #    - map each weight to a per-channel security requirement p_req_ch (continuous)
        #      then convert to continuous committee target m_req_cont via _m_req_cont_from_rho().
        #    - execution m is still integer, and under v5-C we ONLY hard-clamp SEC by m_req (URG/NOR are soft-penalized), and we DO NOT enforce cross-channel ordering.
        #         (i) m_SEC >= ceil(m_req_cont_SEC)
        #        (ii) no enforced m order; we only log order violation
        #      RL still *chooses* m; environment only applies the SEC hard clamp.

        # use local (alpha,beta,gamma) for consistency
        a_eff = float(np.clip(alpha, 0.0, 1.0)) ** float(max(1e-6, self.cfg.p_secure_req_alpha_pow))
        b_eff = float(np.clip(beta, 0.0, 1.0)) ** float(max(1e-6, getattr(self.cfg, 'p_secure_req_beta_pow', 1.0)))
        g_eff = float(np.clip(gamma, 0.0, 1.0)) ** float(max(1e-6, getattr(self.cfg, 'p_secure_req_gamma_pow', 1.0)))

        # allow explicit *_hi; fallback to existing p_secure_req_*
        p_hi_sec = float(getattr(self.cfg, 'p_secure_req_sec_hi', self.cfg.p_secure_req_sec))
        p_hi_urg = float(getattr(self.cfg, 'p_secure_req_urg_hi', self.cfg.p_secure_req_urg))
        p_hi_nor = float(getattr(self.cfg, 'p_secure_req_nor_hi', self.cfg.p_secure_req_nor))

        p_req_sec = float(self.cfg.p_secure_req_sec_min) + a_eff * (p_hi_sec - float(self.cfg.p_secure_req_sec_min))
        # URG/NOR: requirement should *decrease* when beta/gamma increase (prefer latency/energy over security).
        p_req_urg = float(self.cfg.p_secure_req_urg_min) + (1.0 - b_eff) * (p_hi_urg - float(self.cfg.p_secure_req_urg_min))
        p_req_nor = float(self.cfg.p_secure_req_nor_min) + (1.0 - g_eff) * (p_hi_nor - float(self.cfg.p_secure_req_nor_min))
        # optional small boost to demanded channel (SEC only by default)
        if req == 'SEC':
            p_req_sec += float(self.cfg.p_secure_req_req_boost)
        p_req_sec = float(np.clip(p_req_sec, 0.50, 0.999))
        p_req_nor = float(np.clip(p_req_nor, 0.50, 0.999))
        p_req_urg = float(np.clip(p_req_urg, 0.50, 0.999))
        # v5-C: do NOT enforce p_req structural ordering; keep per-channel p_req (clipped)

        p_req = {'SEC': p_req_sec, 'URG': p_req_urg, 'NOR': p_req_nor}

        m_req_cont = {
            'SEC': _m_req_cont_from_rho(rho, p_req['SEC'], self.cfg.m_min, self.cfg.m_max),
            'URG': _m_req_cont_from_rho(rho, p_req['URG'], self.cfg.m_min, self.cfg.m_max),
            'NOR': _m_req_cont_from_rho(rho, p_req['NOR'], self.cfg.m_min, self.cfg.m_max),
        }
        m_req = {k: int(min(self.cfg.m_max, max(self.cfg.m_min, int(math.ceil(v))))) for k, v in m_req_cont.items()}

        # --- v5-C: relax structural constraints, keep only SEC hard m-under ---

        # 1) Do NOT enforce p_req ordering; keep p_req as-is (clipped).

        # 2) Execution m: keep per-channel policy output; only SEC is hard-clamped by m_req.

        # 3) n_used: allocate global n budget fairly (proportional rounding), not URG->NOR->SEC.


        # p_req order violation (for logging only)

        p_order_violation = float(0.0 if (p_req['SEC'] >= p_req['NOR'] >= p_req['URG']) else 1.0)

        # NOTE(v5-C): m_req_cont is derived from (rho, p_req) above; we do NOT override it again with an extra 'pressure' mapping here.

        # executed m (only SEC hard lower bound; URG/NOR are soft-penalized later if under)

        mS = int(np.clip(int(m_map_raw['SEC']), self.cfg.m_min, self.cfg.m_max))

        mU = int(np.clip(int(m_map_raw['URG']), self.cfg.m_min, self.cfg.m_max))

        mN = int(np.clip(int(m_map_raw['NOR']), self.cfg.m_min, self.cfg.m_max))

        mS = int(max(mS, int(m_req['SEC'])))  # SEC hard clamp

        # ordering as constraint (projection): keep SEC highest, NOR middle, URG lowest.
        # This avoids a strong "ordering penalty" that tends to push all m upward.
        if bool(getattr(self.cfg, 'order_as_constraint', True)):
            mm = int(getattr(self.cfg, 'm_order_margin', 0))
            # Enforce: m_SEC >= m_NOR + mm >= m_URG + 2mm.
            # Use a *downward-first* projection to avoid the 'cascade lifting' effect
            # (one large m in a low-priority channel forces all channels to be large).
            mU = int(np.clip(mU, self.cfg.m_min, self.cfg.m_max))
            mN = int(np.clip(mN, self.cfg.m_min, self.cfg.m_max))
            mS = int(np.clip(mS, self.cfg.m_min, self.cfg.m_max))

            # Step 1: try shrinking NOR/URG to satisfy ordering under current SEC.
            if mN > mS - mm:
                mN = int(np.clip(mS - mm, self.cfg.m_min, self.cfg.m_max))
            if mU > mN - mm:
                mU = int(np.clip(mN - mm, self.cfg.m_min, self.cfg.m_max))

            # Step 2: if margin makes it infeasible w.r.t. m_min, minimally lift upward (rare).
            if mU < self.cfg.m_min:
                mU = int(self.cfg.m_min)
            if mN < mU + mm:
                mN = int(np.clip(mU + mm, self.cfg.m_min, self.cfg.m_max))
            if mS < mN + mm:
                mS = int(np.clip(mN + mm, self.cfg.m_min, self.cfg.m_max))

            # Re-apply SEC hard clamp after projection (SEC is security-sensitive).
            mS = int(max(mS, int(m_req['SEC'])))
            # If SEC clamp breaks ordering, shrink lower-priority committees again.
            if mN > mS - mm:
                mN = int(np.clip(mS - mm, self.cfg.m_min, self.cfg.m_max))
            if mU > mN - mm:
                mU = int(np.clip(mN - mm, self.cfg.m_min, self.cfg.m_max))


        m_map = {'SEC': mS, 'URG': mU, 'NOR': mN}


        # --------------- n/m used ---------------

        # pool availability was already measured at the beginning of step() via txpool.q_len()
        # keep a capped view (for logging only)
        n_poolcap = {
            'SEC': int(min(int(n_map_raw['SEC']), int(q0_sec))),
            'URG': int(min(int(n_map_raw['URG']), int(q0_urg))),
            'NOR': int(min(int(n_map_raw['NOR']), int(q0_nor))),
        }
        # global budget (learned): derive a usable [0, n_max] budget from action nS/nU/nN

        # global budget: optionally share a total n budget across channels.
        # - budget_mode='per_channel': each channel uses its own n proposal, capped by its own pool.
        # - budget_mode='global'     : share a single budget_total across channels (proportional to demand).
        budget_mode = str(getattr(self.cfg, 'budget_mode', 'per_channel')).lower()

        # per-channel pool caps
        q_pool = {ch: int(self.txpool.q_len(ch)) if hasattr(self.txpool, 'q_len') else int(self.txpool.size(ch)) for ch in self.CHANNELS}
        n_poolcap = {ch: int(min(int(n_map_raw[ch]), int(q_pool[ch]))) for ch in self.CHANNELS}

        # informational total budget (used only in global mode; in per_channel it is just for logging)
        budget_total_raw = int(n_map_raw['SEC'] + n_map_raw['URG'] + n_map_raw['NOR'])
        budget_total = int(np.clip(budget_total_raw - 3 * int(self.cfg.n_min), 0, int(self.cfg.n_max)))  # [0,n_max]

        def _alloc_proportional(weights: dict, total: int) -> dict:
            if total <= 0:
                return {k: 0 for k in weights}
            ks = list(weights.keys())
            w = np.array([float(max(0.0, weights[k])) for k in ks], dtype=np.float64)
            if float(w.sum()) <= 1e-12:
                w = np.ones_like(w)
            w = w / float(w.sum())
            raw = w * float(total)
            alloc = {k: int(np.floor(raw[i])) for i, k in enumerate(ks)}
            used = int(sum(alloc.values()))
            left = int(total - used)
            if left > 0:
                frac = [(raw[i] - np.floor(raw[i]), ks[i]) for i in range(len(ks))]
                frac.sort(reverse=True)
                for j in range(left):
                    alloc[frac[j % len(frac)][1]] += 1
            return alloc

        if budget_mode in ('global', 'shared', 'total'):
            # greedy order: prioritize current req channel
            order = [req] + [c for c in self.CHANNELS if c != req]
            # demand weights for this round
            w = {ch: float(req_frac.get(ch, 0.0)) for ch in self.CHANNELS}
            n_req = _alloc_proportional(w, total=int(budget_total))

            n_used_ch = {ch: 0 for ch in self.CHANNELS}
            budget_left = int(budget_total)
            for ch in order:
                want = int(n_req[ch])
                take = int(min(want, int(n_poolcap[ch]), int(budget_left)))
                n_used_ch[ch] = take
                budget_left -= take

            # any remaining budget is unused (NO cross-filling beyond each channel's pool)
            budget_used = int(sum(int(n_used_ch[ch]) for ch in self.CHANNELS))
        else:
            # independent per-channel n (NO cross-filling; NO shared total constraint)
            n_used_ch = {ch: int(n_poolcap[ch]) for ch in self.CHANNELS}
            n_req = {ch: int(n_used_ch[ch]) for ch in self.CHANNELS}
            budget_used = int(sum(int(n_used_ch[ch]) for ch in self.CHANNELS))
            budget_left = int(max(0, int(budget_total) - int(budget_used)))





        # 5.9) queue waiting / timeout risk (PRE-BLOCK, before popping txs)
        qr = {'SEC': {'age_norm_mean': 0.0, 'overdue_frac': 0.0},
              'URG': {'age_norm_mean': 0.0, 'overdue_frac': 0.0},
              'NOR': {'age_norm_mean': 0.0, 'overdue_frac': 0.0}}
        if hasattr(self.txpool, 'queue_risk'):
            try:
                qr = self.txpool.queue_risk(now_round=now_round, max_scan=int(getattr(self.cfg, 'risk_max_scan', 2000)))
            except Exception:
                pass
        age_norm_mean = {ch: float(qr.get(ch, {}).get('age_norm_mean', 0.0)) for ch in self.CHANNELS}
        overdue_frac = {ch: float(qr.get(ch, {}).get('overdue_frac', 0.0)) for ch in self.CHANNELS}
        # pre-pop queue lengths (for backlog/queue_cost penalty)
        try:
            q_len_pre = {ch: float(self.txpool.q_len(ch)) for ch in self.CHANNELS}
        except Exception:
            q_len_pre = {ch: float(len(getattr(self.txpool, 'qs', {}).get(ch, []))) for ch in self.CHANNELS}

        # 6) process channels in pipeline order: select txs + consensus
        order = ("URG", "NOR", "SEC")
        res_ch: Dict[str, Dict[str, float]] = {}
        def _txpool_pop(ch: str, n_used: int, now_round: int) -> List[Tx]:
            """TxPool API compatibility: returns picked tx list."""
            tp = self.txpool
            out = None
            if hasattr(tp, "pop_n"):
                out = tp.pop_n(ch=ch, n_used=int(n_used), now_round=int(now_round))
            elif hasattr(tp, "select_for_block"):
                out = tp.select_for_block(ch=ch, n_used=int(n_used), now_round=int(now_round))
            else:
                out = []

            # select_for_block returns list[Tx] in your current txpool.py
            if isinstance(out, list):
                return out
            # pop_n returns (picked, missed)
            if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], list):
                return out[0]
            return []


        popped_map: Dict[str, List[Tx]] = {"SEC": [], "URG": [], "NOR": []}


        for ch in order:
            n_used = int(n_used_ch[ch])
            # pop txs for this block (based on CURRENT pool)
            picked = _txpool_pop(ch=ch, n_used=n_used, now_round=now_round)
            popped_map[ch] = list(picked) if picked is not None else []

            # consensus metrics (block delay)
            res = self.consensus.consensus_round(
                m=int(m_map[ch]),
                n=int(n_used),
                net_bw_mbps=float(self.cfg.net_bw_mbps),
                base_net_lat=float(self.cfg.base_net_lat),
                round_id=int(now_round),
                malicious_ratio=rho,
            )
            res_ch[ch] = res
        # pipeline confirmation time (raw delay): URG confirms first, then NOR, then SEC
        D_urg = float(res_ch["URG"].get("D", 0.0))
        D_nor = float(res_ch["NOR"].get("D", 0.0))
        D_sec = float(res_ch["SEC"].get("D", 0.0))
        T_urg = float(D_urg)
        T_nor = float(D_urg + D_nor)
        T_sec = float(D_urg + D_nor + D_sec)


        # 7) round end: drop expired on REMAINING txs (this makes "URG not enough → expire → penalty" learnable)
        missed_now = self.txpool.drop_expired(now_round=now_round)

        # miss rate after round end drop
        missed_rate_ch: Dict[str, float] = {}
        # TxPool.stats() in this project returns ALL-channel stats (no per-channel miss_rate).
        # We compute miss_rate from "expired at round end" vs (expired + popped + remaining).
        for ch in self.CHANNELS:
            missed_cnt = float(missed_now.get(ch, 0))
            popped_cnt = float(len(popped_map.get(ch, [])))
            remain_cnt = float(self.txpool.q_len(ch))
            denom = float(max(1.0, missed_cnt + popped_cnt + remain_cnt))
            missed_rate_ch[ch] = float(np.clip(missed_cnt / denom, 0.0, 1.0))
        self._last_miss_rate = dict(missed_rate_ch)

        # 8) per-channel utility (channel-specific preference; aggregated by req focus weights)
        eps = 1e-9
        # 8) per-channel utility weights (sum to 1).
        #    SEC: prioritize security with alpha; allocate remaining to delay/energy by beta/gamma ratio.
        #    URG: delay weight increases with beta (urgent -> latency sensitive).
        #    NOR: energy weight increases with gamma (normal -> efficiency sensitive).
        eps = 1e-9
        s_ag = float(alpha + beta + gamma)
        if s_ag <= eps:
            aN = bN = gN = 1.0 / 3.0
        else:
            aN = float(alpha) / s_ag
            bN = float(beta) / s_ag
            gN = float(gamma) / s_ag

        # SEC
        aS_sec = float(np.clip(0.30 + 0.65 * a_eff, 0.20, 0.95))
        rest = float(max(0.0, 1.0 - aS_sec))
        rbg = float(bN + gN)
        frac_b = float(bN / max(eps, rbg))
        aD_sec = float(rest * frac_b)
        aE_sec = float(rest - aD_sec)

        # URG (beta -> stricter delay objective)
        aS_urg = 0.02
        aD_urg = float(np.clip(0.45 + 0.50 * b_eff, 0.35, 0.95))
        aE_urg = float(max(0.0, 1.0 - aS_urg - aD_urg))

        # NOR (gamma -> stricter energy objective)
        aS_nor = 0.05
        aE_nor = float(np.clip(0.40 + 0.55 * g_eff, 0.30, 0.95))
        aD_nor = float(max(0.0, 1.0 - aS_nor - aE_nor))

        pref = {
            'SEC': (float(aS_sec), float(aD_sec), float(aE_sec)),
            'URG': (float(aS_urg), float(aD_urg), float(aE_urg)),
            'NOR': (float(aS_nor), float(aD_nor), float(aE_nor)),
        }

        phi_ch: Dict[str, float] = {}
        # confirmation delay under pipeline (seconds)
        T_map = {'URG': float(T_urg), 'NOR': float(T_nor), 'SEC': float(T_sec)}

        # Normalize raw delay/energy to [0,1] scores for the utility; scales keep reward magnitude stable.
        # D is delay (smaller is better) -> score in (0,1]; E is energy (smaller is better) -> score in (0,1].
        lat_ref = float(getattr(self.cfg, 'lat_ref', 0.0))
        if lat_ref <= 0.0:
            lat_ref = float(getattr(self.cfg, 'lat_scale', 50.0))
        lat_ref = float(max(1e-6, lat_ref))
        eng_ref = float(getattr(self.cfg, 'eng_ref', 0.0))
        if eng_ref <= 0.0:
            eng_ref = float(getattr(self.cfg, 'energy_scale', 20.0))
        eng_ref = float(max(1e-6, eng_ref))
        D_raw_map = {ch: float(max(0.0, float(T_map.get(ch, 0.0)))) for ch in self.CHANNELS}
        E_raw_map = {ch: float(max(0.0, float(res_ch[ch].get('E', 0.0)))) for ch in self.CHANNELS}
        lat_score_map = {ch: float(1.0 / (1.0 + (D_raw_map[ch] / lat_ref))) for ch in self.CHANNELS}
        eff_score_map = {ch: float(1.0 / (1.0 + (E_raw_map[ch] / eng_ref))) for ch in self.CHANNELS}

        for ch in self.CHANNELS:
            aS, aD, aE = pref[ch]
            S = float(res_ch[ch].get('S', 0.0))
            Dm = float(T_map.get(ch, float(res_ch[ch].get('D', 1.0))))
            Em = float(res_ch[ch].get('E', 1.0))
            if bool(getattr(self.cfg, 'de_is_score', False)):
                lat = float(np.clip(Dm, 0.0, 1.0))
                eff = float(np.clip(Em, 0.0, 1.0))
            else:
                lat = float(lat_score_map.get(ch, 0.0))
                eff = float(eff_score_map.get(ch, 0.0))
            phi_ch[ch] = float(aS * S + aD * lat + aE * eff)

        # v6: focus-onehot aggregation (instead of alpha/beta/gamma mixing)
        phi_focus = float(sum(float(w_focus[ch]) * float(phi_ch[ch]) for ch in self.CHANNELS))
        phi = float(phi_focus)

        # 9) penalties / shaping
        q_sec = float(q_len_pre.get('SEC', self.txpool.q_len('SEC') if hasattr(self.txpool,'q_len') else 0.0))
        q_urg = float(q_len_pre.get('URG', self.txpool.q_len('URG') if hasattr(self.txpool,'q_len') else 0.0))
        q_nor = float(q_len_pre.get('NOR', self.txpool.q_len('NOR') if hasattr(self.txpool,'q_len') else 0.0))
        backlog = float((q_sec + q_urg + q_nor) / max(1.0, float(self.cfg.max_txpool_capacity)))
        backlog_w = float((float(w_focus['SEC']) * q_sec + float(w_focus['URG']) * q_urg + float(w_focus['NOR']) * q_nor) / max(1.0, float(self.cfg.max_txpool_capacity)))
        queue_cost = float(self.cfg.queue_penalty) * float(backlog_w)

        # 9.1 delay penalty (confirmation delay under pipeline)
        if bool(getattr(self.cfg, 'de_is_score', False)):
            delay_cost_ch = {ch: float(np.clip(1.0 - float(T_map.get(ch, 0.0)), 0.0, 1.0)) for ch in self.CHANNELS}
        else:
            # raw delay -> normalized cost in [0, +inf); use lat_ref to keep scale stable
            delay_cost_ch = {ch: float(max(0.0, float(D_raw_map.get(ch, 0.0))) / lat_ref) for ch in self.CHANNELS}

        lam_delay = float(getattr(self.cfg, 'lambda_delay', 1.0))

        # energy normalized cost (used for NOR target shaping)
        if bool(getattr(self.cfg, 'de_is_score', False)):
            energy_cost_ch = {ch: float(np.clip(1.0 - float(eff_score_map.get(ch, 0.0)), 0.0, 1.0)) for ch in self.CHANNELS}
        else:
            energy_cost_ch = {ch: float(E_raw_map.get(ch, 0.0)) / float(max(1e-9, eng_ref)) for ch in self.CHANNELS}

        # demand-conditioned targets: URG delay target follows beta, NOR energy target follows gamma.
        target_pow = float(max(1e-6, float(getattr(self.cfg, 'target_pow', 1.0))))
        t_delay_urg = float(getattr(self.cfg, 'delay_target_urg_lo', 0.8)) + (float(getattr(self.cfg, 'delay_target_urg_hi', 1.6)) - float(getattr(self.cfg, 'delay_target_urg_lo', 0.8))) * float((1.0 - b_eff) ** target_pow)
        t_energy_nor = float(getattr(self.cfg, 'energy_target_nor_lo', 0.6)) + (float(getattr(self.cfg, 'energy_target_nor_hi', 1.4)) - float(getattr(self.cfg, 'energy_target_nor_lo', 0.6))) * float((1.0 - g_eff) ** target_pow)
        # hinge on ratio to trigger early (even before hard deadline / miss happens)
        x_d = float(max(0.0, float(delay_cost_ch.get('URG', 0.0)) / max(1e-9, t_delay_urg) - 1.0))
        x_e = float(max(0.0, float(energy_cost_ch.get('NOR', 0.0)) / max(1e-9, t_energy_nor) - 1.0))
        pen_delay_target_ch = {ch: 0.0 for ch in self.CHANNELS}
        pen_energy_target_ch = {ch: 0.0 for ch in self.CHANNELS}
        pen_delay_target_ch['URG'] = float(getattr(self.cfg, 'lambda_delay_target', 0.0)) * float(0.5 + 0.5 * b_eff) * float(x_d * x_d)
        pen_energy_target_ch['NOR'] = float(getattr(self.cfg, 'lambda_energy_target', 0.0)) * float(0.5 + 0.5 * g_eff) * float(x_e * x_e)
        pen_delay_target = float(sum(float(w_focus[ch]) * float(pen_delay_target_ch.get(ch, 0.0)) for ch in self.CHANNELS))
        pen_energy_target = float(sum(float(w_focus[ch]) * float(pen_energy_target_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        pen_delay_ch = {ch: lam_delay * float(delay_cost_ch.get(ch, 0.0)) for ch in self.CHANNELS}
        pen_delay = float(sum(float(w_focus[ch]) * float(pen_delay_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # 9.2 queue waiting / timeout risk (direct scan of TxPool.qs) -- per channel
        lam_age = float(getattr(self.cfg, 'lambda_age', 0.0))
        lam_over = float(getattr(self.cfg, 'lambda_overdue', 0.0))
        pen_q_age_ch = {ch: lam_age * float(age_norm_mean.get(ch, 0.0)) for ch in self.CHANNELS}
        pen_q_overdue_ch = {ch: lam_over * float(overdue_frac.get(ch, 0.0)) for ch in self.CHANNELS}
        pen_q_age = float(sum(float(w_focus[ch]) * float(pen_q_age_ch.get(ch, 0.0)) for ch in self.CHANNELS))
        pen_q_overdue = float(sum(float(w_focus[ch]) * float(pen_q_overdue_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # 9.3 m softcap hinge (prevent committee explosion) -- per channel
        # IMPORTANT: normalize by the softcap itself (not the global range),
        # otherwise when (m_max-m_min) is large the penalty becomes too weak and
        # the policy can keep m far above m_req.
        m_soft_ratio = float(getattr(self.cfg, "m_softcap_ratio", 1.0))
        def _mscap(mr: float, cap_add: float) -> float:
            cap = max(mr * (1.0 + m_soft_ratio), mr + cap_add)
            return float(np.clip(cap, self.cfg.m_min, float(self.cfg.m_max) * 0.95))
        m_softcap = {
            "SEC": _mscap(float(m_req_cont.get("SEC", self.cfg.m_min)), float(getattr(self.cfg, "m_softcap_margin_sec", 0.0))),
            "URG": _mscap(float(m_req_cont.get("URG", self.cfg.m_min)), float(getattr(self.cfg, "m_softcap_margin_other", 0.0))),
            "NOR": _mscap(float(m_req_cont.get("NOR", self.cfg.m_min)), float(getattr(self.cfg, "m_softcap_margin_other", 0.0))),
        }

        def _hinge(x: float) -> float:
            return float(x) if x > 0.0 else 0.0

        m_min = float(self.cfg.m_min)
        m_max = float(self.cfg.m_max)

        lam_m_soft = float(getattr(self.cfg, "lambda_m_softcap", 0.0))

        def _soft_gate(_ch: str) -> float:
            # A soft regularizer (not a hard clamp):
            # - penalize overshoot more for non-demanded channels
            # - relax for SEC when alpha is high (security-demand dominant)
            g_focus = 0.35 if float(w_focus.get(_ch, 0.0)) > 0.5 else 1.0
            if _ch == "SEC":
                g_alpha = float(np.clip(1.0 - 0.85 * float(a_eff), 0.15, 1.0))
                return float(g_focus) * float(g_alpha)
            return float(g_focus)

        pen_m_softcap_ch: Dict[str, float] = {}
        for ch in self.CHANNELS:
            excess = _hinge(float(m_map[ch]) - float(m_softcap[ch]))
            den = max(1.0, float(m_softcap[ch]) - float(m_min))
            pen_m_softcap_ch[ch] = float(lam_m_soft) * float(_soft_gate(ch)) * float((excess / den) ** 2)
        pen_m_softcap = float(sum(float(w_focus[ch]) * float(pen_m_softcap_ch.get(ch, 0.0)) for ch in self.CHANNELS))
        pen_m_softcap_all = float(sum(float(pen_m_softcap_ch.get(ch, 0.0)) for ch in self.CHANNELS) / max(1.0, float(len(self.CHANNELS))))

        # 9.4 n softcap hinge (avoid saturating n beyond effective budget cap)
        # global excess computed from raw decoded n sum, then distributed to channels by raw-n share
        lam_n_soft = float(getattr(self.cfg, 'lambda_n_softcap', 0.0))
        n_soft_base = float(int(self.cfg.n_max) + 3 * int(self.cfg.n_min))
        excess_n = float(max(0.0, float(budget_total_raw) - n_soft_base))
        pen_n_softcap_global = lam_n_soft * float((excess_n / max(1.0, 3.0 * float(self.cfg.n_max))) ** 2)
        sum_nraw = float(max(1.0, float(n_map_raw['SEC'] + n_map_raw['URG'] + n_map_raw['NOR'])))
        pen_n_softcap_ch = {
            ch: float(pen_n_softcap_global) * float(n_map_raw[ch]) / sum_nraw
            for ch in self.CHANNELS
        }
        # focus-weighted (used in aggregated reward)
        pen_n_softcap = float(sum(float(w_focus[ch]) * float(pen_n_softcap_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # 9.5 smooth penalty |Δm| + |Δn| (use raw decoded m/n per channel) -- per channel
        lam_smooth = float(getattr(self.cfg, 'lambda_smooth', 0.0))
        m_range = float(max(1.0, float(self.cfg.m_max - self.cfg.m_min)))
        n_range = float(max(1.0, float(self.cfg.n_max - self.cfg.n_min)))
        prev_m = getattr(self, '_prev_m_map_raw', None)
        prev_n = getattr(self, '_prev_n_map_raw', None)
        if prev_m is None or prev_n is None:
            prev_m = dict(m_map_raw)
            prev_n = dict(n_map_raw)

        dm_ch = {ch: abs(float(m_map_raw[ch]) - float(prev_m.get(ch, m_map_raw[ch]))) / m_range for ch in self.CHANNELS}
        dn_ch = {ch: abs(float(n_map_raw[ch]) - float(prev_n.get(ch, n_map_raw[ch]))) / n_range for ch in self.CHANNELS}
        pen_smooth_ch = {ch: lam_smooth * float(dm_ch[ch] + dn_ch[ch]) for ch in self.CHANNELS}
        pen_smooth = float(sum(float(w_focus[ch]) * float(pen_smooth_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        self._prev_m_map_raw = dict(m_map_raw)
        self._prev_n_map_raw = dict(n_map_raw)

        # 9.6 security shortfall penalty (SEC only)
        S_sec = float(res_ch['SEC'].get('S', 0.0))
        secure_fail = float(max(0.0, float(p_req.get('SEC', 0.0)) - S_sec))
        lam_secure = float(getattr(self.cfg, 'lambda_secure_fail', 5.0))
        pen_secure_fail_ch = {'SEC': lam_secure * secure_fail, 'URG': 0.0, 'NOR': 0.0}
        pen_secure_fail = float(sum(float(w_focus[ch]) * float(pen_secure_fail_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # 9.7 miss penalty (deadline missed) -- per channel
        lam_miss = float(getattr(self.cfg, 'miss_penalty', 2.0))
        pen_miss_ch = {ch: lam_miss * float(missed_rate_ch.get(ch, 0.0)) for ch in self.CHANNELS}
        pen_miss = float(sum(float(w_focus[ch]) * float(pen_miss_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # 9.8 queue backlog penalty (per channel, based on PRE-BLOCK queue length)
        qnorm = float(max(1.0, float(self.cfg.max_txpool_capacity)))
        lam_q = float(getattr(self.cfg, 'queue_penalty', 0.0))
        queue_cost_ch = {
            'SEC': lam_q * float(q_sec) / qnorm,
            'URG': lam_q * float(q_urg) / qnorm,
            'NOR': lam_q * float(q_nor) / qnorm,
        }
        queue_cost = float(sum(float(w_focus[ch]) * float(queue_cost_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # keep legacy terms for logging compatibility
        pen_gap = 0.0
        pen_soft_deadline = 0.0
        pen_alloc_align = 0.0

        # m_req gap penalties (per channel): keep committee size close to requirement
        lam_m_over = float(getattr(self.cfg, 'lambda_m_over', 0.0))
        lam_m_under = float(getattr(self.cfg, 'lambda_m_under', 0.0))
        mm = int(getattr(self.cfg, 'm_order_margin', 0))
        m_over_ch = {ch: 0.0 for ch in self.CHANNELS}
        m_under_ch = {ch: 0.0 for ch in self.CHANNELS}
        pen_m_over_ch = {ch: 0.0 for ch in self.CHANNELS}
        pen_m_under_ch = {ch: 0.0 for ch in self.CHANNELS}
        if (lam_m_over > 0.0) or (lam_m_under > 0.0):
            for ch in self.CHANNELS:
                m_now = float(m_map.get(ch, 0))
                mr = float(m_req.get(ch, m_now))
                # UNDER: penalize falling below required committee
                under = float(max(0.0, (mr - m_now) / max(1.0, mr)))
                # OVER: penalize exceeding required committee (allow +mm slack for ordering margin)
                over = float(max(0.0, (m_now - (mr + mm)) / max(1.0, float(m_max - (mr + mm)))))
                m_under_ch[ch] = under
                m_over_ch[ch] = over
                pen_m_under_ch[ch] = lam_m_under * float(under ** 2)
                pen_m_over_ch[ch] = lam_m_over * float(over ** 2)
        pen_m_under = float(sum(float(w_focus[ch]) * float(pen_m_under_ch.get(ch, 0.0)) for ch in self.CHANNELS))
        pen_m_over = float(sum(float(w_focus[ch]) * float(pen_m_over_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # structural order violation metrics (pre/post).
        # Order is enforced by projection above when order_as_constraint=True.
        mm = int(getattr(self.cfg, 'm_order_margin', 0))
        vio_pre = max(0.0, float(m_map_raw['URG'] + mm - m_map_raw['NOR'])) + max(0.0, float(m_map_raw['NOR'] + mm - m_map_raw['SEC']))
        vio_post = max(0.0, float(m_map['URG'] + mm - m_map['NOR'])) + max(0.0, float(m_map['NOR'] + mm - m_map['SEC']))
        m_order_range = float(max(1.0, float(self.cfg.m_max - self.cfg.m_min)))
        m_order_violation_pre = float(vio_pre / m_order_range)
        m_order_violation = float(vio_post / m_order_range)
        # ordering penalty: penalize RAW (pre-projection) violations so PPO actually learns ordering
        lam_m_order = float(getattr(self.cfg, "lambda_m_order", 0.0))
        pen_m_order = lam_m_order * float(m_order_violation_pre) if lam_m_order > 0.0 else 0.0

        # slack pressure (early-trigger hinge) + per-channel soft-deadline penalty
        st_all2 = self._txpool_stats_all(now_round=int(self.round_id))
        slack_q_on = bool(float(getattr(self.cfg, 'slack_quantile', 0.0)) > 0.0)
        press_pow = float(max(1e-6, float(getattr(self.cfg, 'slack_press_pow', 2.0))))
        thr_base = float(max(1e-6, float(getattr(self.cfg, 'slack_thr', 8.0))))
        thr_sec = thr_base + float(getattr(self.cfg, 'slack_thr_gain_alpha', 0.0)) * float(a_eff)
        thr_urg = thr_base + float(getattr(self.cfg, 'slack_thr_gain_beta', 0.0)) * float(b_eff)
        thr_nor = thr_base + float(getattr(self.cfg, 'slack_thr_gain_gamma', 0.0)) * float(g_eff)

        def _slack_press_from(st: Dict[str, float], thr: float) -> float:
            s = float(st.get('min_slack', 0.0))
            if slack_q_on and ('slack_q' in st):
                s = float(st.get('slack_q', s))
            thr = float(max(1e-6, thr))
            x = float(max(0.0, (thr - s) / max(1.0, thr)))
            return float(x ** press_pow)

        sp_sec = _slack_press_from(st_all2.get('SEC', {}), thr_sec)
        sp_urg = _slack_press_from(st_all2.get('URG', {}), thr_urg)
        sp_nor = _slack_press_from(st_all2.get('NOR', {}), thr_nor)

        lam_soft = float(getattr(self.cfg, 'lambda_soft_deadline', 0.0))
        pen_soft_deadline_ch = {ch: 0.0 for ch in self.CHANNELS}
        pen_soft_deadline_ch['SEC'] = lam_soft * float(0.5 + 0.5 * a_eff) * float(sp_sec)
        pen_soft_deadline_ch['URG'] = lam_soft * float(0.5 + 0.5 * b_eff) * float(sp_urg)
        pen_soft_deadline_ch['NOR'] = lam_soft * float(0.5 + 0.5 * g_eff) * float(sp_nor)
        pen_soft_deadline = float(sum(float(w_focus[ch]) * float(pen_soft_deadline_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # 10) reward (per-channel rewards every step; env returns req-focused aggregate)
        # per-channel total cost (private + distributed global terms)
        cost_ch = {
            ch: float(queue_cost_ch.get(ch, 0.0))
                + float(pen_delay_ch.get(ch, 0.0))
                + float(pen_q_age_ch.get(ch, 0.0))
                + float(pen_q_overdue_ch.get(ch, 0.0))
                + float(pen_soft_deadline_ch.get(ch, 0.0))
                + float(pen_delay_target_ch.get(ch, 0.0))
                + float(pen_energy_target_ch.get(ch, 0.0))
                + float(pen_m_softcap_ch.get(ch, 0.0))
                + float(pen_m_under_ch.get(ch, 0.0))
                + float(pen_m_over_ch.get(ch, 0.0))
                + float(pen_n_softcap_ch.get(ch, 0.0))
                + float(pen_smooth_ch.get(ch, 0.0))
                + float(pen_secure_fail_ch.get(ch, 0.0))
                + float(pen_miss_ch.get(ch, 0.0))
            for ch in self.CHANNELS
        }

        r_unscaled_ch = {ch: float(phi_ch.get(ch, 0.0)) - float(cost_ch.get(ch, 0.0)) for ch in self.CHANNELS}
        # scaled per-channel rewards (used for channel agents)
        r_ch_raw = {
            ch: float(self.cfg.reward_scale) * float(r_unscaled_ch.get(ch, 0.0))
                + float(self.cfg.throughput_bonus) * float(n_used_ch.get(ch, 0.0))
            for ch in self.CHANNELS
        }

        # env-level reward is req-focused aggregate (kept for compatibility/logging)
        r_unscaled = float(sum(float(w_focus[ch]) * float(r_unscaled_ch.get(ch, 0.0)) for ch in self.CHANNELS))
        cost_total = float(sum(float(w_focus[ch]) * float(cost_ch.get(ch, 0.0)) for ch in self.CHANNELS))

        # Enforce system-level constraints regardless of current req focus:
        # - average m softcap penalty across channels
        # - global n softcap penalty across channels
        # - committee ordering constraint
        pen_n_softcap_all = float(pen_n_softcap_global)
        pen_global_extra = float(
            (float(pen_m_softcap_all) - float(pen_m_softcap))
            + (float(pen_n_softcap_all) - float(pen_n_softcap))
            + float(pen_m_order)
        )
        r_unscaled -= pen_global_extra
        cost_total += pen_global_extra

        throughput = float(n_used_ch['SEC'] + n_used_ch['URG'] + n_used_ch['NOR'])
        r_raw = float(self.cfg.reward_scale) * float(r_unscaled) + float(self.cfg.throughput_bonus) * float(throughput)

        # reward stabilization (compress + clip) to keep PPO numerically stable
        def _transform_reward(x: float) -> float:
            xt = float(x)
            rt = str(getattr(self.cfg, 'reward_transform', 'none')).lower()
            if rt == 'log':
                xt = float(np.sign(xt) * np.log1p(abs(xt)))
            elif rt == 'sqrt':
                xt = float(np.sign(xt) * np.sqrt(abs(xt) + 1e-9))
            elif rt == 'tanh':
                scale = float(getattr(self.cfg, 'reward_tanh_scale', 20.0))
                scale = max(1e-6, scale)
                xt = float(np.tanh(xt / scale))

            clip = getattr(self.cfg, 'reward_clip', None)
            if clip is not None:
                try:
                    c = float(clip)
                    if c > 0:
                        xt = float(np.clip(xt, -c, c))
                except Exception:
                    pass
            return float(xt)

        r_ch = {ch: _transform_reward(float(r_ch_raw.get(ch, 0.0))) for ch in self.CHANNELS}
        r = _transform_reward(r_raw)

        self.ep_return += r
        done = bool(self.step_id >= int(self.cfg.max_steps))

# 11) advance to next round + generate new txs for NEXT decision state
        self.round_id = int(self.round_id) + 1
        self._generate_transactions(now_round=int(self.round_id))

        # update demand weights for NEXT state (fix state/req mismatch)
        try:
            self._compute_demand_vector()
        except Exception:
            pass

        # next state (agent will act on this pool)
        s = self._get_state()
        mask = np.ones((3,), dtype=np.float32)
        g = dict(getattr(self, "_last_gen_cnt", {"SEC": 0, "URG": 0, "NOR": 0}))
        info: Dict[str, Any] = {
            "episode_return": float(self.ep_return),
            "rho": float(rho),

            # pools at decision time (start of round, before popping)
            "pool0_SEC": int(q0_sec),
            "pool0_URG": int(q0_urg),
            "pool0_NOR": int(q0_nor),

            "gen_SEC": int(g.get("SEC", 0)),
            "gen_URG": int(g.get("URG", 0)),
            "gen_NOR": int(g.get("NOR", 0)),
            "gen_total": int(getattr(self, "_last_gen_total", 0)),

            # actually popped this round (tx count)
            "popped_SEC": int(len(popped_map.get("SEC", []))),
            "popped_URG": int(len(popped_map.get("URG", []))),
            "popped_NOR": int(len(popped_map.get("NOR", []))),

            # demand
            "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
            "req": str(req),
            "w_focus_SEC": float(w_focus["SEC"]),
            "w_focus_URG": float(w_focus["URG"]),
            "w_focus_NOR": float(w_focus["NOR"]),
            "w_dem_SEC": float(w_dem["SEC"]),
            "w_dem_URG": float(w_dem["URG"]),
            "w_dem_NOR": float(w_dem["NOR"]),
            "w_pen_SEC": float(w_pen["SEC"]),
            "w_pen_URG": float(w_pen["URG"]),
            "w_pen_NOR": float(w_pen["NOR"]),
            "scenario": str(getattr(self, "_scenario", "")) if getattr(self, "_scenario", None) is not None else "",
            "req_src": str(getattr(self, "_req_src", "pool")),
            "thr_priv": float(self._thr_priv),
            "thr_urg": float(self._thr_urg),

            # pipeline timing
            "T_URG": float(T_urg),
            "T_NOR": float(T_nor),
            "T_SEC": float(T_sec),
            "D_max": float(max(T_urg, T_nor, T_sec)),
            "D_pipeline": float(T_sec),

            # delay cost (derived from T_map)
            "delay_cost_SEC": float(delay_cost_ch.get("SEC", 0.0)),
            "delay_cost_URG": float(delay_cost_ch.get("URG", 0.0)),
            "delay_cost_NOR": float(delay_cost_ch.get("NOR", 0.0)),
            "energy_cost_SEC": float(energy_cost_ch.get("SEC", 0.0)),
            "energy_cost_URG": float(energy_cost_ch.get("URG", 0.0)),
            "energy_cost_NOR": float(energy_cost_ch.get("NOR", 0.0)),
            "t_delay_urg": float(t_delay_urg),
            "t_energy_nor": float(t_energy_nor),
            "pen_delay_target": float(pen_delay_target),
            "pen_energy_target": float(pen_energy_target),

            # pools (end of round, AFTER drop_expired, BEFORE next arrivals)
            "pool_SEC": int(q_sec),
            "pool_URG": int(q_urg),
            "pool_NOR": int(q_nor),

            # pools for next state (AFTER next arrivals)
            "pool_next_SEC": int(self.txpool.q_len("SEC")),
            "pool_next_URG": int(self.txpool.q_len("URG")),
            "pool_next_NOR": int(self.txpool.q_len("NOR")),

            # arrivals generated for next round (approx = next_pool - pool_end)
            "arrive_SEC": int(max(0, self.txpool.q_len("SEC") - int(q_sec))),
            "arrive_URG": int(max(0, self.txpool.q_len("URG") - int(q_urg))),
            "arrive_NOR": int(max(0, self.txpool.q_len("NOR") - int(q_nor))),

            "miss_rate_SEC": float(missed_rate_ch["SEC"]),
            "miss_rate_URG": float(missed_rate_ch["URG"]),
            "miss_rate_NOR": float(missed_rate_ch["NOR"]),

            # queue risk (mean age/ttl and overdue fraction)
            "age_norm_mean_SEC": float(age_norm_mean.get("SEC", 0.0)),
            "age_norm_mean_URG": float(age_norm_mean.get("URG", 0.0)),
            "age_norm_mean_NOR": float(age_norm_mean.get("NOR", 0.0)),
            "overdue_frac_SEC": float(overdue_frac.get("SEC", 0.0)),
            "overdue_frac_URG": float(overdue_frac.get("URG", 0.0)),
            "overdue_frac_NOR": float(overdue_frac.get("NOR", 0.0)),

            # action decoded
            "m_SEC": int(m_map["SEC"]), "n_SEC": int(n_map["SEC"]), "n_used_SEC": int(n_used_ch["SEC"]),
            "m_URG": int(m_map["URG"]), "n_URG": int(n_map["URG"]), "n_used_URG": int(n_used_ch["URG"]),
            "m_NOR": int(m_map["NOR"]), "n_NOR": int(n_map["NOR"]), "n_used_NOR": int(n_used_ch["NOR"]),
            "m_raw_SEC": int(m_map_raw["SEC"]), "m_raw_URG": int(m_map_raw["URG"]), "m_raw_NOR": int(m_map_raw["NOR"]),
            "m_req_cont_SEC": float(m_req_cont["SEC"]), "m_req_cont_URG": float(m_req_cont["URG"]), "m_req_cont_NOR": float(m_req_cont["NOR"]),
            "m_req_SEC": int(m_req["SEC"]), "m_req_URG": int(m_req["URG"]), "m_req_NOR": int(m_req["NOR"]),
            "p_req_SEC": float(p_req["SEC"]), "p_req_URG": float(p_req["URG"]), "p_req_NOR": float(p_req["NOR"]),
            "p_order_violation": float(p_order_violation),
            "n_req_SEC": int(n_req["SEC"]), "n_req_URG": int(n_req["URG"]), "n_req_NOR": int(n_req["NOR"]),

            # mapped n after pool-cap, before global budget allocation
            "n_poolcap_SEC": int(n_poolcap['SEC']),
            "n_poolcap_URG": int(n_poolcap['URG']),
            "n_poolcap_NOR": int(n_poolcap['NOR']),

            "budget_total": int(budget_total),
            "budget_left": int(budget_left),
            "budget_used": int(int(budget_total) - int(budget_left)),

            # consensus metrics (per channel)
            "S_SEC": float(res_ch["SEC"].get("S", 0.0)),
            "D_SEC": float(res_ch["SEC"].get("D", 0.0)),
            "E_SEC": float(res_ch["SEC"].get("E", 0.0)),
            "S_URG": float(res_ch["URG"].get("S", 0.0)),
            "D_URG": float(res_ch["URG"].get("D", 0.0)),
            "E_URG": float(res_ch["URG"].get("E", 0.0)),
            "S_NOR": float(res_ch["NOR"].get("S", 0.0)),
            "D_NOR": float(res_ch["NOR"].get("D", 0.0)),
            "E_NOR": float(res_ch["NOR"].get("E", 0.0)),

            # utility
            "phi_SEC": float(phi_ch["SEC"]),
            "phi_URG": float(phi_ch["URG"]),
            "phi_NOR": float(phi_ch["NOR"]),
            "phi": float(phi),
            "phi_focus": float(phi_focus),

            # penalties
            "backlog": float(backlog),
            "backlog_w": float(backlog_w),
            "queue_cost": float(queue_cost),
            "cost_total": float(cost_total),
            "reward_unscaled": float(r_unscaled),
            "reward_raw": float(r_raw),
            "reward_clip": float(self.cfg.reward_clip) if self.cfg.reward_clip is not None else float("inf"),
            "reward_is_clipped": float(abs(r_raw) > float(self.cfg.reward_clip)) if self.cfg.reward_clip is not None else 0.0,
            "reward_SEC": float(r_ch.get("SEC", 0.0)),
            "reward_ch_raw": {k: float(v) for k, v in r_ch_raw.items()},
            # "reward_ch_is_clipped": {ch: (float(abs(r_ch_raw.get(ch, 0.0)) > float(self.cfg.reward_clip)) if self.cfg.reward_clip is not None else 0.0) for ch in self.channels},
            "reward_URG": float(r_ch.get("URG", 0.0)),
            "reward_NOR": float(r_ch.get("NOR", 0.0)),
            "reward_unscaled_SEC": float(r_unscaled_ch.get("SEC", 0.0)),
            "reward_unscaled_URG": float(r_unscaled_ch.get("URG", 0.0)),
            "reward_unscaled_NOR": float(r_unscaled_ch.get("NOR", 0.0)),
            "cost_SEC": float(cost_ch.get("SEC", 0.0)),
            "cost_URG": float(cost_ch.get("URG", 0.0)),
            "cost_NOR": float(cost_ch.get("NOR", 0.0)),
            "pen_n_softcap_global": float(pen_n_softcap_global),
            "pen_gap": float(pen_gap),
            "m_order_violation_pre": float(m_order_violation_pre),
            "m_order_violation": float(m_order_violation_pre),  # RAW (pre-projection)
            "m_order_violation_exec": float(m_order_violation),  # executed (post-projection)
            "pen_m_order": float(pen_m_order),
            "pen_m_softcap_all": float(pen_m_softcap_all),
            "pen_global_extra": float(pen_global_extra),
            "pen_m_over": float(pen_m_over),
            "pen_m_under": float(pen_m_under),
            "pen_secure_fail": float(pen_secure_fail),
            "pen_miss": float(pen_miss),
            "pen_soft_deadline": float(pen_soft_deadline),
            "pen_alloc_align": float(pen_alloc_align),
            "pen_delay": float(pen_delay),
            "pen_q_age": float(pen_q_age),
            "pen_q_overdue": float(pen_q_overdue),
            "pen_m_softcap": float(pen_m_softcap),
            "pen_n_softcap": float(pen_n_softcap),
            "pen_smooth": float(pen_smooth),
            "slack_press_SEC": float(sp_sec),
            "slack_press_URG": float(sp_urg),
            "slack_press_NOR": float(sp_nor),

            # raw demand debug
            **dbg_dem,
        }

        return s, mask, r, done, info

    # -----------------------
    # internals
    # -----------------------

    def _normalize_action(self, action: Any) -> np.ndarray:
        """
        Accepts:
          - list/tuple/np.ndarray with 6 floats in [0,1]
          - torch tensor (converted by numpy()) upstream; here we treat as array-like.
        """
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], (int, np.integer)):
            # allow old (ch, m_norm, n_norm) tuple by expanding (fallback)
            _, m_norm, n_norm = action
            a = np.asarray([m_norm, n_norm, m_norm, n_norm, m_norm, n_norm], dtype=np.float32)
        else:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 6:
            raise ValueError(f"Scheme-B action must have 6 dims, got {a.size}")
        a = np.clip(a, 0.0, 1.0).astype(np.float32)
        return a

    @staticmethod
    def _decode_int(x01: float, lo: int, hi: int) -> int:
        lo = int(lo)
        hi = int(hi)
        if hi <= lo:
            return int(lo)
        x01 = float(np.clip(float(x01), 0.0, 1.0))
        return int(round(lo + x01 * (hi - lo)))

    def _generate_transactions(self, now_round: int) -> None:
        bs = int(max(0, self.cfg.tx_gen_rate))
        if bs <= 0:
            return

        txs: List[Tx] = []

        # NEW: count generated tx by channel (this batch)
        gen_cnt = {"SEC": 0, "URG": 0, "NOR": 0}

        # Effective deadline ranges (scenario may shrink ranges to create learning signal)
        sec_dl = tuple(self.cfg.sec_deadline_range)
        urg_dl = tuple(self.cfg.urg_deadline_range)
        nor_dl = tuple(self.cfg.nor_deadline_range)
        if bool(getattr(self.cfg, 'use_curriculum', False)) and self._scenario:
            def _shrink(dl, factor: float):
                lo, hi = int(dl[0]), int(dl[1])
                factor = float(np.clip(factor, 0.1, 1.0))
                lo2 = max(1, int(round(lo * factor)))
                hi2 = max(lo2, int(round(hi * factor)))
                return (lo2, hi2)
            if self._scenario == 'URG':
                urg_dl = _shrink(urg_dl, float(getattr(self.cfg, 'curriculum_deadline_shrink_urg', 0.65)))
            elif self._scenario == 'SEC':
                sec_dl = _shrink(sec_dl, float(getattr(self.cfg, 'curriculum_deadline_shrink_sec', 0.90)))

        if self.data_mode == "physionet" and self.stream is not None:
            # curriculum may relax quantiles to avoid single-channel degeneracy
            q_priv_eff = float(self.cfg.q_priv)
            q_urg_eff = float(self.cfg.q_urg)
            if bool(getattr(self.cfg, 'use_curriculum', False)) and self._scenario:
                if self._scenario == 'SEC':
                    q_priv_eff = float(min(q_priv_eff, float(getattr(self.cfg, 'curriculum_q_priv', 0.75))))
                elif self._scenario == 'URG':
                    q_urg_eff = float(min(q_urg_eff, float(getattr(self.cfg, 'curriculum_q_urg', 0.85))))

            demands = self.stream.sample_batch(
                batch_size=bs,
                q_priv=float(q_priv_eff),
                q_urg=float(q_urg_eff),
                min_thr=float(self.cfg.min_thr),
                sec_deadline_range=tuple(sec_dl),
                urg_deadline_range=tuple(urg_dl),
                nor_deadline_range=tuple(nor_dl),
            )

            # compute per-batch thresholds (for debug + SEC trigger)
            privs = np.asarray([float(d.privacy) for d in demands], dtype=np.float32)
            urgs = np.asarray([float(d.urgency) for d in demands], dtype=np.float32)
            thr_priv = float(max(float(self.cfg.min_thr), float(np.quantile(privs, float(q_priv_eff))))) if privs.size > 0 else float(self.cfg.min_thr)
            thr_urg = float(max(float(self.cfg.min_thr), float(np.quantile(urgs, float(q_urg_eff))))) if urgs.size > 0 else float(self.cfg.min_thr)

            # EMA smooth (prevents thr from jumping too wildly)
            self._thr_priv = float((1.0 - self._ema_thr) * self._thr_priv + self._ema_thr * thr_priv)
            self._thr_urg = float((1.0 - self._ema_thr) * self._thr_urg + self._ema_thr * thr_urg)

            # IMPORTANT:
            #   Your current PhysioNet stream may output a degenerate channel label distribution
            #   (e.g., always "NOR"), which makes req always NOR and prevents RL from learning
            #   meaningful 3-channel allocation.
            #
            #   To guarantee correct coupling with TxPool (3 channels) WITHOUT changing any
            #   external interfaces, we (re)classify each tx by its privacy/urgency against
            #   per-batch quantile thresholds (EMA-smoothed).
            #
            #   If cfg.override_channel=True, we still respect d.channel, BUT we will upgrade
            #   a "NOR" label to "SEC"/"URG" when the feature-based classifier strongly
            #   indicates it, so high-privacy / high-urgency txs don't get stuck in NOR.

            def _classify_by_feat(priv: float, urg: float) -> str:
                if float(priv) >= float(self._thr_priv):
                    return "SEC"
                if float(urg) >= float(self._thr_urg):
                    return "URG"
                return "NOR"

            for d in demands:
                ch_raw = str(getattr(d, "channel", "NOR"))
                if ch_raw not in self.CHANNELS:
                    ch_raw = "NOR"

                ch_feat = _classify_by_feat(float(d.privacy), float(d.urgency))

                if bool(self.cfg.override_channel):
                    # keep stream label, but avoid "NOR" swallowing true SEC/URG
                    ch = ch_raw if not (ch_raw == "NOR" and ch_feat != "NOR") else ch_feat
                else:
                    ch = ch_feat
                    # allow a little stochasticity for exploration
                    u = float(self.rng.rand())
                    if u < 0.03:
                        ch = "SEC"
                    elif u < 0.15:
                        ch = "URG"

                # deadline offset: if we change channel relative to the stream label,
                # resample from cfg ranges to keep semantics consistent.
                if ch == ch_raw and hasattr(d, "deadline_offset"):
                    deadline_round = int(now_round + int(d.deadline_offset))
                else:
                    if ch == "SEC":
                        dlo, dhi = sec_dl
                    elif ch == "URG":
                        dlo, dhi = urg_dl
                    else:
                        dlo, dhi = nor_dl
                    deadline_round = int(now_round + int(self.rng.randint(int(dlo), int(dhi) + 1)))

                gen_cnt[str(ch)] = gen_cnt.get(str(ch), 0) + 1

                txs.append(Tx(
                    tx_id=int(self.rng.randint(0, 1_000_000_000)),
                    ch=str(ch),
                    created_round=int(now_round),
                    deadline_round=int(deadline_round),
                    urgency=float(d.urgency),
                    privacy=float(d.privacy),
                    complexity=float(d.complexity),
                    size_bytes=int(200 + 800 * self.rng.rand()),
                    record_id=str(d.record_id),
                ))

        else:
            # synthetic fallback
            for _ in range(bs):
                urg = float(self.rng.rand())
                priv = float(self.rng.rand())
                comp = float(self.rng.rand())

                # thresholds using running EMA (still meaningful)
                thr_priv = float(self._thr_priv)
                thr_urg = float(self._thr_urg)

                # channel assignment
                if priv >= thr_priv:
                    ch = "SEC"
                    dlo, dhi = sec_dl
                elif urg >= thr_urg:
                    ch = "URG"
                    dlo, dhi = urg_dl
                else:
                    ch = "NOR"
                    dlo, dhi = nor_dl

                deadline_round = int(now_round + int(self.rng.randint(int(dlo), int(dhi) + 1)))
                txs.append(Tx(
                    tx_id=int(self.rng.randint(0, 1_000_000_000)),
                    ch=str(ch),
                    created_round=int(now_round),
                    deadline_round=int(deadline_round),
                    urgency=float(urg),
                    privacy=float(priv),
                    complexity=float(comp),
                    size_bytes=int(200 + 800 * self.rng.rand()),
                    record_id=str(-1),
                ))

        self._last_gen_cnt = dict(gen_cnt)
        self._last_gen_total = int(bs)

        if hasattr(self.txpool, 'add_many'):
            self.txpool.add_many(txs)
        else:
            for tx in txs:
                # legacy fallback
                if hasattr(self.txpool, 'add'):
                    self.txpool.add(tx)
                else:
                    # last-resort direct insert
                    try:
                        self.txpool.qs[str(tx.ch)].append(tx)
                    except Exception:
                        pass

    def _compute_demand_vector(self) -> Tuple[float, float, float, str, Dict[str, Any]]:
        # queue shares
        q_sec = float(self.txpool.q_len("SEC"))
        q_urg = float(self.txpool.q_len("URG"))
        q_nor = float(self.txpool.q_len("NOR"))
        q_sum = float(max(1.0, q_sec + q_urg + q_nor))

        share_sec = q_sec / q_sum
        share_urg = q_urg / q_sum
        share_nor = q_nor / q_sum

        # stats
        st_all = self._txpool_stats_all(now_round=int(self.round_id))
        st_sec = st_all.get("SEC", {})
        st_urg = st_all.get("URG", {})
        st_nor = st_all.get("NOR", {})

        priv_sec = float(st_sec.get("priv_mean", 0.0))
        urg_urg = float(st_urg.get("urg_mean", 0.0))
        comp_nor = float(st_nor.get("comp_mean", 0.0))

        # slack: min_slack can be very noisy -> allow quantile
        def _slack_score(st: Dict[str, float]) -> float:
            if float(self.cfg.slack_quantile) > 0.0:
                return float(np.clip(float(st.get("slack_q", 0.0)), -50.0, 50.0))
            return float(np.clip(float(st.get("min_slack", 0.0)), -50.0, 50.0))

        slack_sec = _slack_score(st_sec)
        slack_urg = _slack_score(st_urg)
        slack_nor = _slack_score(st_nor)

        # normalize slack: more negative slack => more urgent
        # map slack to [0,1] via piecewise
        def _slack_act(s: float) -> float:
            # slack <= 0 => full pressure
            if s <= 0.0:
                return 1.0
            # slack > 0 reduces pressure quickly
            return float(np.exp(-0.30 * float(s)))

        s_act_sec = _slack_act(slack_sec)
        s_act_urg = _slack_act(slack_urg)
        s_act_nor = _slack_act(slack_nor)

        # SEC: threshold-triggered privacy activation
        thr_priv = float(np.clip(self._thr_priv, 0.0, 0.99))
        priv_norm = 0.0
        if priv_sec > thr_priv:
            priv_norm = float((priv_sec - thr_priv) / max(1e-6, (1.0 - thr_priv)))
        priv_norm = float(np.clip(priv_norm, 0.0, 1.0))

        # URG: threshold-triggered urgency activation
        thr_urg = float(np.clip(self._thr_urg, 0.0, 0.99))
        urg_norm = 0.0
        if urg_urg > thr_urg:
            urg_norm = float((urg_urg - thr_urg) / max(1e-6, (1.0 - thr_urg)))
        urg_norm = float(np.clip(urg_norm, 0.0, 1.0))

        comp_norm = float(np.clip(comp_nor, 0.0, 1.0))

        # channel pressures
        p_sec = (
            float(self.cfg.w_sec_pressure) * share_sec
            + float(self.cfg.w_priv_mean) * priv_norm
            + float(self.cfg.w_slack_sec) * s_act_sec
        )
        p_urg = (
            float(self.cfg.w_urg_pressure) * share_urg
            + float(self.cfg.w_urg_mean) * urg_norm
            + float(self.cfg.w_slack_urg) * s_act_urg
        )
        p_nor = (
            float(self.cfg.w_nor_pressure) * share_nor
            + float(self.cfg.w_comp_mean) * comp_norm
            + float(self.cfg.w_slack_nor) * s_act_nor
        )

        # avoid all-zeros
        p_sec = float(max(1e-6, p_sec))
        p_urg = float(max(1e-6, p_urg))
        p_nor = float(max(1e-6, p_nor))

        s = float(p_sec + p_urg + p_nor)
        alpha = float(p_sec / s)
        beta = float(p_urg / s)
        gamma = float(p_nor / s)

        req = "SEC"
        if beta >= alpha and beta >= gamma:
            req = "URG"
        elif gamma >= alpha and gamma >= beta:
            req = "NOR"
        # curriculum: optionally force diverse (alpha,beta,gamma) / req during training
        req_src = "pool"
        if bool(getattr(self.cfg, 'use_curriculum', False)) and getattr(self, '_scenario', None):
            p0 = float(getattr(self.cfg, 'force_req_prob', 0.0))
            pmin = float(getattr(self.cfg, 'force_req_prob_min', 0.0))
            # When using deterministic curriculum cycling, keep forcing probability high so logs show diverse req modes.
            if bool(getattr(self.cfg, 'curriculum_force_cycle', False)):
                p0 = max(p0, 0.95)
            anneal = int(getattr(self.cfg, 'force_req_anneal_episodes', 0))
            scale = 1.0
            if anneal > 0:
                scale = float(max(0.0, 1.0 - float(self.episode_id) / float(anneal)))
            p_force = float(p0 * scale + pmin * (1.0 - scale))
            if float(self.rng.rand()) < p_force:
                # Force the *req scenario* while keeping (alpha,beta,gamma) continuous.
                # This avoids overfitting to a few hard-coded corners and improves interpolation/generalization.
                req = str(self._scenario)
                req_src = "curriculum"

                # Current pool-derived demand (keeps realism)
                w_pool = np.asarray([alpha, beta, gamma], dtype=np.float32)

                # Scenario anchor (only an anchor, not a fixed point)
                if req == "SEC":
                    w_tgt = np.asarray([0.70, 0.15, 0.15], dtype=np.float32)
                elif req == "URG":
                    w_tgt = np.asarray([0.15, 0.70, 0.15], dtype=np.float32)
                else:
                    w_tgt = np.asarray([0.20, 0.25, 0.55], dtype=np.float32)

                # Sample a continuous weight vector around the anchor using Dirichlet.
                conc = float(getattr(self.cfg, 'curriculum_dirichlet_conc', 25.0))
                conc = float(max(1.0, conc))
                w_samp = self.rng.dirichlet((w_tgt * conc).astype(np.float64)).astype(np.float32)

                # Optional: occasionally sample a completely random mixture to cover the whole simplex.
                p_u = float(getattr(self.cfg, 'curriculum_uniform_prob', 0.05))
                if p_u > 0.0 and float(self.rng.rand()) < p_u:
                    w_samp = self.rng.dirichlet(np.asarray([1.0, 1.0, 1.0], dtype=np.float64)).astype(np.float32)

                # Blend anchor-sample with pool-derived weights; anneal from curriculum -> pool.
                mix_max = float(getattr(self.cfg, 'curriculum_mix_max', 1.0))
                mix_min = float(getattr(self.cfg, 'curriculum_mix_min', 0.0))
                mix = float(mix_max * scale + mix_min * (1.0 - scale))
                w = (1.0 - mix) * w_pool + mix * w_samp
                w = np.clip(w, 1e-4, None)
                w = w / float(w.sum())

                alpha, beta, gamma = float(w[0]), float(w[1]), float(w[2])
        self._req_src = req_src

        dbg = {
            "share_sec": float(share_sec),
            "share_urg": float(share_urg),
            "share_nor": float(share_nor),
            "priv_sec": float(priv_sec),
            "urg_urg": float(urg_urg),
            "comp_nor": float(comp_nor),
            "slack_sec": float(slack_sec),
            "slack_urg": float(slack_urg),
            "slack_nor": float(slack_nor),
            "p_sec": float(p_sec),
            "p_urg": float(p_urg),
            "p_nor": float(p_nor),
        }

        self._last_alpha, self._last_beta, self._last_gamma, self._last_req = alpha, beta, gamma, req
        return alpha, beta, gamma, req, dbg

    def _get_state(self) -> np.ndarray:
        """
        State contains:
          - pool sizes (3)
          - per-channel stats: priv/urg/comp mean (3)
          - per-channel slack & miss rate (6)
          - demand weights alpha,beta,gamma (3)
          - thresholds thr_priv/thr_urg (2)
          - round progress (1)
        Total = 18
        """
        q_sec = float(self.txpool.q_len("SEC"))
        q_urg = float(self.txpool.q_len("URG"))
        q_nor = float(self.txpool.q_len("NOR"))
        cap = float(max(1.0, self.cfg.max_txpool_capacity))

        st_all = self._txpool_stats_all(now_round=int(self.round_id))
        st_sec = st_all.get("SEC", {})
        st_urg = st_all.get("URG", {})
        st_nor = st_all.get("NOR", {})

        # means
        priv_sec = float(st_sec.get("priv_mean", 0.0))
        urg_urg = float(st_urg.get("urg_mean", 0.0))
        comp_nor = float(st_nor.get("comp_mean", 0.0))

        # slack/miss
        slack_sec = float(np.clip(float(st_sec.get("min_slack", 0.0)), -50.0, 50.0))
        slack_urg = float(np.clip(float(st_urg.get("min_slack", 0.0)), -50.0, 50.0))
        slack_nor = float(np.clip(float(st_nor.get("min_slack", 0.0)), -50.0, 50.0))

        # v5: soft-deadline pressure (0..1, larger => closer to deadline)
        def _slack_press_from(st: Dict[str, float], fallback: float) -> float:
            s = float(fallback)
            if float(getattr(self.cfg, "slack_quantile", 0.0)) > 0.0 and ("slack_q" in st):
                s = float(st.get("slack_q", s))
            s = float(np.clip(s, -50.0, 50.0))
            if s <= 0.0:
                return 1.0
            return float(np.exp(-float(self.cfg.soft_deadline_k) * s))

        press_sec = _slack_press_from(st_sec, slack_sec)
        press_urg = _slack_press_from(st_urg, slack_urg)
        press_nor = _slack_press_from(st_nor, slack_nor)

        mr = getattr(self, "_last_miss_rate", {"SEC": 0.0, "URG": 0.0, "NOR": 0.0})
        miss_sec = float(np.clip(float(mr.get("SEC", 0.0)), 0.0, 1.0))
        miss_urg = float(np.clip(float(mr.get("URG", 0.0)), 0.0, 1.0))
        miss_nor = float(np.clip(float(mr.get("NOR", 0.0)), 0.0, 1.0))

        # last demand
        a = float(self._last_alpha)
        b = float(self._last_beta)
        g = float(self._last_gamma)

        x = np.asarray([
            q_sec / cap, q_urg / cap, q_nor / cap,
            priv_sec, urg_urg, comp_nor,
            slack_sec / 50.0, slack_urg / 50.0, slack_nor / 50.0,
            press_sec, press_urg, press_nor,
            miss_sec, miss_urg, miss_nor,
            a, b, g,
            float(self._thr_priv), float(self._thr_urg),
            float(self.round_id) / 1000.0,
        ], dtype=np.float32)
        return x
