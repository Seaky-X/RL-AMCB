# -*- coding: utf-8 -*-
"""
env/consensus.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import math
import numpy as np

from env.vrf import VRFKeypair, vrf_prove, vrf_verify


@dataclass
class Node:
    node_id: int
    keypair: VRFKeypair
    is_malicious: bool
    bandwidth_mbps: float
    compute: float
    reputation: float


def _binom_tail_prob(m: int, p: float, k_min: int) -> float:
    """P(X >= k_min), X ~ Binomial(m, p)."""
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    s = 0.0
    for k in range(k_min, m + 1):
        s += math.comb(m, k) * (p ** k) * ((1.0 - p) ** (m - k))
    return float(np.clip(s, 0.0, 1.0))


class ConsensusModule:
    def __init__(self,
                 total_nodes: int = 50,
                 base_malicious_ratio: float = 0.20,
                 offline_prob: float = 0.05,
                 seed: int = 0):
        self.total_nodes = int(total_nodes)
        self.base_malicious_ratio = float(base_malicious_ratio)
        self.offline_prob = float(np.clip(offline_prob, 0.0, 0.5))
        self.rng = np.random.RandomState(int(seed))

        self.nodes: List[Node] = []
        for i in range(self.total_nodes):
                self.nodes.append(Node(
                node_id=int(i),
                keypair=kp,
                is_malicious=False,
                bandwidth_mbps=float(self.rng.uniform(10.0, 100.0)),
                compute=float(self.rng.uniform(0.5, 2.0)),
                reputation=float(self.rng.uniform(0.5, 1.0)),
            ))

    def sample_dynamic_malicious_ratio(self) -> float:
        jitter = float(self.rng.normal(loc=0.0, scale=0.04))
        return float(np.clip(self.base_malicious_ratio + jitter, 0.0, 0.49))

    def _sample_active_nodes(self) -> List[Node]:
        active = []
        for nd in self.nodes:
            if self.rng.rand() < self.offline_prob:
                continue
            active.append(nd)
        if len(active) < 4:
            active = self.nodes[:min(4, len(self.nodes))]
        return active

    def _mark_malicious(self, nodes: List[Node], malicious_ratio: float) -> None:
        malicious_ratio = float(np.clip(malicious_ratio, 0.0, 0.49))
        for nd in nodes:
            nd.is_malicious = False

        k = int(round(len(nodes) * malicious_ratio))
        k = int(np.clip(k, 0, len(nodes)))
        if k <= 0:
            return

        idx = self.rng.choice(len(nodes), size=k, replace=False)
        for i in idx:
            nodes[int(i)].is_malicious = True

    def _vrf_leader(self, nodes: List[Node], round_id: int) -> Node:
        """Pick leader by min VRF output y (verified)."""
        best = None
        best_y = None
        msg = f"round:{int(round_id)}".encode("utf-8")

        for nd in nodes:
            # vrf_prove(sk, alpha) -> (y, pi)
            y, pi = vrf_prove(nd.keypair.sk, msg)
            # vrf_verify(pk, alpha, pi) -> (ok, y2)
            ok, y2 = vrf_verify(nd.keypair.pk, msg, pi)
            if not ok:
                continue
            if best is None or y2 < best_y:
                best = nd
                best_y = y2

        return best if best is not None else nodes[0]

    def consensus_round(self,
                        m: int,
                        n: int,
                        net_bw_mbps: float,
                        base_net_lat: float,
                        round_id: int = 0,
                        malicious_ratio: Optional[float] = None) -> Dict[str, float]:
        m = int(max(4, m))
        n = int(max(0, n))

        active = self._sample_active_nodes()
        if len(active) <= 0:
            active = self.nodes[:4]

        m_eff = int(min(m, len(active)))
        m_eff = int(max(4, m_eff))

        if malicious_ratio is None:
            malicious_ratio = self.sample_dynamic_malicious_ratio()
        malicious_ratio = float(np.clip(malicious_ratio, 0.0, 0.49))
        self._mark_malicious(active, malicious_ratio)

        # committee (simple shuffle pick)
        idx = np.arange(len(active))
        self.rng.shuffle(idx)
        committee = [active[int(i)] for i in idx[:m_eff]]

        leader = self._vrf_leader(committee, round_id=int(round_id))

        # --- security (PBFT-ish): need >= 2/3 honest + 1 ---
        p_honest = float(np.clip(1.0 - malicious_ratio, 1e-9, 1.0 - 1e-9))
        k_honest_min = int(math.floor(2.0 * m_eff / 3.0) + 1)
        p_secure = _binom_tail_prob(m=m_eff, p=p_honest, k_min=k_honest_min)
        p_secure = float(np.clip(p_secure, 0.0, 1.0))
        S = float(p_secure)

        # --- latency (simplified): prop + vote + exec ---
        bw = float(max(1e-6, net_bw_mbps))
        lat0 = float(max(1e-6, base_net_lat))

        # message complexity ~ O(m^2)
        msg_factor = (m_eff * m_eff) / 2500.0
        blk_factor = float(max(1.0, n))  # proportional to block size

        t_prop = (blk_factor * 8.0) / (bw * 1000.0)
        t_vote = lat0 * (1.0 + msg_factor)
        t_exec = 0.001 * blk_factor
        t_cons = float(max(0.0, t_prop + t_vote + t_exec))

        # --- energy/cost (simplified) ---
        e_comm = float(msg_factor * (1.0 + t_prop))
        e_comp = float((m_eff / 50.0) * (1.0 + blk_factor / 300.0))
        e_cons = float(max(0.0, e_comm + e_comp))

        # normalized: bigger is better
        D = float(np.clip(1.0 / (1.0 + t_cons), 0.0, 1.0))
        E = float(np.clip(1.0 / (1.0 + e_cons), 0.0, 1.0))

        return {
            "S": float(S),
            "D": float(D),
            "E": float(E),
            "p_secure": float(S),

            "t_cons": float(t_cons),
            "e_cons": float(e_cons),

            "malicious_ratio": float(malicious_ratio),

            "leader_id": float(leader.node_id),
            "leader_is_malicious": int(bool(leader.is_malicious)),

            # compatibility aliases
            "p_hat": float(malicious_ratio),
            "leader_mal": int(bool(leader.is_malicious)),
        }
