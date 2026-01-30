# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np


@dataclass
class Tx:
    """Transaction object used by BlockchainEnv.

    NOTE: Keep fields stable to maintain compatibility with existing env versions.
    """
    tx_id: int
    created_round: int
    deadline_round: int
    ch: str
    urgency: float
    privacy: float
    complexity: float
    size_bytes: int
    record_id: str = ""


@dataclass
class TxPoolConfig:
    # total max capacity across all channels
    max_capacity: int = 10000


class TxPool:
    """3-channel tx pool (SEC/URG/NOR).

    Compatibility:
      - TxPool(cfg=TxPoolConfig(...), seed=0)
      - TxPool(max_capacity=..., seed=0)
      - TxPool(q_sec=..., q_urg=..., q_nor=..., seed=0)  (legacy, ignored)

    The environment only assumes:
      - qs dict exists with keys {SEC, URG, NOR}
      - add_many / add
      - drop_expired
      - select_for_block (or pop_n in some legacy envs)
      - q_len / total_len / stats
    """

    CHANNELS = ("SEC", "URG", "NOR")

    def __init__(
        self,
        cfg: Optional[TxPoolConfig] = None,
        seed: int = 0,
        max_capacity: Optional[int] = None,
        q_sec: Optional[int] = None,
        q_urg: Optional[int] = None,
        q_nor: Optional[int] = None,
        **kwargs: Any,
    ):
        if cfg is None:
            cfg = TxPoolConfig(max_capacity=int(max_capacity) if max_capacity is not None else 10000)
        # legacy args are accepted but we use a single global capacity
        self.cfg = cfg
        self.rng = np.random.RandomState(int(seed))
        self.qs: Dict[str, List[Tx]] = {"SEC": [], "URG": [], "NOR": []}

    # ---- basic ops ----
    def reset(self) -> None:
        self.clear()

    def clear(self) -> None:
        for k in self.qs:
            self.qs[k].clear()

    def total_len(self) -> int:
        return int(sum(len(v) for v in self.qs.values()))

    def q_len(self, ch: str) -> int:
        return int(len(self.qs.get(str(ch), [])))


    # backward-compat: some env versions call txpool.size(ch)
    def size(self, ch: str) -> int:
        return int(self.q_len(ch))

    def mask(self) -> np.ndarray:
        # 1 if channel non-empty
        return np.asarray(
            [
                1.0 if self.q_len("SEC") > 0 else 0.0,
                1.0 if self.q_len("URG") > 0 else 0.0,
                1.0 if self.q_len("NOR") > 0 else 0.0,
            ],
            dtype=np.float32,
        )

    def add(self, tx: Tx) -> None:
        self.add_many([tx])

    def add_many(self, txs: List[Tx]) -> None:
        """Append txs. If overflowing, drop from low-priority queues first."""
        for tx in txs:
            ch = str(getattr(tx, "ch", "NOR"))
            if ch not in self.qs:
                continue

            # enforce global capacity: drop NOR -> URG -> SEC
            while self.total_len() >= int(self.cfg.max_capacity):
                dropped = False
                for drop_ch in ("NOR", "URG", "SEC"):
                    if self.qs[drop_ch]:
                        self.qs[drop_ch].pop()  # drop tail
                        dropped = True
                        break
                if not dropped:
                    break

            self.qs[ch].append(tx)

    # ---- expiry + selection ----
    def drop_expired(self, now_round: int) -> Dict[str, int]:
        """Remove tx with deadline_round <= now_round.

        deadline_round is treated as an *inclusive* deadline at round end.
        """
        missed = {"SEC": 0, "URG": 0, "NOR": 0}
        nr = int(now_round)
        for ch in self.CHANNELS:
            q = self.qs[ch]
            if not q:
                continue
            keep: List[Tx] = []
            m = 0
            for tx in q:
                if int(tx.deadline_round) <= nr:
                    m += 1
                else:
                    keep.append(tx)
            self.qs[ch] = keep
            missed[ch] = int(m)
        return missed

    def queue_risk(self, now_round: int, max_scan: int = 2000) -> Dict[str, Dict[str, float]]:
        """Compute queue waiting / timeout risk per channel.

        This function directly traverses TxPool.qs[ch] as requested.

        Metrics:
          - age_norm_mean: mean( age / ttl ), where age = now_round - created_round,
            ttl = max(1, deadline_round - created_round).
          - overdue_frac: fraction with now_round >= deadline_round (i.e., at or past deadline).

        To avoid heavy O(N) scanning when queues are huge, we scan up to max_scan
        transactions per channel using a simple stride.
        """
        nr = int(now_round)
        max_scan = int(max(1, max_scan))
        out: Dict[str, Dict[str, float]] = {"SEC": {"age_norm_mean": 0.0, "overdue_frac": 0.0},
                                            "URG": {"age_norm_mean": 0.0, "overdue_frac": 0.0},
                                            "NOR": {"age_norm_mean": 0.0, "overdue_frac": 0.0}}
        for ch in self.CHANNELS:
            q = self.qs.get(ch, [])
            if not q:
                continue
            L = len(q)
            stride = max(1, L // max_scan)
            age_sum = 0.0
            od_cnt = 0
            cnt = 0
            for tx in q[::stride]:
                # age and normalized age
                age = float(max(0, nr - int(tx.created_round)))
                ttl = float(max(1, int(tx.deadline_round) - int(tx.created_round)))
                age_sum += float(age / ttl)
                if nr >= int(tx.deadline_round):
                    od_cnt += 1
                cnt += 1
                if cnt >= max_scan:
                    break
            if cnt > 0:
                out[ch] = {
                    "age_norm_mean": float(age_sum / float(cnt)),
                    "overdue_frac": float(od_cnt / float(cnt)),
                }
        return out

    def _sort_key(self, tx: Tx) -> Tuple[int, int]:
        # earliest deadline first, tie by created_round
        return (int(tx.deadline_round), int(tx.created_round))

    def select_for_block(self, ch: str, n_used: int, now_round: int) -> List[Tx]:
        """Pop up to n_used txs from channel ch (earliest deadlines first)."""
        ch = str(ch)
        if ch not in self.qs:
            return []
        q = self.qs[ch]
        k = int(n_used)
        if not q or k <= 0:
            return []
        q.sort(key=self._sort_key)
        kk = int(min(k, len(q)))
        picked = q[:kk]
        self.qs[ch] = q[kk:]
        return picked

    # ---- stats ----
    def _channel_arrays(self, ch: str, now_round: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = self.qs.get(ch, [])
        if not q:
            z = np.zeros((0,), dtype=np.float32)
            return z, z, z, z
        priv = np.asarray([float(t.privacy) for t in q], dtype=np.float32)
        urg = np.asarray([float(t.urgency) for t in q], dtype=np.float32)
        comp = np.asarray([float(t.complexity) for t in q], dtype=np.float32)
        slack = np.asarray([max(0.0, float(t.deadline_round - int(now_round))) for t in q], dtype=np.float32)
        return priv, urg, comp, slack

    def stats(self, ch: Optional[str] = None, now_round: int = 0, slack_quantile: float = 0.0, **kwargs: Any) -> Union[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Return per-channel stats.

        Supports both call styles used by older env variants:
          - stats(now_round=...)
          - stats('SEC', now_round=...)

        Keys are stable:
          q_len, q_len_norm, q_share,
          priv_mean, urg_mean, comp_mean,
          slack_mean, min_slack, slack_q
        """
        total = float(max(1, self.total_len()))
        out: Dict[str, Dict[str, float]] = {}
        nr = int(now_round)
        cap = float(max(1, int(self.cfg.max_capacity)))
        for c in self.CHANNELS:
            qlen = float(self.q_len(c))
            priv, urg, comp, slack = self._channel_arrays(c, nr)
            out[c] = {
                "q_len": float(qlen),
                "q_len_norm": float(np.clip(qlen / cap, 0.0, 1.0)),
                "q_share": float(qlen / total),
                "priv_mean": float(priv.mean()) if priv.size else 0.0,
                "urg_mean": float(urg.mean()) if urg.size else 0.0,
                "comp_mean": float(comp.mean()) if comp.size else 0.0,
                "slack_mean": float(slack.mean()) if slack.size else 0.0,
                "slack_q": float(np.quantile(slack, float(np.clip(slack_quantile, 0.0, 1.0)))) if (slack.size and float(slack_quantile) > 0.0) else 0.0,
                "min_slack": float(slack.min()) if slack.size else 0.0,
            }

        if ch is None:
            return out
        cc = str(ch)
        return out.get(cc, {
            "q_len": 0.0,
            "q_len_norm": 0.0,
            "q_share": 0.0,
            "priv_mean": 0.0,
            "urg_mean": 0.0,
            "comp_mean": 0.0,
            "slack_mean": 0.0,
            "slack_q": 0.0,
            "min_slack": 0.0,
        })

    # ---- legacy alias ----
    def pop_n(self, ch: str, n_used: int, now_round: int):
        picked = self.select_for_block(ch=ch, n_used=n_used, now_round=now_round)
        missed = {"SEC": 0, "URG": 0, "NOR": 0}
        return picked, missed