# -*- coding: utf-8 -*-
"""
env/physionet2012.py

PhysioNet 2012 ICU (set-a / set-b) streaming loader.

Key fix:
- Channel assignment MUST NOT use a too-high fixed threshold (otherwise URG becomes empty).
- We assign channel per-batch using quantiles (q_priv / q_urg).
- Threshold is capped to (max - eps) if the whole batch is low-valued, avoiding empty SEC/URG.

No extra libs beyond numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import glob
import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class RecordDemand:
    record_id: str
    urgency: float
    privacy: float
    complexity: float
    channel: str = ""           # "SEC" / "URG" / "NOR"
    deadline_offset: int = 0    # rounds until deadline


def _parse_outcomes(outcomes_path: str) -> Dict[str, Dict[str, float]]:
    """
    Outcomes-a.txt format:
    RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death
    """
    out: Dict[str, Dict[str, float]] = {}
    if (not outcomes_path) or (not os.path.isfile(outcomes_path)):
        return out

    with open(outcomes_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != len(header):
                continue
            d = {header[i]: parts[i] for i in range(len(header))}
            rid = str(d.get("RecordID", "")).strip()
            if not rid:
                continue

            def _f(key: str) -> float:
                try:
                    return float(d.get(key, "nan"))
                except Exception:
                    return float("nan")

            out[rid] = {
                "saps": _f("SAPS-I"),
                "sofa": _f("SOFA"),
                "los": _f("Length_of_stay"),
                "death": _f("In-hospital_death"),
            }
    return out


def derive_demand_scores(timeseries: List[Tuple[str, float, float]],
                         outcome: Optional[Dict[str, float]],
                         rng: np.random.RandomState) -> Tuple[float, float, float]:
    if not timeseries:
        u = 0.2 + 0.1 * rng.rand()
        p = 0.2 + 0.1 * rng.rand()
        c = 0.2 + 0.1 * rng.rand()
        return _clip01(u), _clip01(p), _clip01(c)

    params = [p for (p, _, _) in timeseries]
    n_points = len(params)
    n_unique = len(set(params))
    comp = _clip01(0.10 + 0.35 * (n_points / 200.0) + 0.55 * (n_unique / 35.0))

    vals = np.asarray([v for (_, _, v) in timeseries], dtype=np.float32)
    abnormal = float(np.mean(np.abs(vals) > 2.0)) if vals.size > 0 else 0.0
    abnormal = float(np.clip(abnormal, 0.0, 1.0))

    sev = 0.0
    if outcome:
        death = outcome.get("death", float("nan"))
        sofa = outcome.get("sofa", float("nan"))
        saps = outcome.get("saps", float("nan"))
        if not np.isnan(death):
            sev += 0.40 * float(np.clip(death, 0.0, 1.0))
        if not np.isnan(sofa):
            sev += 0.30 * float(np.clip(sofa / 20.0, 0.0, 1.0))
        if not np.isnan(saps):
            sev += 0.30 * float(np.clip(saps / 60.0, 0.0, 1.0))
    sev = _clip01(sev)

    urg = _clip01(0.18 + 0.70 * abnormal + 0.12 * comp + 0.25 * sev + 0.02 * (rng.rand() - 0.5))
    priv = _clip01(0.35 + 0.25 * sev + 0.20 * comp + 0.05 * (rng.rand() - 0.5))
    return float(urg), float(priv), float(comp)


class PhysioNet2012Stream:
    def __init__(self,
                 set_dir: str,
                 outcomes_path: str,
                 seed: int = 0,
                 shuffle: bool = True,
                 cache: bool = True):
        self.set_dir = str(set_dir)
        self.outcomes_path = str(outcomes_path)
        self.rng = np.random.RandomState(int(seed))
        self.shuffle = bool(shuffle)
        self.cache = bool(cache)

        self._files: List[str] = sorted(glob.glob(os.path.join(self.set_dir, "*.txt")))
        self._n = len(self._files)
        self._idx = 0

        self._outcomes = _parse_outcomes(self.outcomes_path)
        self._cache_ts: Dict[str, List[Tuple[str, float, float]]] = {}

        if self.shuffle:
            self.rng.shuffle(self._files)

    def __len__(self) -> int:
        return int(self._n)

    def _read_timeseries(self, path: str) -> List[Tuple[str, float, float]]:
        rid = os.path.splitext(os.path.basename(path))[0]
        if self.cache and rid in self._cache_ts:
            return self._cache_ts[rid]

        ts: List[Tuple[str, float, float]] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                _ = f.readline()
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 3:
                        continue
                    t_str, p, v_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    try:
                        hh, mm = t_str.split(":")
                        t = float(hh) + float(mm) / 60.0
                    except Exception:
                        t = 0.0
                    try:
                        v = float(v_str)
                    except Exception:
                        continue
                    ts.append((p, t, v))
        except Exception:
            ts = []

        if self.cache:
            self._cache_ts[rid] = ts
        return ts

    def _get_one(self) -> RecordDemand:
        if self._n == 0:
            u = 0.2 + 0.6 * self.rng.rand()
            p = 0.2 + 0.6 * self.rng.rand()
            c = 0.2 + 0.6 * self.rng.rand()
            return RecordDemand(record_id="synthetic", urgency=float(u), privacy=float(p), complexity=float(c))

        if self._idx >= self._n:
            self._idx = 0
            if self.shuffle:
                self.rng.shuffle(self._files)

        path = self._files[self._idx]
        self._idx += 1

        rid = os.path.splitext(os.path.basename(path))[0]
        ts = self._read_timeseries(path)
        outcome = self._outcomes.get(rid, None)

        urg, priv, comp = derive_demand_scores(ts, outcome, self.rng)
        return RecordDemand(record_id=str(rid), urgency=float(urg), privacy=float(priv), complexity=float(comp))

    def sample_batch(self,
                     batch_size: int,
                     q_priv: float = 0.80,
                     q_urg: float = 0.80,
                     min_thr: float = 0.55,
                     sec_deadline_range: Tuple[int, int] = (10, 20),
                     urg_deadline_range: Tuple[int, int] = (2, 6),
                     nor_deadline_range: Tuple[int, int] = (6, 12),
                     enforce_mix: Optional[Tuple[float, float, float]] = None,
                     enforce_min_per_ch: int = 0,
                     deadline_compress: float = 1.0) -> List[RecordDemand]:
        bs = max(1, int(batch_size))
        dem: List[RecordDemand] = [self._get_one() for _ in range(bs)]

        privs = np.asarray([d.privacy for d in dem], dtype=np.float32)
        urgs = np.asarray([d.urgency for d in dem], dtype=np.float32)

        thr_priv = float(np.quantile(privs, float(np.clip(q_priv, 0.0, 1.0))))
        thr_urg = float(np.quantile(urgs, float(np.clip(q_urg, 0.0, 1.0))))

        # avoid empty SEC/URG when batch is low-valued: cap to (max - eps)
        max_priv = float(np.max(privs)) if privs.size > 0 else 0.0
        max_urg = float(np.max(urgs)) if urgs.size > 0 else 0.0
        thr_priv = float(max(float(min_thr), thr_priv))
        thr_urg = float(max(float(min_thr), thr_urg))
        thr_priv = float(min(thr_priv, max(0.0, max_priv - 1e-6)))
        thr_urg = float(min(thr_urg, max(0.0, max_urg - 1e-6)))

        # ---------------- curriculum / injection (optional) ----------------
        # enforce_mix = (p_sec, p_urg, p_nor). When provided, we will:
        # 1) pick SEC from top-privacy samples,
        # 2) pick URG from top-urgency among the remaining,
        # 3) assign the rest to NOR.
        if enforce_mix is not None:
            p = np.asarray(enforce_mix, dtype=np.float32).reshape(-1)
            if p.size != 3 or not np.isfinite(p).all() or float(p.sum()) <= 0.0:
                p = np.asarray([0.33, 0.33, 0.34], dtype=np.float32)
            p = p / float(p.sum())

            minc = int(max(0, enforce_min_per_ch))
            # initial counts (rounded)
            sec_cnt = int(round(float(batch_size) * float(p[0])))
            urg_cnt = int(round(float(batch_size) * float(p[1])))

            sec_cnt = max(sec_cnt, minc)
            urg_cnt = max(urg_cnt, minc)

            # keep at least minc for NOR if requested
            max_non_nor = batch_size - minc
            if sec_cnt + urg_cnt > max_non_nor:
                excess = (sec_cnt + urg_cnt) - max_non_nor
                while excess > 0 and (sec_cnt > minc or urg_cnt > minc):
                    if (sec_cnt - minc) >= (urg_cnt - minc) and sec_cnt > minc:
                        sec_cnt -= 1
                    elif urg_cnt > minc:
                        urg_cnt -= 1
                    else:
                        break
                    excess -= 1

            nor_cnt = batch_size - sec_cnt - urg_cnt
            if nor_cnt < minc:
                need = minc - nor_cnt
                while need > 0 and (sec_cnt > minc or urg_cnt > minc):
                    if sec_cnt > urg_cnt and sec_cnt > minc:
                        sec_cnt -= 1
                    elif urg_cnt > minc:
                        urg_cnt -= 1
                    else:
                        break
                    need -= 1
                nor_cnt = batch_size - sec_cnt - urg_cnt

            priv = np.asarray([float(d.privacy) for d in dem], dtype=np.float32)
            urg = np.asarray([float(d.urgency) for d in dem], dtype=np.float32)

            sec_order = np.argsort(-priv)
            sec_set = set([int(i) for i in sec_order[:sec_cnt]])

            remain = [i for i in range(batch_size) if i not in sec_set]
            urg_scores = urg[remain]
            urg_order_local = np.argsort(-urg_scores)[:urg_cnt]
            urg_set = set([int(remain[int(j)]) for j in urg_order_local])

            # rest is NOR
            nor_set = set(range(batch_size)) - sec_set - urg_set

            def _sample_deadline(rng_tuple):
                lo, hi = int(rng_tuple[0]), int(rng_tuple[1])
                lo = int(max(1, lo))
                hi = int(max(lo + 1, hi))
                off = int(self.rng.randint(lo, hi))
                c = float(max(1e-3, deadline_compress))
                off = int(max(1, round(float(off) * c)))
                return off

            for i, d in enumerate(dem):
                if i in sec_set:
                    d.channel = "SEC"
                    d.deadline_offset = _sample_deadline(sec_deadline_range)
                elif i in urg_set:
                    d.channel = "URG"
                    d.deadline_offset = _sample_deadline(urg_deadline_range)
                else:
                    d.channel = "NOR"
                    d.deadline_offset = _sample_deadline(nor_deadline_range)
            return dem

        # ---------------- default (data-driven thresholding) ----------------
        c = float(max(1e-3, deadline_compress))
        for d in dem:
            if d.privacy >= thr_priv:
                d.channel = "SEC"
                lo, hi = sec_deadline_range
            elif d.urgency >= thr_urg:
                d.channel = "URG"
                lo, hi = urg_deadline_range
            else:
                d.channel = "NOR"
                lo, hi = nor_deadline_range

            lo = int(max(1, lo))
            hi = int(max(lo + 1, hi))
            off = int(self.rng.randint(lo, hi))
            d.deadline_offset = int(max(1, round(float(off) * c)))

        return dem
