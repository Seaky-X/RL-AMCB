# rl/preprocess_physionet_gpt4.py
from __future__ import annotations
import os, glob, json, argparse
from typing import Dict, List, Tuple
import numpy as np

from env.llm_examiner import GPTExaminer
from env.physionet2012 import _parse_outcomes  # 你们文件里已有这个函数


def read_timeseries(path: str) -> List[Tuple[str, float, float]]:
    ts = []
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
    return ts


def robust_outlier_ratio(ts: List[Tuple[str, float, float]], rng: np.random.RandomState, max_points: int = 2000) -> float:
    if not ts:
        return 0.0
    # subsample to keep cheap
    if len(ts) > max_points:
        idx = rng.choice(len(ts), size=max_points, replace=False)
        samp = [ts[i] for i in idx]
    else:
        samp = ts

    by_p: Dict[str, List[float]] = {}
    for p, _, v in samp:
        by_p.setdefault(p, []).append(v)

    out_cnt = 0
    tot = 0
    for p, vals in by_p.items():
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size < 8:
            continue
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad < 1e-6:
            continue
        z = np.abs(arr - med) / (1.4826 * mad)
        out_cnt += int(np.sum(z > 3.0))
        tot += int(arr.size)

    if tot <= 0:
        return 0.0
    return float(np.clip(out_cnt / float(tot), 0.0, 1.0))


def build_summary(record_id: str, ts: List[Tuple[str, float, float]], outcome: Dict[str, float], rng: np.random.RandomState) -> Dict:
    params = [p for (p, _, _) in ts]
    times = [t for (_, t, _) in ts]
    n_points = len(ts)
    n_unique = len(set(params)) if params else 0
    dur = (max(times) - min(times)) if times else 0.0

    # top params by count (keep compact)
    top = {}
    if params:
        uniq, cnt = np.unique(np.asarray(params), return_counts=True)
        order = np.argsort(-cnt)[:8]
        top = {str(uniq[i]): int(cnt[i]) for i in order}

    out_ratio = robust_outlier_ratio(ts, rng=rng)

    summary = {
        "record_id": record_id,
        "modality": "timeseries",
        "n_points": n_points,
        "n_unique_params": n_unique,
        "duration_hours": float(dur),
        "top_params_count": top,
        "robust_outlier_ratio": float(out_ratio),
        # outcomes (if available)
        "death": float(outcome.get("death", float("nan"))) if outcome else None,
        "sofa": float(outcome.get("sofa", float("nan"))) if outcome else None,
        "saps": float(outcome.get("saps", float("nan"))) if outcome else None,
    }
    return summary


def load_done_ids(out_path: str) -> set:
    done = set()
    if not os.path.isfile(out_path):
        return done
    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rid = str(d.get("record_id", "")).strip()
                if rid:
                    done.add(rid)
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set_dir", type=str, required=True)
    ap.add_argument("--outcomes", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_records", type=int, default=0)   # 0 = all
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rng = np.random.RandomState(int(args.seed))
    files = sorted(glob.glob(os.path.join(str(args.set_dir), "*.txt")))
    outcomes = _parse_outcomes(str(args.outcomes))

    examiner = GPTExaminer(model=str(args.model))

    done = load_done_ids(args.out_path) if args.resume else set()

    n_written = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    with open(args.out_path, "a", encoding="utf-8") as fw:
        for fp in files:
            rid = os.path.splitext(os.path.basename(fp))[0]
            if rid in done:
                continue

            ts = read_timeseries(fp)
            out = outcomes.get(rid, {})
            summary = build_summary(rid, ts, out, rng=rng)

            try:
                ann = examiner.annotate(summary)
            except Exception as e:
                # 写入失败也落盘，后续可重跑或回退
                ann = {"urgency": 0.2, "privacy": 0.2, "complexity": 0.2, "confidence": 0.0}

            row = {
                "record_id": rid,
                "urgency": float(ann["urgency"]),
                "privacy": float(ann["privacy"]),
                "complexity": float(ann["complexity"]),
                "confidence": float(ann.get("confidence", 0.5)),
            }
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

            if args.max_records > 0 and n_written >= int(args.max_records):
                break

    print(f"[DONE] wrote={n_written} -> {args.out_path}")


if __name__ == "__main__":
    main()
