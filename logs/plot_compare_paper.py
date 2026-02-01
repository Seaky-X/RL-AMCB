
# -*- coding: utf-8 -*-
"""
logs/plot_compare_paper.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PALETTE = ["#ffbe7a", "#3480b8", "#9bbf8a", "#c2bdde", "#c82423"]


def apply_paper_style(font: str = "Times New Roman", base: int = 9) -> None:
    plt.rcParams.update({
        "font.family": font,
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": base - 1,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _grid(ax):
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.6)


def _safe_series(df: pd.DataFrame, col: str) -> np.ndarray:
    s = pd.to_numeric(df.get(col, pd.Series([], dtype=float)), errors="coerce")
    return s.fillna(method="ffill").fillna(0.0).to_numpy(dtype=float)


def _per_episode(df: pd.DataFrame, metric: str, how: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (episodes, values) aggregated per episode.
    how in {"last","mean","sum","max"}.
    """
    if "episode" not in df.columns or metric not in df.columns:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    g = df.groupby("episode", sort=True)[metric]
    if how == "last":
        y = g.last()
    elif how == "sum":
        y = g.sum()
    elif how == "max":
        y = g.max()
    else:
        y = g.mean()
    x = y.index.to_numpy(dtype=int)
    yv = pd.to_numeric(y, errors="coerce").fillna(method="ffill").fillna(0.0).to_numpy(dtype=float)
    return x, yv


def _rolling(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or y.size == 0:
        return y
    s = pd.Series(y)
    return s.rolling(win, min_periods=max(1, win // 3), center=False).mean().to_numpy(dtype=float)


def _plot_methods(methods: List[Dict], x: str, y: str, how: str, out: Path,
                  fig_w: float, fig_h: float, roll: int, ylabel: str, fname: str,
                  ylog: bool = False, ylim0: bool = False, hline0: bool = False):
    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    _grid(ax)

    for i, m in enumerate(methods):
        df = m["df"]
        label = m["label"]
        color = m["color"]
        xx, yy = _per_episode(df, y, how)
        if yy.size == 0:
            continue
        yy = _rolling(yy, roll)
        ax.plot(xx, yy, label=label, color=color)

    if hline0:
        ax.axhline(0.0, linestyle=":", linewidth=1.0, color="black", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if ylim0:
        ax.set_ylim(bottom=0.0)
    if ylog:
        ax.set_yscale("log")

    ax.legend(frameon=False, ncol=2)
    plt.tight_layout(pad=0.15)

    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / f"{fname}.pdf", dpi=300)
    plt.savefig(out / f"{fname}.png", dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs="+", required=True,
                    help='Each item: "Label:path/to.csv"')
    ap.add_argument("--out_dir", type=str, default="logs/figs_cmp")
    ap.add_argument("--roll", type=int, default=25)
    ap.add_argument("--font", type=str, default="Times New Roman")
    ap.add_argument("--base_font", type=int, default=9)
    ap.add_argument("--column", type=int, default=1, choices=[1, 2], help="1 single-column, 2 double-column")
    ap.add_argument("--fig_w", type=float, default=None, help="Override figure width in inches")
    ap.add_argument("--fig_h", type=float, default=None, help="Override figure height in inches")
    args = ap.parse_args()

    apply_paper_style(args.font, args.base_font)

    # IEEE/ACM common widths: single ~3.35in, double ~6.9in
    default_w = 3.35 if args.column == 1 else 6.9
    default_h = 2.15 if args.column == 1 else 2.5
    fig_w = float(args.fig_w) if args.fig_w is not None else default_w
    fig_h = float(args.fig_h) if args.fig_h is not None else default_h

    methods: List[Dict] = []
    for i, spec in enumerate(args.runs):
        if ":" not in spec:
            raise SystemExit(f'Bad --runs item: "{spec}". Use "Label:path.csv".')
        label, path = spec.split(":", 1)
        p = Path(path).expanduser()
        if not p.exists():
            raise SystemExit(f"CSV not found: {p}")
        df = pd.read_csv(p)
        methods.append({"label": label, "path": str(p), "df": df, "color": PALETTE[i % len(PALETTE)]})

    out = Path(args.out_dir)

    # 1) Return (episode)
    _plot_methods(methods, x="episode", y="episode_return", how="last", out=out,
                  fig_w=fig_w, fig_h=fig_h, roll=args.roll,
                  ylabel="Episode return", fname="cmp_return", ylim0=False)

    # 2) phi (mean per episode)
    _plot_methods(methods, x="episode", y="phi", how="mean", out=out,
                  fig_w=fig_w, fig_h=fig_h, roll=args.roll,
                  ylabel="Utility (phi)", fname="cmp_phi", ylim0=False)

    # 3) URG miss rate (mean per episode)
    _plot_methods(methods, x="episode", y="miss_rate_URG", how="mean", out=out,
                  fig_w=fig_w, fig_h=fig_h, roll=args.roll,
                  ylabel="URG miss rate", fname="cmp_miss_urg", ylim0=True)

    # 4) SEC provisioning gap (m_SEC - m_req_SEC), mean per episode
    # If m_req_SEC missing, will skip.
    for m in methods:
        df = m["df"]
        if "m_SEC" in df.columns and "m_req_SEC" in df.columns:
            df = df.copy()
            df["gap_m_sec"] = pd.to_numeric(df["m_SEC"], errors="coerce") - pd.to_numeric(df["m_req_SEC"], errors="coerce")
            m["df"] = df
    _plot_methods(methods, x="episode", y="gap_m_sec", how="mean", out=out,
                  fig_w=fig_w, fig_h=fig_h, roll=args.roll,
                  ylabel=r"SEC provisioning gap ($m_{\mathrm{SEC}}-m_{\mathrm{req}}$)",
                  fname="cmp_gap_m_sec", ylim0=False, hline0=True)

    # 5) backlog_w (mean per episode)
    _plot_methods(methods, x="episode", y="backlog_w", how="mean", out=out,
                  fig_w=fig_w, fig_h=fig_h, roll=args.roll,
                  ylabel="Demand-weighted backlog", fname="cmp_backlogw", ylim0=True)

    print(f"[OK] saved comparison figs to: {out.resolve()}")


if __name__ == "__main__":
    main()
