#!/usr/bin/env python3
"""
sweep.py — Grid search over gate parameters to find the best configuration.

For each combination of gate parameters, generates the same set of scenarios
and scores them.  Outputs a ranked results table (CSV + console) and a heatmap
of pass-rate vs. the two most important dimensions.

Usage
-----
  python sweep.py                        # default grid, 15 scenarios, seed 42
  python sweep.py --count 30 --seed 99
  python sweep.py --out-dir results/sweep01
  python sweep.py --quick                # tiny 2x2x2 grid for a fast sanity check
  python sweep.py --gate hpf             # high-pass filtered dB gate (method 10)
  python sweep.py --gate bandpass        # speech-band (300-3400 Hz) gate (method 9)
  python sweep.py --gate dual_tc         # dual time-constant envelope gate (method 6)
"""

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gate import (
    GateConfig, run_gate,
    PercentileGateConfig, run_gate_percentile,
    MinStatsGateConfig, run_gate_min_stats,
    HpfDbGateConfig, run_gate_hpf,
    BandpassDbGateConfig, run_gate_bandpass,
    DualTcGateConfig, run_gate_dual_tc,
    PeakRmsGateConfig, run_gate_peak_rms,
    RollingLinGateConfig, run_gate_rolling_lin,
    TwoStageGateConfig, run_gate_two_stage,
)
from scenario import build_recipes, mix_audio
from score import ScoreThresholds, score_scenario


# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

FULL_GRID = {
    "open_margin_db":      [6.0, 8.0, 10.0, 12.0, 14.0],
    "close_margin_db":     [2.0, 4.0, 6.0, 8.0],
    "release_ms":          [150.0, 300.0, 500.0, 800.0],
    "hold_ms":             [100.0, 250.0, 400.0],
    "noise_ema_tc_closed": [0.1, 0.3, 0.5, 1.0],
    # noise_ema_tc_open is kept fixed; sweeping it adds 4x combinations for small gain
}

QUICK_GRID = {
    "open_margin_db":      [8.0, 12.0],
    "close_margin_db":     [3.0, 6.0],
    "release_ms":          [300.0, 600.0],
    "hold_ms":             [150.0, 300.0],
    "noise_ema_tc_closed": [0.3, 0.8],
}

# ---------------------------------------------------------------------------
# HPF gate parameter grids  (method 10)
# ---------------------------------------------------------------------------

FULL_GRID_HPF = {
    "open_margin_db":      [6.0, 8.0, 10.0, 12.0],
    "close_margin_db":     [2.0, 4.0, 6.0],
    "release_ms":          [300.0, 500.0, 800.0],
    "hold_ms":             [100.0, 250.0],
    "noise_ema_tc_closed": [0.3, 1.0],
    "cutoff_hz":           [80.0, 120.0, 200.0],
    # 4 × 3 × 3 × 2 × 2 × 3 = 432 combinations
}

QUICK_GRID_HPF = {
    "open_margin_db":      [8.0, 12.0],
    "close_margin_db":     [3.0, 6.0],
    "release_ms":          [300.0, 800.0],
    "hold_ms":             [100.0, 250.0],
    "noise_ema_tc_closed": [0.3, 1.0],
    "cutoff_hz":           [80.0, 200.0],
    # 2^6 = 64 combinations
}

# ---------------------------------------------------------------------------
# Bandpass gate parameter grids  (method 9)
# ---------------------------------------------------------------------------

FULL_GRID_BANDPASS = {
    "open_margin_db":      [6.0, 8.0, 10.0, 12.0],
    "close_margin_db":     [2.0, 4.0, 6.0],
    "release_ms":          [300.0, 500.0, 800.0],
    "hold_ms":             [100.0, 250.0],
    "noise_ema_tc_closed": [0.3, 1.0],
    "low_hz":              [150.0, 300.0],
    "high_hz":             [3400.0, 4000.0],
    # 4 × 3 × 3 × 2 × 2 × 2 × 2 = 576 combinations
}

QUICK_GRID_BANDPASS = {
    "open_margin_db":      [8.0, 12.0],
    "close_margin_db":     [3.0, 6.0],
    "release_ms":          [300.0, 800.0],
    "hold_ms":             [100.0, 250.0],
    "noise_ema_tc_closed": [0.3, 1.0],
    "low_hz":              [150.0, 300.0],
    "high_hz":             [3400.0, 4000.0],
    # 2^7 = 128 combinations
}

# ---------------------------------------------------------------------------
# Dual-TC gate parameter grids  (method 6)
# ---------------------------------------------------------------------------

FULL_GRID_DUAL_TC = {
    "open_margin_db":      [6.0, 8.0, 10.0, 12.0],
    "close_margin_db":     [2.0, 4.0, 6.0],
    "release_ms":          [300.0, 500.0, 800.0],
    "hold_ms":             [100.0, 250.0],
    "noise_ema_tc_closed": [0.3, 1.0],
    "attack_tc":           [0.005, 0.020, 0.050],
    "decay_tc":            [0.15, 0.30, 0.60],
    # 4 × 3 × 3 × 2 × 2 × 3 × 3 = 1296 combinations
}

QUICK_GRID_DUAL_TC = {
    "open_margin_db":      [8.0, 12.0],
    "close_margin_db":     [3.0, 6.0],
    "release_ms":          [300.0, 800.0],
    "hold_ms":             [100.0, 250.0],
    "noise_ema_tc_closed": [0.3, 1.0],
    "attack_tc":           [0.010, 0.050],
    "decay_tc":            [0.15, 0.60],
    # 2^7 = 128 combinations
}

# ---------------------------------------------------------------------------
# Percentile gate parameter grids
# ---------------------------------------------------------------------------

FULL_GRID_PERCENTILE = {
    "open_margin_db":  [6.0, 8.0, 10.0, 12.0],
    "close_margin_db": [2.0, 4.0, 6.0],
    "release_ms":      [300.0, 500.0, 800.0],
    "hold_ms":         [100.0, 250.0],
    "window_sec":      [1.0, 2.0, 4.0],
    "percentile":      [15.0, 25.0, 35.0],
    # 4 × 3 × 3 × 2 × 3 × 3 = 648 combinations
}

# ---------------------------------------------------------------------------
# Rolling linear regression gate parameter grids  (method 14)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Method 5 — Minimum-statistics gate
# ---------------------------------------------------------------------------

FULL_GRID_MIN_STATS = {
    "open_margin_db":   [8.0, 10.0, 12.0, 15.0],
    "close_margin_db":  [3.0, 5.0, 7.0],
    "release_ms":       [200.0, 400.0, 700.0],
    "hold_ms":          [100.0, 200.0],
    "min_window_sec":   [1.0, 2.0, 3.0],
    # 4 × 3 × 3 × 2 × 3 = 216 combinations
}

QUICK_GRID_MIN_STATS = {
    "open_margin_db":   [8.0, 12.0],
    "close_margin_db":  [3.0, 7.0],
    "release_ms":       [200.0, 700.0],
    "hold_ms":          [100.0, 200.0],
    "min_window_sec":   [1.0, 3.0],
    # 2^5 = 32 combinations
}

# ---------------------------------------------------------------------------
# Method 7 — Peak + RMS hybrid gate
# ---------------------------------------------------------------------------

FULL_GRID_PEAK_RMS = {
    "open_margin_db":      [8.0, 10.0, 12.0, 15.0],
    "close_margin_db":     [3.0, 5.0, 7.0],
    "peak_open_margin_db": [12.0, 16.0, 20.0],
    "release_ms":          [200.0, 400.0, 700.0],
    "hold_ms":             [100.0, 200.0],
    # 4 × 3 × 3 × 3 × 2 = 216 combinations
}

QUICK_GRID_PEAK_RMS = {
    "open_margin_db":      [8.0, 12.0],
    "close_margin_db":     [3.0, 7.0],
    "peak_open_margin_db": [12.0, 20.0],
    "release_ms":          [200.0, 700.0],
    "hold_ms":             [100.0, 200.0],
    # 2^5 = 32 combinations
}

# ---------------------------------------------------------------------------
# Method 15 — Two-stage open/sustain gate
# ---------------------------------------------------------------------------

FULL_GRID_TWO_STAGE = {
    "open_margin_db":    [10.0, 12.0, 15.0, 18.0],
    "sustain_margin_db": [4.0, 6.0, 8.0],
    "close_margin_db":   [1.0, 2.0, 4.0],
    "release_ms":        [250.0, 400.0, 600.0],
    "hold_ms":           [150.0, 250.0],
    # 4 × 3 × 3 × 3 × 2 = 216 combinations
}

QUICK_GRID_TWO_STAGE = {
    "open_margin_db":    [10.0, 15.0],
    "sustain_margin_db": [4.0, 8.0],
    "close_margin_db":   [1.0, 4.0],
    "release_ms":        [250.0, 600.0],
    "hold_ms":           [150.0, 250.0],
    # 2^5 = 32 combinations
}

# ---------------------------------------------------------------------------
# Rolling linear regression gate parameter grids  (method 14)
# ---------------------------------------------------------------------------

FULL_GRID_ROLLING_LIN = {
    # Tuned for float32 normalised audio (RMS ~0.001–0.15).
    # sqrt(varY) of a quiet region ≈ 0.001–0.005, so stddev_open of 2–20
    # places the open threshold 0.002–0.1 above the quiet floor.
    # abs_slope_max likewise scaled down ~1000× from the original PCM grid.
    "stddev_open":   [2.0, 5.0, 10.0, 20.0],
    "close_perc":    [0.3, 0.5, 0.7],
    "history_secs":  [0.5, 1.0, 2.0],
    "abs_slope_max": [1e-6, 1e-5, 1e-4],
    "win_secs":      [0.2, 0.5, 1.0],
    # 4 × 3 × 3 × 3 × 3 = 324 combinations
}

QUICK_GRID_ROLLING_LIN = {
    "stddev_open":   [2.0, 10.0],
    "close_perc":    [0.3, 0.7],
    "history_secs":  [0.5, 2.0],
    "abs_slope_max": [1e-6, 1e-4],
    "win_secs":      [0.2, 1.0],
    # 2^5 = 32 combinations
}

QUICK_GRID_PERCENTILE = {
    "open_margin_db":  [8.0, 12.0],
    "close_margin_db": [3.0, 6.0],
    "release_ms":      [300.0, 800.0],
    "hold_ms":         [100.0, 250.0],
    "window_sec":      [2.0, 4.0],
    "percentile":      [20.0, 30.0],
    # 2^6 = 64 combinations
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _aggregate(scores: list) -> dict:
    """Summarise a list of ScenarioScore objects into aggregate metrics."""
    n = len(scores)
    if n == 0:
        return {}
    passed = sum(1 for s in scores if s.passed)
    return {
        "pass_rate":            round(passed / n, 4),
        "passed":               passed,
        "failed":               n - passed,
        "mean_missed_speech":   round(sum(s.missed_speech_sec for s in scores) / n, 4),
        "mean_false_open":      round(sum(s.false_open_sec for s in scores) / n, 4),
        "mean_start_delay_ms":  round(sum(s.max_start_delay_sec * 1000 for s in scores) / n, 2),
        "mean_early_close_ms":  round(sum(s.max_early_close_sec * 1000 for s in scores) / n, 2),
        "total_scenarios":      n,
    }


# ---------------------------------------------------------------------------
# Single grid point evaluation
# ---------------------------------------------------------------------------

def _evaluate(recipes, audio_cache, cfg, thresholds: ScoreThresholds) -> list:
    """Run gate + score for all recipes and return list of ScenarioScore.

    *cfg* may be a GateConfig, PercentileGateConfig, HpfDbGateConfig,
    BandpassDbGateConfig, or DualTcGateConfig.
    """
    scores = []
    for recipe in recipes:
        audio = audio_cache[recipe["scenario_id"]]
        if isinstance(cfg, PercentileGateConfig):
            predicted, _ = run_gate_percentile(audio, recipe["sample_rate"], config=cfg)
        elif isinstance(cfg, HpfDbGateConfig):
            predicted, _ = run_gate_hpf(audio, recipe["sample_rate"], config=cfg)
        elif isinstance(cfg, BandpassDbGateConfig):
            predicted, _ = run_gate_bandpass(audio, recipe["sample_rate"], config=cfg)
        elif isinstance(cfg, DualTcGateConfig):
            predicted, _ = run_gate_dual_tc(audio, recipe["sample_rate"], config=cfg)
        elif isinstance(cfg, MinStatsGateConfig):
            predicted, _ = run_gate_min_stats(audio, recipe["sample_rate"], config=cfg)
        elif isinstance(cfg, PeakRmsGateConfig):
            predicted, _ = run_gate_peak_rms(audio, recipe["sample_rate"], config=cfg)
        elif isinstance(cfg, TwoStageGateConfig):
            predicted, _ = run_gate_two_stage(audio, recipe["sample_rate"], config=cfg)
        elif isinstance(cfg, RollingLinGateConfig):
            predicted, _ = run_gate_rolling_lin(audio, recipe["sample_rate"], config=cfg)
        else:
            predicted, _ = run_gate(audio, recipe["sample_rate"], config=cfg)
        sc = score_scenario(
            recipe["scenario_id"],
            recipe["truth_segments"],
            predicted,
            recipe["duration_sec"],
            thresholds,
        )
        scores.append(sc)
    return scores


# ---------------------------------------------------------------------------
# Heatmap plot
# ---------------------------------------------------------------------------

def _heatmap(results: list[dict], x_key: str, y_key: str, out_path: Path) -> None:
    """2-D pass-rate heatmap for two parameters, marginalised over the rest."""
    x_vals = sorted(set(r["params"][x_key] for r in results))
    y_vals = sorted(set(r["params"][y_key] for r in results))

    grid = np.zeros((len(y_vals), len(x_vals)))
    counts = np.zeros_like(grid, dtype=int)
    for r in results:
        xi = x_vals.index(r["params"][x_key])
        yi = y_vals.index(r["params"][y_key])
        grid[yi, xi] += r["pass_rate"]
        counts[yi, xi] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(counts > 0, grid / counts, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Pass rate: {x_key} vs {y_key}\n(averaged over other params)")
    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            val = grid[yi, xi]
            if not np.isnan(val):
                ax.text(xi, yi, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, label="pass rate")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Grid-search gate parameters")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/sweep")
    parser.add_argument("--count", type=int, default=15,
                        help="Scenarios per grid point (same seed = same scenarios)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Use a small 2-value grid for a fast sanity check")
    parser.add_argument("--gate",
                        choices=["ema", "percentile", "min_stats", "hpf", "bandpass",
                                 "dual_tc", "peak_rms", "rolling_lin", "two_stage"],
                        default="ema",
                        help="Gate algorithm to sweep (default: ema)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.gate == "percentile":
        grid = QUICK_GRID_PERCENTILE if args.quick else FULL_GRID_PERCENTILE
    elif args.gate == "min_stats":
        grid = QUICK_GRID_MIN_STATS if args.quick else FULL_GRID_MIN_STATS
    elif args.gate == "hpf":
        grid = QUICK_GRID_HPF if args.quick else FULL_GRID_HPF
    elif args.gate == "bandpass":
        grid = QUICK_GRID_BANDPASS if args.quick else FULL_GRID_BANDPASS
    elif args.gate == "dual_tc":
        grid = QUICK_GRID_DUAL_TC if args.quick else FULL_GRID_DUAL_TC
    elif args.gate == "peak_rms":
        grid = QUICK_GRID_PEAK_RMS if args.quick else FULL_GRID_PEAK_RMS
    elif args.gate == "rolling_lin":
        grid = QUICK_GRID_ROLLING_LIN if args.quick else FULL_GRID_ROLLING_LIN
    elif args.gate == "two_stage":
        grid = QUICK_GRID_TWO_STAGE if args.quick else FULL_GRID_TWO_STAGE
    else:
        grid = QUICK_GRID if args.quick else FULL_GRID
    param_keys = list(grid.keys())
    combinations = list(itertools.product(*[grid[k] for k in param_keys]))
    total = len(combinations)

    print(f"Grid: {' × '.join(f'{k}({len(grid[k])})' for k in param_keys)}")
    print(f"Total combinations: {total}")
    print(f"Scenarios per point: {args.count}  (seed={args.seed})\n")

    # Load index
    index_path = Path(args.data_dir) / "index.json"
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}")
    with open(index_path) as fh:
        index = json.load(fh)

    thresholds = ScoreThresholds()

    # Generate recipes once — same for every grid point
    recipes = build_recipes(index, args.count, seed=args.seed)

    # Pre-mix all audio once and cache — avoids redundant I/O per grid point
    print("Pre-mixing audio for all scenarios ...")
    audio_cache = {}
    for recipe in recipes:
        audio_cache[recipe["scenario_id"]] = mix_audio(recipe)
    print(f"  cached {len(audio_cache)} clips\n")

    # --- Sweep ---
    all_results = []
    t0 = time.monotonic()

    for idx, combo in enumerate(combinations):
        params = dict(zip(param_keys, combo))
        if args.gate == "percentile":
            cfg = PercentileGateConfig(
                open_margin_db=params["open_margin_db"],
                close_margin_db=params["close_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
                window_sec=params["window_sec"],
                percentile=params["percentile"],
            )
            extra = f"win={params['window_sec']:.1f}s  pct={params['percentile']:.0f}"
        elif args.gate == "min_stats":
            cfg = MinStatsGateConfig(
                open_margin_db=params["open_margin_db"],
                close_margin_db=params["close_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
                min_window_sec=params["min_window_sec"],
            )
            extra = f"win={params['min_window_sec']:.1f}s"
        elif args.gate == "peak_rms":
            cfg = PeakRmsGateConfig(
                open_margin_db=params["open_margin_db"],
                close_margin_db=params["close_margin_db"],
                peak_open_margin_db=params["peak_open_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
            )
            extra = f"pkmargin={params['peak_open_margin_db']:.0f}dB"
        elif args.gate == "two_stage":
            cfg = TwoStageGateConfig(
                open_margin_db=params["open_margin_db"],
                sustain_margin_db=params["sustain_margin_db"],
                close_margin_db=params["close_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
            )
            extra = f"sust={params['sustain_margin_db']:.0f}dB"
        elif args.gate == "hpf":
            cfg = HpfDbGateConfig(
                open_margin_db=params["open_margin_db"],
                close_margin_db=params["close_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
                noise_ema_tc_closed=params["noise_ema_tc_closed"],
                cutoff_hz=params["cutoff_hz"],
            )
            extra = f"tc={params['noise_ema_tc_closed']:.1f}s  hp={params['cutoff_hz']:.0f}Hz"
        elif args.gate == "bandpass":
            cfg = BandpassDbGateConfig(
                open_margin_db=params["open_margin_db"],
                close_margin_db=params["close_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
                noise_ema_tc_closed=params["noise_ema_tc_closed"],
                low_hz=params["low_hz"],
                high_hz=params["high_hz"],
            )
            extra = f"tc={params['noise_ema_tc_closed']:.1f}s  bp={params['low_hz']:.0f}-{params['high_hz']:.0f}Hz"
        elif args.gate == "dual_tc":
            cfg = DualTcGateConfig(
                open_margin_db=params["open_margin_db"],
                close_margin_db=params["close_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
                noise_ema_tc_closed=params["noise_ema_tc_closed"],
                attack_tc=params["attack_tc"],
                decay_tc=params["decay_tc"],
            )
            extra = f"tc={params['noise_ema_tc_closed']:.1f}s  atk={params['attack_tc']*1000:.0f}ms  dec={params['decay_tc']*1000:.0f}ms"
        elif args.gate == "rolling_lin":
            cfg = RollingLinGateConfig(
                stddev_open=params["stddev_open"],
                close_perc=params["close_perc"],
                history_secs=params["history_secs"],
                abs_slope_max=params["abs_slope_max"],
                win_secs=params["win_secs"],
            )
            extra = (f"std={params['stddev_open']:.0f}  cperc={params['close_perc']:.1f}  "
                     f"hist={params['history_secs']:.1f}s  slp={params['abs_slope_max']:.5f}  "
                     f"win={params['win_secs']:.2f}s")
        else:
            cfg = GateConfig(
                open_margin_db=params["open_margin_db"],
                close_margin_db=params["close_margin_db"],
                release_ms=params["release_ms"],
                hold_ms=params["hold_ms"],
                noise_ema_tc_closed=params.get("noise_ema_tc_closed", 0.3),
            )
            extra = f"tc={params.get('noise_ema_tc_closed', 0.3):.1f}s"
        scores = _evaluate(recipes, audio_cache, cfg, thresholds)
        agg = _aggregate(scores)
        row = {"params": params, **agg}
        all_results.append(row)

        elapsed = time.monotonic() - t0
        eta = elapsed / (idx + 1) * (total - idx - 1)
        if args.gate == "rolling_lin":
            print(
                f"[{idx+1:4d}/{total}]  "
                f"{extra}  "
                f"pass={agg['pass_rate']:5.1%}  "
                f"ETA {eta:.0f}s"
            )
        else:
            print(
                f"[{idx+1:4d}/{total}]  "
                f"open={params['open_margin_db']:4.1f}  close={params['close_margin_db']:3.1f}  "
                f"rel={params['release_ms']:5.0f}ms  hold={params['hold_ms']:5.0f}ms  "
                f"{extra}  "
                f"pass={agg['pass_rate']:5.1%}  "
                f"ETA {eta:.0f}s"
            )
        # Note: min_stats/peak_rms/two_stage all have open/close/rel/hold so use the else branch

    elapsed_total = time.monotonic() - t0
    print(f"\nSweep complete in {elapsed_total:.1f}s")

    # --- Sort by pass_rate desc, then mean_missed_speech asc ---
    all_results.sort(key=lambda r: (-r["pass_rate"], r["mean_missed_speech"]))

    # --- Top 10 ---
    if args.gate == "percentile":
        hdr_extra = f"{'Window':>7}  {'Pct':>5}"
    elif args.gate == "min_stats":
        hdr_extra = f"{'MinWin':>6}"
    elif args.gate == "hpf":
        hdr_extra = f"{'EMAcls':>6}  {'HPFHz':>7}"
    elif args.gate == "bandpass":
        hdr_extra = f"{'EMAcls':>6}  {'LowHz':>6}  {'HighHz':>7}"
    elif args.gate == "dual_tc":
        hdr_extra = f"{'EMAcls':>6}  {'AtkMs':>6}  {'DecMs':>6}"
    elif args.gate == "peak_rms":
        hdr_extra = f"{'PkMgn':>6}"
    elif args.gate == "two_stage":
        hdr_extra = f"{'SustMgn':>7}"
    elif args.gate == "rolling_lin":
        hdr_extra = f"{'Std':>6}  {'CPct':>5}  {'Hist':>5}  {'Slope':>9}  {'Win':>5}"
    else:
        hdr_extra = f"{'EMAcls':>6}"
    col_width = 115 if args.gate == "rolling_lin" else 105
    if args.gate == "rolling_lin":
        print(f"\n{'Rank':<5} {'PassRate':>8}  {'MissSpch':>9}  {'FalseOpen':>9}  {hdr_extra}")
    else:
        print(f"\n{'Rank':<5} {'PassRate':>8}  {'MissSpch':>9}  {'FalseOpen':>9}  "
              f"{'OpenMgn':>7}  {'ClsMgn':>6}  {'Rel':>6}  {'Hold':>5}  {hdr_extra}")
    print("-" * col_width)
    for rank, r in enumerate(all_results[:10], 1):
        p = r["params"]
        if args.gate == "percentile":
            row_extra = f"{p['window_sec']:>7.1f}  {p['percentile']:>5.0f}"
        elif args.gate == "hpf":
            row_extra = f"{p['noise_ema_tc_closed']:>6.2f}  {p['cutoff_hz']:>7.0f}"
        elif args.gate == "bandpass":
            row_extra = f"{p['noise_ema_tc_closed']:>6.2f}  {p['low_hz']:>6.0f}  {p['high_hz']:>7.0f}"
        elif args.gate == "dual_tc":
            row_extra = f"{p['noise_ema_tc_closed']:>6.2f}  {p['attack_tc']*1000:>6.0f}  {p['decay_tc']*1000:>6.0f}"
        elif args.gate == "min_stats":
            row_extra = f"{p['min_window_sec']:>6.1f}"
        elif args.gate == "peak_rms":
            row_extra = f"{p['peak_open_margin_db']:>6.0f}"
        elif args.gate == "two_stage":
            row_extra = f"{p['sustain_margin_db']:>7.0f}"
        elif args.gate == "rolling_lin":
            row_extra = (f"{p['stddev_open']:>6.0f}  {p['close_perc']:>5.1f}  "
                         f"{p['history_secs']:>5.1f}  {p['abs_slope_max']:>9.5f}  {p['win_secs']:>5.2f}")
        else:
            row_extra = f"{p.get('noise_ema_tc_closed', 0.3):>6.2f}"
        if args.gate == "rolling_lin":
            print(
                f"{rank:<5} {r['pass_rate']:>8.1%}  {r['mean_missed_speech']:>9.3f}  "
                f"{r['mean_false_open']:>9.3f}  "
                f"{row_extra}"
            )
        else:
            print(
                f"{rank:<5} {r['pass_rate']:>8.1%}  {r['mean_missed_speech']:>9.3f}  "
                f"{r['mean_false_open']:>9.3f}  "
                f"{p['open_margin_db']:>7.1f}  {p['close_margin_db']:>6.1f}  "
                f"{p['release_ms']:>6.0f}  {p['hold_ms']:>5.0f}  "
                f"{row_extra}"
            )

    # --- Write CSV ---
    csv_path = out_dir / "sweep_results.csv"
    csv_fields = param_keys + [
        "pass_rate", "passed", "failed",
        "mean_missed_speech", "mean_false_open",
        "mean_start_delay_ms", "mean_early_close_ms",
        "total_scenarios",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            row = {**r["params"], **{k: r[k] for k in csv_fields if k not in r["params"]}}
            writer.writerow(row)

    # --- Write JSON summary ---
    json_path = out_dir / "sweep_results.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "seed": args.seed,
                "scenario_count": args.count,
                "grid": grid,
                "best": all_results[:5],
                "all": all_results,
            },
            fh,
            indent=2,
        )

    # --- Heatmaps ---
    print(f"\nGenerating heatmaps ...")
    _NO_OPEN_CLOSE_PARAMS = {"rolling_lin"}
    if args.gate not in _NO_OPEN_CLOSE_PARAMS:
        _heatmap(all_results, "open_margin_db", "close_margin_db",
                 out_dir / "heatmap_open_vs_close.png")
    if args.gate == "percentile":
        _heatmap(all_results, "window_sec", "percentile",
                 out_dir / "heatmap_window_vs_percentile.png")
        _heatmap(all_results, "open_margin_db", "window_sec",
                 out_dir / "heatmap_open_vs_window.png")
    elif args.gate == "hpf":
        _heatmap(all_results, "cutoff_hz", "open_margin_db",
                 out_dir / "heatmap_cutoff_vs_open.png")
        _heatmap(all_results, "noise_ema_tc_closed", "open_margin_db",
                 out_dir / "heatmap_tc_vs_open.png")
    elif args.gate == "bandpass":
        _heatmap(all_results, "low_hz", "high_hz",
                 out_dir / "heatmap_low_vs_high.png")
        _heatmap(all_results, "noise_ema_tc_closed", "open_margin_db",
                 out_dir / "heatmap_tc_vs_open.png")
    elif args.gate == "dual_tc":
        _heatmap(all_results, "attack_tc", "decay_tc",
                 out_dir / "heatmap_attack_vs_decay.png")
        _heatmap(all_results, "noise_ema_tc_closed", "open_margin_db",
                 out_dir / "heatmap_tc_vs_open.png")
    elif args.gate == "min_stats":
        _heatmap(all_results, "open_margin_db", "min_window_sec",
                 out_dir / "heatmap_open_vs_minwin.png")
        _heatmap(all_results, "open_margin_db", "release_ms",
                 out_dir / "heatmap_open_vs_release.png")
    elif args.gate == "peak_rms":
        _heatmap(all_results, "open_margin_db", "peak_open_margin_db",
                 out_dir / "heatmap_open_vs_peak.png")
        _heatmap(all_results, "open_margin_db", "release_ms",
                 out_dir / "heatmap_open_vs_release.png")
    elif args.gate == "two_stage":
        _heatmap(all_results, "open_margin_db", "sustain_margin_db",
                 out_dir / "heatmap_open_vs_sustain.png")
        _heatmap(all_results, "sustain_margin_db", "release_ms",
                 out_dir / "heatmap_sustain_vs_release.png")
    elif args.gate == "rolling_lin":
        _heatmap(all_results, "stddev_open", "close_perc",
                 out_dir / "heatmap_stddev_vs_closeperc.png")
        _heatmap(all_results, "history_secs", "win_secs",
                 out_dir / "heatmap_history_vs_win.png")
        _heatmap(all_results, "abs_slope_max", "stddev_open",
                 out_dir / "heatmap_slope_vs_stddev.png")
    else:
        _heatmap(all_results, "noise_ema_tc_closed", "open_margin_db",
                 out_dir / "heatmap_tc_vs_open.png")

    print(f"\nResults CSV  : {csv_path}")
    print(f"Results JSON : {json_path}")
    print(f"Heatmaps     : {out_dir}/heatmap_*.png")

    best = all_results[0]
    print(f"\nBest config  : {best['params']}")
    print(f"             pass_rate={best['pass_rate']:.1%}  "
          f"missed={best['mean_missed_speech']:.3f}s  "
          f"false_open={best['mean_false_open']:.3f}s")


if __name__ == "__main__":
    main()
