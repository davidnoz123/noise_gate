#!/usr/bin/env python3
"""
run.py — Noise gate test harness: generate scenarios, run gate, score, report.

Usage
-----
  python run.py                          # smoke test: 10 scenarios, default gate
  python run.py --count 20 --seed 99
  python run.py --save-failures          # write WAV for each failing scenario
  python run.py --save-scenario quiet_speaker_const_000042_0003
  python run.py --out-dir results/run01
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from gate import AdaptiveDbGate, GateConfig, run_gate
from scenario import build_recipes, mix_audio
from score import ScoreThresholds, score_scenario, score_to_dict

import soundfile as sf


# ---------------------------------------------------------------------------
# Failure plots
# ---------------------------------------------------------------------------

def _plot_scenario(
    recipe: dict,
    audio: np.ndarray,
    trace: list[dict],
    score,
    out_path: Path,
) -> None:
    sr = recipe["sample_rate"]
    duration = recipe["duration_sec"]
    t = np.arange(len(audio)) / sr

    # Compute dB envelope from trace
    times = [row["time_sec"] for row in trace]
    frame_db = [row["frame_db"] for row in trace]
    noise_db = [row["noise_db"] for row in trace]
    open_thresh = [row["open_thresh_db"] for row in trace]
    close_thresh = [row["close_thresh_db"] for row in trace]
    gate_state = [row["gate_state"] for row in trace]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"{recipe['scenario_id']}  |  "
        f"{'PASS' if score.passed else 'FAIL: ' + '; '.join(score.fatal_reasons)}",
        fontsize=9,
        color="green" if score.passed else "red",
    )

    # --- Top: waveform + truth/predicted overlays ---
    ax1.plot(t, audio, color="#aaaaaa", linewidth=0.4, label="mixed audio")

    for ts, te in recipe["truth_segments"]:
        ax1.axvspan(ts, te, alpha=0.25, color="steelblue", label="truth speech")

    for ps, pe in score.predicted_segments:
        ax1.axvspan(ps, pe, alpha=0.2, color="orange", label="gate open")

    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.05, 1.05)

    # Deduplicate legend
    handles, labels = ax1.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax1.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right")

    # --- Bottom: dB activity + thresholds + gate state ---
    ax2.plot(times, frame_db, color="#444444", linewidth=0.6, label="frame dB")
    ax2.plot(times, noise_db, color="purple", linewidth=1.0, linestyle="--", label="noise floor")
    ax2.plot(times, open_thresh, color="red", linewidth=0.8, linestyle=":", label="open thresh")
    ax2.plot(times, close_thresh, color="darkorange", linewidth=0.8, linestyle=":", label="close thresh")

    # Gate state shading
    gate_arr = np.array(gate_state, dtype=float)
    ax2.fill_between(times, -80, -10, where=gate_arr > 0.5,
                     alpha=0.12, color="orange", label="gate open")

    ax2.set_ylabel("dBFS")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, duration)
    ax2.set_ylim(-80, 5)
    ax2.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Noise gate test harness")
    parser.add_argument("--data-dir", default="data", metavar="DIR")
    parser.add_argument("--out-dir", default="results", metavar="DIR")
    parser.add_argument("--count", type=int, default=10, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-failures", action="store_true",
                        help="Save mixed WAV for each failing scenario")
    parser.add_argument("--save-scenario", metavar="SCENARIO_ID",
                        help="Save mixed WAV for a specific scenario ID")
    # Gate tuning
    parser.add_argument("--open-margin-db", type=float, default=10.0)
    parser.add_argument("--close-margin-db", type=float, default=4.0)
    parser.add_argument("--release-ms", type=float, default=300.0)
    parser.add_argument("--hold-ms", type=float, default=150.0)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index_path = data_dir / "index.json"
    if not index_path.exists():
        sys.exit(f"Index not found: {index_path}\nRun acquire.py and label.py first.")
    with open(index_path) as fh:
        index = json.load(fh)

    # Gate config
    gate_cfg = GateConfig(
        open_margin_db=args.open_margin_db,
        close_margin_db=args.close_margin_db,
        release_ms=args.release_ms,
        hold_ms=args.hold_ms,
    )
    thresholds = ScoreThresholds()

    # Generate recipes
    print(f"Generating {args.count} scenario recipes (seed={args.seed}) ...")
    recipes = build_recipes(index, args.count, seed=args.seed)

    # Run
    results = []
    pass_count = 0

    print(f"{'Scenario':<42} {'Pass':>5}  {'MissedSp':>9}  {'FalseOpen':>9}  "
          f"{'StartDelay':>11}  {'EarlyClose':>11}")
    print("-" * 95)

    for recipe in recipes:
        audio = mix_audio(recipe)
        predicted, trace = run_gate(audio, recipe["sample_rate"], config=gate_cfg)

        sc = score_scenario(
            recipe["scenario_id"],
            recipe["truth_segments"],
            predicted,
            recipe["duration_sec"],
            thresholds,
        )

        if sc.passed:
            pass_count += 1

        d = score_to_dict(sc)
        d["recipe"] = recipe
        d["trace"] = trace
        results.append(d)

        status = "PASS" if sc.passed else "FAIL"
        print(
            f"{recipe['scenario_id']:<42} {status:>5}  "
            f"{sc.missed_speech_sec:>9.3f}  {sc.false_open_sec:>9.3f}  "
            f"{sc.max_start_delay_sec*1000:>10.0f}ms  "
            f"{sc.max_early_close_sec*1000:>10.0f}ms"
        )

        # Save WAV if requested
        if args.save_failures and not sc.passed:
            wav_path = out_dir / f"{recipe['scenario_id']}.wav"
            sf.write(str(wav_path), audio, recipe["sample_rate"])

        if args.save_scenario and recipe["scenario_id"] == args.save_scenario:
            wav_path = out_dir / f"{recipe['scenario_id']}.wav"
            sf.write(str(wav_path), audio, recipe["sample_rate"])
            print(f"  -> saved: {wav_path}")

    print("-" * 95)
    pass_rate = pass_count / len(results) * 100 if results else 0
    print(f"\nResults: {pass_count}/{len(results)} passed ({pass_rate:.0f}%)\n")

    # --- Write summary JSON ---
    summary = {
        "seed": args.seed,
        "count": args.count,
        "passed": pass_count,
        "failed": len(results) - pass_count,
        "pass_rate": round(pass_rate / 100, 4),
        "gate_config": {
            "open_margin_db": gate_cfg.open_margin_db,
            "close_margin_db": gate_cfg.close_margin_db,
            "release_ms": gate_cfg.release_ms,
            "hold_ms": gate_cfg.hold_ms,
        },
        "scenarios": [
            {k: v for k, v in d.items() if k not in ("trace", "recipe")}
            for d in results
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- Write summary CSV ---
    csv_path = out_dir / "summary.csv"
    csv_fields = [
        "scenario_id", "passed", "missed_speech_sec", "false_open_sec",
        "max_start_delay_ms", "max_early_close_ms", "max_end_overhang_ms",
        "fragmentation_count", "fatal_reasons",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for d in results:
            row = {k: d.get(k, "") for k in csv_fields}
            row["fatal_reasons"] = "; ".join(d.get("fatal_reasons", []))
            writer.writerow(row)

    # --- Failure plots (worst 5 failures) ---
    failures = [d for d in results if not d["passed"]]
    failures_sorted = sorted(
        failures,
        key=lambda d: (d["missed_speech_sec"], d["false_open_sec"]),
        reverse=True,
    )
    plot_count = min(5, len(failures_sorted))
    if plot_count:
        print(f"Generating failure plots for {plot_count} worst scenario(s) ...")
        for d in failures_sorted[:plot_count]:
            from score import ScenarioScore
            sc = ScenarioScore(
                scenario_id=d["scenario_id"],
                truth_segments=d["truth_segments"],
                predicted_segments=d["predicted_segments"],
                duration_sec=d["recipe"]["duration_sec"],
                missed_speech_sec=d["missed_speech_sec"],
                false_open_sec=d["false_open_sec"],
                max_start_delay_sec=d["max_start_delay_ms"] / 1000,
                max_early_close_sec=d["max_early_close_ms"] / 1000,
                max_end_overhang_sec=d["max_end_overhang_ms"] / 1000,
                fragmentation_count=d["fragmentation_count"],
                fatal_reasons=d["fatal_reasons"],
                passed=False,
            )
            audio = mix_audio(d["recipe"])
            plot_path = out_dir / f"failure_{d['scenario_id']}.png"
            _plot_scenario(d["recipe"], audio, d["trace"], sc, plot_path)
            print(f"  -> {plot_path}")
    else:
        print("No failures — all scenarios passed.")

    print(f"\nSummary JSON : {summary_path}")
    print(f"Summary CSV  : {csv_path}")


if __name__ == "__main__":
    main()
