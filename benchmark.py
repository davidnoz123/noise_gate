#!/usr/bin/env python3
"""
benchmark.py — Compare adaptive dB gate against Silero VAD on the same mixed scenarios.

For each scenario the mixed audio is written to a temporary WAV file and passed
to Silero VAD.  Both the gate and Silero VAD are then scored with the same
score.py logic so the results are directly comparable.

Silero VAD acts as an upper-bound reference: it shows the best achievable pass
rate given the scenario difficulty, not the gate algorithm's limitations.

Usage
-----
  python benchmark.py                     # 10 scenarios, best-sweep gate config
  python benchmark.py --count 20 --seed 99
  python benchmark.py --open-margin-db 6 --close-margin-db 4 --release-ms 800
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from gate import GateConfig, run_gate
from scenario import build_recipes, mix_audio
from score import ScoreThresholds, score_scenario

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


SAMPLE_RATE = 16000
FRAME_SIZE = 320  # 20 ms at 16 kHz


# ---------------------------------------------------------------------------
# Silero VAD helper
# ---------------------------------------------------------------------------

def run_silero_on_audio(audio: np.ndarray, model) -> list[list[float]]:
    """
    Write *audio* to a temporary WAV file, run Silero VAD, return
    [[start_sec, end_sec], ...].  The temp file is deleted afterwards.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)  # release OS file descriptor before soundfile opens it
    try:
        sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
        wav = read_audio(tmp_path, sampling_rate=SAMPLE_RATE)
        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
        )
        return [[round(t["start"], 4), round(t["end"], 4)] for t in timestamps]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Per-scenario result
# ---------------------------------------------------------------------------

def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark: adaptive dB gate vs Silero VAD on identical mixed scenarios"
    )
    parser.add_argument("--data-dir", default="data", metavar="DIR")
    parser.add_argument("--count", type=int, default=10, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=42)
    # Gate parameters — defaults taken from best sweep result
    parser.add_argument("--open-margin-db", type=float, default=6.0)
    parser.add_argument("--close-margin-db", type=float, default=4.0)
    parser.add_argument("--release-ms", type=float, default=800.0)
    parser.add_argument("--hold-ms", type=float, default=100.0)
    parser.add_argument("--noise-ema-tc-closed", type=float, default=1.0,
                        help="EMA time constant (s) while gate is closed")
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Load index
    # ------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    index_path = data_dir / "index.json"
    if not index_path.exists():
        sys.exit(f"Index not found: {index_path}\nRun acquire.py and label.py first.")
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)

    # ------------------------------------------------------------------
    # Gate config
    # ------------------------------------------------------------------
    gate_cfg = GateConfig(
        open_margin_db=args.open_margin_db,
        close_margin_db=args.close_margin_db,
        release_ms=args.release_ms,
        hold_ms=args.hold_ms,
        noise_ema_tc_closed=args.noise_ema_tc_closed,
    )
    thresholds = ScoreThresholds()

    # ------------------------------------------------------------------
    # Generate recipes
    # ------------------------------------------------------------------
    print(f"Generating {args.count} scenario recipes (seed={args.seed}) ...")
    recipes = build_recipes(index, args.count, seed=args.seed)

    # ------------------------------------------------------------------
    # Load Silero VAD model once
    # ------------------------------------------------------------------
    print("Loading Silero VAD model ...")
    vad_model = load_silero_vad()

    # ------------------------------------------------------------------
    # Run both predictors on every scenario
    # ------------------------------------------------------------------
    COL = 44
    print()
    print(
        f"{'Scenario':<{COL}} {'Gate':>6}  {'Silero':>6}  "
        f"{'Miss(G)':>8}  {'Miss(V)':>8}  {'FO(G)':>7}  {'FO(V)':>7}"
    )
    print("-" * (COL + 52))

    gate_passes = 0
    vad_passes = 0
    rows = []

    for recipe in recipes:
        audio = mix_audio(recipe)
        sid = recipe["scenario_id"]
        truth = recipe["truth_segments"]
        dur = recipe["duration_sec"]

        # Gate (in-memory)
        gate_preds, _ = run_gate(audio, SAMPLE_RATE, frame_size=FRAME_SIZE, config=gate_cfg)
        gate_sc = score_scenario(sid, truth, gate_preds, dur, thresholds)

        # Silero VAD (via temp WAV)
        vad_preds = run_silero_on_audio(audio, vad_model)
        vad_sc = score_scenario(sid, truth, vad_preds, dur, thresholds)

        if gate_sc.passed:
            gate_passes += 1
        if vad_sc.passed:
            vad_passes += 1

        print(
            f"{sid:<{COL}} {_status(gate_sc.passed):>6}  {_status(vad_sc.passed):>6}  "
            f"{gate_sc.missed_speech_sec:>8.3f}  {vad_sc.missed_speech_sec:>8.3f}  "
            f"{gate_sc.false_open_sec:>7.3f}  {vad_sc.false_open_sec:>7.3f}"
        )

        rows.append({
            "scenario_id": sid,
            "gate_pass": gate_sc.passed,
            "vad_pass": vad_sc.passed,
            "gate_fatal": gate_sc.fatal_reasons,
            "vad_fatal": vad_sc.fatal_reasons,
            "gate_missed_sec": round(gate_sc.missed_speech_sec, 4),
            "vad_missed_sec": round(vad_sc.missed_speech_sec, 4),
            "gate_false_open_sec": round(gate_sc.false_open_sec, 4),
            "vad_false_open_sec": round(vad_sc.false_open_sec, 4),
        })

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(recipes)
    gate_pct = gate_passes / total * 100 if total else 0.0
    vad_pct = vad_passes / total * 100 if total else 0.0

    print("-" * (COL + 52))
    print(f"{'PASS RATE':<{COL}} {gate_pct:>5.1f}%  {vad_pct:>5.1f}%")
    print()

    # Cases where gate fails but Silero passes — these are gate algorithm failures
    gate_only_failures = [r for r in rows if not r["gate_pass"] and r["vad_pass"]]
    # Cases where both fail — scenario is inherently hard
    both_fail = [r for r in rows if not r["gate_pass"] and not r["vad_pass"]]
    # Cases where gate passes but Silero fails (gate is more lenient)
    vad_only_failures = [r for r in rows if r["gate_pass"] and not r["vad_pass"]]

    print(f"Gate config  : open={args.open_margin_db}dB  close={args.close_margin_db}dB  "
          f"rel={args.release_ms:.0f}ms  hold={args.hold_ms:.0f}ms  "
          f"tc_closed={args.noise_ema_tc_closed}s")
    print(f"Gate pass    : {gate_passes}/{total}  ({gate_pct:.1f}%)")
    print(f"Silero pass  : {vad_passes}/{total}  ({vad_pct:.1f}%)")
    print()
    print(f"Gate fails, Silero passes  ({len(gate_only_failures)}) — fixable by better gate algorithm:")
    for r in gate_only_failures:
        print(f"  {r['scenario_id']}  fatal={r['gate_fatal']}")
    print()
    print(f"Both fail  ({len(both_fail)}) — inherently hard scenarios:")
    for r in both_fail:
        print(f"  {r['scenario_id']}  gate_fatal={r['gate_fatal']}  vad_fatal={r['vad_fatal']}")
    if vad_only_failures:
        print()
        print(f"Silero fails, gate passes  ({len(vad_only_failures)}) — gate is more lenient here:")
        for r in vad_only_failures:
            print(f"  {r['scenario_id']}  vad_fatal={r['vad_fatal']}")


if __name__ == "__main__":
    main()
