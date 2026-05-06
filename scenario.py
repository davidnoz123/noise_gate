#!/usr/bin/env python3
"""
scenario.py — Deterministic scenario recipe generation and on-the-fly audio mixing.

A scenario recipe is a JSON-serialisable dict that fully describes how to
re-create a mixed audio clip.  Generated WAV files are never stored by default;
the recipe is the durable artifact.

Public API
----------
  build_recipes(index, count, seed, sample_rate) -> list[dict]
  mix_audio(recipe, index_by_id)                 -> np.ndarray (mono float32, 16 kHz)
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


# ---------------------------------------------------------------------------
# Gain envelope generation
# ---------------------------------------------------------------------------

def _make_envelope(n_samples: int, spec: dict, rng: np.random.Generator) -> np.ndarray:
    """Return a linear-amplitude envelope array of length n_samples."""
    etype = spec.get("type", "constant")

    if etype == "constant":
        return np.ones(n_samples, dtype=np.float32)

    if etype == "linear":
        start_db = spec.get("start_db", 0.0)
        end_db = spec.get("end_db", 0.0)
        start_amp = 10 ** (start_db / 20)
        end_amp = 10 ** (end_db / 20)
        return np.linspace(start_amp, end_amp, n_samples, dtype=np.float32)

    if etype == "slow_sine":
        depth_db = spec.get("depth_db", 6.0)
        period_sec = spec.get("period_sec", 4.0)
        sr = spec.get("_sr", 16000)
        t = np.arange(n_samples) / sr
        depth_amp = 10 ** (depth_db / 20)
        # oscillates between 1/depth and 1
        env = 1.0 / depth_amp + (1.0 - 1.0 / depth_amp) * (
            0.5 + 0.5 * np.sin(2 * math.pi * t / period_sec)
        )
        return env.astype(np.float32)

    if etype == "random_walk":
        # Smooth random walk in dB space.
        # smooth_sec controls how slowly the level drifts:
        #   0.05 s => jittery frame-to-frame variation
        #   0.5 s  => moderate drift (default)
        #   2.0 s  => very slow swell
        step_db = spec.get("step_db", 1.0)
        low_db = spec.get("low_db", -12.0)
        high_db = spec.get("high_db", 0.0)
        smooth_sec = spec.get("smooth_sec", 0.5)
        walk = np.zeros(n_samples)
        db = 0.0
        for i in range(n_samples):
            db += rng.normal(0, step_db)
            db = float(np.clip(db, low_db, high_db))
            walk[i] = db
        from scipy.ndimage import uniform_filter1d
        sr = spec.get("_sr", 16000)
        smooth_samples = max(1, int(sr * smooth_sec))
        walk = uniform_filter1d(walk, size=smooth_samples)
        return (10 ** (walk / 20)).astype(np.float32)

    if etype == "burst":
        # A sudden loud event: Gaussian-shaped amplitude spike at a random time.
        # burst_db    : peak amplitude of the burst relative to baseline (dB, positive)
        # burst_sec   : half-width of the Gaussian in seconds (controls sharpness)
        # burst_at_sec: centre of the burst; if None, drawn randomly
        sr = spec.get("_sr", 16000)
        burst_db = spec.get("burst_db", 12.0)
        burst_sec = spec.get("burst_sec", 0.15)   # 150 ms half-width by default
        if spec.get("burst_at_sec") is not None:
            centre = int(spec["burst_at_sec"] * sr)
        else:
            centre = int(rng.uniform(0.1, 0.9) * n_samples)
        sigma = max(1, int(burst_sec * sr))
        t = np.arange(n_samples)
        burst_amp = 10 ** (burst_db / 20)
        env = 1.0 + (burst_amp - 1.0) * np.exp(-0.5 * ((t - centre) / sigma) ** 2)
        return env.astype(np.float32)

    raise ValueError(f"Unknown envelope type: {etype!r}")


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------

def _load_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file as mono float32, return (samples, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return mono, sr


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _loop_to_length(audio: np.ndarray, n: int) -> np.ndarray:
    """Tile *audio* until it is at least *n* samples, then trim."""
    if len(audio) == 0:
        return np.zeros(n, dtype=np.float32)
    reps = math.ceil(n / len(audio))
    return np.tile(audio, reps)[:n]


# ---------------------------------------------------------------------------
# Scenario recipe building
# ---------------------------------------------------------------------------

# Each entry:
#   (type_name, speech_gain_db, speech_envelope_type,
#    noise_gain_db, noise_envelope_type)
#
# speech_envelope_type controls how the speaker's own level varies during the clip:
#   "constant"     — fixed level throughout
#   "linear"       — speaker walks closer/farther (monotonic ramp)
#   "random_walk"  — natural mic-distance variation (slow smooth walk)
#
# noise_envelope_type controls background level variation:
#   "constant"     — steady background
#   "linear"       — noise level rises or falls across the clip
#   "slow_sine"    — cyclic noise swell (e.g. traffic, AC hum)
#   "random_walk"  — stochastic drift (e.g. crowd, wind)
#   "burst"        — sudden impulse event (clatter, door slam)
_SCENARIO_TYPES = [
    # name                       sp_db  sp_env          ns_db   ns_env
    ("speech_only",              -6.0,  "constant",    -60.0,  "constant"),
    ("speech_const_noise",       -6.0,  "constant",    -24.0,  "constant"),
    ("speech_const_noise",       -9.0,  "constant",    -20.0,  "constant"),
    ("quiet_speech_noise",      -18.0,  "constant",    -20.0,  "constant"),
    ("speech_rising_noise",      -6.0,  "constant",    -30.0,  "linear"),
    ("speech_sine_noise",        -6.0,  "constant",    -22.0,  "slow_sine"),
    ("short_cmd_noise",          -6.0,  "constant",    -20.0,  "constant"),
    ("speech_rand_noise",        -8.0,  "constant",    -22.0,  "random_walk"),
    # Speaker level variation
    ("fading_speaker",           -6.0,  "linear",      -24.0,  "constant"),
    ("approaching_speaker",      -6.0,  "linear",      -24.0,  "constant"),
    ("natural_level_variation",  -8.0,  "random_walk", -22.0,  "constant"),
    ("fading_speaker_rand_noise",-6.0,  "linear",      -22.0,  "random_walk"),
    # Hard burst events
    ("speech_burst_noise",       -6.0,  "constant",    -26.0,  "burst"),
    ("quiet_speech_burst",      -16.0,  "constant",    -26.0,  "burst"),
    # Combined: fading speech into rising noise
    ("fade_speech_rise_noise",   -6.0,  "linear",      -30.0,  "linear"),
]


def _stable_scenario_seed(base_seed: int, index: int) -> int:
    h = hashlib.sha256(f"{base_seed}/{index}".encode()).hexdigest()
    return int(h[:8], 16)


def build_recipes(
    index: dict,
    count: int,
    seed: int = 42,
    sample_rate: int = 16000,
    duration_sec: float = 8.0,
) -> list[dict]:
    """
    Build *count* deterministic scenario recipes from *index*.

    Parameters
    ----------
    index       : loaded data/index.json dict
    count       : number of scenarios to generate
    seed        : top-level RNG seed (stored in each recipe)
    sample_rate : target sample rate for mixed audio
    duration_sec: duration of each generated clip
    """
    speech_files = [
        f for f in index["files"]
        if ("speech" in f.get("dataset", "").lower() or "libri" in f.get("dataset", "").lower())
        and f.get("speech_segments")
    ]
    noise_files = [
        f for f in index["files"]
        if "esc" in f.get("dataset", "").lower() or "noise" in f.get("local_path", "").lower()
    ]

    if not speech_files:
        raise ValueError("No labelled speech files in index. Run label.py first.")
    if not noise_files:
        raise ValueError("No noise files in index.")

    recipes = []
    for i in range(count):
        scenario_seed = _stable_scenario_seed(seed, i)
        rng = np.random.default_rng(scenario_seed)

        stype = _SCENARIO_TYPES[i % len(_SCENARIO_TYPES)]
        type_name, speech_db, speech_env_type, noise_db, noise_env_type = stype

        speech_entry = speech_files[i % len(speech_files)]
        noise_entry = noise_files[i % len(noise_files)]

        # Pick a random speech segment from the file
        segments = speech_entry["speech_segments"]
        seg = segments[int(rng.integers(len(segments)))]
        seg_dur = seg[1] - seg[0]

        # Insert speech somewhere inside the clip with margin
        max_insert = max(0.0, duration_sec - seg_dur - 0.3)
        insert_at = float(rng.uniform(0.3, max(0.31, max_insert)))

        # Noise offset: random start within the noise file
        noise_dur = noise_entry.get("duration_sec") or 5.0
        noise_offset = float(rng.uniform(0.0, max(0.0, noise_dur - 1.0)))

        # --- Speech envelope ---
        speech_envelope: dict = {"type": speech_env_type}
        if speech_env_type == "linear":
            # Randomise direction: fading away or approaching
            if type_name == "approaching_speaker" or rng.random() > 0.5:
                speech_envelope["start_db"] = float(rng.uniform(-12, -6))
                speech_envelope["end_db"] = float(rng.uniform(-3, 0))
            else:  # fading
                speech_envelope["start_db"] = float(rng.uniform(-3, 0))
                speech_envelope["end_db"] = float(rng.uniform(-14, -8))
        elif speech_env_type == "random_walk":
            speech_envelope["step_db"] = 0.4
            speech_envelope["low_db"] = -8.0
            speech_envelope["high_db"] = 3.0
            speech_envelope["smooth_sec"] = float(rng.uniform(0.3, 1.2))

        # --- Noise envelope ---
        noise_envelope: dict = {"type": noise_env_type}
        if noise_env_type == "linear":
            noise_envelope["start_db"] = float(rng.uniform(-6, 0))
            noise_envelope["end_db"] = float(rng.uniform(-3, 6))
        elif noise_env_type == "slow_sine":
            noise_envelope["depth_db"] = float(rng.uniform(4, 10))
            noise_envelope["period_sec"] = float(rng.uniform(3, 7))
        elif noise_env_type == "random_walk":
            noise_envelope["step_db"] = 0.5
            noise_envelope["low_db"] = -8.0
            noise_envelope["high_db"] = 4.0
            noise_envelope["smooth_sec"] = float(rng.uniform(0.2, 1.5))
        elif noise_env_type == "burst":
            noise_envelope["burst_db"] = float(rng.uniform(8, 18))
            noise_envelope["burst_sec"] = float(rng.uniform(0.05, 0.3))   # 50–300 ms half-width

        truth_start = round(insert_at + seg[0] - seg[0], 4)  # = insert_at + 0 offset within clip
        # truth relative to mixed-audio timeline: insert_at marks where seg[0] lands
        truth_start = round(insert_at, 4)
        truth_end = round(insert_at + seg_dur, 4)

        scenario_id = f"{type_name}_{seed:06d}_{i:04d}"

        recipe = {
            "scenario_id": scenario_id,
            "type": type_name,
            "seed": scenario_seed,
            "base_seed": seed,
            "scenario_index": i,
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "speech": {
                "source_id": speech_entry["source_id"],
                "local_path": speech_entry["local_path"],
                "source_segment": seg,       # [start, end] in source file
                "insert_at_sec": insert_at,
                "gain_db": speech_db,
                "gain_envelope": speech_envelope,
            },
            "noise": {
                "source_id": noise_entry["source_id"],
                "local_path": noise_entry["local_path"],
                "offset_sec": noise_offset,
                "gain_db": noise_db,
                "gain_envelope": noise_envelope,
            },
            "truth_segments": [[truth_start, truth_end]],
        }
        recipes.append(recipe)

    return recipes


# ---------------------------------------------------------------------------
# On-the-fly audio generation
# ---------------------------------------------------------------------------

def mix_audio(recipe: dict) -> np.ndarray:
    """
    Reconstruct the mixed audio for *recipe* in memory.
    Returns a mono float32 numpy array at recipe['sample_rate'].
    """
    sr = recipe["sample_rate"]
    n_total = int(recipe["duration_sec"] * sr)
    rng = np.random.default_rng(recipe["seed"])
    mixed = np.zeros(n_total, dtype=np.float32)

    # --- speech ---
    sp = recipe["speech"]
    speech_raw, speech_sr = _load_mono_float32(Path(sp["local_path"]))
    speech_raw = _resample(speech_raw, speech_sr, sr)

    seg_start_sam = int(sp["source_segment"][0] * sr)
    seg_end_sam = int(sp["source_segment"][1] * sr)
    seg_end_sam = min(seg_end_sam, len(speech_raw))
    speech_clip = speech_raw[seg_start_sam:seg_end_sam]

    insert_sam = int(sp["insert_at_sec"] * sr)
    end_sam = min(insert_sam + len(speech_clip), n_total)
    clip_len = end_sam - insert_sam

    gain_amp = 10 ** (sp["gain_db"] / 20)
    env_spec = dict(sp["gain_envelope"])
    env_spec["_sr"] = sr
    envelope = _make_envelope(clip_len, env_spec, rng) * gain_amp
    mixed[insert_sam:end_sam] += speech_clip[:clip_len] * envelope

    # --- noise ---
    ns = recipe["noise"]
    if ns["gain_db"] > -50:  # -60 dB = effectively silent (speech_only type)
        noise_raw, noise_sr = _load_mono_float32(Path(ns["local_path"]))
        noise_raw = _resample(noise_raw, noise_sr, sr)

        offset_sam = int(ns["offset_sec"] * sr)
        noise_from = noise_raw[offset_sam:]
        noise_looped = _loop_to_length(noise_from, n_total)

        gain_amp_n = 10 ** (ns["gain_db"] / 20)
        env_spec_n = dict(ns["gain_envelope"])
        env_spec_n["_sr"] = sr
        envelope_n = _make_envelope(n_total, env_spec_n, rng) * gain_amp_n
        mixed += noise_looped * envelope_n

    # clip-protect
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed / peak * 0.99

    return mixed


# ---------------------------------------------------------------------------
# CLI: generate and optionally save recipes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Generate scenario recipes from data/index.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-audio", metavar="SCENARIO_ID",
                        help="Save mixed WAV for this scenario ID (debug)")
    args = parser.parse_args()

    index_path = Path(args.data_dir) / "index.json"
    with open(index_path) as fh:
        index = json.load(fh)

    recipes = build_recipes(index, args.count, seed=args.seed)
    print(f"Generated {len(recipes)} scenario recipes (seed={args.seed})\n")
    for r in recipes:
        print(f"  {r['scenario_id']:<40}  truth={r['truth_segments']}")

    if args.save_audio:
        match = next((r for r in recipes if r["scenario_id"] == args.save_audio), None)
        if match is None:
            print(f"Scenario {args.save_audio!r} not found.", file=sys.stderr)
            sys.exit(1)
        audio = mix_audio(match)
        out = Path(args.data_dir) / f"{args.save_audio}.wav"
        sf.write(str(out), audio, match["sample_rate"])
        print(f"\nSaved: {out}")
