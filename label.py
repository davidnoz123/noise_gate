#!/usr/bin/env python3
"""
label.py — Run Silero VAD on indexed clean speech files and write speech_segments
back into data/index.json.

Only processes files with dataset == 'LibriSpeech-test-clean' (or any dataset
containing 'speech' in the name) that don't already have speech_segments labelled.

Usage
-----
  python label.py                  # label all unlabelled speech files in data/index.json
  python label.py --force          # re-label even if speech_segments already present
  python label.py --data-dir /tmp/ngdata
"""

import argparse
import json
import time
from pathlib import Path

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


def label_file(path: Path, model) -> list[list[float]]:
    """Return [[start_sec, end_sec], ...] for speech regions in *path*."""
    wav = read_audio(str(path), sampling_rate=16000)
    timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        return_seconds=True,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=100,
        speech_pad_ms=30,
    )
    return [[round(t["start"], 4), round(t["end"], 4)] for t in timestamps]


def is_speech_dataset(entry: dict) -> bool:
    dataset = entry.get("dataset", "").lower()
    return "speech" in dataset or "libri" in dataset


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Label clean speech files with Silero VAD and store segments in index.json"
    )
    parser.add_argument("--data-dir", default="data", metavar="DIR")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-label files that already have speech_segments",
    )
    args = parser.parse_args(argv)

    index_path = Path(args.data_dir) / "index.json"
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}\nRun acquire.py first.")

    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)

    candidates = [
        e for e in index["files"]
        if is_speech_dataset(e) and (args.force or e.get("speech_segments") is None)
    ]

    if not candidates:
        print("All speech files already labelled. Use --force to re-label.")
        return

    print(f"Loading Silero VAD model ...")
    model = load_silero_vad()
    print(f"Labelling {len(candidates)} file(s) ...\n")

    labelled = 0
    for entry in candidates:
        path = Path(entry["local_path"])
        if not path.exists():
            print(f"  [warn] file not found, skipping: {path}")
            continue
        try:
            segments = label_file(path, model)
            entry["speech_segments"] = segments
            entry["speech_segments_source"] = "silero-vad-6"
            entry["speech_segments_labelled_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            total = round(sum(e - s for s, e in segments), 3)
            print(f"  {path.name:<35}  {segments}  ({total} s speech)")
            labelled += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {path.name}: {exc}")

    index["labelled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    print(f"\nDone. Labelled {labelled} file(s). Index written to: {index_path}")


if __name__ == "__main__":
    main()
