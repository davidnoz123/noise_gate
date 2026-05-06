#!/usr/bin/env python3
"""
acquire.py — Download a deterministic small selection of clean speech and background
noise for the noise gate test harness.

Sources
-------
Speech : LibriSpeech test-clean  — single tar.gz archive (~346 MB, downloaded once)
Noise  : ESC-50                  — individual WAV files from GitHub (only selected files)

Output layout (default: data/)
------------------------------
data/
  archives/
    test-clean.tar.gz             LibriSpeech archive (kept as download cache)
  listings/
    esc50_listing.json            cached ESC-50 metadata (built from GitHub CSV)
    librispeech_index.json        index built after local extraction
  speech/
    LibriSpeech/test-clean/...    extracted FLAC files
  noise/
    <esc50_filename>.wav          downloaded ESC-50 files (selected subset only)
  index.json                      combined index of all selected files

Selection method
----------------
Items are ranked by SHA256(namespace + "/" + source_id) and the first N are taken.
This gives repeatable, bias-free selection that expands gracefully as N grows.

Usage
-----
  python acquire.py                        # defaults: 5 speech, 5 noise
  python acquire.py --speech 10 --noise 10
  python acquire.py --data-dir /tmp/ngdata --force
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional dependency: soundfile (for audio metadata)
# ---------------------------------------------------------------------------
try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

# ---------------------------------------------------------------------------
# Source URLs and licence metadata
# ---------------------------------------------------------------------------

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_LICENSE = "CC BY 4.0"
LIBRISPEECH_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

ESC50_LISTING_URL = (
    "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
)
ESC50_AUDIO_URL = (
    "https://github.com/karolpiczak/ESC-50/raw/master/audio/{filename}"
)
ESC50_LICENSE = "CC BY-NC 3.0"
ESC50_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/3.0/"

# ---------------------------------------------------------------------------
# Hash-based deterministic selection  (design doc 04)
# ---------------------------------------------------------------------------

def stable_key(namespace: str, source_id: str) -> str:
    """Return SHA256 hex digest of 'namespace/source_id'."""
    text = f"{namespace}/{source_id}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def stable_sample(items: list[dict], count: int, namespace: str) -> list[dict]:
    """Return up to *count* items ranked deterministically by stable_key."""
    ranked = sorted(items, key=lambda item: stable_key(namespace, item["source_id"]))
    return ranked[:count]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Listing cache helpers
# ---------------------------------------------------------------------------

def load_cached_listing(cache_path: Path) -> Optional[dict]:
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_listing(cache_path: Path, data: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

# ---------------------------------------------------------------------------
# HTTP / download utilities
# ---------------------------------------------------------------------------

def _opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "noise-gate-harness/0.1")]
    return opener


def fetch_text(url: str) -> tuple[str, dict]:
    """Fetch URL and return (text_content, http_meta_dict)."""
    print(f"  [fetch] {url}")
    with _opener().open(url, timeout=30) as resp:
        content = resp.read().decode("utf-8")
        meta = {
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
            "content_length": resp.headers.get("Content-Length"),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    return content, meta


def download_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download *url* to *dest*, skipping if already present (unless force=True)."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"  [skip]     {dest.name}  (already downloaded)")
        return dest

    print(f"  [download] {dest.name}")
    print(f"             {url}")
    try:
        with _opener().open(url, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            downloaded = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(1 << 16)  # 64 KB
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r             {downloaded / 1e6:.1f} MB"
                            f" / {total / 1e6:.1f} MB  ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
            print()
    except urllib.error.URLError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc

    return dest

# ---------------------------------------------------------------------------
# Safe tar extraction  (guards against path traversal)
# ---------------------------------------------------------------------------

def _safe_extract(tar: tarfile.TarFile, dest: Path, suffix: str = ".flac") -> int:
    """Extract only regular files with *suffix*, rejecting absolute/traversal paths."""
    count = 0
    for member in tar.getmembers():
        if not member.isfile():
            continue
        if not member.name.endswith(suffix):
            continue
        norm = os.path.normpath(member.name)
        if os.path.isabs(norm) or norm.startswith(".."):
            print(f"  [security] skipping suspicious member: {member.name!r}")
            continue
        out_path = dest / norm
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        count += 1
    return count

# ---------------------------------------------------------------------------
# Audio indexing  (design doc 02)
# ---------------------------------------------------------------------------

def index_audio_file(
    path: Path,
    source_id: str,
    dataset: str,
    license_name: str,
    license_url: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "source_id": source_id,
        "dataset": dataset,
        "local_path": str(path),
        "filename": path.name,
        "license": license_name,
        "license_url": license_url,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "duration_sec": None,
        "sample_rate": None,
        "channels": None,
        "format": None,
    }
    if _HAS_SOUNDFILE:
        try:
            with sf.SoundFile(path) as sf_file:
                entry["duration_sec"] = round(len(sf_file) / sf_file.samplerate, 4)
                entry["sample_rate"] = sf_file.samplerate
                entry["channels"] = sf_file.channels
                entry["format"] = sf_file.format
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not read audio metadata for {path.name}: {exc}")
    return entry

# ---------------------------------------------------------------------------
# ESC-50  (individual file repository)
# ---------------------------------------------------------------------------

def acquire_esc50(data_dir: Path, count: int, *, force: bool = False) -> list[dict]:
    """
    Fetch ESC-50 listing from GitHub, select *count* items deterministically,
    download only those audio files.
    """
    listing_cache = data_dir / "listings" / "esc50_listing.json"
    noise_dir = data_dir / "noise"

    cached = load_cached_listing(listing_cache)
    if cached and not force:
        print("ESC-50 listing: using cache")
        items = cached["items"]
        listing_sha256 = cached["listing_sha256"]
    else:
        print("ESC-50 listing: fetching from GitHub ...")
        content, http_meta = fetch_text(ESC50_LISTING_URL)
        listing_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()

        reader = csv.DictReader(io.StringIO(content))
        items = [
            {
                "source_id": row["filename"],
                "filename": row["filename"],
                "url": ESC50_AUDIO_URL.format(filename=row["filename"]),
                "category": row.get("category", ""),
                "esc10": row.get("esc10", ""),
            }
            for row in reader
        ]
        listing_data = {
            "source": "ESC-50",
            "listing_url": ESC50_LISTING_URL,
            "listing_sha256": listing_sha256,
            **http_meta,
            "item_count": len(items),
            "items": items,
        }
        save_listing(listing_cache, listing_data)
        print(f"  cached {len(items)} items  (sha256 prefix: {listing_sha256[:12]})")

    selected = stable_sample(items, count, namespace="esc50")
    print(f"ESC-50: selected {len(selected)} of {len(items)} items")

    index_entries = []
    for item in selected:
        dest = noise_dir / item["filename"]
        download_file(item["url"], dest, force=force)
        entry = index_audio_file(
            dest,
            item["source_id"],
            "ESC-50",
            ESC50_LICENSE,
            ESC50_LICENSE_URL,
        )
        entry["category"] = item.get("category", "")
        entry["listing_sha256"] = listing_sha256
        index_entries.append(entry)

    return index_entries

# ---------------------------------------------------------------------------
# LibriSpeech test-clean  (single archive dataset)
# ---------------------------------------------------------------------------

def acquire_librispeech(data_dir: Path, count: int, *, force: bool = False) -> list[dict]:
    """
    Download LibriSpeech test-clean archive (once), extract FLAC files, build a
    local index, then select *count* items deterministically.
    """
    archive_path = data_dir / "archives" / "test-clean.tar.gz"
    extract_dir = data_dir / "speech"
    listing_cache = data_dir / "listings" / "librispeech_index.json"
    extract_marker = data_dir / "listings" / ".librispeech_extracted"

    # 1. Download archive (never re-download unless missing)
    print(f"LibriSpeech archive: ~346 MB")
    download_file(LIBRISPEECH_URL, archive_path, force=False)

    # 2. Extract (skip if marker exists)
    if extract_marker.exists() and not force:
        print("LibriSpeech: already extracted")
    else:
        print("LibriSpeech: extracting FLAC files (this may take a moment) ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            count_extracted = _safe_extract(tar, extract_dir, suffix=".flac")
        extract_marker.parent.mkdir(parents=True, exist_ok=True)
        extract_marker.touch()
        print(f"  extracted {count_extracted} FLAC files")

    # 3. Build or load local file index
    cached = load_cached_listing(listing_cache)
    if cached and not force:
        print("LibriSpeech: using cached file index")
        items = cached["items"]
    else:
        print("LibriSpeech: indexing extracted files ...")
        flac_files = sorted(extract_dir.rglob("*.flac"))
        items = [
            {"source_id": str(p.relative_to(extract_dir)).replace("\\", "/")}
            for p in flac_files
        ]
        index_data = {
            "source": "LibriSpeech-test-clean",
            "archive_url": LIBRISPEECH_URL,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "item_count": len(items),
            "items": items,
        }
        save_listing(listing_cache, index_data)
        print(f"  indexed {len(items)} files")

    # 4. Hash-select and build index entries
    selected = stable_sample(items, count, namespace="librispeech-test-clean")
    print(f"LibriSpeech: selected {len(selected)} of {len(items)} files")

    index_entries = []
    for item in selected:
        path = extract_dir / item["source_id"]
        if not path.exists():
            candidates = list(extract_dir.rglob(Path(item["source_id"]).name))
            if candidates:
                path = candidates[0]
            else:
                print(f"  [warn] file not found: {item['source_id']}")
                continue
        entry = index_audio_file(
            path,
            item["source_id"],
            "LibriSpeech-test-clean",
            LIBRISPEECH_LICENSE,
            LIBRISPEECH_LICENSE_URL,
        )
        index_entries.append(entry)

    return index_entries

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download a deterministic small selection of speech and noise audio."
    )
    parser.add_argument(
        "--speech",
        type=int,
        default=5,
        metavar="N",
        help="Number of clean speech files to select (default: 5)",
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=5,
        metavar="N",
        help="Number of noise files to select (default: 5)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        metavar="DIR",
        help="Root directory for all downloaded files (default: data/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch listings and re-index even if cached data exists",
    )
    parser.add_argument(
        "--noise-only",
        action="store_true",
        help="Download only noise samples (skip the large LibriSpeech archive)",
    )
    args = parser.parse_args(argv)

    if not _HAS_SOUNDFILE:
        print(
            "[warn] soundfile not installed — audio metadata (duration, sample rate, "
            "channels) will not be recorded.\n"
            "       Install with: pip install soundfile\n"
        )

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict] = []

    # --- Noise: ESC-50 ---
    print("\n=== ESC-50 (noise) ===")
    noise_entries = acquire_esc50(data_dir, args.noise, force=args.force)
    all_entries.extend(noise_entries)
    print(f"ESC-50: done ({len(noise_entries)} files indexed)")

    # --- Speech: LibriSpeech ---
    if not args.noise_only:
        print("\n=== LibriSpeech test-clean (speech) ===")
        speech_entries = acquire_librispeech(data_dir, args.speech, force=args.force)
        all_entries.extend(speech_entries)
        print(f"LibriSpeech: done ({len(speech_entries)} files indexed)")
    else:
        print("\n[skip] LibriSpeech (--noise-only flag set)")

    # --- Write combined index ---
    index_path = data_dir / "index.json"
    index_doc = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "speech_count": args.speech,
        "noise_count": args.noise,
        "total_files": len(all_entries),
        "files": all_entries,
    }
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index_doc, fh, indent=2)

    print(f"\n=== Done ===")
    print(f"Index written to: {index_path}")
    print(f"Total files indexed: {len(all_entries)}")
    if not _HAS_SOUNDFILE:
        print("Tip: install soundfile to capture duration/sample-rate metadata.")


if __name__ == "__main__":
    main()
