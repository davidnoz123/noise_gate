# Listing Cache and Hash-Based Sample Selection

## Core idea

Remote listings should be cached separately from downloaded audio.

Sample selection should be deterministic and based on stable identifiers such as filenames, URLs, repository item IDs, or dataset-relative paths.

## Why hash-based selection?

Hash-based selection gives:

- repeatable sample selection
- no accidental alphabetical bias
- stable expansion from 100 to 1000 samples
- simple reproducibility
- no central database needed initially

Example selection rule:

```python
score = sha256(namespace + "/" + source_id).hexdigest()
```

Then sort by score and take the first N.

## Basic selection pseudocode

```python
import hashlib

def stable_key(namespace: str, source_id: str) -> str:
    text = f"{namespace}/{source_id}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()

def stable_sample(items, count: int, namespace: str):
    return sorted(items, key=lambda item: stable_key(namespace, item.source_id))[:count]
```

## Stratified selection later

Later, add category-aware selection:

- choose N cafe samples
- choose N fan samples
- choose N traffic samples
- choose N speech-like samples

Within each category, still use hash-based selection.

## Cached listing metadata

A cached listing should contain enough information to avoid unnecessary refreshes and reproduce old runs.

Useful fields:

```json
{
  "source": "example_repository",
  "listing_url": "https://example.com/listing",
  "fetched_at": "2026-05-07T00:00:00Z",
  "etag": "...",
  "last_modified": "...",
  "content_length": 12345,
  "listing_sha256": "...",
  "items": [
    {
      "source_id": "relative/path/or/remote/id.wav",
      "filename": "example.wav",
      "url": "https://example.com/example.wav",
      "duration_sec": 5.0,
      "license": "..."
    }
  ]
}
```

## ETag and Last-Modified

Use HTTP cache metadata where available:

- ETag
- Last-Modified
- Content-Length

These are useful for avoiding unnecessary refetches, but they are not the final source of truth.

Use layers:

1. ETag / Last-Modified to avoid unnecessary refresh
2. listing hash to detect actual listing changes
3. file hash to verify downloaded content
4. scenario seed + selected source IDs to reproduce tests

## Important reproducibility issue

If a remote listing changes, the first N items by hash can change because new files may enter ahead of old files.

Therefore each run should record:

- listing hash or listing version ID
- selected speech source IDs
- selected noise source IDs
- scenario seed
- gate configuration
- code version when available

Old runs must be reproducible even if the remote source later changes.
