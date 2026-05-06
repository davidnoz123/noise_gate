# Claude Prompt — Download Automation

Use this after the core proof-of-concept loop works.

```text
Extend the noise gate test harness with sample discovery and download automation.

Keep discovery, selection, download, indexing, scenario generation, and evaluation conceptually separate.

Do not store generated mixed WAV files by default. Scenario recipes remain the durable artifact.

Add support for remote source listings/manifests.

Discovery requirements:
- Fetch/cache remote listings where possible.
- Record ETag, Last-Modified, Content-Length, fetch time, and listing SHA256 when available.
- Do not treat Last-Modified as authoritative; use it only to avoid unnecessary refreshes.
- Store the cached listing content or parsed listing in a reproducible way.
- Record source URL and licence metadata.

Selection requirements:
- Select source items deterministically using SHA256(namespace + source_id).
- Avoid alphabetical or unseeded random selection.
- Store selected source IDs in run/scenario manifests.
- If a refreshed listing changes, old runs must still be reproducible from their recorded selected IDs and listing hash.

Download requirements:
- If the source provides individual file URLs, download only selected files.
- If a dataset is distributed only as a single archive, download the archive once and apply deterministic selection after indexing locally.
- Do not re-download existing files unless force refresh is requested.
- Verify hash/size when possible.

Indexing requirements:
- Record duration, sample rate, channel count, file hash, local path, source ID, and licence/source metadata.

Initial source support:
- Clean speech: LibriSpeech test-clean or a similarly simple clean speech source.
- Noise: ESC-50 or another small environmental noise source.

Keep the implementation conservative and auditable.
```
