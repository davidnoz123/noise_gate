# Claude Prompt — Initial Proof of Concept

Use this as the first prompt to Claude.

```text
We are building a Python test harness for developing a noise gate for a real-time speech-to-text app.

Important: do not decide or overcommit to a final project folder layout. Keep the code simple and movable. Focus on clear behavioural boundaries and a working proof of concept.

The harness should eventually:
- discover/download clean speech and background noise samples
- cache listings and source metadata
- select samples deterministically using stable filename/source-ID hashing
- generate synthetic noisy speech scenarios from recipes
- generate mixed audio on the fly in memory
- run different noise gate implementations
- score predicted gate-open regions against ground-truth speech regions
- produce summary reports and failure plots

For this first proof of concept, build the smallest useful loop:
1. Accept a local clean speech folder and a local background noise folder.
2. Index audio files with duration, sample rate, channel count, and hash.
3. Select a deterministic subset using SHA256(namespace + source_id).
4. Generate scenario JSON recipes using explicit seeds.
5. Do not store generated mixed WAV files by default.
6. Reconstruct mixed audio in memory from the recipe when running a scenario.
7. Implement one simple frame-based adaptive RMS or dB noise gate.
8. Score predicted open segments against truth segments.
9. Produce a summary CSV/JSON.
10. Produce diagnostic PNG plots for worst failures.

Use 16 kHz mono internally.
Use 20 ms frames initially.
Use deterministic random seeds.
Keep generated scenario recipes as the durable artifact.
Generated WAV export should only be an explicit debug option.

Do not use advanced ML.
Do not build Android integration.
Do not build a GUI.
Do not implement the final folder layout yet.
Avoid unnecessary dependencies.

The gate should expose enough debug trace data for plotting:
- frame time
- activity/RMS/dB
- noise floor
- open threshold
- close threshold
- gate state
- reason code if practical

The scoring should emphasize product impact:
- missed speech
- late open
- early close
- false-open time
- end overhang
- fragmentation

Opening early should be penalized less than opening late.
Closing late should be penalized less than closing early.

After building the proof of concept, create a tiny smoke test with a handful of local files and a few generated scenarios.
```
