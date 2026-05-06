# Noise Gate Test Harness — Claude Handoff Pack

This pack describes a Python project to build a reproducible test harness for developing and comparing noise-gating methods for a speech-to-text Android/Vosk-style app.

The project is not primarily about deciding the final repository layout. The first goal is to create a small, reliable engineering loop:

1. discover/download reusable clean speech and background noise sources
2. select samples reproducibly
3. generate noisy speech scenarios from recipes
4. run one or more noise gate implementations
5. score the gate output against known speech regions
6. produce failure reports and plots

Generated mixed WAV files should **not** be stored by default. The scenario recipe is the durable artifact. Audio should normally be generated on the fly in memory.

## Project priorities

The harness should help answer:

- Does a gate miss speech?
- Does it clip the beginning of an utterance?
- Does it close mid-utterance?
- Does it false-open on background noise?
- Does it stay open too long?
- Does it still work when speech/noise volumes change dynamically?
- Which gate method works best across many synthetic scenarios?

## Important design stance

Do not prematurely lock in a final file/folder layout. Keep boundaries clear and implementation movable.

Focus first on capabilities, data contracts, reproducibility, and testability.
