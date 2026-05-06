# Open Design Decisions

These should not be prematurely locked in.

## Storage

- JSON files vs SQLite vs something else
- whether generated scenario recipes are individual files or batch manifests
- whether failure artifacts are stored per run or centrally

## Labelling

- manual clean speech labels
- automatic VAD-derived labels
- forced alignment
- hybrid label review workflow

## Source acquisition

- archive-first datasets
- individual-file repositories
- Freesound/HuggingFace APIs
- local customer/device recordings

## Audio representation

- frame RMS
- dB
- filtered RMS
- spectral band energy
- pre-emphasis/high-pass filtering

## Scenario generation

- how many scenarios per source pair
- how to weight easy vs hard scenarios
- how to represent dynamic gain envelopes
- whether to generate adversarial scenarios

## Gate API

- event-based output vs segment output
- whether gates own frame extraction
- how debug traces are represented
- whether the gate sees raw frame or feature frame

## Reporting

- static HTML
- markdown report
- CSV/JSON only
- plots only for failures or for all scenarios

## Recognition evaluation

- whether to include Vosk in the harness
- how to score recognition text
- whether gate evaluation and recognizer evaluation are separate stages
