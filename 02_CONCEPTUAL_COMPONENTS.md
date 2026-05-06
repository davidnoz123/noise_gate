# Conceptual Components

The implementation should keep these concepts separate, even if the first proof-of-concept uses simple modules or scripts.

## Discovery

Discovers available source audio from remote repositories or local folders.

Responsibilities:

- fetch/cache remote listings
- record source URLs
- record licence information
- record HTTP metadata when available
- produce a stable list of candidate source items

## Selection

Chooses a deterministic subset of candidate source items.

Responsibilities:

- deterministic selection based on stable source IDs
- filename/source-ID hashing
- optional category balancing later
- record selected IDs in run/scenario manifests

## Download

Downloads selected source audio or dataset archives.

Responsibilities:

- resumable downloads
- avoid re-downloading existing files
- verify file size/hash when available
- cache original source audio
- keep licensing/source metadata

## Indexing

Reads local source audio metadata.

Responsibilities:

- duration
- sample rate
- channel count
- file hash
- maybe RMS/loudness later
- optional speech/noise category metadata

## Clean speech labelling

Provides ground-truth speech segments for clean speech.

Responsibilities:

- allow manual labels
- allow automatic labels later
- make labels editable without regenerating everything
- use tolerance zones during scoring

## Scenario generation

Creates deterministic recipes for synthetic noisy mixtures.

Responsibilities:

- choose speech source
- choose noise source
- define offsets
- define gain envelopes
- define sample rate
- define duration
- derive ground-truth speech regions in mixed-audio time
- store recipe, not generated WAV by default

## Gate execution

Runs a gate implementation on generated audio.

Responsibilities:

- produce predicted open/closed segments
- optionally produce frame-level debug traces
- not depend on how the scenario was generated

## Scoring

Compares predicted gate-open segments against truth.

Responsibilities:

- missed speech
- false-open time
- start delay
- early close
- end overhang
- fragmentation
- pass/fail classification
- fatal reason

## Reporting

Makes results actionable.

Responsibilities:

- summary CSV/JSON
- ranked failure list
- diagnostic plots
- optional debug WAVs only when explicitly requested
