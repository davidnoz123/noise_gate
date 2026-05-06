# Reporting and Failure Analysis

The harness should make failures easy to understand.

## Standard run outputs

Each run should produce:

- summary JSON
- summary CSV
- ranked failure list
- per-method aggregate metrics
- worst failure plots

## Failure plot contents

For each bad scenario, generate a plot showing:

- mixed waveform or envelope
- clean speech envelope if available
- background noise envelope if available
- frame activity
- noise floor estimate
- open threshold
- close threshold
- truth speech regions
- predicted gate-open regions
- failure annotation

The plot should make it obvious whether the problem was:

- threshold too high
- threshold too low
- noise floor adapted during speech
- background burst caused false trigger
- quiet first syllable was clipped
- release time was too short
- speech-like background fooled the gate

## Optional audio export

Do not save generated audio by default.

Allow explicit export for:

- worst failures
- one scenario ID
- manual debugging

Useful optional exports:

- mixed audio snippet
- clean speech snippet
- noise snippet
- gated audio or recognizer input chunk

## Run manifest

Each run should record:

- harness version/code version if available
- gate method
- gate configuration
- source listing versions
- selected source IDs
- scenario seed
- scenario IDs
- timestamp
- environment summary

This makes old results auditable and reproducible.
