# Terms and Definitions

## Source audio

Original clean speech or background noise files downloaded or supplied locally.

## Listing

A cached representation of available remote source files or dataset items.

## Selection

The deterministic subset of source items chosen from a listing or local index.

## Scenario recipe

A JSON-like specification that defines how to construct one synthetic noisy audio test case.

It includes:

- source IDs
- offsets
- gains
- gain envelopes
- seed
- sample rate
- duration
- truth segments

## Generated audio

The actual mixed waveform produced from a scenario recipe.

This should normally be generated on the fly in memory and not stored.

## Truth segments

The expected speech-active time intervals in the generated scenario.

## Predicted segments

The gate-open intervals emitted by a noise gate.

## False open

A predicted gate-open interval when no truth speech is present.

## Missed speech

Truth speech time not covered by predicted gate-open intervals.

## Late open

The gate opens after speech has already started.

## Early close

The gate closes before speech has ended.

## End overhang

The gate remains open after speech has ended.

## Fragmentation

One truth utterance is split into multiple predicted gate-open segments.

## Pre-roll

A short buffer of audio retained before gate open, used to avoid clipping the first syllable.

## Endpointing

Logic for deciding that an utterance has ended.
