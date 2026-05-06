# First Milestone

The first milestone should prove the full loop without over-engineering.

## Inputs

Use a tiny local or downloaded sample set:

- a small set of clean speech WAV/FLAC files
- a small set of background noise WAV files
- simple clean speech labels

Labels can initially be approximate or derived from single-utterance files.

## Build

Create a proof-of-concept harness that can:

1. discover/index source audio
2. select a deterministic subset
3. generate scenario recipes
4. generate mixed audio on the fly
5. run a simple adaptive RMS or dB gate
6. score against truth
7. produce summary output
8. produce failure plots

## Initial gate

Use a simple adaptive frame-based gate.

Suggested default:

```text
sample rate: 16000 Hz
frame size: 20 ms
high-pass optional later
frame RMS or dB activity
EMA noise floor
strict open threshold
relaxed close/sustain threshold
release hold
pre-roll metadata
```

## Initial scenario types

Start with:

- clean speech only
- speech + constant noise
- speech + rising noise
- quiet speech + noise
- short command + noise
- speech with pause + noise

## Smoke test

A tiny smoke test should run very quickly:

```text
3 clean speech clips
3 noise clips
10 generated scenarios
1 gate method
summary printed
worst failure plot generated if any fail
```

## Success criteria

The first milestone is successful if:

- the same seed produces the same scenario recipes
- generated audio can be reconstructed without saving WAVs
- the gate produces predicted segments
- metrics are calculated
- failure plots can be inspected
- the design has not committed prematurely to a rigid final architecture
