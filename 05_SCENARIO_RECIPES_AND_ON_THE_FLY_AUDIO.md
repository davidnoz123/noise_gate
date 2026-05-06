# Scenario Recipes and On-the-Fly Audio Generation

## Do not store generated WAVs by default

Generated mixed WAVs are low-value and can consume a lot of space.

The durable artifact is the scenario recipe.

Generated audio should normally be reconstructed in memory at evaluation time.

## When to save generated WAVs

Saving generated audio should be optional and explicit:

- save worst failures
- save one scenario by ID
- debug mode
- manual listening/export

Possible flags:

```text
--save-failures
--save-scenario <scenario_id>
--debug-audio
```

## Scenario recipe fields

A scenario recipe should contain enough to regenerate the same audio.

Example:

```json
{
  "scenario_id": "quiet_speaker_cafe_00017",
  "seed": 123456,
  "sample_rate": 16000,
  "duration_sec": 8.0,

  "speech": {
    "source_id": "librispeech/test-clean/...",
    "source_file": "...",
    "source_segments": [[0.0, 2.4]],
    "insert_at_sec": 1.25,
    "gain_db": -10.0,
    "gain_envelope": {
      "type": "linear",
      "start_db": -12.0,
      "end_db": -6.0
    }
  },

  "noise": {
    "source_id": "esc50/audio/...",
    "source_file": "...",
    "offset_sec": 0.7,
    "gain_db": -24.0,
    "gain_envelope": {
      "type": "slow_sine",
      "depth_db": 8.0,
      "period_sec": 5.0
    }
  },

  "truth_segments": [
    [1.25, 3.65]
  ]
}
```

## Generation steps

For each scenario:

1. Load clean speech.
2. Convert to mono.
3. Resample to target sample rate.
4. Apply speech gain/envelope.
5. Load background noise.
6. Convert to mono.
7. Resample to target sample rate.
8. Loop/crop noise to scenario duration.
9. Apply noise gain/envelope.
10. Mix speech and noise.
11. Clip-protect or scale down if needed.
12. Keep generated audio in memory unless debug output is requested.

## Gain envelopes to support

Start simple:

- constant gain
- linear ramp
- slow sine
- random smooth walk
- burst events

Useful scenario types:

- constant SNR
- rising background noise
- falling background noise
- voice fading away
- voice getting louder
- sudden clatter
- speech-like background
- long pause mid-utterance
- short command
- quiet speaker

## Ground truth

Truth segments should be derived from clean speech labels plus insertion offset.

If clean speech contains:

```json
[[0.2, 1.6]]
```

and is inserted at 3.0 sec, truth becomes:

```json
[[3.2, 4.6]]
```

Scoring should allow tolerance zones because exact acoustic speech boundaries are fuzzy.
