# Gate Methods to Try

The harness should compare multiple gate families. Do not design the harness around only one algorithm.

Each gate should conceptually accept audio frames and produce gate-open/gate-closed events or predicted segments.

## 1. Fixed RMS threshold gate

Baseline only.

Rule:

```text
open if frame_rms >= threshold_open
close if frame_rms < threshold_close for N frames
```

Expected weakness: fails when mic distance or background noise changes.

## 2. Adaptive RMS noise-floor gate

First serious production candidate.

Rule:

```text
noise_floor = EMA(frame_rms)
open if frame_rms > noise_floor * open_ratio
close if frame_rms < noise_floor * close_ratio for release time
```

Important variation:

```text
adapt quickly while closed
adapt slowly or not at all while open
```

## 3. Adaptive dB gate

Same concept, but in decibels.

Rule:

```text
frame_db = 20 * log10(rms + epsilon)
noise_db = EMA(frame_db)
open if frame_db > noise_db + open_margin_db
close if frame_db < noise_db + close_margin_db
```

Example initial values:

```text
open_margin_db: 8 to 12 dB
close_margin_db: 3 to 6 dB
release: 200 to 500 ms
```

## 4. Percentile / quantile noise-floor gate

Use recent low-energy percentiles instead of average.

Rule:

```text
noise_floor = 20th percentile of recent frame energy
```

Then gate above percentile + margin.

This may be more robust to bursts than an EMA.

## 5. Minimum-statistics noise estimator

Track the low envelope/minimum statistics of recent energy.

Good for intermittent speech, but can lag when the environment changes.

## 6. Dual-time-constant envelope follower

Use fast attack and slow release smoothing.

Useful to reduce frame-level chatter.

Can be combined with adaptive noise floor.

## 7. Peak + RMS hybrid gate

Open if either RMS or peak evidence suggests speech.

Useful for quiet consonants or short commands.

## 8. Zero-crossing assisted gate

Use zero-crossing rate as a weak helper feature.

Can help reject low-frequency rumble or bumps.

Do not make it the main decision signal initially.

## 9. Spectral band energy gate

Measure energy in speech-relevant bands, roughly:

```text
300 Hz to 3400 Hz
```

or a broader speech band.

Can reject low-frequency handling noise and rumble.

## 10. High-pass-filtered RMS/dB gate

A simple practical improvement:

```text
high-pass around 80 to 150 Hz
then compute RMS/dB
```

Likely useful for tablets/phones because handling noise and rumble can dominate raw energy.

## 11. WebRTC VAD wrapper

Use WebRTC VAD as a strong lightweight baseline.

Good comparison method.

May or may not meet quiet/distant speech requirements.

## 12. Silero or neural VAD

Useful as a higher-quality reference.

May be too heavy for deployment, but useful in the harness.

## 13. Denoise-before-gate

Pipeline:

```text
raw audio -> denoiser -> gate -> recognizer
```

Likely later-stage only.

## 14. Existing rolling-regression quiet-region gate

The existing experimental gate estimates quiet/stable regions and derives dynamic thresholds from them.

Include it later as an adapter, not as the first harness dependency.

Risks:

- complexity
- tuning difficulty
- assumptions about stability/slope
- harder Android/C# port

## 15. Two-stage open/sustain gate

Use one rule to open and a more relaxed rule to stay open.

This is highly relevant for speech:

```text
strict open rule
relaxed sustain rule
release hold
```

Avoiding mid-utterance closure is more important than closing at the earliest possible instant.

## 16. Pre-roll and endpointing

Mandatory in practice.

Keep a ring buffer of recent audio:

```text
300 to 700 ms pre-roll
```

When gate opens, prepend this to recognizer input.

Endpoint after sustained silence.

## 17. Adaptive gain / normalization

Useful for distant speech, but can raise background noise.

Test both:

- gate before AGC
- AGC before gate
- gate on raw/filtered signal but normalize only speech chunks for recognizer

## 18. Tiny learned classifier

Later, use harness-generated labelled frames to train a small classifier.

Features:

- RMS
- dB above noise
- peak
- zero-crossing rate
- speech-band energy
- spectral flatness
- deltas

Avoid this until handcrafted methods are benchmarked.

## Suggested bake-off order

1. Fixed RMS baseline
2. Adaptive RMS EMA
3. Adaptive dB EMA
4. Percentile/minimum-statistics
5. High-pass + adaptive dB
6. WebRTC VAD
7. Two-stage open/sustain gate
8. Existing rolling-regression adapter
9. Spectral band energy gate
10. Tiny learned classifier
