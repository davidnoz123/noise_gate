# Scoring and Metrics

## Scoring philosophy

For speech recognition, the costs are asymmetric.

Opening early is usually acceptable.
Opening late is bad.
Closing late is often acceptable.
Closing early is very bad.
False opening briefly may be acceptable.
False opening for a long time is bad.

## Core metrics

For each scenario:

```text
speech_total_sec
missed_speech_sec
false_open_sec
max_start_delay_sec
max_early_close_sec
max_end_overhang_sec
predicted_segment_count
truth_segment_count
fragmentation_count
fatal_reason
passed
```

## Fatal failure examples

Fail a scenario if:

- an utterance is completely missed
- gate opens more than 250 ms after speech start
- gate closes more than 150 ms before speech end
- gate closes mid-utterance
- false-open time exceeds a configured limit
- one utterance is fragmented too severely

Initial thresholds are tunable. The point is to encode product pain rather than generic mathematical precision.

## Segment comparison

Given truth segments and predicted segments:

1. compute overlap
2. compute missed speech
3. compute false-open time
4. for each truth segment, find best overlapping predicted segment
5. estimate start delay
6. estimate early close / end overhang
7. detect fragmentation

## Tolerance zones

Exact acoustic boundaries are fuzzy.

Use tolerances such as:

```text
start tolerance: 50 to 150 ms
end tolerance: 100 to 300 ms
```

The harness should report raw values and pass/fail values.

## Later: recognizer outcome metrics

Eventually add Vosk evaluation:

- did the recognizer capture the intended text?
- did first-word clipping hurt recognition?
- did false-open noise produce garbage text?
- did long open segments confuse endpointing?

Gate metrics are not the final product metric. They are the first engineering metric.
