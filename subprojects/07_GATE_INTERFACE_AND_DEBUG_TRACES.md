# Gate Interface and Debug Traces

## Conceptual gate interface

Avoid binding the harness to a specific algorithm.

Conceptually, a gate should support:

```text
reset(sample_rate, config)
process_frame(frame, frame_start_sec)
finish()
get_predicted_segments()
get_debug_trace()
```

This does not prescribe exact classes or folders. It defines the behavioural boundary.

## Input

A gate receives frame-based audio.

Suggested default:

```text
sample rate: 16000 Hz
frame size: 20 ms
samples per frame: 320
```

Frame-based processing is much easier to test and closer to practical real-time use than per-sample processing.

## Output

The gate should output either:

- open/close events
- or final predicted open segments

Example:

```json
{
  "predicted_segments": [
    [1.18, 3.92],
    [5.10, 6.44]
  ]
}
```

## Debug traces

Each gate should optionally emit frame-level debug traces.

Useful fields:

```text
time_sec
frame_rms
frame_db
noise_floor
open_threshold
close_threshold
gate_state
reason_code
```

Reason codes are very useful in failure plots.

Examples:

```text
OPEN_ENERGY_ABOVE_MARGIN
CLOSE_RELEASE_EXPIRED
HOLD_RECENT_SPEECH
NOISE_FLOOR_UPDATE_SKIPPED_WHILE_OPEN
FALSE_OPEN_CANDIDATE
```

## Existing experimental gate adapter

The existing rolling-regression gate should be wrapped behind the same conceptual interface.

Do not refactor it into the harness first.

Adapter idea:

```text
frame -> activity measure -> existing observe_ng_activity(activity, pos_in_secs)
```

The first adapter can use frame RMS or dB as the activity measure.

The old code can remain as an experimental implementation. The harness should treat it as one candidate among many.
