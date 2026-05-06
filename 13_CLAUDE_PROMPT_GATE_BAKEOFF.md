# Claude Prompt — Gate Bake-Off

Use this after the harness can generate scenarios and run one simple gate.

```text
Extend the harness so it can compare multiple noise gate implementations on the same scenarios.

Do not redesign the whole project structure. Add only the minimum needed abstraction so gates can be swapped.

Each gate should conceptually:
- reset with sample rate and configuration
- process 20 ms audio frames
- emit open/close state or predicted segments
- optionally emit debug traces for plotting

Add these gate methods:
1. fixed RMS threshold baseline
2. adaptive RMS EMA gate
3. adaptive dB EMA gate
4. high-pass + adaptive dB gate if practical
5. two-stage open/sustain gate if practical

Keep WebRTC VAD, Silero, learned classifiers, and the existing rolling-regression gate as later extensions.

For each method/configuration, run the same scenario recipes and produce:
- per-scenario metrics
- aggregate metrics
- ranked failure list
- comparison summary

Scoring priorities:
1. missed speech is severe
2. late open is severe
3. early close is severe
4. mid-utterance closure is severe
5. long false open is bad
6. small early open and small late close are acceptable

The report should make it easy to see which gate method is better and why.
```
