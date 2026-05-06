#!/usr/bin/env python3
"""
score.py — Score predicted gate-open segments against ground-truth speech segments.

Scoring philosophy (doc 08)
---------------------------
  Opening early   < penalised less than opening late
  Closing late    < penalised less than closing early
  Missing speech  = fatal
  Late open       = fatal if > start_delay_limit_ms
  Early close     = fatal if > early_close_limit_ms

Public API
----------
  result = score_scenario(truth_segments, predicted_segments, duration_sec, thresholds)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreThresholds:
    # Fatal limits
    start_delay_limit_ms: float = 250.0
    early_close_limit_ms: float = 150.0
    false_open_limit_sec: float = 3.0
    fragmentation_limit: int = 3       # max extra predicted segments per truth segment

    # Tolerance zones used when computing start/end alignment
    start_tolerance_ms: float = 100.0
    end_tolerance_ms: float = 200.0


@dataclass
class ScenarioScore:
    scenario_id: str
    truth_segments: list
    predicted_segments: list
    duration_sec: float

    speech_total_sec: float = 0.0
    covered_speech_sec: float = 0.0
    missed_speech_sec: float = 0.0
    false_open_sec: float = 0.0

    max_start_delay_sec: float = 0.0
    max_early_close_sec: float = 0.0
    max_end_overhang_sec: float = 0.0

    predicted_segment_count: int = 0
    truth_segment_count: int = 0
    fragmentation_count: int = 0

    fatal_reasons: list = field(default_factory=list)
    passed: bool = True


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _union_length(segments: list) -> float:
    """Total length of merged intervals."""
    if not segments:
        return 0.0
    merged = sorted(segments)
    result = 0.0
    cur_s, cur_e = merged[0]
    for s, e in merged[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            result += cur_e - cur_s
            cur_s, cur_e = s, e
    result += cur_e - cur_s
    return result


def _covered_by_predicted(truth_s, truth_e, predicted) -> float:
    """How much of [truth_s, truth_e] is covered by any predicted segment."""
    intervals = [
        (max(truth_s, p_s), min(truth_e, p_e))
        for p_s, p_e in predicted
        if p_s < truth_e and p_e > truth_s
    ]
    return _union_length(intervals)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_scenario(
    scenario_id: str,
    truth_segments: list,
    predicted_segments: list,
    duration_sec: float,
    thresholds: Optional[ScoreThresholds] = None,
) -> ScenarioScore:
    thr = thresholds or ScoreThresholds()
    result = ScenarioScore(
        scenario_id=scenario_id,
        truth_segments=truth_segments,
        predicted_segments=predicted_segments,
        duration_sec=duration_sec,
        truth_segment_count=len(truth_segments),
        predicted_segment_count=len(predicted_segments),
    )

    # --- Speech coverage ---
    speech_total = _union_length(truth_segments)
    result.speech_total_sec = round(speech_total, 4)

    covered = sum(
        _covered_by_predicted(ts, te, predicted_segments)
        for ts, te in truth_segments
    )
    result.covered_speech_sec = round(covered, 4)
    result.missed_speech_sec = round(max(0.0, speech_total - covered), 4)

    # --- False-open time ---
    # Predicted time that does not overlap any truth segment
    pred_total = _union_length(predicted_segments)
    false_open = pred_total - covered
    result.false_open_sec = round(max(0.0, false_open), 4)

    # --- Per-truth-segment analysis ---
    start_delay_limit = thr.start_delay_limit_ms / 1000.0
    early_close_limit = thr.early_close_limit_ms / 1000.0
    start_tol = thr.start_tolerance_ms / 1000.0
    end_tol = thr.end_tolerance_ms / 1000.0

    for ts, te in truth_segments:
        truth_dur = te - ts

        # Find all predicted segments that overlap this truth segment
        overlapping = [
            (ps, pe) for ps, pe in predicted_segments
            if ps < te and pe > ts
        ]

        if not overlapping:
            result.fatal_reasons.append(
                f"MISSED_UTTERANCE [{ts:.2f}-{te:.2f}]"
            )
            continue

        # Fragmentation: more overlapping segments than expected
        extra = len(overlapping) - 1
        result.fragmentation_count += extra
        if extra >= thr.fragmentation_limit:
            result.fatal_reasons.append(
                f"FRAGMENTATION [{ts:.2f}-{te:.2f}] ({len(overlapping)} segments)"
            )

        # Best matching segment = highest overlap
        best = max(overlapping, key=lambda p: _overlap(ts, te, p[0], p[1]))
        ps, pe = best

        # Start delay: how late did the gate open after truth start?
        start_delay = max(0.0, ps - ts)
        result.max_start_delay_sec = max(result.max_start_delay_sec, start_delay)
        if start_delay > start_delay_limit:
            result.fatal_reasons.append(
                f"LATE_OPEN [{ts:.2f}-{te:.2f}] delay={start_delay*1000:.0f}ms"
            )

        # Early close: did the gate close before truth end?
        early_close = max(0.0, te - pe)
        result.max_early_close_sec = max(result.max_early_close_sec, early_close)
        if early_close > early_close_limit:
            result.fatal_reasons.append(
                f"EARLY_CLOSE [{ts:.2f}-{te:.2f}] early_by={early_close*1000:.0f}ms"
            )

        # End overhang: gate stays open past truth end
        overhang = max(0.0, pe - te)
        result.max_end_overhang_sec = max(result.max_end_overhang_sec, overhang)

    # --- Fatal threshold checks ---
    if result.missed_speech_sec > 0.05:  # >50ms missed = fail
        if not any("MISSED" in r for r in result.fatal_reasons):
            result.fatal_reasons.append(
                f"MISSED_SPEECH {result.missed_speech_sec:.3f}s"
            )

    if result.false_open_sec > thr.false_open_limit_sec:
        result.fatal_reasons.append(
            f"EXCESSIVE_FALSE_OPEN {result.false_open_sec:.2f}s"
        )

    result.max_start_delay_sec = round(result.max_start_delay_sec, 4)
    result.max_early_close_sec = round(result.max_early_close_sec, 4)
    result.max_end_overhang_sec = round(result.max_end_overhang_sec, 4)
    result.passed = len(result.fatal_reasons) == 0

    return result


def score_to_dict(s: ScenarioScore) -> dict:
    return {
        "scenario_id": s.scenario_id,
        "passed": s.passed,
        "fatal_reasons": s.fatal_reasons,
        "speech_total_sec": s.speech_total_sec,
        "missed_speech_sec": s.missed_speech_sec,
        "false_open_sec": s.false_open_sec,
        "max_start_delay_ms": round(s.max_start_delay_sec * 1000, 1),
        "max_early_close_ms": round(s.max_early_close_sec * 1000, 1),
        "max_end_overhang_ms": round(s.max_end_overhang_sec * 1000, 1),
        "fragmentation_count": s.fragmentation_count,
        "predicted_segments": s.predicted_segments,
        "truth_segments": s.truth_segments,
    }
