#!/usr/bin/env python3
"""
gate.py — Noise gate implementations (frame-based).

Shared interface for all gate classes
--------------------------------------
  gate.reset(sample_rate, frame_size)
  gate.process_frame(frame, frame_start_sec)
  gate.finish()
  gate.get_predicted_segments()  -> [[start, end], ...]
  gate.get_debug_trace()         -> list of per-frame dicts

Available gates
---------------
  AdaptiveDbGate      (method 3)  — EMA noise floor in dB
  PercentileGate      (method 4)  — rolling-window percentile noise floor
"""

import collections
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


EPSILON = 1e-9  # prevent log(0)


@dataclass
class GateConfig:
    # Thresholds (dB above estimated noise floor)
    open_margin_db: float = 10.0
    close_margin_db: float = 4.0

    # EMA time constants (seconds)
    noise_ema_tc_closed: float = 0.3   # adapt quickly while closed
    noise_ema_tc_open: float = 4.0     # adapt very slowly while open

    # Timing
    release_ms: float = 300.0          # must be below threshold for this long before closing
    hold_ms: float = 150.0             # hold open this long after each above-threshold frame

    # Initial noise floor estimate (dBFS)
    initial_noise_db: float = -40.0


class AdaptiveDbGate:
    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()
        self._sr: int = 16000
        self._frame_size: int = 320
        self._reset_state()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, sample_rate: int = 16000, frame_size: int = 320) -> None:
        self._sr = sample_rate
        self._frame_size = frame_size
        self._reset_state()

    def process_frame(self, frame: np.ndarray, frame_start_sec: float) -> None:
        cfg = self.config
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        frame_db = 20.0 * math.log10(rms + EPSILON)

        # EMA coefficient depends on gate state
        tc = cfg.noise_ema_tc_closed if not self._open else cfg.noise_ema_tc_open
        frame_sec = self._frame_size / self._sr
        alpha = 1.0 - math.exp(-frame_sec / tc)
        self._noise_db += alpha * (frame_db - self._noise_db)

        open_thresh = self._noise_db + cfg.open_margin_db
        close_thresh = self._noise_db + cfg.close_margin_db

        release_frames = max(1, int((cfg.release_ms / 1000.0) / frame_sec))
        hold_frames = max(1, int((cfg.hold_ms / 1000.0) / frame_sec))

        prev_open = self._open
        reason = ""

        if not self._open:
            if frame_db >= open_thresh:
                self._open = True
                self._hold_counter = hold_frames
                self._below_counter = 0
                reason = "OPEN_ENERGY_ABOVE_MARGIN"
            else:
                reason = "CLOSED_BELOW_THRESHOLD"
        else:
            # While open: reset hold counter whenever frame is above close threshold
            if frame_db >= close_thresh:
                self._hold_counter = hold_frames
                self._below_counter = 0
                reason = "OPEN_HOLD_REFRESH"
            else:
                self._below_counter += 1
                self._hold_counter = max(0, self._hold_counter - 1)
                if self._hold_counter == 0 and self._below_counter >= release_frames:
                    self._open = False
                    reason = "CLOSE_RELEASE_EXPIRED"
                else:
                    reason = "OPEN_HOLD_ACTIVE" if self._hold_counter > 0 else "OPEN_RELEASE_COUNTING"

        # Record segment boundaries
        frame_end_sec = frame_start_sec + frame_sec
        if self._open and not prev_open:
            self._seg_start = frame_start_sec
        elif not self._open and prev_open:
            self._segments.append([round(self._seg_start, 4), round(frame_start_sec, 4)])
            self._seg_start = None

        self._trace.append({
            "time_sec": round(frame_start_sec, 4),
            "frame_db": round(frame_db, 2),
            "noise_db": round(self._noise_db, 2),
            "open_thresh_db": round(open_thresh, 2),
            "close_thresh_db": round(close_thresh, 2),
            "gate_state": 1 if self._open else 0,
            "reason": reason,
        })
        self._last_frame_end = frame_end_sec

    def finish(self) -> None:
        """Close any still-open segment at end of stream."""
        if self._open and self._seg_start is not None:
            self._segments.append([round(self._seg_start, 4), round(self._last_frame_end, 4)])
            self._open = False
            self._seg_start = None

    def get_predicted_segments(self) -> list[list[float]]:
        return list(self._segments)

    def get_debug_trace(self) -> list[dict]:
        return list(self._trace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._noise_db: float = self.config.initial_noise_db
        self._open: bool = False
        self._seg_start: Optional[float] = None
        self._hold_counter: int = 0
        self._below_counter: int = 0
        self._segments: list[list[float]] = []
        self._trace: list[dict] = []
        self._last_frame_end: float = 0.0


# ---------------------------------------------------------------------------
# Convenience: run gate on a full audio array
# ---------------------------------------------------------------------------

def run_gate(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_size: int = 320,
    config: Optional[GateConfig] = None,
) -> tuple[list[list[float]], list[dict]]:
    """
    Run the gate on *audio* (mono float32) and return (predicted_segments, trace).
    """
    gate = AdaptiveDbGate(config)
    gate.reset(sample_rate, frame_size)

    n = len(audio)
    pos = 0
    while pos + frame_size <= n:
        frame = audio[pos : pos + frame_size]
        frame_start = pos / sample_rate
        gate.process_frame(frame, frame_start)
        pos += frame_size

    # Handle final partial frame
    if pos < n:
        frame = np.zeros(frame_size, dtype=np.float32)
        frame[: n - pos] = audio[pos:]
        gate.process_frame(frame, pos / sample_rate)

    gate.finish()
    return gate.get_predicted_segments(), gate.get_debug_trace()


# ===========================================================================
# Method 4 — Percentile / quantile noise-floor gate
# ===========================================================================

@dataclass
class PercentileGateConfig:
    """Configuration for the rolling-percentile noise floor gate.

    The noise floor is estimated as the Nth percentile of the most recent
    *window_sec* seconds of frame_db values.  Unlike EMA, the percentile is
    insensitive to the upper tail, so speech bursts cannot inflate the noise
    floor estimate and cause the gate to miss quieter subsequent speech.
    """
    # Thresholds (dB above estimated noise floor)
    open_margin_db: float = 10.0
    close_margin_db: float = 4.0

    # Rolling window
    window_sec: float = 2.0        # how much history to keep
    percentile: float = 20.0       # e.g. 20 → 20th percentile = low-energy floor

    # Timing
    release_ms: float = 300.0
    hold_ms: float = 150.0

    # Startup: used as noise floor until window contains at least this many frames
    initial_noise_db: float = -40.0


class PercentileGate:
    """Noise gate using a rolling percentile as the noise floor estimate.

    Identical open/close state machine to AdaptiveDbGate; only the noise floor
    estimator differs.
    """

    def __init__(self, config: Optional[PercentileGateConfig] = None):
        self.config = config or PercentileGateConfig()
        self._sr: int = 16000
        self._frame_size: int = 320
        self._reset_state()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, sample_rate: int = 16000, frame_size: int = 320) -> None:
        self._sr = sample_rate
        self._frame_size = frame_size
        self._reset_state()

    def process_frame(self, frame: np.ndarray, frame_start_sec: float) -> None:
        cfg = self.config
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        frame_db = 20.0 * math.log10(rms + EPSILON)

        # Update rolling window
        self._window.append(frame_db)

        # Percentile noise floor — uses whatever is in the window so far
        self._noise_db = float(np.percentile(self._window, cfg.percentile))

        open_thresh = self._noise_db + cfg.open_margin_db
        close_thresh = self._noise_db + cfg.close_margin_db

        frame_sec = self._frame_size / self._sr
        release_frames = max(1, int((cfg.release_ms / 1000.0) / frame_sec))
        hold_frames = max(1, int((cfg.hold_ms / 1000.0) / frame_sec))

        prev_open = self._open
        reason = ""

        if not self._open:
            if frame_db >= open_thresh:
                self._open = True
                self._hold_counter = hold_frames
                self._below_counter = 0
                reason = "OPEN_ENERGY_ABOVE_MARGIN"
            else:
                reason = "CLOSED_BELOW_THRESHOLD"
        else:
            if frame_db >= close_thresh:
                self._hold_counter = hold_frames
                self._below_counter = 0
                reason = "OPEN_HOLD_REFRESH"
            else:
                self._below_counter += 1
                self._hold_counter = max(0, self._hold_counter - 1)
                if self._hold_counter == 0 and self._below_counter >= release_frames:
                    self._open = False
                    reason = "CLOSE_RELEASE_EXPIRED"
                else:
                    reason = "OPEN_HOLD_ACTIVE" if self._hold_counter > 0 else "OPEN_RELEASE_COUNTING"

        frame_end_sec = frame_start_sec + frame_sec
        if self._open and not prev_open:
            self._seg_start = frame_start_sec
        elif not self._open and prev_open:
            self._segments.append([round(self._seg_start, 4), round(frame_start_sec, 4)])
            self._seg_start = None

        self._trace.append({
            "time_sec": round(frame_start_sec, 4),
            "frame_db": round(frame_db, 2),
            "noise_db": round(self._noise_db, 2),
            "open_thresh_db": round(open_thresh, 2),
            "close_thresh_db": round(close_thresh, 2),
            "gate_state": 1 if self._open else 0,
            "reason": reason,
        })
        self._last_frame_end = frame_end_sec

    def finish(self) -> None:
        if self._open and self._seg_start is not None:
            self._segments.append([round(self._seg_start, 4), round(self._last_frame_end, 4)])
            self._open = False
            self._seg_start = None

    def get_predicted_segments(self) -> list[list[float]]:
        return list(self._segments)

    def get_debug_trace(self) -> list[dict]:
        return list(self._trace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        cfg = self.config
        frame_sec = self._frame_size / self._sr
        maxlen = max(1, int(cfg.window_sec / frame_sec))
        # Pre-fill with initial_noise_db so the first frame sees a sane floor
        self._window: collections.deque = collections.deque(
            [cfg.initial_noise_db] * maxlen, maxlen=maxlen
        )
        self._noise_db: float = cfg.initial_noise_db
        self._open: bool = False
        self._seg_start: Optional[float] = None
        self._hold_counter: int = 0
        self._below_counter: int = 0
        self._segments: list[list[float]] = []
        self._trace: list[dict] = []
        self._last_frame_end: float = 0.0


def run_gate_percentile(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_size: int = 320,
    config: Optional[PercentileGateConfig] = None,
) -> tuple[list[list[float]], list[dict]]:
    """Run the percentile gate on *audio* and return (predicted_segments, trace)."""
    gate = PercentileGate(config)
    gate.reset(sample_rate, frame_size)

    n = len(audio)
    pos = 0
    while pos + frame_size <= n:
        gate.process_frame(audio[pos : pos + frame_size], pos / sample_rate)
        pos += frame_size

    if pos < n:
        frame = np.zeros(frame_size, dtype=np.float32)
        frame[: n - pos] = audio[pos:]
        gate.process_frame(frame, pos / sample_rate)

    gate.finish()
    return gate.get_predicted_segments(), gate.get_debug_trace()
