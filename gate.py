#!/usr/bin/env python3
"""
gate.py — Adaptive dB noise gate (frame-based).

Interface
---------
  gate = AdaptiveDbGate(config)
  gate.reset(sample_rate)
  for frame in frames:
      gate.process_frame(frame, frame_start_sec)
  gate.finish()
  segments = gate.get_predicted_segments()   # [[start, end], ...]
  trace    = gate.get_debug_trace()          # list of per-frame dicts

Algorithm (doc 06, method 3)
----------------------------
  frame_db   = 20 * log10(rms(frame) + epsilon)
  noise_db   = EMA(frame_db)  — adapts fast while closed, slow while open
  open  if frame_db > noise_db + open_margin_db
  close if frame_db < noise_db + close_margin_db  for >= release_frames
  hold  for hold_frames after last open event (prevents fragmentation)
"""

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
