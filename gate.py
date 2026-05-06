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
  DualTcGate          (method 6)  — fast-attack / slow-decay envelope follower
  BandpassDbGate      (method 9)  — 300-3400 Hz speech-band filtered gate
  HpfDbGate           (method 10) — high-pass filtered EMA gate
  RollingLinGate      (method 14) — rolling linear regression quiet-region gate
"""

import collections
import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


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


# ===========================================================================
# Method 10 — High-pass filtered dB gate
# ===========================================================================

@dataclass
class HpfDbGateConfig:
    """Configuration for the high-pass filtered adaptive dB gate.

    A Butterworth high-pass filter is applied to each frame *before* computing
    RMS/dB.  This strips low-frequency rumble and handling noise (~0-150 Hz)
    from the energy measurement so the noise floor estimate and open/close
    thresholds are based only on speech-band energy.  The gate logic itself
    (EMA noise floor, margins, release, hold) is identical to AdaptiveDbGate.
    """
    # High-pass filter
    cutoff_hz: float = 120.0
    filter_order: int = 2

    # Thresholds (dB above estimated noise floor, measured on HPF signal)
    open_margin_db: float = 10.0
    close_margin_db: float = 4.0

    # EMA time constants (seconds)
    noise_ema_tc_closed: float = 0.3
    noise_ema_tc_open: float = 4.0

    # Timing
    release_ms: float = 300.0
    hold_ms: float = 150.0

    # Initial noise floor estimate (dBFS)
    initial_noise_db: float = -40.0


class HpfDbGate:
    """Adaptive dB gate that measures energy on a high-pass filtered copy of
    each frame.  The filter state is maintained across frames for correct IIR
    continuity.
    """

    def __init__(self, config: Optional[HpfDbGateConfig] = None):
        self.config = config or HpfDbGateConfig()
        self._sr: int = 16000
        self._frame_size: int = 320
        self._sos = None
        self._zi = None
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

        # Filter frame through HPF (IIR state carried across calls)
        filtered, self._zi = sosfilt(self._sos, frame.astype(np.float64), zi=self._zi)
        rms = float(np.sqrt(np.mean(filtered ** 2)))
        frame_db = 20.0 * math.log10(rms + EPSILON)

        # EMA noise floor
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
        nyq = self._sr / 2.0
        self._sos = butter(cfg.filter_order, cfg.cutoff_hz / nyq,
                           btype="highpass", output="sos")
        # sosfilt_zi returns (n_sections, 2) — correct shape for 1D input
        self._zi = sosfilt_zi(self._sos) * 0.0
        self._noise_db: float = cfg.initial_noise_db
        self._open: bool = False
        self._seg_start: Optional[float] = None
        self._hold_counter: int = 0
        self._below_counter: int = 0
        self._segments: list[list[float]] = []
        self._trace: list[dict] = []
        self._last_frame_end: float = 0.0


def run_gate_hpf(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_size: int = 320,
    config: Optional[HpfDbGateConfig] = None,
) -> tuple[list[list[float]], list[dict]]:
    """Run the high-pass filtered gate on *audio* and return (predicted_segments, trace)."""
    gate = HpfDbGate(config)
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


# ===========================================================================
# Method 9 — Bandpass (speech-band) dB gate
# ===========================================================================

@dataclass
class BandpassDbGateConfig:
    """Configuration for the speech-band filtered adaptive dB gate.

    A Butterworth bandpass filter is applied to each frame before computing
    RMS/dB.  The pass-band covers the core telephone / speech band
    (default 300-3400 Hz), which rejects both:
      - low-frequency rumble / handling noise below 300 Hz, and
      - broadband hiss / environmental noise above 3400 Hz.

    The gate logic (EMA noise floor, margins, release, hold) is identical
    to AdaptiveDbGate and HpfDbGate.
    """
    # Bandpass filter
    low_hz: float = 300.0
    high_hz: float = 3400.0
    filter_order: int = 2

    # Thresholds (dB above estimated noise floor, measured on bandpassed signal)
    open_margin_db: float = 10.0
    close_margin_db: float = 4.0

    # EMA time constants (seconds)
    noise_ema_tc_closed: float = 0.3
    noise_ema_tc_open: float = 4.0

    # Timing
    release_ms: float = 300.0
    hold_ms: float = 150.0

    # Initial noise floor estimate (dBFS)
    initial_noise_db: float = -40.0


class BandpassDbGate:
    """Adaptive dB gate that measures energy on a bandpass-filtered copy of
    each frame (default 300-3400 Hz speech band).  IIR filter state is
    maintained across frames for continuity.
    """

    def __init__(self, config: Optional[BandpassDbGateConfig] = None):
        self.config = config or BandpassDbGateConfig()
        self._sr: int = 16000
        self._frame_size: int = 320
        self._sos = None
        self._zi = None
        self._reset_state()

    def reset(self, sample_rate: int = 16000, frame_size: int = 320) -> None:
        self._sr = sample_rate
        self._frame_size = frame_size
        self._reset_state()

    def process_frame(self, frame: np.ndarray, frame_start_sec: float) -> None:
        cfg = self.config

        filtered, self._zi = sosfilt(self._sos, frame.astype(np.float64), zi=self._zi)
        rms = float(np.sqrt(np.mean(filtered ** 2)))
        frame_db = 20.0 * math.log10(rms + EPSILON)

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

    def _reset_state(self) -> None:
        cfg = self.config
        nyq = self._sr / 2.0
        self._sos = butter(cfg.filter_order,
                           [cfg.low_hz / nyq, cfg.high_hz / nyq],
                           btype="bandpass", output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0
        self._noise_db: float = cfg.initial_noise_db
        self._open: bool = False
        self._seg_start: Optional[float] = None
        self._hold_counter: int = 0
        self._below_counter: int = 0
        self._segments: list[list[float]] = []
        self._trace: list[dict] = []
        self._last_frame_end: float = 0.0


def run_gate_bandpass(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_size: int = 320,
    config: Optional[BandpassDbGateConfig] = None,
) -> tuple[list[list[float]], list[dict]]:
    """Run the speech-band filtered gate on *audio* and return (predicted_segments, trace)."""
    gate = BandpassDbGate(config)
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


# ===========================================================================
# Method 6 — Dual time-constant envelope follower gate
# ===========================================================================

@dataclass
class DualTcGateConfig:
    """Configuration for the dual time-constant envelope follower gate.

    The raw frame_db signal is smoothed with an asymmetric envelope follower
    (fast attack, slow release) *before* being compared to the noise floor.
    This reduces frame-level chatter: a single loud frame can open the gate
    quickly, but the envelope decays slowly so transient dips mid-utterance
    don't trigger a close.

    The noise floor itself is still estimated by EMA (same as AdaptiveDbGate).
    """
    # Envelope follower time constants (seconds)
    attack_tc: float = 0.01     # fast: ~10 ms to track speech onset
    decay_tc: float = 0.3       # slow: ~300 ms to ride through inter-word gaps

    # Thresholds (dB above estimated noise floor, applied to *smoothed* envelope)
    open_margin_db: float = 10.0
    close_margin_db: float = 4.0

    # EMA noise floor time constants (seconds)
    noise_ema_tc_closed: float = 0.3
    noise_ema_tc_open: float = 4.0

    # Timing
    release_ms: float = 300.0
    hold_ms: float = 150.0

    # Initial values (dBFS)
    initial_noise_db: float = -40.0


class DualTcGate:
    """Adaptive dB gate with a fast-attack / slow-decay envelope smoother
    applied to the frame energy before the open/close decision.

    This reduces chatter from short transients and inter-word pauses.
    """

    def __init__(self, config: Optional[DualTcGateConfig] = None):
        self.config = config or DualTcGateConfig()
        self._sr: int = 16000
        self._frame_size: int = 320
        self._reset_state()

    def reset(self, sample_rate: int = 16000, frame_size: int = 320) -> None:
        self._sr = sample_rate
        self._frame_size = frame_size
        self._reset_state()

    def process_frame(self, frame: np.ndarray, frame_start_sec: float) -> None:
        cfg = self.config
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        raw_db = 20.0 * math.log10(rms + EPSILON)

        frame_sec = self._frame_size / self._sr

        # Asymmetric envelope follower on the dB signal
        if raw_db > self._env_db:
            tc = cfg.attack_tc
        else:
            tc = cfg.decay_tc
        env_alpha = 1.0 - math.exp(-frame_sec / tc)
        self._env_db += env_alpha * (raw_db - self._env_db)
        frame_db = self._env_db  # smoothed energy used for open/close decision

        # EMA noise floor (tracks the smoothed envelope)
        noise_tc = cfg.noise_ema_tc_closed if not self._open else cfg.noise_ema_tc_open
        noise_alpha = 1.0 - math.exp(-frame_sec / noise_tc)
        self._noise_db += noise_alpha * (frame_db - self._noise_db)

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
            "raw_db": round(raw_db, 2),
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

    def _reset_state(self) -> None:
        cfg = self.config
        self._env_db: float = cfg.initial_noise_db   # envelope follower state
        self._noise_db: float = cfg.initial_noise_db
        self._open: bool = False
        self._seg_start: Optional[float] = None
        self._hold_counter: int = 0
        self._below_counter: int = 0
        self._segments: list[list[float]] = []
        self._trace: list[dict] = []
        self._last_frame_end: float = 0.0


def run_gate_dual_tc(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_size: int = 320,
    config: Optional[DualTcGateConfig] = None,
) -> tuple[list[list[float]], list[dict]]:
    """Run the dual-TC envelope gate on *audio* and return (predicted_segments, trace)."""
    gate = DualTcGate(config)
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


# ===========================================================================
# Method 14 — Rolling linear regression quiet-region gate
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _releq(a: float, b: float, tolerance: float) -> bool:
    """Relative equality: |a-b| / max(|a|, |b|, 1e-15) <= tolerance."""
    return abs(a - b) / max(abs(a), abs(b), 1e-15) <= tolerance


def _round_sig_figs(x: float, n: int) -> float:
    """Round *x* to *n* significant figures."""
    if x == 0.0:
        return 0.0
    d = math.ceil(math.log10(abs(x)))
    factor = 10 ** (n - d)
    return round(x * factor) / factor


# ---------------------------------------------------------------------------
# Rolling OLS regression over a sliding window
# ---------------------------------------------------------------------------

class RollingLinearRegression:
    """Incremental ordinary-least-squares regression over a sliding window.

    Based on https://gist.github.com/NikolayIT/d86118a3a0cb3f5ed63d674a350d75f2

    Maintains five running sums (sumOfX, sumOfY, sumOfXSq, sumOfYSq,
    sumCodeviates) over the most recent ``window_size`` (x, y) pairs.
    All regression statistics are lazily-evaluated properties derived from
    those sums in O(1).  Floating-point drift is corrected every
    ``observe_count_recalculate_modulus`` observations by recomputing from
    scratch.

    Note: the effective window once rolling is ``window_size - 1`` pairs
    (the oldest is evicted on the frame that would make it full).

    Parameters
    ----------
    window_size :
        Number of (x, y) pairs in the sliding window.
    translate_x_to_xo :
        If True (default), x-dependent sums are computed relative to the
        oldest x in the window, improving numerical stability for large
        monotonically-increasing x values (e.g. pos_in_secs at 16 kHz).
    sklearn_linear_model :
        If not None, slope/yIntercept/R² are cross-checked against sklearn
        on every recalculation cycle (debug / development only; leave None
        in production).

    Computed properties (O(1), all derived from running sums)
    ----------------------------------------------------------
    slope, yIntercept : least-squares fit parameters
    varY              : population variance of y in the current window
    rSquared          : coefficient of determination R²
    meanX, meanY      : window means
    ssX, ssY          : total sum of squared deviations from the mean
    """

    _props_installed: bool = False
    _prop_exprs: dict = {
        "sumOfX_":        "self.sumOfX        if not self.translate_x_to_xo else self.sumOfX        - self.window[0][0] * self.count",
        "sumOfXSq_":      "self.sumOfXSq      if not self.translate_x_to_xo else self.sumOfXSq      - self.window[0][0] * self.sumOfX * 2.0 + (self.window[0][0] ** 2) * self.count",
        "sumCodeviates_": "self.sumCodeviates if not self.translate_x_to_xo else self.sumCodeviates - self.window[0][0] * self.sumOfY",
        "sumOfYSq_":      "self.sumOfYSq",
        "ssX":            "self.sumOfXSq_ - ((self.sumOfX_ ** 2) / self.count)",
        "ssY":            "self.sumOfYSq - ((self.sumOfY * self.sumOfY) / self.count)",
        "varY":           "self.ssY / self.count",
        "rNumerator":     "(self.count * self.sumCodeviates_) - (self.sumOfX_ * self.sumOfY)",
        "rDenom":         "(self.count * self.sumOfXSq_ - (self.sumOfX_ ** 2)) * (self.count * self.sumOfYSq - (self.sumOfY * self.sumOfY))",
        "sCo":            "self.sumCodeviates_ - ((self.sumOfX_ * self.sumOfY) / self.count)",
        "meanX":          "self.sumOfX_ / self.count",
        "meanY":          "self.sumOfY / self.count",
        "dblR":           "self.rNumerator / math.sqrt(self.rDenom)",
        "rSquared":       "self.dblR ** 2",
        "yIntercept":     "self.meanY - ((self.sCo / self.ssX) * self.meanX)",
        "slope":          "self.sCo / self.ssX",
    }

    def __init__(self, window_size: int = 20, translate_x_to_xo: bool = True,
                 sklearn_linear_model=None):
        """Initialise the regressor.

        Installs all entries from ``_prop_exprs`` as class-level ``property``
        objects on first instantiation (cost paid once per class, not per
        instance).
        """
        self.window_size = window_size
        self.translate_x_to_xo = translate_x_to_xo
        self.sklearn_linear_model = sklearn_linear_model
        self.window: list = []
        self.x_implied = None
        self.observe_count = 0
        self.debug_tested_count = 0
        self.count = 0.0
        self.sumOfX = 0.0
        self.sumOfY = 0.0
        self.sumOfXSq = 0.0
        self.sumOfYSq = 0.0
        self.sumCodeviates = 0.0
        self.observe_count_recalculate_modulus = 5096
        if not self.__class__._props_installed:
            for name, expr in sorted(self.__class__._prop_exprs.items()):
                setattr(self.__class__, name, eval(f"property(lambda self: {expr})"))
            self.__class__._props_installed = True

    def observe_value(self, y: float, x: float = None) -> None:
        """Add one (x, y) observation to the sliding window.

        Parameters
        ----------
        y :
            Dependent variable (e.g. per-frame RMS activity).
        x :
            Independent variable (e.g. pos_in_secs).  If None on the first
            call, sequential integers (0, 1, 2, …) are used for all calls.
            The choice of implied vs explicit x must be consistent per instance.

        Notes
        -----
        * Appends (x, y) and updates all five running sums in O(1).
        * Once the window reaches ``window_size``, the oldest pair is evicted
          (effective window is window_size - 1 when rolling).
        * Every ``observe_count_recalculate_modulus`` calls the sums are
          recomputed from scratch to bound floating-point accumulation error.
          Validation assertions (currently enabled) verify the recomputed sums
          agree with the rolling sums to within 1e-10 relative tolerance.
        * If ``sklearn_linear_model`` was supplied, slope/yIntercept/R² are
          cross-checked against sklearn on each recalculation (debug only).
        """
        if self.x_implied is None:
            self.x_implied = (x is None)
        if not self.x_implied:
            if x is None:
                raise ValueError("x must be provided (was explicit on first call)")
            self.window.append((x, y))
        else:
            if x is not None:
                raise ValueError("x must not be provided (was implied on first call)")
            x = self.observe_count
            self.window.append((x, y))
        self.observe_count += 1

        if self.observe_count < self.window_size:
            self.count = self.observe_count
        else:
            x_old, y_old = self.window.pop(0)
            self.sumCodeviates -= x_old * y_old
            self.sumOfX -= x_old
            self.sumOfY -= y_old
            self.sumOfXSq -= x_old * x_old
            self.sumOfYSq -= y_old * y_old

        self.sumCodeviates += x * y
        self.sumOfX += x
        self.sumOfY += y
        self.sumOfXSq += x * x
        self.sumOfYSq += y * y

        if self.observe_count % self.observe_count_recalculate_modulus == 0:
            if self.observe_count >= len(self.window):
                sum_c = sum_x = sum_y = sum_xsq = sum_ysq = 0.0
                for xw, yw in self.window:
                    sum_c   += xw * yw
                    sum_x   += xw
                    sum_y   += yw
                    sum_xsq += xw * xw
                    sum_ysq += yw * yw

                # --- Validation assertions (temporarily enabled) ---
                if not _releq(self.sumCodeviates, sum_c,   1e-10):
                    raise AssertionError(f"sumCodeviates drift: rolling={self.sumCodeviates}  recomputed={sum_c}")
                if not _releq(self.sumOfX,        sum_x,   1e-10):
                    raise AssertionError(f"sumOfX drift: rolling={self.sumOfX}  recomputed={sum_x}")
                if not _releq(self.sumOfY,        sum_y,   1e-10):
                    raise AssertionError(f"sumOfY drift: rolling={self.sumOfY}  recomputed={sum_y}")
                if not _releq(self.sumOfXSq,      sum_xsq, 1e-10):
                    raise AssertionError(f"sumOfXSq drift: rolling={self.sumOfXSq}  recomputed={sum_xsq}")
                if not _releq(self.sumOfYSq,      sum_ysq, 1e-10):
                    raise AssertionError(f"sumOfYSq drift: rolling={self.sumOfYSq}  recomputed={sum_ysq}")

                self.sumCodeviates = sum_c
                self.sumOfX        = sum_x
                self.sumOfY        = sum_y
                self.sumOfXSq      = sum_xsq
                self.sumOfYSq      = sum_ysq

                if self.sklearn_linear_model is not None:
                    xx_arr, yy_arr = zip(*self.window)
                    yy_arr = np.array(yy_arr)
                    xx_arr = np.array(xx_arr)
                    if not _releq(float(np.mean(yy_arr)),  self.meanY, 1e-10):
                        raise AssertionError("meanY mismatch vs numpy")
                    if not _releq(float(np.var(yy_arr)),   self.varY,  1e-10):
                        raise AssertionError("varY mismatch vs numpy")
                    xx_fit = (xx_arr - xx_arr[0] if self.translate_x_to_xo else xx_arr).reshape(-1, 1)
                    model = self.sklearn_linear_model.LinearRegression()
                    try:
                        model.fit(xx_fit, yy_arr)
                        if not math.isnan(self.slope) and not _releq(float(model.coef_[0]), self.slope, 1e-4):
                            raise AssertionError(f"slope mismatch: sklearn={model.coef_[0]}  rlr={self.slope}")
                        if not math.isnan(self.yIntercept) and not _releq(float(model.intercept_), self.yIntercept, 1e-4):
                            raise AssertionError(f"yIntercept mismatch: sklearn={model.intercept_}  rlr={self.yIntercept}")
                        self.debug_tested_count += 1
                    except ZeroDivisionError:
                        pass

    @classmethod
    def testbed(cls, c_sharp_interface=None):
        """Self-test: validates slope/yIntercept against sklearn for many
        window/sequence/modulus size combinations.

        Requires: sklearn.
        Optional: c_sharp_interface with a matching RollingLinearRegression.
        """
        import random
        print("import sklearn.linear_model ...")
        import sklearn.linear_model
        print("import sklearn.linear_model end")
        c_sharp_rlr = None if c_sharp_interface is None else c_sharp_interface.RollingLinearRegression

        for window_size in range(2, 10):
            for sequence_size in range(10):
                for recalc_mod in range(1, 10):
                    rnd = random.Random(123)
                    a = [rnd.random() for _ in range(sequence_size)]
                    print(window_size, sequence_size, recalc_mod)
                    for translate_x_to_xo in [True, False]:
                        rlr = cls(window_size, translate_x_to_xo=translate_x_to_xo,
                                  sklearn_linear_model=sklearn.linear_model)
                        rlr.observe_count_recalculate_modulus = recalc_mod
                        vals_py = []
                        for v in a:
                            rlr.observe_value(v)
                            try:
                                vals_py.append((rlr.slope, rlr.yIntercept))
                            except ZeroDivisionError:
                                vals_py.append((math.nan, math.nan))

                        if c_sharp_rlr is not None:
                            vals_cs = []
                            rlr_cs = c_sharp_rlr(window_size, translate_x_to_xo)
                            rlr_cs.observe_count_recalculate_modulus = recalc_mod
                            for v in a:
                                rlr_cs.observe_value(v)
                                vals_cs.append((rlr_cs.slope, rlr_cs.yIntercept))
                            for k, (v_py, v_cs) in enumerate(zip(vals_py, vals_cs)):
                                if v_py != v_cs:
                                    if not any(math.isnan(vv) for vv in v_py) and \
                                       not any(math.isnan(vv) for vv in v_cs):
                                        if not all(_releq(vp, vc, 1e-9) for vp, vc in zip(v_py, v_cs)):
                                            raise AssertionError(
                                                f"C# mismatch at k={k}: py={v_py}  cs={v_cs}"
                                            )
        print("RollingLinearRegression.testbed passed")


# ---------------------------------------------------------------------------
# Base: two-threshold hysteresis gate, sample-level interface
# ---------------------------------------------------------------------------

class _NoiseGateBase:
    """Two-threshold hysteresis gate (base implementation, sample-level interface).

    State machine
    -------------
    CLOSED → OPEN  : activity >= threshold_open
    OPEN   → CLOSED: activity < threshold_close continuously for >= hold_secs
                      (hold timer resets if activity returns to threshold_open)

    Returns ``(noise_gate_beg, noise_gate_end)`` event flags per sample.
    Both flags are mutually exclusive and are never both True in one call.
    """

    def __init__(self, threshold_open: float, threshold_close: float, hold_secs: float):
        """
        Parameters
        ----------
        threshold_open :
            Activity level at which the gate opens.
        threshold_close :
            Activity level below which the hold timer starts.  Must be
            <= threshold_open (asserted at runtime when validation is enabled).
        hold_secs :
            Duration (seconds) that activity must remain below threshold_close
            before the gate closes.  Prevents premature closure during
            inter-word pauses.
        """
        self.threshold_open  = threshold_open
        self.threshold_close = threshold_close
        self.hold_secs       = hold_secs
        self.threshold_open_pos_in_secs          = None
        self.below_threshold_close_pos_in_secs   = None
        self.pos_in_secs_prev                    = float("-inf")
        self.call_count = 0

    def observe_ng_activity(self, activity: float, pos_in_secs: float):
        """Advance the gate state machine by one sample.

        Parameters
        ----------
        activity :
            Non-negative scalar activity value for this sample.
        pos_in_secs :
            Monotonically non-decreasing timestamp in seconds.

        Returns
        -------
        (noise_gate_beg, noise_gate_end) : tuple[bool, bool]
            noise_gate_beg=True  — gate just transitioned CLOSED → OPEN.
            noise_gate_end=True  — gate just transitioned OPEN   → CLOSED.

        Raises
        ------
        AssertionError
            If activity < 0, timestamps are non-monotonic, or
            threshold_open < threshold_close.  (Validation temporarily enabled.)
        """
        self.call_count += 1

        # --- Runtime validation assertions (temporarily enabled) ---
        if activity < 0:
            raise AssertionError(f"activity < 0: {activity}")
        if self.pos_in_secs_prev > pos_in_secs:
            raise AssertionError(
                f"pos_in_secs not monotonic: prev={self.pos_in_secs_prev}  cur={pos_in_secs}"
            )
        if self.threshold_open < self.threshold_close:
            raise AssertionError(
                f"threshold_open < threshold_close: {self.threshold_open} < {self.threshold_close}"
            )

        noise_gate_beg, noise_gate_end = False, False
        self.pos_in_secs_prev = pos_in_secs

        if self.threshold_open_pos_in_secs is None:
            if activity >= self.threshold_open:
                noise_gate_beg = True
                self.threshold_open_pos_in_secs        = pos_in_secs
                self.below_threshold_close_pos_in_secs = None
        elif activity >= self.threshold_open:
            self.below_threshold_close_pos_in_secs = None
        elif activity < self.threshold_close:
            if self.below_threshold_close_pos_in_secs is None:
                self.below_threshold_close_pos_in_secs = pos_in_secs
            if (pos_in_secs - self.below_threshold_close_pos_in_secs) >= self.hold_secs:
                noise_gate_end = True
                self.threshold_open_pos_in_secs        = None
                self.below_threshold_close_pos_in_secs = None

        return noise_gate_beg, noise_gate_end


# ---------------------------------------------------------------------------
# _NoiseGate2: rolling-regression quiet-region estimator
# ---------------------------------------------------------------------------

class _NoiseGate2(_NoiseGateBase):
    """Rolling-regression quiet-region gate (method 14 / existing experimental gate).

    Automatically estimates a dynamic noise floor by finding the quietest,
    most statistically stable window in recent audio, then sets open/close
    thresholds relative to that estimate.

    Algorithm overview
    ------------------
    1. **Activity preprocessing** (per observation)

       Per-frame RMS is fed into a short rolling linear regression (``rlr``,
       window = ``rolling_linear_regression_window_secs``).  When
       ``use_rms_activity=True`` (default), the windowed RMS of ``rlr`` feeds
       a second regression (``rlr_rms``), giving ``activity_xxx`` a smoother,
       less impulsive character.

    2. **Quiet-region candidate selection**

       A regression snapshot is accepted only when all three hold:
       * ``|slope| < abs_slope_threshold_max``  — signal is near-flat
       * ``yIntercept > 0``                      — level is positive
       * ``varY > 0``                             — window has real variance

    3. **Candidate bookkeeping**

       Each accepted snapshot is stored as a ``_HeapElem`` in two parallel
       structures:
       * ``elems_list`` (FIFO deque) — time-ordered expiry; entries older
         than ``history_secs_threshold_max`` are marked inactive.
       * ``tuple_heap`` (min-heap) — keyed by
         ``(round_sig_figs(varY, 2), round_sig_figs(|slope|, 2), elem)``
         so the heap root is always the snapshot with the lowest variance
         (most stable noise estimate).  Inactive entries are lazily removed.

    4. **Threshold derivation**

       From the heap root (best active snapshot)::

           threshold_open  = best.yIntercept
                           + stddev_2_add_to_threshold_open * sqrt(best.varY)
           threshold_close = threshold_close_perc_of_threshold_open * threshold_open

    5. **Adaptive hold time**

       After gate-open, ``hold_secs`` ramps linearly from ``hold_secs_min``
       to ``hold_secs_max`` over ``hold_secs_dur``, then stays at the maximum.

    6. **Gate decision**

       Delegates to ``_NoiseGateBase.observe_ng_activity`` with the current
       ``activity_xxx`` and dynamic thresholds.

    Control parameters (``control_set_00``, modifiable after construction)
    -----------------------------------------------------------------------
    abs_slope_threshold_max                : float = 0.0001
    history_secs_threshold_max             : float = 1.0 s
    stddev_2_add_to_threshold_open         : float = 150.0  (× estimated noise stddev)
    threshold_close_perc_of_threshold_open : float = 0.4
    hold_secs_min                          : float = 0.01 s
    hold_secs_max                          : float = 0.2 s
    hold_secs_dur                          : float = 0.15 s
    rolling_linear_regression_window_secs  : float = 0.04 s
    """

    class _HeapElem:
        """Snapshot of one quiet-region regression estimate.

        Stored simultaneously in ``elems_list`` (FIFO, for time-based expiry)
        and ``tuple_heap`` (min-heap, for variance-ordered selection).

        Attributes
        ----------
        ordinal    : ``-pos_in_secs`` at creation; negative so the FIFO list
                     expires oldest entries from its front.
        is_active  : Set to False when the entry ages out; allows lazy heap
                     deletion without an O(N) rebuild.
        yIntercept : Estimated noise floor level from the regression.
        varY       : Population variance of y — the primary heap sort key
                     (lower = more stable / quieter estimate).
        """
        __slots__ = ["ordinal", "is_active", "yIntercept", "varY"]

        def __init__(self, ordinal: float, yIntercept: float, varY: float):
            self.ordinal    = ordinal
            self.is_active  = True
            self.yIntercept = yIntercept
            self.varY       = varY

        def __lt__(self, other):
            # Tie-break for heapq when (varY, slope) values are equal
            return self.ordinal < other.ordinal

    control_set_00 = dict(
        abs_slope_threshold_max                = 0.0001,
        history_secs_threshold_max             = 1.0,
        stddev_2_add_to_threshold_open         = 150.0,
        threshold_close_perc_of_threshold_open = 0.4,
        hold_secs_min                          = 0.01,
        hold_secs_max                          = 0.2,
        hold_secs_dur                          = 0.15,
        rolling_linear_regression_window_secs  = 0.04,
    )

    def __init__(self, orig_sr: int, rolling_linear_regression_window_size: int = None,
                 use_rms_activity: bool = True):
        """
        Parameters
        ----------
        orig_sr :
            Sample rate of the audio (Hz).  Used to convert
            ``rolling_linear_regression_window_secs`` to a sample count.
        rolling_linear_regression_window_size :
            Override the regression window size in samples.  If None,
            derived from ``orig_sr × rolling_linear_regression_window_secs``.
        use_rms_activity :
            If True (default), compute windowed RMS from ``rlr.sumOfYSq``
            and feed it through a second regression (``rlr_rms``) to obtain
            a smoother, less impulsive ``activity_xxx``.
        """
        self.orig_sr                             = orig_sr
        self.rolling_linear_regression_window_size = rolling_linear_regression_window_size
        self.use_rms_activity                    = use_rms_activity
        self.yIntercept                          = 0.0
        self.varY                                = 0.0
        self.activity_xxx                        = 0.0
        self.elems_list                          = None
        self.tuple_heap                          = None
        self.rlr                                 = None
        self.rlr_rms                             = None
        self.rlr_xxx                             = None
        self.activity_raw_max                    = 0.0
        self.active_count                        = 0
        self.hit_count_aa                        = 0
        self.hit_count_bb                        = 0
        self.hit_count_cc                        = 0
        self.elem_min_bak                        = None
        self._dump_file_name                     = None
        self._ff                                 = None
        self._ff_buff                            = None

        for k in sorted(self.control_set_00):
            setattr(self, k, self.control_set_00[k])

        threshold_open  = 1.0
        threshold_close = self.threshold_close_perc_of_threshold_open * threshold_open
        super().__init__(threshold_open, threshold_close, self.hold_secs_min)

    def reset(self) -> None:
        """Reinitialise all rolling state while preserving control parameters.

        Recreates ``rlr`` (and optionally ``rlr_rms``) from scratch and clears
        all candidate bookkeeping structures.  Call explicitly to restart the
        gate on a new audio stream without constructing a new instance.
        """
        self.threshold_open  = 1.0
        self.threshold_close = self.threshold_close_perc_of_threshold_open * self.threshold_open
        self.hold_secs       = self.hold_secs_min

        win_size = self.rolling_linear_regression_window_size
        if win_size is None:
            win_size = int(self.orig_sr * self.rolling_linear_regression_window_secs)
        if win_size < 2:
            win_size = 2

        self.yIntercept       = 0.0
        self.varY             = 0.0
        self.activity_xxx     = 0.0
        self.elems_list       = []
        self.tuple_heap       = []
        self.activity_raw_max = 0.0
        self.active_count     = 0
        self.hit_count_aa     = 0
        self.hit_count_bb     = 0
        self.hit_count_cc     = 0
        self.elem_min_bak     = None

        # Recreate regressors
        self.rlr_xxx = self.rlr = RollingLinearRegression(win_size)
        if self.use_rms_activity:
            self.rlr_xxx = self.rlr_rms = RollingLinearRegression(win_size)
        else:
            self.rlr_rms = None

        # Reset base-class state
        self.threshold_open_pos_in_secs        = None
        self.below_threshold_close_pos_in_secs = None
        self.pos_in_secs_prev                  = float("-inf")
        self.call_count                        = 0

    def observe_ng_activity(self, activity_raw: float, pos_in_secs: float):
        """Process one observation and advance the gate.

        Parameters
        ----------
        activity_raw :
            Non-negative scalar.  Pass ``abs(sample)`` for PCM, or per-frame
            RMS when calling at frame rate.
        pos_in_secs :
            Monotonically non-decreasing timestamp in seconds.

        Returns
        -------
        (noise_gate_beg, noise_gate_end) : tuple[bool, bool]
            Gate transition flags; forwarded from ``_NoiseGateBase``.

        Side effects
        ------------
        Updates ``activity_xxx``, ``yIntercept``, ``varY``,
        ``threshold_open``, ``threshold_close``, and ``hold_secs`` in place.
        """
        if activity_raw < 0.0:
            raise ValueError(f"activity_raw must be >= 0, got {activity_raw}")

        if self.activity_raw_max < activity_raw:
            self.activity_raw_max = activity_raw

        if self.rlr is None:
            self.reset()

        self.rlr.observe_value(activity_raw, pos_in_secs)

        # Compute smoothed activity_xxx
        if self.rlr_rms is None:
            self.activity_xxx = activity_raw
        else:
            # Windowed RMS from the running sum-of-squares
            # (sumOfYSq can go slightly negative due to rolling cancellation)
            self.activity_xxx = (
                0.0 if self.rlr.sumOfYSq <= 0.0
                else math.sqrt(self.rlr.sumOfYSq / self.rlr.count)
            )
            self.rlr_rms.observe_value(self.activity_xxx, pos_in_secs)

        self.yIntercept = 0.0
        self.varY       = 0.0

        try:
            abs_slope = abs(self.rlr_xxx.slope)
        except ZeroDivisionError:
            abs_slope = float("inf")

        if abs_slope < self.abs_slope_threshold_max:
            self.hit_count_aa += 1
            try:
                self.yIntercept = yIntercept = self.rlr_xxx.yIntercept
            except ZeroDivisionError:
                self.yIntercept = yIntercept = -1e-15

            if yIntercept > 0.0:
                self.hit_count_bb += 1
                self.varY = varY = self.rlr_xxx.varY

                if varY > 0.0:
                    self.hit_count_cc += 1
                    self.active_count += 1
                    elem_new = self._HeapElem(-pos_in_secs, yIntercept, varY)
                    self.elems_list.append(elem_new)

                    sigf = 2
                    heapq.heappush(
                        self.tuple_heap,
                        (_round_sig_figs(varY, sigf), _round_sig_figs(abs_slope, sigf), elem_new)
                    )

                    # Expire entries older than history_secs_threshold_max
                    while self.elems_list:
                        elem_tmp = self.elems_list[0]
                        if -elem_tmp.ordinal > pos_in_secs - self.history_secs_threshold_max:
                            break
                        if elem_tmp.is_active:
                            self.active_count -= 1
                            elem_tmp.is_active = False
                        self.elems_list.pop(0)

                    # Get the best (lowest-variance) active element from the heap
                    elem_min = None
                    while self.tuple_heap:
                        elem_min = self.tuple_heap[0][2]
                        if elem_min.is_active:
                            break
                        heapq.heappop(self.tuple_heap)

                    self.elem_min_bak = elem_min
                    if not self.tuple_heap:
                        raise RuntimeError(
                            f"tuple_heap is empty (active_count={self.active_count})"
                        )

                    self.threshold_open  = (
                        elem_min.yIntercept
                        + self.stddev_2_add_to_threshold_open * math.sqrt(elem_min.varY)
                    )
                    self.threshold_close = (
                        self.threshold_close_perc_of_threshold_open * self.threshold_open
                    )

        # Adaptive hold time: ramp from hold_secs_min to hold_secs_max over hold_secs_dur
        if self.threshold_open_pos_in_secs is None:
            self.hold_secs = self.hold_secs_min
        else:
            t = pos_in_secs - self.threshold_open_pos_in_secs
            if t < self.hold_secs_dur:
                self.hold_secs = (
                    self.hold_secs_min
                    + (self.hold_secs_max - self.hold_secs_min) * t / self.hold_secs_dur
                )
            else:
                self.hold_secs = self.hold_secs_max

        return super().observe_ng_activity(self.activity_xxx, pos_in_secs)

    # ------------------------------------------------------------------
    # Binary file dump / replay
    # ------------------------------------------------------------------

    def file_dump_beg(self, user_header_string: str = "",
                      dump_file_name: str = "NoiseGate.bin") -> None:
        """Open a binary dump file and begin recording ``(activity_raw, pos_in_secs)`` pairs.

        The file stores a length-prefixed UTF-8 header followed by pairs of
        little-endian float64 values.  Call ``file_dump_end`` to flush and close.

        Parameters
        ----------
        user_header_string :
            Arbitrary metadata (e.g. the source audio path) written to the header.
        dump_file_name :
            Output file path.  Raises if already open.
        """
        import os, struct
        if self._dump_file_name is not None:
            raise RuntimeError("file_dump_beg called while already open")
        self._dump_file_name = os.path.abspath(dump_file_name)
        self._ff      = open(self._dump_file_name, "wb")
        self._ff_buff = []
        encoded = user_header_string.encode("utf-8")
        self._ff.write(struct.pack(f"<i{len(encoded)}s", len(encoded), encoded))
        print(f"_NoiseGate2:{self._dump_file_name} open")

    def file_dump_end(self) -> None:
        """Flush the write buffer, close the dump file, and clear dump state."""
        import struct
        if self._dump_file_name is None:
            raise RuntimeError("file_dump_end called while not open")
        ff, ff_buff, name = self._ff, self._ff_buff, self._dump_file_name
        self._ff = self._ff_buff = self._dump_file_name = None
        if ff_buff:
            ff.write(struct.pack(f"<{2*len(ff_buff)}d", *[v for vv in ff_buff for v in vv]))
        ff.close()
        print(f"_NoiseGate2:{name} close")

    def close(self) -> None:
        """Safe cleanup: close the dump file if one is open.  Idempotent."""
        if hasattr(self, "_dump_file_name") and self._dump_file_name is not None:
            self.file_dump_end()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def get_dump_file_contents(cls, file_name: str = "NoiseGate.bin"):
        """Deserialise a binary dump file produced by ``file_dump_beg``/``file_dump_end``.

        Returns
        -------
        user_header_string : str
        activity_lst       : np.ndarray shape (N,) — recorded activity_raw values
        pos_in_secs_step   : int                   — fixed step between timestamps
        """
        import os, struct
        file_name = os.path.abspath(file_name)
        with open(file_name, "rb") as ff:
            b = ff.read()
        beg  = 0
        len_n = struct.unpack("<i", b[beg:beg+4])[0]; beg += 4
        user_header_string = b[beg:beg+len_n].decode("utf-8"); beg += len_n
        n_doubles = (len(b) - beg) // 8
        pairs = struct.unpack(f"<{n_doubles}d", b[beg:])
        if len(b) != beg + n_doubles * 8:
            raise ValueError(f"file length mismatch: {len(b)} vs {beg + n_doubles * 8}")
        data = np.array(pairs).reshape(-1, 2)
        pos_in_secs_step = None
        activity_lst = []
        for k, row in enumerate(data):
            activity_lst.append(float(row[0]))
            if k > 0:
                step = int(data[k, 1] - data[k-1, 1])
                if pos_in_secs_step is None and step > 0:
                    pos_in_secs_step = step
                elif pos_in_secs_step is not None and step > 0 and step != pos_in_secs_step:
                    raise ValueError(
                        f"non-uniform pos_in_secs_step at k={k}: {step} vs {pos_in_secs_step}"
                    )
        if pos_in_secs_step is None:
            raise ValueError(
                f"could not determine pos_in_secs_step from {len(data)} rows in '{file_name}'"
            )
        return user_header_string, np.array(activity_lst), pos_in_secs_step

    # ------------------------------------------------------------------
    # Testbed (interactive use; dependencies not required at import time)
    # ------------------------------------------------------------------

    @classmethod
    def testbed_run_one(cls, ng2_main, y_pad: np.ndarray, orig_sr: int, file_name: str,
                        y_pad_noise: np.ndarray = None,
                        y_pad_play_audio: np.ndarray = None,
                        orig_sr_play_audio: int = None,
                        ng2_cshp=None) -> None:
        """Run the gate sample-by-sample on one audio array and log segment events.

        Iterates through ``y_pad``, calling ``observe_ng_activity(abs(sample),
        pos_in_secs)`` for each sample.  On gate-close events the non-silence
        segment is logged (and optionally played back if ``NpAudioPlayer`` is
        importable).

        Parameters
        ----------
        ng2_main :
            A ``_NoiseGate2`` instance to exercise.
        y_pad :
            1-D numpy audio array.
        orig_sr :
            Sample rate of ``y_pad`` (Hz).
        file_name :
            Label used in log output.
        y_pad_noise :
            If not None, mixed into ``y_pad`` at a dynamically varying scale
            to simulate background noise.
        y_pad_play_audio :
            If not None, playback on gate-close events uses this array
            (at ``orig_sr_play_audio``) rather than ``y_pad``.
        ng2_cshp :
            Optional C# port implementing the same interface; enables
            per-sample numerical cross-checking.
        """
        import datetime, sys

        # Optional visualisation / audio dependencies
        try:
            from pcm_plotter import PCMPlotter  # type: ignore
        except ImportError:
            PCMPlotter = None
        try:
            from np_audio_player import NpAudioPlayer  # type: ignore
        except ImportError:
            NpAudioPlayer = None

        pos_in_secs_step     = 1.0 / orig_sr
        samples_per_chunk    = int(200.0 * orig_sr)   # plot update interval
        include_semilogy     = True

        y_pad_max_abs  = float(np.max(np.abs(y_pad)))
        y_pad_mean_abs = float(np.mean(np.abs(y_pad)))
        print("min max avg sr sec file %.8e %.8e %.8e %6d %.3e %s" % (
            float(np.min(np.abs(y_pad))), y_pad_max_abs, y_pad_mean_abs,
            orig_sr, len(y_pad) / orig_sr, file_name))

        pcmp = (
            PCMPlotter(y_pad_mean_abs=0.1,
                       samples_per_plot_chunk=samples_per_chunk,
                       include_semilogy_plots=include_semilogy)
            if PCMPlotter is not None else None
        )

        if y_pad_noise is not None:
            if y_pad_max_abs > 1.0:
                raise ValueError("np.max(np.abs(y_pad)) > 1.0")
            noise_adj   = 0.0045 - 0.002 * y_pad_max_abs
            y_pad_noise = y_pad_noise.copy() * noise_adj / float(np.mean(np.abs(y_pad_noise)))
            y_pn_pos, y_pn_nxt  = 0, 0
            y_pn_scale           = noise_adj * 80.0
            y_pn_scale_min       = y_pn_scale
            y_pn_scale_max       = y_pn_scale * 30.0
            y_pn_scale_sgn       = -1.0

        log_count = 0
        t_beg = t_nxt = datetime.datetime.now()
        t_tot = 0.0
        k_beg = k_end = 0
        pos_in_secs = 0.0

        def _play_audio(which: str, k_b: int, k_e: int) -> None:
            nonlocal log_count, t_tot, t_nxt
            t_tot += (datetime.datetime.now() - t_nxt).total_seconds()
            if y_pad_play_audio is None:
                yy, sr_play = y_pad[k_b:k_e], orig_sr
                k_bx, k_ex = k_b, k_e
            else:
                k_bx = int((k_b / orig_sr) * orig_sr_play_audio)
                k_ex = int((k_e / orig_sr) * orig_sr_play_audio)
                yy, sr_play = y_pad_play_audio[k_bx:k_ex], orig_sr_play_audio
            duration = (k_e - k_b) / orig_sr
            if log_count % 20 == 0:
                print(f"{'which':<20} {'k_beg':>8} {'k_bx':>8} {'k_ex':>8} "
                      f"{'smp/sec':>12} {'start_s':>10} {'dur_s':>8} {'mean_yy':>10}")
            print(f"{which:<20} {k_b:>8d} {k_bx:>8d} {k_ex:>8d} "
                  f"{(k_e / t_tot if t_tot > 0 else 0):>12.2f} "
                  f"{k_b / orig_sr:>10.4f} {duration:>8.4f} "
                  f"{float(np.mean(np.abs(yy))):>10.6f}  {file_name}", end="")
            sys.stdout.flush()
            log_count += 1
            if NpAudioPlayer is not None and duration >= 0.2:
                ap = NpAudioPlayer(yy.dtype, sr_play, silent=True)
                ap.start_play(yy)
                while ap.is_alive():
                    ap.join(0.001)
                ap.kill()
            print("")
            t_nxt = datetime.datetime.now()

        noise_gate_beg_x = noise_gate_end_x = False
        y_pad = y_pad.copy()
        k = 0
        for k, activity_raw in enumerate(y_pad):
            if y_pad_noise is not None:
                if y_pn_pos >= len(y_pad_noise):
                    y_pn_pos, y_pn_nxt = 0, 0
                if y_pn_pos >= y_pn_nxt:
                    y_pn_nxt += int(orig_sr * 0.25)
                    if y_pn_scale_sgn > 0:
                        y_pn_scale *= 1.05
                    else:
                        y_pn_scale /= 1.05
                    if y_pn_scale < y_pn_scale_min:
                        y_pn_scale_sgn = 1.0
                    elif y_pn_scale > y_pn_scale_max:
                        y_pn_scale_sgn = -1.0
                y_pad[k]    += y_pn_scale * y_pad_noise[y_pn_pos]
                y_pn_pos    += 1
                activity_raw = y_pad[k]

            activity_abs = abs(float(activity_raw))
            noise_gate_beg_o, noise_gate_end_o = ng2_main.observe_ng_activity(
                activity_abs, pos_in_secs
            )
            if ng2_cshp is not None:
                noise_gate_beg_c, noise_gate_end_c = ng2_cshp.observe_ng_activity(
                    activity_abs, pos_in_secs
                )
            noise_gate_beg_x = noise_gate_beg_o
            noise_gate_end_x = noise_gate_end_o

            if noise_gate_beg_x:
                if noise_gate_end_x:
                    raise RuntimeError("noise_gate_beg_x and noise_gate_end_x both True")
                k_beg = k
            elif noise_gate_end_x:
                k_end = k
                _play_audio("non-silence", k_beg, k_end)

            if pcmp is not None:
                pcmp.observe_pcm_activity(
                    k, ng2_main.activity_xxx,
                    noise_gate_beg=noise_gate_beg_x, noise_gate_end=noise_gate_end_x,
                    activity2=None, ng2=ng2_main,
                    subplot_activity_lst_=[ng2_main.yIntercept, ng2_main.varY]
                )
            pos_in_secs += pos_in_secs_step

        if pcmp is not None:
            pcmp.observe_pcm_activity(
                k, ng2_main.activity_xxx,
                noise_gate_beg=noise_gate_beg_x, noise_gate_end=noise_gate_end_x,
                activity2=None, ng2=ng2_main,
                subplot_activity_lst_=[ng2_main.yIntercept, ng2_main.varY],
                force=True
            )
        elapsed = t_tot + (datetime.datetime.now() - t_nxt).total_seconds()
        print("%9d %7.2f %15.2f" % (k, elapsed, 0.0 if elapsed == 0.0 else k / elapsed))

    @classmethod
    def testbed(cls, c_sharp_interface=None) -> None:
        """Interactive testbed: load audio files and run ``testbed_run_one``.

        Hard-coded file paths target the original development environment;
        adjust ``fnl`` and noise paths as needed.

        Requires: librosa.
        Optional: PCMPlotter, NpAudioPlayer, c_sharp_interface.
        """
        import importlib, os
        try:
            import librosa  # type: ignore
        except ImportError as exc:
            raise ImportError("testbed requires librosa: pip install librosa") from exc

        sample_rate = 16000
        _cache_orig: dict = {}
        _cache_16k:  dict = {}

        def _get_y_pad(path: str):
            if path not in _cache_orig:
                _cache_orig[path] = librosa.load(path, sr=None, mono=True)
            y, sr = _cache_orig[path]
            if path not in _cache_16k:
                _cache_16k[path] = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
            return _cache_16k[path], sample_rate

        def _fnl(path: str):
            if os.path.isfile(path):
                y, sr = _get_y_pad(path)
                yield path, (y, sr, None, None, True)
            elif os.path.isdir(path):
                for fn in sorted(os.listdir(path)):
                    p = os.path.abspath(os.path.join(path, fn))
                    if os.path.isfile(p):
                        y, sr = _get_y_pad(p)
                        yield p, (y, sr, None, None, True)
            else:
                raise FileNotFoundError(path)

        # NOTE: adjust these paths to your local audio files before running
        y_pad_noise = None  # set to np.ndarray to enable noise mixing
        fnl = _fnl(
            r"C:\lexi_core\SynologyDrive\voices\spanish_adhoc"
            r"\like_and_unlike_to_abajo_etc_umidigi.mp3"
        )

        for file_name, (y_pad, orig_sr, y_pad_play_audio, orig_sr_play_audio, use_rms) in fnl:
            ng2_main = cls(orig_sr, use_rms_activity=use_rms)
            ng2_cshp = (
                None if c_sharp_interface is None
                else c_sharp_interface._NoiseGate2(orig_sr, use_rms_activity=use_rms)
            )
            cls.testbed_run_one(
                ng2_main, y_pad, orig_sr, file_name,
                y_pad_noise=y_pad_noise,
                y_pad_play_audio=y_pad_play_audio,
                orig_sr_play_audio=orig_sr_play_audio,
                ng2_cshp=ng2_cshp,
            )


# ---------------------------------------------------------------------------
# Frame-level adapter
# ---------------------------------------------------------------------------

@dataclass
class RollingLinGateConfig:
    """Configuration for the rolling linear regression quiet-region gate adapter.

    The gate operates at frame rate (one RMS value per 20 ms frame) rather
    than sample rate.  The regression window and history are specified in
    seconds; they are converted to frame counts internally.

    Parameters
    ----------
    stddev_open :
        Gate-open threshold expressed as a multiple of the estimated noise
        standard deviation added to the estimated noise floor::

            threshold_open = best.yIntercept + stddev_open * sqrt(best.varY)

    close_perc :
        Close threshold as a fraction of the open threshold
        (threshold_close = close_perc × threshold_open).
    history_secs :
        How far back (seconds) regression snapshots are retained as
        quiet-region candidates.
    abs_slope_max :
        Maximum |slope| of the regression line to accept a window as a
        quiet-region candidate (units: RMS / second).
    win_secs :
        Regression window duration (seconds).  Controls how many consecutive
        frames are included in each least-squares fit.
    hold_secs_min :
        Minimum gate hold time immediately after opening (seconds).
    hold_secs_max :
        Gate hold time after ``hold_secs_dur`` has elapsed (seconds).
    hold_secs_dur :
        Time (seconds) over which hold ramps from min to max after gate-open.
    use_rms_activity :
        If True (default), the windowed RMS of the first regression feeds
        a second regression, giving a smoother activity signal.
    """
    stddev_open:      float = 150.0
    close_perc:       float = 0.4
    history_secs:     float = 1.0
    abs_slope_max:    float = 0.0001
    win_secs:         float = 0.5
    hold_secs_min:    float = 0.05
    hold_secs_max:    float = 0.3
    hold_secs_dur:    float = 0.15
    use_rms_activity: bool  = True


class RollingLinGate:
    """Frame-level adapter wrapping ``_NoiseGate2`` in the standard gate interface.

    Converts each 20 ms frame to a single RMS value and feeds it to
    ``_NoiseGate2.observe_ng_activity`` (frame rate ≈ 50 Hz rather than
    16 kHz).  This keeps the sweep tractable while preserving the core
    rolling-regression quiet-region algorithm.

    For sample-level processing (e.g. interactive debugging or full
    fidelity evaluation), use ``_NoiseGate2.testbed_run_one`` directly.
    """

    def __init__(self, config: Optional[RollingLinGateConfig] = None):
        self.config = config or RollingLinGateConfig()
        self._sr         = 16000
        self._frame_size = 320
        self._ng2: Optional[_NoiseGate2] = None
        self._reset_state()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, sample_rate: int = 16000, frame_size: int = 320) -> None:
        self._sr         = sample_rate
        self._frame_size = frame_size
        self._reset_state()

    def process_frame(self, frame: np.ndarray, frame_start_sec: float) -> None:
        frame_sec = self._frame_size / self._sr
        frame_mid = frame_start_sec + frame_sec / 2.0

        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        ng_beg, ng_end = self._ng2.observe_ng_activity(rms, frame_mid)

        if ng_beg:
            self._seg_start = frame_start_sec
        elif ng_end:
            if self._seg_start is not None:
                self._segments.append(
                    [round(self._seg_start, 4), round(frame_start_sec, 4)]
                )
                self._seg_start = None

        self._trace.append({
            "time_sec":        round(frame_start_sec, 4),
            "activity_raw":    round(rms, 6),
            "activity_xxx":    round(self._ng2.activity_xxx, 6),
            "yIntercept":      round(self._ng2.yIntercept, 6),
            "varY":            round(self._ng2.varY, 8),
            "threshold_open":  round(self._ng2.threshold_open, 6),
            "threshold_close": round(self._ng2.threshold_close, 6),
            "gate_state":      1 if self._ng2.threshold_open_pos_in_secs is not None else 0,
        })
        self._last_frame_end = frame_start_sec + frame_sec

    def finish(self) -> None:
        if self._ng2 is not None and self._ng2.threshold_open_pos_in_secs is not None:
            if self._seg_start is not None:
                self._segments.append(
                    [round(self._seg_start, 4), round(self._last_frame_end, 4)]
                )
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

        # Convert seconds → samples for the regression window
        win_samples = max(2, int(self._sr * cfg.win_secs))

        self._ng2 = _NoiseGate2(self._sr,
                                rolling_linear_regression_window_size=win_samples,
                                use_rms_activity=cfg.use_rms_activity)
        self._ng2.abs_slope_threshold_max                = cfg.abs_slope_max
        self._ng2.history_secs_threshold_max             = cfg.history_secs
        self._ng2.stddev_2_add_to_threshold_open         = cfg.stddev_open
        self._ng2.threshold_close_perc_of_threshold_open = cfg.close_perc
        self._ng2.hold_secs_min                          = cfg.hold_secs_min
        self._ng2.hold_secs_max                          = cfg.hold_secs_max
        self._ng2.hold_secs_dur                          = cfg.hold_secs_dur
        self._ng2.reset()

        self._seg_start:    Optional[float] = None
        self._segments:     list[list[float]] = []
        self._trace:        list[dict] = []
        self._last_frame_end: float = 0.0


def run_gate_rolling_lin(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_size: int = 320,
    config: Optional[RollingLinGateConfig] = None,
) -> tuple[list[list[float]], list[dict]]:
    """Run the rolling linear regression gate on *audio* and return (segments, trace)."""
    gate = RollingLinGate(config)
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
