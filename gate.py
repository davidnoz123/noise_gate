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
"""

import collections
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
