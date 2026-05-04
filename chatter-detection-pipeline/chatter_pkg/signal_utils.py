"""
Low-level signal-processing primitives used across the chatter pipeline.

These functions stay intentionally small and reusable: filtering, framing, FFT-based
measures, coherence, stability, and other numerical helpers live here.
"""

import math
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


def butter_filter(x: np.ndarray, fs: float, kind: str, f0: float, f1: Optional[float] = None, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth high-pass or band-pass filter to a signal.
    """
    nyq = 0.5 * fs
    if kind == "highpass":
        b, a = signal.butter(order, f0 / nyq, btype="highpass")
    elif kind == "bandpass":
        assert f1 is not None
        b, a = signal.butter(order, [f0 / nyq, f1 / nyq], btype="bandpass")
    else:
        raise ValueError(kind)
    return signal.filtfilt(b, a, x)


def preprocess_signal(x: np.ndarray, fs: float, highpass_hz: float) -> np.ndarray:
    """
    Detrend and high-pass filter the raw vibration signal before analysis.
    """
    x = np.asarray(x, dtype=float)
    x = signal.detrend(x, type="constant")
    x = butter_filter(x, fs, "highpass", highpass_hz, order=3)
    return x.astype(np.float32)


def frame_signal(x: np.ndarray, fs: float, window_sec: float, hop_sec: float) -> np.ndarray:
    """
    Split a 1D signal into overlapping analysis windows.
    """
    win = int(round(window_sec * fs))
    hop = int(round(hop_sec * fs))
    if len(x) < win:
        return np.empty((0, win), dtype=float)
    starts = np.arange(0, len(x) - win + 1, hop)
    return np.stack([x[s:s + win] for s in starts], axis=0)


def compute_fft(frame: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute frequency bins, magnitude spectrum, and power spectrum for one frame.
    """
    w = signal.get_window("hann", frame.size, fftbins=True)
    X = rfft(frame * w)
    freqs = rfftfreq(frame.size, d=1.0 / fs)
    mag = np.abs(X)
    power = mag ** 2
    return freqs, mag, power


def band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """
    Return a boolean mask for frequencies that fall inside a requested band.
    """
    return (freqs >= band[0]) & (freqs < band[1])


def band_energy(freqs: np.ndarray, power: np.ndarray, band: Tuple[float, float]) -> float:
    """
    Sum spectral power inside a frequency band.
    """
    idx = band_mask(freqs, band)
    return float(np.sum(power[idx])) if np.any(idx) else 0.0


def compute_alpha(hist: list[float], hop_sec: float, n_frames: int) -> float:
    """
    Estimate short-horizon exponential growth from recent band-energy history.
    """
    if len(hist) < n_frames:
        return 0.0
    y = np.log(np.maximum(np.asarray(hist[-n_frames:]), 1e-12))
    t = np.arange(n_frames) * hop_sec
    num = float(np.sum((t - t.mean()) * (y - y.mean())))
    den = float(np.sum((t - t.mean()) ** 2) + 1e-12)
    return num / den


def spectral_flatness(mag: np.ndarray) -> float:
    """
    Measure how tone-like or noise-like a spectrum is.
    """
    m = np.maximum(mag, 1e-12)
    return float(np.exp(np.mean(np.log(m))) / (np.mean(m) + 1e-12))


def coherence_in_band(x: np.ndarray, y: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    """
    Compute mean cross-sensor coherence inside one frequency band.
    """
    f, cxy = signal.coherence(x, y, fs=fs, nperseg=min(1024, len(x)))
    idx = band_mask(f, band)
    return float(np.nanmean(cxy[idx])) if np.any(idx) else 0.0


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    """
    Correlation helper that avoids division issues for nearly constant signals.
    """
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def freq_stability(history: list[float], n: int) -> float:
    """
    Measure how stable a tracked frequency has been over recent frames.
    """
    vals = np.asarray(history[-n:], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.inf
    return float(np.std(vals))


def ema_update(prev: Optional[float], current: Optional[float], alpha: float) -> float:
    """
    Update an exponential moving average while handling missing values cleanly.
    """
    if current is None or not np.isfinite(current):
        return float(prev) if prev is not None and np.isfinite(prev) else np.nan
    if prev is None or not np.isfinite(prev):
        return float(current)
    return float(alpha * current + (1.0 - alpha) * prev)
