"""
Feature extraction and detector assembly for the offline chatter pipeline.

This module is where raw signals are converted into frame-wise features, band scores,
state-machine outputs, and operator-friendly control columns. It is the core of the
signal-processing side of the project.
"""

import numpy as np
import pandas as pd

from .candidate_logic import add_band_decisions
from .events import extract_events
from .state_logic import apply_state_machine


def dominant_state_rank(state: str) -> int:
    """
    Give each detector state an integer rank so states can be compared safely.
    """
    return {"STABLE": 0, "WATCH": 1, "WARNING": 2, "ALARM": 3, "SAFETY": 4}.get(str(state), 0)


def robust_nanmedian(x: pd.Series) -> float:
    """
    Median helper that ignores NaNs and non-numeric values.
    """
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _baseline_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the early-run slice used for baseline and safety calibration.
    """
    from .config import CFG

    n = len(df)
    if n == 0:
        return df
    count = int(round(max(CFG.baseline_min_frames, CFG.baseline_fraction * n)))
    count = min(max(count, 1), min(CFG.baseline_max_frames, n))
    return df.iloc[:count].copy()


def calibrate_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """
    Estimate run-specific safety thresholds from the baseline slice.
    
    At the moment this mainly derives a robust RMS threshold used by the state machine's
    safety override.
    """
    baseline = _baseline_slice(df)
    vib = pd.to_numeric(baseline["vib_rms_mean"], errors="coerce")
    vib = vib[np.isfinite(vib)]

    if vib.empty:
        safety_rms = 0.03
    else:
        med = float(np.median(vib))
        q99 = float(np.quantile(vib, 0.99))
        mad = float(np.median(np.abs(vib - med)) + 1e-12)
        robust_thr = med + 12.0 * mad
        safety_rms = max(q99, robust_thr, 0.03)

    return {
        "safety_rms": safety_rms,
        "baseline_frames_used": int(len(baseline)),
    }


def extract_features(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Turn raw sensor streams into frame-wise spectral and statistical features.
    
    This is the main feature-engineering step of the detector. It preprocesses the signals,
    frames them, tracks candidate peaks in the third and fifth chatter bands, computes
    quality/growth/coherence features, and then applies the downstream decision logic.
    """
    from .config import CFG
    from .signal_utils import (
        preprocess_signal,
        frame_signal,
        compute_fft,
        band_energy,
        compute_alpha,
        coherence_in_band,
        corrcoef_safe,
        freq_stability,
        spectral_flatness,
    )
    from .tracking import TrackManager, pick_tracked_peak
    from scipy.stats import kurtosis, skew
    import pandas as pd

    fs = float(data["fs"])
    os_sig = preprocess_signal(data["os"], fs, CFG.highpass_hz)
    ds_sig = preprocess_signal(data["ds"], fs, CFG.highpass_hz)
    os_frames = frame_signal(os_sig, fs, CFG.window_sec, CFG.hop_sec)
    ds_frames = frame_signal(ds_sig, fs, CFG.window_sec, CFG.hop_sec)

    if len(os_frames) == 0:
        raise ValueError("Signal too short for chosen window/hop.")

    def aux_frames(key: str):
        """
        Frame an auxiliary process signal so it lines up with the vibration windows.
        """
        arr = np.asarray(data[key], dtype=float)
        xf = frame_signal(arr, fs, CFG.window_sec, CFG.hop_sec)
        return xf if len(xf) == len(os_frames) else None

    spd_frames = aux_frames("speed")
    ten_frames = aux_frames("tension")
    thk_frames = aux_frames("thickness")
    time_arr = pd.to_datetime(data["time"])

    alpha_n = max(2, int(round(CFG.alpha_horizon_sec / CFG.hop_sec)))
    e3_hist, e5_hist = [], []
    freq_hist3, freq_hist5 = [], []
    track3 = TrackManager("third")
    track5 = TrackManager("fifth")
    prev_e3 = prev_e5 = None
    rows = []

    for i, (fr_os, fr_ds) in enumerate(zip(os_frames, ds_frames)):
        x_mean = 0.5 * (fr_os + fr_ds)

        vib_rms_os = float(np.sqrt(np.mean(fr_os ** 2)))
        vib_rms_ds = float(np.sqrt(np.mean(fr_ds ** 2)))
        vib_rms_mean = float(np.sqrt(np.mean(x_mean ** 2)))

        freqs, mag, power = compute_fft(x_mean, fs)
        e_wide = band_energy(freqs, power, CFG.wideband)

        c3 = pick_tracked_peak(
            freqs, mag, power, CFG.third_search,
            track3.smoothed_freq, track3.active_strength, "third"
        )
        c5 = pick_tracked_peak(
            freqs, mag, power, CFG.fifth_search,
            track5.smoothed_freq, track5.active_strength, "fifth"
        )

        t3 = track3.update(c3)
        t5 = track5.update(c5)

        e3 = band_energy(
            freqs, power,
            (t3["freq"] - CFG.narrow_band_halfwidth_hz, t3["freq"] + CFG.narrow_band_halfwidth_hz)
        ) if np.isfinite(t3["freq"]) else 0.0

        e5 = band_energy(
            freqs, power,
            (t5["freq"] - CFG.narrow_band_halfwidth_hz, t5["freq"] + CFG.narrow_band_halfwidth_hz)
        ) if np.isfinite(t5["freq"]) else 0.0

        # light smoothing to reduce spike-driven growth
        e3_hist.append(e3)
        e5_hist.append(e5)
        e3_smooth = float(np.mean(e3_hist[-3:]))
        e5_smooth = float(np.mean(e5_hist[-3:]))

        g3 = (e3_smooth / prev_e3) if (prev_e3 is not None and prev_e3 > 1e-12) else 1.0
        g5 = (e5_smooth / prev_e5) if (prev_e5 is not None and prev_e5 > 1e-12) else 1.0
        prev_e3, prev_e5 = e3_smooth, e5_smooth

        a3 = compute_alpha([max(v, 1e-12) for v in e3_hist[-alpha_n:]], CFG.hop_sec, min(len(e3_hist), alpha_n)) if len(e3_hist) >= 2 else 0.0
        a5 = compute_alpha([max(v, 1e-12) for v in e5_hist[-alpha_n:]], CFG.hop_sec, min(len(e5_hist), alpha_n)) if len(e5_hist) >= 2 else 0.0

        a3 = float(np.clip(a3, -2.0, 2.0))
        a5 = float(np.clip(a5, -2.0, 2.0))

        coh3 = coherence_in_band(fr_os, fr_ds, fs, CFG.third_search)
        coh5 = coherence_in_band(fr_os, fr_ds, fs, CFG.fifth_search)

        freq_hist3.append(float(t3["freq"]) if np.isfinite(t3["freq"]) else np.nan)
        freq_hist5.append(float(t5["freq"]) if np.isfinite(t5["freq"]) else np.nan)

        stab3 = freq_stability(freq_hist3, CFG.track_stability_frames_third)
        stab5 = freq_stability(freq_hist5, CFG.track_stability_frames_fifth)

        consistent3 = int(np.isfinite(t3["freq"]) and stab3 <= CFG.min_freq_stability_third_hz)
        consistent5 = int(np.isfinite(t5["freq"]) and stab5 <= CFG.min_freq_stability_fifth_hz)

        row = {
            "frame": i,
            "t_sec": float(i * CFG.hop_sec),
            "timestamp": pd.Timestamp(
                time_arr[min(int(round(i * CFG.hop_sec * fs + (CFG.window_sec * fs) / 2.0)), len(time_arr) - 1)]
            ).isoformat(),
            "vib_rms_os": vib_rms_os,
            "vib_rms_ds": vib_rms_ds,
            "vib_rms_mean": vib_rms_mean,
            "rms_ratio_os_ds": float(max(vib_rms_os, vib_rms_ds) / (min(vib_rms_os, vib_rms_ds) + 1e-12)),
            "corr_os_ds": corrcoef_safe(fr_os, fr_ds),
            "spec_flatness": spectral_flatness(mag),
            "kurtosis": float(kurtosis(x_mean, fisher=False, bias=False)),
            "skewness": float(skew(x_mean, bias=False)),
            "e_wide": e_wide,
            "e_third": e3_smooth,
            "e_fifth": e5_smooth,
            "alpha_third": a3,
            "alpha_fifth": a5,
            "growth_third": g3,
            "growth_fifth": g5,
            "dom_third_hz": t3["freq"],
            "dom_fifth_hz": t5["freq"],
            "dom_third_raw_hz": t3.get("raw_freq", np.nan),
            "dom_fifth_raw_hz": t5.get("raw_freq", np.nan),
            "third_valid": float(t3["valid"]),
            "fifth_valid": float(t5["valid"]),
            "third_held": int(t3.get("held", 0)),
            "fifth_held": int(t5.get("held", 0)),
            "third_energy_ratio": t3["energy_ratio"],
            "fifth_energy_ratio": t5["energy_ratio"],
            "third_prominence_ratio": t3["prominence_ratio"],
            "fifth_prominence_ratio": t5["prominence_ratio"],
            "third_snr_db": t3["snr_db"],
            "fifth_snr_db": t5["snr_db"],
            "third_local_concentration": t3["local_concentration"],
            "fifth_local_concentration": t5["local_concentration"],
            "third_freq_stability_hz": stab3,
            "fifth_freq_stability_hz": stab5,
            "third_consistent": consistent3,
            "fifth_consistent": consistent5,
            "coh_third": coh3,
            "coh_fifth": coh5,
        }

        row["impulse_like"] = int(
            row["kurtosis"] > 5.5
            and row["spec_flatness"] > 0.45
            and max(row["third_local_concentration"], row["fifth_local_concentration"]) < 0.50
        )

        if spd_frames is not None:
            row["speed_mean"] = float(np.mean(spd_frames[i]))
            row["speed_std"] = float(np.std(spd_frames[i]))
        if ten_frames is not None:
            row["tension_mean"] = float(np.mean(ten_frames[i]))
            row["tension_std"] = float(np.std(ten_frames[i]))
        if thk_frames is not None:
            row["thickness_mean"] = float(np.mean(thk_frames[i]))
            row["thickness_std"] = float(np.std(thk_frames[i]))

        rows.append(row)

    df = pd.DataFrame(rows)
    thresholds = calibrate_thresholds(df)
    df = add_band_decisions(df)
    df = apply_state_machine(df, safety_rms=thresholds["safety_rms"])

    if "control_message" not in df.columns:
        df["control_message"] = df["state_message"]
    if "control_level" not in df.columns:
        df["control_level"] = df["control_action"]
    if "speed_setpoint" not in df.columns:
        df["speed_setpoint"] = df["speed_mean"] if "speed_mean" in df.columns else np.nan
    if "tension_setpoint" not in df.columns:
        df["tension_setpoint"] = df["tension_mean"] if "tension_mean" in df.columns else np.nan

    return df