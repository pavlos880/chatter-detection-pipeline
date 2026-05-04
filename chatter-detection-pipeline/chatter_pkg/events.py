"""
Helpers for converting frame-wise detector states into event-level summaries.

A run may contain thousands of frames but only a few meaningful chatter episodes.
This module groups those frames into human-readable events.
"""

import numpy as np
import pandas as pd

from .config import CFG
from .state_logic import dominant_state_rank


def _robust_nanmedian(x: pd.Series) -> float:
    """
    Median helper used when event summaries contain missing numeric values.
    """
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _build_event(seg: pd.DataFrame, state_col: str = "state"):
    """
    Summarise one contiguous active segment into a single event dictionary.
    """
    if seg.empty:
        return None

    start_t = float(seg["t_sec"].iloc[0])
    end_t = float(seg["t_sec"].iloc[-1])
    duration = float(end_t - start_t + CFG.hop_sec)

    if duration < CFG.min_event_duration_sec:
        return None

    peak_idx = int(np.argmax(seg["risk_score"].to_numpy()))
    peak_row = seg.iloc[peak_idx]
    peak_state = max(seg[state_col].astype(str), key=dominant_state_rank)

    watch_start = seg.loc[seg[state_col] == "WATCH", "t_sec"]
    warn_start = seg.loc[seg[state_col] == "WARNING", "t_sec"]
    alarm_start = seg.loc[seg[state_col].isin(["ALARM", "SAFETY"]), "t_sec"]

    dom_band = str(peak_row.get("dominant_band", "none"))
    dom_freq = _robust_nanmedian(seg["dom_third_hz"] if dom_band == "third" else seg["dom_fifth_hz"])

    return {
        "start_t_sec": start_t,
        "end_t_sec": end_t,
        "duration_sec": duration,
        "watch_start_t_sec": float(watch_start.iloc[0]) if len(watch_start) else np.nan,
        "warning_start_t_sec": float(warn_start.iloc[0]) if len(warn_start) else np.nan,
        "alarm_start_t_sec": float(alarm_start.iloc[0]) if len(alarm_start) else np.nan,
        "peak_risk_t_sec": float(peak_row["t_sec"]),
        "peak_risk_score": float(seg["risk_score"].max()),
        "peak_state": peak_state,
        "dominant_band": dom_band,
        "dominant_frequency_hz": dom_freq,
        "valid_fraction": float(
            ((seg["dominant_band"] != "none") & np.isfinite(seg["dominant_frequency_hz"])).mean()
        ),
        "max_vib_rms": float(seg["vib_rms_mean"].max()),
    }


def _merge_events(events: list[dict]) -> list[dict]:
    """
    Merge nearby events separated by only a short quiet gap.
    """
    if not events:
        return []

    merged = [events[0]]

    for ev in events[1:]:
        prev = merged[-1]

        if ev["start_t_sec"] - prev["end_t_sec"] <= CFG.min_event_gap_sec:
            prev["end_t_sec"] = max(prev["end_t_sec"], ev["end_t_sec"])
            prev["duration_sec"] = float(prev["end_t_sec"] - prev["start_t_sec"] + CFG.hop_sec)

            if ev["peak_risk_score"] >= prev["peak_risk_score"]:
                prev["peak_risk_score"] = ev["peak_risk_score"]
                prev["peak_risk_t_sec"] = ev["peak_risk_t_sec"]
                prev["dominant_band"] = ev["dominant_band"]
                prev["dominant_frequency_hz"] = ev["dominant_frequency_hz"]

            prev["peak_state"] = max([prev["peak_state"], ev["peak_state"]], key=dominant_state_rank)
            prev["max_vib_rms"] = max(prev["max_vib_rms"], ev["max_vib_rms"])
        else:
            merged.append(ev)

    return merged


def extract_events(df: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    """
    Extract event-level chatter episodes from the frame-wise state column.
    """
    raw = []
    active = False
    start_idx = None

    for i, row in df.iterrows():
        is_on = row[state_col] in ("WARNING", "ALARM", "SAFETY")

        if is_on and not active:
            active = True
            start_idx = i
        elif not is_on and active:
            ev = _build_event(df.iloc[start_idx:i].copy(), state_col=state_col)
            if ev is not None:
                raw.append(ev)
            active = False
            start_idx = None

    if active and start_idx is not None:
        ev = _build_event(df.iloc[start_idx:].copy(), state_col=state_col)
        if ev is not None:
            raw.append(ev)

    out = pd.DataFrame(_merge_events(raw))
    if out.empty:
        return out

    start = pd.to_numeric(out["start_t_sec"], errors="coerce")
    warn = pd.to_numeric(out["warning_start_t_sec"], errors="coerce")
    alarm = pd.to_numeric(out["alarm_start_t_sec"], errors="coerce")
    watch = pd.to_numeric(out["watch_start_t_sec"], errors="coerce")

    out["warning_to_alarm_sec"] = alarm - warn
    out["watch_to_warning_sec"] = warn - watch
    out["lead_time_sec"] = start - warn

    return out