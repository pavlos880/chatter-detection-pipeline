"""
Simple advisory layer built on top of the detector state.

This module does NOT close the loop or change the plant automatically.
It only turns detector states into conservative operator-facing suggestions.
"""

import numpy as np
import pandas as pd


# Speed reduction factors per state.
# These are conservative prototype values — not validated against the mill's
# actual stability boundary.
_SPEED_FACTORS = {
    "STABLE":  1.00,
    "WATCH":   1.00,
    "WARNING": 0.98,   # −2% — prepare the operator
    "ALARM":   0.95,   # −5% — act now
    "SAFETY":  0.95,   # same as ALARM for advisory layer
}

_CONTROL_MODES = {
    "STABLE":  "HOLD",
    "WATCH":   "WATCH",
    "WARNING": "PREPARE",
    "ALARM":   "SLOWDOWN",
    "SAFETY":  "SLOWDOWN",
}

_CONTROL_TEXTS = {
    "HOLD":     "Stable. Hold nominal settings.",
    "WATCH":    "Watch state. Monitor trend and prepare corrective action if needed.",
    "PREPARE":  "Warning state. Prepare small speed reduction if persistence continues.",
    "SLOWDOWN": "Alarm or safety state. Recommend operator review and speed reduction.",
}


def apply_simple_advisory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map detector state to recommended speed setpoints and operator messages.

    All operations are vectorized (no Python-level row loop).

    Columns added:
      recommended_speed_setpoint   — speed_mean × state factor (NaN if no speed data)
      recommended_tension_setpoint — tension_mean unchanged (NaN if no tension data)
      recommended_control_mode     — one of HOLD / WATCH / PREPARE / SLOWDOWN
      recommended_control_text     — human-readable message string
      speed_setpoint               — alias for recommended_speed_setpoint
      tension_setpoint             — alias for recommended_tension_setpoint
      control_message              — alias for recommended_control_text
      control_level                — alias for recommended_control_mode
    """
    out = df.copy()

    # Ensure the process columns exist even if they were not in the raw data
    if "speed_mean" not in out.columns:
        out["speed_mean"] = np.nan
    if "tension_mean" not in out.columns:
        out["tension_mean"] = np.nan

    # ── Speed factor: vectorized lookup ─────────────────────────────────────
    # Map each state to its factor, default 1.0 for unknown states
    speed_factors = out["state"].map(_SPEED_FACTORS).fillna(1.0)

    # ── Control mode and text ───────────────────────────────────────────────
    control_modes = out["state"].map(_CONTROL_MODES).fillna("HOLD")
    control_texts = control_modes.map(_CONTROL_TEXTS).fillna(_CONTROL_TEXTS["HOLD"])

    # ── Recommended setpoints ───────────────────────────────────────────────
    # Multiply element-wise; NaN propagates naturally for missing speed data
    out["recommended_speed_setpoint"]   = out["speed_mean"] * speed_factors
    out["recommended_tension_setpoint"] = out["tension_mean"]  # no tension action yet
    out["recommended_control_mode"]     = control_modes
    out["recommended_control_text"]     = control_texts

    # Aliases expected by the rest of the pipeline
    out["speed_setpoint"]   = out["recommended_speed_setpoint"]
    out["tension_setpoint"] = out["recommended_tension_setpoint"]
    out["control_message"]  = out["recommended_control_text"]
    out["control_level"]    = out["recommended_control_mode"]

    return out
