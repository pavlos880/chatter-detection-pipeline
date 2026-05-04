"""
KPI computation and diagnostic plot generation.

Changes from original:
  - peak_quality_vs_time.png is now two subplots so local concentration
    (0-1 scale) is visible alongside SNR and prominence (0-30+ scale).
  - coherence_vs_time.png now shows warning threshold lines.
  - risk_score_vs_time.png now shows watch / warning threshold lines.
  - NEW: dashboard_summary.png — 5-panel overview combining all key signals.
  - save_plots() imports CFG so threshold lines stay in sync with config.
"""

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CFG
from .paths import mkdir


STATE_TO_NUM = {"STABLE": 0, "WATCH": 1, "WARNING": 2, "ALARM": 3, "SAFETY": 4}
STATE_COLORS = {
    "STABLE":  "#70AD47",
    "WATCH":   "#FFC000",
    "WARNING": "#ED7D31",
    "ALARM":   "#C00000",
    "SAFETY":  "#7030A0",
}


# Helpers

def _safe_numeric(series_like, default: float = np.nan) -> pd.Series:
    """Convert to numeric, replace non-finite with default."""
    s = pd.to_numeric(series_like, errors="coerce")
    if isinstance(s, pd.Series):
        return s.fillna(default)
    return pd.Series(s).fillna(default)


def _state_series_to_num(s: pd.Series) -> pd.Series:
    return s.astype(str).map(STATE_TO_NUM).fillna(0).astype(float)


def _state_color_series(s: pd.Series) -> list[str]:
    return [STATE_COLORS.get(str(v), "#70AD47") for v in s]


# KPI / Summary

def compute_kpis(df: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    """
    Compute descriptor-internal KPIs (no ground-truth labels required).

    NOTE: These KPIs describe what the detector did. They are NOT validated
    performance numbers. For validated metrics, use evaluation.py against
    manually or synthetically labelled intervals.
    """
    if df.empty:
        return _empty_kpis()

    duration_sec = float(pd.to_numeric(df["t_sec"], errors="coerce").max()) or 0.0
    hours = max(duration_sec / 3600.0, 1e-9)
    state_series = df["state"].astype(str) if "state" in df.columns else pd.Series(dtype=str)

    warning_frames  = int((state_series == "WARNING").sum())
    alarm_frames    = int((state_series.isin(["ALARM", "SAFETY"])).sum())
    watch_frames    = int((state_series == "WATCH").sum())
    nonstable_frames = int((state_series != "STABLE").sum())

    if events.empty:
        return {
            "n_events": 0,
            "early_warning_rate": np.nan,
            "mean_lead_time_sec": np.nan,
            "median_lead_time_sec": np.nan,
            "detection_rate_pct": np.nan,
            "avg_lead_time_sec": np.nan,
            "false_warnings_per_hour": float(warning_frames / hours),
            "warning_frames_per_hour": float(warning_frames / hours),
            "watch_frames_per_hour":   float(watch_frames / hours),
            "alarm_frames_per_hour":   float(alarm_frames / hours),
            "nonstable_frames_per_hour": float(nonstable_frames / hours),
            "internal_mean_watch_to_warning_sec": np.nan,
            "internal_mean_warning_to_alarm_sec": np.nan,
        }

    w2w = pd.to_numeric(events.get("watch_to_warning_sec",   pd.Series(dtype=float)), errors="coerce")
    w2a = pd.to_numeric(events.get("warning_to_alarm_sec",   pd.Series(dtype=float)), errors="coerce")
    w2w = w2w[np.isfinite(w2w)]
    w2a = w2a[np.isfinite(w2a)]
    internal_leads = w2a if len(w2a) else w2w

    return {
        "n_events": int(len(events)),
        "early_warning_rate": float(np.mean(internal_leads > 0)) if len(internal_leads) else np.nan,
        "mean_lead_time_sec": float(np.mean(internal_leads))   if len(internal_leads) else np.nan,
        "median_lead_time_sec": float(np.median(internal_leads)) if len(internal_leads) else np.nan,
        "detection_rate_pct": np.nan,
        "avg_lead_time_sec":  float(np.mean(internal_leads))   if len(internal_leads) else np.nan,
        "false_warnings_per_hour": float(warning_frames / hours),
        "warning_frames_per_hour": float(warning_frames / hours),
        "watch_frames_per_hour":   float(watch_frames / hours),
        "alarm_frames_per_hour":   float(alarm_frames / hours),
        "nonstable_frames_per_hour": float(nonstable_frames / hours),
        "internal_mean_watch_to_warning_sec": float(np.mean(w2w)) if len(w2w) else np.nan,
        "internal_mean_warning_to_alarm_sec": float(np.mean(w2a)) if len(w2a) else np.nan,
    }


def _empty_kpis() -> dict[str, Any]:
    keys = [
        "n_events", "early_warning_rate", "mean_lead_time_sec", "median_lead_time_sec",
        "detection_rate_pct", "avg_lead_time_sec", "false_warnings_per_hour",
        "warning_frames_per_hour", "watch_frames_per_hour", "alarm_frames_per_hour",
        "nonstable_frames_per_hour", "internal_mean_watch_to_warning_sec",
        "internal_mean_warning_to_alarm_sec",
    ]
    d = {k: np.nan for k in keys}
    d["n_events"] = 0
    return d


def summary(df: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    """One-line snapshot of a run — saved as summary.json."""
    return {
        "frames":              int(len(df)),
        "duration_sec":        float(df["t_sec"].max()) if len(df) else 0.0,
        "n_warning_frames":    int((df["state"] == "WARNING").sum()),
        "n_alarm_frames":      int((df["state"].isin(["ALARM", "SAFETY"])).sum()),
        "n_watch_frames":      int((df["state"] == "WATCH").sum()),
        "n_events":            int(len(events)),
        "third_valid_fraction": float((df["third_valid"] >= 0.5).mean()) if "third_valid" in df.columns else np.nan,
        "fifth_valid_fraction": float((df["fifth_valid"] >= 0.5).mean()) if "fifth_valid" in df.columns else np.nan,
        "mean_risk_score":     float(df["risk_score"].mean()) if "risk_score" in df.columns else np.nan,
        "max_risk_score":      float(df["risk_score"].max())  if "risk_score" in df.columns else np.nan,
        "mean_coh_third":      float(_safe_numeric(df["coh_third"]).mean())  if "coh_third" in df.columns else np.nan,
        "mean_coh_fifth":      float(_safe_numeric(df["coh_fifth"]).mean())  if "coh_fifth" in df.columns else np.nan,
    }


# Individual plots

def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate and save all diagnostic plots for one run."""
    mkdir(out_dir)
    t = df["t_sec"]

    _plot_dominant_freq(t, df, out_dir)
    _plot_risk_score(t, df, out_dir)
    _plot_band_scores(t, df, out_dir)
    _plot_coherence(t, df, out_dir)
    _plot_growth(t, df, out_dir)
    _plot_state(t, df, out_dir)
    _plot_peak_quality(t, df, out_dir)      # FIXED: two subplots
    _plot_validity_flags(t, df, out_dir)
    _plot_dashboard(t, df, out_dir)          # NEW: 5-panel overview


def _plot_dominant_freq(t, df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, df["dom_third_hz"], label="3rd candidate", color="#2E75B6")
    ax.plot(t, df["dom_fifth_hz"], label="5th candidate", color="#ED7D31")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Dominant frequency [Hz]")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "dominant_frequency_vs_time.png", dpi=150)
    plt.close()


def _plot_risk_score(t, df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, df["risk_score"], color="#C00000", linewidth=0.8)
    ax.axhline(y=CFG.min_watch_risk_third,   color="orange", linestyle="--",
               alpha=0.7, label=f"Watch threshold ({CFG.min_watch_risk_third})")
    ax.axhline(y=CFG.min_warning_risk_third, color="red",    linestyle="--",
               alpha=0.7, label=f"Warning threshold ({CFG.min_warning_risk_third})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Risk score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "risk_score_vs_time.png", dpi=150)
    plt.close()


def _plot_band_scores(t, df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, df["score_third"], label="3rd score", color="#2E75B6")
    ax.plot(t, df["score_fifth"], label="5th score", color="#ED7D31")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Band score")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "band_scores_vs_time.png", dpi=150)
    plt.close()


def _plot_coherence(t, df, out_dir):
    if not {"coh_third", "coh_fifth"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, _safe_numeric(df["coh_third"]), label="3rd coherence", color="#2E75B6")
    ax.plot(t, _safe_numeric(df["coh_fifth"]), label="5th coherence", color="#ED7D31")
    # Threshold lines — these are the gates the WARNING trigger must pass
    ax.axhline(y=CFG.min_coherence_third, color="#2E75B6", linestyle="--", alpha=0.6,
               label=f"3rd warning gate ({CFG.min_coherence_third})")
    ax.axhline(y=CFG.min_coherence_fifth, color="#ED7D31", linestyle="--", alpha=0.6,
               label=f"5th warning gate ({CFG.min_coherence_fifth})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Coherence")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "coherence_vs_time.png", dpi=150)
    plt.close()


def _plot_growth(t, df, out_dir):
    cols = [c for c in ["alpha_third", "alpha_fifth"] if c in df.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    if "alpha_third" in df.columns:
        ax.plot(t, _safe_numeric(df["alpha_third"]), label="3rd alpha (growth rate)", color="#2E75B6")
    if "alpha_fifth" in df.columns:
        ax.plot(t, _safe_numeric(df["alpha_fifth"]), label="5th alpha (growth rate)", color="#ED7D31")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axhline(y=CFG.min_growth_alpha_third, color="#2E75B6", linestyle="--", alpha=0.5,
               label=f"3rd growth gate ({CFG.min_growth_alpha_third})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Growth rate alpha [1/s]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "growth_vs_time.png", dpi=150)
    plt.close()


def _plot_state(t, df, out_dir):
    state_cols = [c for c in ["third_state", "fifth_state", "state"] if c in df.columns]
    if not state_cols:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    if "third_state" in df.columns:
        ax.plot(t, _state_series_to_num(df["third_state"]), label="3rd state", alpha=0.7)
    if "fifth_state" in df.columns:
        ax.plot(t, _state_series_to_num(df["fifth_state"]), label="5th state", alpha=0.7)
    if "state" in df.columns:
        ax.plot(t, _state_series_to_num(df["state"]), label="overall state", linewidth=2)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["STABLE", "WATCH", "WARNING", "ALARM", "SAFETY"])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "state_vs_time.png", dpi=150)
    plt.close()


def _plot_peak_quality(t, df, out_dir):
    """
    FIXED: Two subplots so local concentration (0-1) is visible.

    Top panel   — SNR [dB] and prominence ratio (both can exceed 20).
    Bottom panel — local spectral concentration (0 to 1) with threshold lines.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: SNR and prominence
    if "third_snr_db" in df.columns:
        ax1.plot(t, _safe_numeric(df["third_snr_db"]),        label="3rd SNR [dB]",    color="#2E75B6")
    if "fifth_snr_db" in df.columns:
        ax1.plot(t, _safe_numeric(df["fifth_snr_db"]),        label="5th SNR [dB]",    color="#ED7D31")
    if "third_prominence_ratio" in df.columns:
        ax1.plot(t, _safe_numeric(df["third_prominence_ratio"]), label="3rd prominence", color="#2E75B6", alpha=0.5, linestyle="--")
    if "fifth_prominence_ratio" in df.columns:
        ax1.plot(t, _safe_numeric(df["fifth_prominence_ratio"]), label="5th prominence", color="#ED7D31", alpha=0.5, linestyle="--")
    ax1.axhline(y=CFG.min_peak_snr_db_third, color="#2E75B6", linestyle=":", alpha=0.6,
                label=f"3rd SNR gate ({CFG.min_peak_snr_db_third} dB)")
    ax1.set_ylabel("SNR [dB] / Prominence ratio")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # Bottom: local concentration only — THIS WAS INVISIBLE IN THE OLD PLOT
    if "third_local_concentration" in df.columns:
        ax2.plot(t, _safe_numeric(df["third_local_concentration"]),
                 label="3rd local concentration", color="#2E75B6")
    if "fifth_local_concentration" in df.columns:
        ax2.plot(t, _safe_numeric(df["fifth_local_concentration"]),
                 label="5th local concentration", color="#ED7D31")
    ax2.axhline(y=CFG.min_local_concentration_third, color="#2E75B6", linestyle="--", alpha=0.7,
                label=f"3rd gate ({CFG.min_local_concentration_third})")
    ax2.axhline(y=CFG.min_local_concentration_fifth, color="#ED7D31", linestyle="--", alpha=0.7,
                label=f"5th gate ({CFG.min_local_concentration_fifth})")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Local spectral concentration")
    ax2.set_xlabel("Time [s]")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "peak_quality_vs_time.png", dpi=150)
    plt.close()


def _plot_validity_flags(t, df, out_dir):
    cols = [c for c in ["third_valid", "fifth_valid", "third_held", "fifth_held"] if c in df.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"third_valid": "#2E75B6", "fifth_valid": "#ED7D31",
              "third_held": "#70AD47",  "fifth_held": "#C00000"}
    for c in cols:
        ax.plot(t, _safe_numeric(df[c]), label=c.replace("_", " "), color=colors.get(c))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flag value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "validity_and_hold_flags_vs_time.png", dpi=150)
    plt.close()


def _plot_dashboard(t, df, out_dir):
    """
    NEW: 5-panel summary dashboard — the single most useful figure for the report.

    Panel 1 — Vibration RMS (overall energy)
    Panel 2 — Third and fifth band energy (where chatter energy concentrates)
    Panel 3 — Cross-sensor coherence with warning gate lines
    Panel 4 — Risk score with watch / warning gate lines
    Panel 5 — Detector state (color-coded bar chart)
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Chatter Detector — Run Summary Dashboard", fontsize=14, fontweight="bold", y=0.995)

    #  Panel 1: RMS 
    ax = axes[0]
    if "vib_rms_mean" in df.columns:
        ax.plot(t, _safe_numeric(df["vib_rms_mean"]), color="#2E75B6", linewidth=0.8)
    ax.set_ylabel("Vibration RMS\n[g or m/s²]", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Vibration RMS (wideband)", fontsize=9, loc="left")

    #  Panel 2: Band energies 
    ax = axes[1]
    if "e_third" in df.columns:
        ax.plot(t, _safe_numeric(df["e_third"]), label="3rd band energy", color="#2E75B6", linewidth=0.8)
    if "e_fifth" in df.columns:
        ax.plot(t, _safe_numeric(df["e_fifth"]), label="5th band energy", color="#ED7D31", linewidth=0.8)
    ax.set_ylabel("Band energy", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Chatter-band energy (3rd: 100–250 Hz, 5th: 450–1200 Hz)", fontsize=9, loc="left")

    #  Panel 3: Coherence + threshold lines 
    ax = axes[2]
    if "coh_third" in df.columns:
        ax.plot(t, _safe_numeric(df["coh_third"]), label="3rd coherence", color="#2E75B6", linewidth=0.8)
    if "coh_fifth" in df.columns:
        ax.plot(t, _safe_numeric(df["coh_fifth"]), label="5th coherence", color="#ED7D31", linewidth=0.8)
    ax.axhline(y=CFG.min_coherence_third, color="#2E75B6", linestyle="--", alpha=0.6,
               label=f"3rd warning gate ({CFG.min_coherence_third})")
    ax.axhline(y=CFG.min_coherence_fifth, color="#ED7D31", linestyle="--", alpha=0.6,
               label=f"5th warning gate ({CFG.min_coherence_fifth})")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Cross-sensor\ncoherence", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("OS–DS coherence (high = correlated structural vibration)", fontsize=9, loc="left")

    #  Panel 4: Risk score + threshold lines 
    ax = axes[3]
    ax.plot(t, _safe_numeric(df["risk_score"]), color="#C00000", linewidth=0.8)
    ax.axhline(y=CFG.min_watch_risk_third,   color="orange", linestyle="--", alpha=0.7,
               label=f"Watch gate ({CFG.min_watch_risk_third})")
    ax.axhline(y=CFG.min_warning_risk_third, color="red",    linestyle="--", alpha=0.7,
               label=f"Warning gate ({CFG.min_warning_risk_third})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Risk score", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Risk score (max of weighted band scores)", fontsize=9, loc="left")

    #  Panel 5: State as color bar 
    ax = axes[4]
    if "state" in df.columns:
        state_num    = _state_series_to_num(df["state"])
        bar_colors   = _state_color_series(df["state"])
        ax.bar(t, state_num + 0.6, width=float(CFG.hop_sec) * 0.95,
               color=bar_colors, align="center")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["STABLE", "WATCH", "WARNING", "ALARM", "SAFETY"], fontsize=8)
    ax.set_ylabel("Detector state", fontsize=9)
    ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.2)
    ax.set_title("State machine output (persistence-filtered)", fontsize=9, loc="left")

    plt.tight_layout()
    plt.savefig(out_dir / "dashboard_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
