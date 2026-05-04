"""
Band-level decision rules for turning raw spectral features into detector triggers.

This module converts frame-level spectral features into transparent detector flags:
peak quality, coherence, growth, stability, WATCH trigger, WARNING trigger, and
ALARM trigger.

The third-band logic is intentionally more permissive than the fifth-band logic so
that strong synthetic third-band chatter can produce a WARNING/ALARM response.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG


@dataclass
class BandRules:
    """
    Container for the thresholds and persistence settings used by one chatter band.
    """

    name: str
    min_energy_ratio: float
    min_prominence_ratio: float
    min_snr_db: float
    min_local_concentration: float
    min_coherence: float
    min_growth_alpha: float
    min_growth_ratio: float
    min_watch_risk: float
    min_warning_risk: float
    require_growth_for_warning: bool
    require_coherence_for_warning: bool
    band_weight: float
    watch_persistence_frames: int
    warning_persistence_frames: int
    alarm_persistence_frames: int
    release_frames: int


def rules_for_band(name: str) -> BandRules:
    """
    Return the active rule set for the requested chatter band.
    """

    if name == "third":
        return BandRules(
            name="third",
            min_energy_ratio=CFG.min_band_energy_ratio_third,
            min_prominence_ratio=CFG.min_peak_prominence_ratio_third,
            min_snr_db=CFG.min_peak_snr_db_third,
            min_local_concentration=CFG.min_local_concentration_third,
            min_coherence=CFG.min_coherence_third,
            min_growth_alpha=CFG.min_growth_alpha_third,
            min_growth_ratio=CFG.min_growth_ratio_third,
            min_watch_risk=CFG.min_watch_risk_third,
            min_warning_risk=CFG.min_warning_risk_third,
            require_growth_for_warning=CFG.require_growth_for_warning_third,
            require_coherence_for_warning=CFG.require_coherence_for_warning_third,
            band_weight=CFG.third_band_weight,
            watch_persistence_frames=CFG.watch_persistence_frames_third,
            warning_persistence_frames=CFG.warning_persistence_frames_third,
            alarm_persistence_frames=CFG.alarm_persistence_frames_third,
            release_frames=CFG.release_frames_third,
        )

    if name == "fifth":
        return BandRules(
            name="fifth",
            min_energy_ratio=CFG.min_band_energy_ratio_fifth,
            min_prominence_ratio=CFG.min_peak_prominence_ratio_fifth,
            min_snr_db=CFG.min_peak_snr_db_fifth,
            min_local_concentration=CFG.min_local_concentration_fifth,
            min_coherence=CFG.min_coherence_fifth,
            min_growth_alpha=CFG.min_growth_alpha_fifth,
            min_growth_ratio=CFG.min_growth_ratio_fifth,
            min_watch_risk=CFG.min_watch_risk_fifth,
            min_warning_risk=CFG.min_warning_risk_fifth,
            require_growth_for_warning=CFG.require_growth_for_warning_fifth,
            require_coherence_for_warning=CFG.require_coherence_for_warning_fifth,
            band_weight=CFG.fifth_band_weight,
            watch_persistence_frames=CFG.watch_persistence_frames_fifth,
            warning_persistence_frames=CFG.warning_persistence_frames_fifth,
            alarm_persistence_frames=CFG.alarm_persistence_frames_fifth,
            release_frames=CFG.release_frames_fifth,
        )

    raise ValueError(f"Unknown chatter band: {name}")


def _valid_streak(mask: pd.Series) -> pd.Series:
    """
    Count consecutive True frames.

    Example:
        False, True, True, False, True -> 0, 1, 2, 0, 1
    """

    mask_bool = mask.astype(bool)
    streak = (
        (mask_bool.groupby((~mask_bool).cumsum()).cumcount() + 1)
        * mask_bool.astype(int)
    )
    return streak.astype(int)


def add_band_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw band features into detector flags and scores.

    The output includes:
      - third_peak_ok / fifth_peak_ok
      - third_coherence_ok / fifth_coherence_ok
      - third_growth_ok / fifth_growth_ok
      - third_stability_ok / fifth_stability_ok
      - score_third / score_fifth
      - third_watch_trigger / fifth_watch_trigger
      - third_warning_trigger / fifth_warning_trigger
      - third_alarm_trigger / fifth_alarm_trigger
      - risk_score
    """

    out = df.copy()

    for band in ("third", "fifth"):
        rules = rules_for_band(band)

        valid = out[f"{band}_valid"] >= 1.0
        not_held = out[f"{band}_held"] == 0

        energy_ok = out[f"{band}_energy_ratio"] >= rules.min_energy_ratio
        prominence_ok = out[f"{band}_prominence_ratio"] >= rules.min_prominence_ratio
        snr_ok = out[f"{band}_snr_db"] >= rules.min_snr_db
        concentration_ok = (
            out[f"{band}_local_concentration"] >= rules.min_local_concentration
        )

        peak_ok = (
            valid
            & not_held
            & energy_ok
            & prominence_ok
            & snr_ok
            & concentration_ok
        )

        coherence_ok = peak_ok & (out[f"coh_{band}"] >= rules.min_coherence)

        growth_ok = peak_ok & (
            (out[f"alpha_{band}"] >= rules.min_growth_alpha)
            | (out[f"growth_{band}"] >= rules.min_growth_ratio)
        )

        stability_ok = peak_ok & (out[f"{band}_consistent"] > 0)

        # Important difference:
        # Third-band chatter onset may show a coherent peak before frequency tracking
        # becomes perfectly stable. Therefore, for third band, growth can substitute
        # for stability in the reliability gate.
        if band == "third":
            reliable = coherence_ok & (stability_ok | growth_ok)
        else:
            reliable = coherence_ok & stability_ok
            reliable = reliable & (
                out[f"{band}_local_concentration"]
                >= max(rules.min_local_concentration, 0.80)
            )

        score = (
            0.35 * peak_ok.astype(float)
            + 0.20 * coherence_ok.astype(float)
            + 0.20 * growth_ok.astype(float)
            + 0.25 * stability_ok.astype(float)
        )

        # Reduce confidence for impulsive frames if that diagnostic exists.
        if "impulse_like" in out.columns:
            score = np.where(out["impulse_like"] > 0, 0.8 * score, score)

        valid_streak = _valid_streak(peak_ok)

        risk = rules.band_weight * score

        watch_trigger = (
            peak_ok
            & (risk >= rules.min_watch_risk)
            & (valid_streak >= 2)
        )

        if band == "third":
            # Third-band warning:
            # Requires coherent narrowband evidence and either growth or stable tracking.
            # This avoids the old problem where the detector stayed silent even when
            # synthetic third-band chatter was present.
            warning_trigger = (
                reliable
                & (risk >= rules.min_warning_risk)
                & (valid_streak >= 2)
                & (growth_ok | stability_ok)
            )

            # Alarm remains stricter than warning.
            alarm_trigger = (
                warning_trigger
                & coherence_ok
                & (valid_streak >= 5)
            )

        else:
            # Fifth band remains stricter because it is easier to create false positives
            # from high-frequency machine noise.
            warning_trigger = (
                reliable
                & (risk >= rules.min_warning_risk)
                & (valid_streak >= 3)
            )

            if rules.require_growth_for_warning:
                warning_trigger = warning_trigger & growth_ok

            if rules.require_coherence_for_warning:
                warning_trigger = warning_trigger & coherence_ok

            alarm_trigger = (
                warning_trigger
                & growth_ok
                & coherence_ok
                & (valid_streak >= 5)
            )

        out[f"{band}_peak_ok"] = peak_ok.astype(int)
        out[f"{band}_coherence_ok"] = coherence_ok.astype(int)
        out[f"{band}_growth_ok"] = growth_ok.astype(int)
        out[f"{band}_stability_ok"] = stability_ok.astype(int)
        out[f"{band}_reliable"] = reliable.astype(int)
        out[f"{band}_valid_streak"] = valid_streak.astype(int)

        out[f"score_{band}"] = score.astype(float)
        out[f"weighted_score_{band}"] = risk.astype(float)

        out[f"{band}_watch_trigger"] = watch_trigger.astype(int)
        out[f"{band}_warning_trigger"] = warning_trigger.astype(int)
        out[f"{band}_alarm_trigger"] = alarm_trigger.astype(int)

    out["dominant_band"] = np.where(
        out["weighted_score_third"] >= out["weighted_score_fifth"],
        "third",
        "fifth",
    )

    out["dominant_frequency_hz"] = np.where(
        out["dominant_band"] == "third",
        out["dom_third_hz"],
        out["dom_fifth_hz"],
    )

    out["risk_score"] = out[["weighted_score_third", "weighted_score_fifth"]].max(axis=1)
    out["warning_index"] = out["risk_score"]

    out["hard_candidate"] = (
        (out["third_alarm_trigger"] > 0)
        | (out["fifth_alarm_trigger"] > 0)
    ).astype(int)

    return out