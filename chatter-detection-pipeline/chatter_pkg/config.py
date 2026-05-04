"""
All tunable thresholds, search bands, persistence rules, and control-related settings
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """
    Project-wide detector configuration.

    Fields are grouped by purpose:
      1. Sampling / windowing
      2. Frequency search bands
      3. Tracking parameters
      4. Peak-quality thresholds
      5. State-machine persistence
      6. Baseline / safety calibration
      7. Scoring weights and risk gates
      8. Event extraction
    """

    # 1. Sampling / windowing
    fs_fallback: float = 5000.0
    window_sec: float = 2.0
    hop_sec: float = 0.25
    highpass_hz: float = 20.0

    # 2. Frequency search bands
    wideband: Tuple[float, float] = (20.0, 1500.0)
    third_search: Tuple[float, float] = (100.0, 250.0)
    fifth_search: Tuple[float, float] = (450.0, 1200.0)
    narrow_band_halfwidth_hz: float = 12.0
    local_context_halfwidth_hz: float = 45.0
    alpha_horizon_sec: float = 2.0

    # 3a. Tracking — third band
    switch_confirm_frames_third: int = 3
    track_stability_frames_third: int = 5
    smoothing_alpha_third: float = 0.25
    max_track_jump_hz_third: float = 18.0
    big_jump_strength_ratio_third: float = 1.25
    switch_margin_score_ratio_third: float = 1.10

    # 3b. Tracking — fifth band
    switch_confirm_frames_fifth: int = 4
    track_stability_frames_fifth: int = 5
    smoothing_alpha_fifth: float = 0.30
    max_track_jump_hz_fifth: float = 45.0
    big_jump_strength_ratio_fifth: float = 1.35
    switch_margin_score_ratio_fifth: float = 1.20

    # 4a. Peak quality — third band
    # Relaxed so the rule-based detector can respond to strong synthetic third-band chatter.
    min_band_energy_ratio_third: float = 0.030
    min_peak_prominence_ratio_third: float = 3.5
    min_peak_snr_db_third: float = 9.5
    min_local_concentration_third: float = 0.68
    min_coherence_third: float = 0.45
    min_freq_stability_third_hz: float = 22.0
    min_growth_alpha_third: float = 0.004
    min_growth_ratio_third: float = 1.01

    # 4b. Peak quality — fifth band
    # Keep fifth band stricter because high-frequency false positives are easier.
    min_band_energy_ratio_fifth: float = 0.10
    min_peak_prominence_ratio_fifth: float = 5.0
    min_peak_snr_db_fifth: float = 16.0
    min_local_concentration_fifth: float = 0.78
    min_coherence_fifth: float = 0.55
    min_freq_stability_fifth_hz: float = 18.0
    min_growth_alpha_fifth: float = 0.03
    min_growth_ratio_fifth: float = 1.12

    # 5a. State-machine persistence — third band
    # Shorter persistence so synthetic onset can actually reach WARNING/ALARM.
    watch_persistence_frames_third: int = 3
    warning_persistence_frames_third: int = 3
    alarm_persistence_frames_third: int = 5
    release_frames_third: int = 5
    hold_grace_frames_third: int = 2

    # 5b. State-machine persistence — fifth band
    watch_persistence_frames_fifth: int = 3
    warning_persistence_frames_fifth: int = 10
    alarm_persistence_frames_fifth: int = 7
    release_frames_fifth: int = 6
    hold_grace_frames_fifth: int = 1

    # 6. Baseline / safety calibration
    baseline_fraction: float = 0.20
    baseline_min_frames: int = 20
    baseline_max_frames: int = 240
    safety_rms_quantile: float = 0.999

    # 7. Scoring weights and risk gates
    third_band_weight: float = 1.00
    fifth_band_weight: float = 0.75

    min_watch_risk_third: float = 0.35
    min_watch_risk_fifth: float = 0.42

    min_warning_risk_third: float = 0.75
    min_warning_risk_fifth: float = 0.82

    # For third-band synthetic chatter, growth should help but should not completely block WARNING.
    require_growth_for_warning_third: bool = False
    require_growth_for_warning_fifth: bool = True

    require_coherence_for_warning_third: bool = True
    require_coherence_for_warning_fifth: bool = True

    # 8. Event extraction
    min_event_gap_sec: float = 3.0
    min_event_duration_sec: float = 1.0
    min_event_valid_fraction: float = 0.45


CFG = Config()