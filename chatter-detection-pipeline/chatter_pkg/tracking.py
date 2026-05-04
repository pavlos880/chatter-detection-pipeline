"""
Peak picking and frequency tracking utilities for chatter bands.

Instead of accepting the loudest peak in every frame, this module tries to follow
a physically consistent peak over time and reject obvious jumps.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .config import CFG
from .signal_utils import band_mask, ema_update


@dataclass
class PeakCandidate:
    """
    Simple structured representation of one candidate spectral peak.
    """
    freq: float = np.nan
    raw_freq: float = np.nan
    strength: float = 0.0
    valid: float = 0.0
    energy_ratio: float = 0.0
    prominence_ratio: float = 0.0
    snr_db: float = -np.inf
    local_concentration: float = 0.0
    score: float = 0.0
    held: int = 0


@dataclass
class TrackManager:
    """
    Track one chatter-band frequency smoothly across frames.
    
    The tracker tries to avoid noisy peak jumping by allowing short holds, requiring
    confirmation for large jumps, and smoothing accepted updates.
    """
    band_name: str
    active_freq: Optional[float] = None
    smoothed_freq: Optional[float] = None
    active_strength: float = 0.0
    hold_count: int = 0
    pending_freq: Optional[float] = None
    pending_strength: float = 0.0
    pending_count: int = 0
    history: list[float] = field(default_factory=list)

    def _band_params(self):
        """
        Fetch the tracking parameters that belong to the current band.
        """
        if self.band_name == "third":
            return {
                "hold_grace_frames": CFG.hold_grace_frames_third,
                "max_track_jump_hz": CFG.max_track_jump_hz_third,
                "big_jump_strength_ratio": CFG.big_jump_strength_ratio_third,
                "switch_confirm_frames": CFG.switch_confirm_frames_third,
                "smoothing_alpha": CFG.smoothing_alpha_third,
            }
        return {
            "hold_grace_frames": CFG.hold_grace_frames_fifth,
            "max_track_jump_hz": CFG.max_track_jump_hz_fifth,
            "big_jump_strength_ratio": CFG.big_jump_strength_ratio_fifth,
            "switch_confirm_frames": CFG.switch_confirm_frames_fifth,
            "smoothing_alpha": CFG.smoothing_alpha_fifth,
        }

    def update(self, candidate: dict[str, float]) -> dict[str, float]:
        """
        Update the active track using the newest candidate peak information.
        """
        p = self._band_params()

        cand_freq = float(candidate.get("freq", np.nan))
        cand_valid = bool(candidate.get("valid", 0.0)) and np.isfinite(cand_freq)
        cand_strength = float(candidate.get("strength", 0.0))
        active = self.active_freq is not None and np.isfinite(self.active_freq)
        output = dict(candidate)

        if not cand_valid:
            if active and self.hold_count < p["hold_grace_frames"]:
                self.hold_count += 1
                held = float(
                    self.smoothed_freq
                    if self.smoothed_freq is not None and np.isfinite(self.smoothed_freq)
                    else self.active_freq
                )
                output.update({"freq": held, "raw_freq": held, "valid": 0.5, "held": 1})
                self.history.append(held)
                return output

            self.pending_freq = None
            self.pending_strength = 0.0
            self.pending_count = 0
            output.update({"freq": np.nan, "raw_freq": np.nan, "held": 0})
            self.history.append(np.nan)
            return output

        self.hold_count = 0
        output["held"] = 0

        if not active:
            self.active_freq = cand_freq
            self.active_strength = cand_strength
            self.smoothed_freq = cand_freq
        else:
            delta = abs(cand_freq - float(self.active_freq))

            if delta <= p["max_track_jump_hz"]:
                self.active_freq = cand_freq
                self.active_strength = cand_strength
                self.pending_freq = None
                self.pending_strength = 0.0
                self.pending_count = 0
            else:
                strength_margin_ok = cand_strength >= self.active_strength * p["big_jump_strength_ratio"]

                if strength_margin_ok:
                    if self.pending_freq is not None and abs(cand_freq - self.pending_freq) <= p["max_track_jump_hz"]:
                        self.pending_count += 1
                    else:
                        self.pending_freq = cand_freq
                        self.pending_strength = cand_strength
                        self.pending_count = 1

                    if self.pending_count >= p["switch_confirm_frames"]:
                        self.active_freq = self.pending_freq
                        self.active_strength = self.pending_strength
                        self.pending_freq = None
                        self.pending_strength = 0.0
                        self.pending_count = 0
                else:
                    output.update({
                        "freq": float(self.smoothed_freq if self.smoothed_freq is not None else self.active_freq),
                        "raw_freq": cand_freq,
                        "valid": 0.5,
                        "held": 1,
                    })
                    self.pending_freq = None
                    self.pending_strength = 0.0
                    self.pending_count = 0
                    self.history.append(float(output["freq"]))
                    return output

        self.smoothed_freq = ema_update(self.smoothed_freq, self.active_freq, p["smoothing_alpha"])
        output["freq"] = float(self.smoothed_freq)
        self.history.append(float(self.smoothed_freq))
        return output


def pick_tracked_peak(
    freqs: np.ndarray,
    mag: np.ndarray,
    power: np.ndarray,
    search_band: tuple[float, float],
    prev_freq: Optional[float],
    prev_strength: float,
    band_name: str,
) -> dict[str, float]:
    """
    Score candidate peaks inside a search band and return the best one.
    """
    idx = band_mask(freqs, search_band)
    empty = PeakCandidate().__dict__.copy()
    if not np.any(idx):
        return empty

    if band_name == "third":
        min_band_energy_ratio = CFG.min_band_energy_ratio_third
        min_peak_prominence_ratio = CFG.min_peak_prominence_ratio_third
        min_peak_snr_db = CFG.min_peak_snr_db_third
        min_local_concentration = CFG.min_local_concentration_third
        max_track_jump_hz = CFG.max_track_jump_hz_third
        switch_margin_score_ratio = CFG.switch_margin_score_ratio_third
        big_jump_strength_ratio = CFG.big_jump_strength_ratio_third
    else:
        min_band_energy_ratio = CFG.min_band_energy_ratio_fifth
        min_peak_prominence_ratio = CFG.min_peak_prominence_ratio_fifth
        min_peak_snr_db = CFG.min_peak_snr_db_fifth
        min_local_concentration = CFG.min_local_concentration_fifth
        max_track_jump_hz = CFG.max_track_jump_hz_fifth
        switch_margin_score_ratio = CFG.switch_margin_score_ratio_fifth
        big_jump_strength_ratio = CFG.big_jump_strength_ratio_fifth

    f = freqs[idx]
    m = mag[idx]
    pwr = power[idx]
    total_band_energy = float(np.sum(pwr) + 1e-12)

    peak_ids, props = find_peaks(m, prominence=np.mean(m) * 0.5 if np.mean(m) > 0 else None)
    if len(peak_ids) == 0:
        peak_ids = np.array([int(np.argmax(m))])
        prominences = np.array([max(0.0, m[peak_ids[0]] - np.median(m))])
    else:
        prominences = props.get("prominences", np.zeros(len(peak_ids)))

    candidates = []
    noise_floor = float(np.median(m) + 1e-12)
    local_mean = float(np.mean(m) + 1e-12)

    for local_i, prom in zip(peak_ids, prominences):
        pf = float(f[local_i])
        amp = float(m[local_i])

        nb = (f >= pf - CFG.narrow_band_halfwidth_hz) & (f <= pf + CFG.narrow_band_halfwidth_hz)
        local_ctx = (f >= pf - CFG.local_context_halfwidth_hz) & (f <= pf + CFG.local_context_halfwidth_hz)

        e_narrow = float(np.sum(pwr[nb]))
        e_local = float(np.sum(pwr[local_ctx]) + 1e-12)

        energy_ratio = e_narrow / total_band_energy
        prominence_ratio = amp / local_mean
        snr_db = 20.0 * math.log10(amp / noise_floor)
        local_concentration = e_narrow / e_local

        valid_gate = (
            energy_ratio >= min_band_energy_ratio
            and prominence_ratio >= min_peak_prominence_ratio
            and snr_db >= min_peak_snr_db
            and local_concentration >= min_local_concentration
        )

        continuity_bonus = 0.0
        if prev_freq is not None and np.isfinite(prev_freq):
            delta = abs(pf - prev_freq)
            continuity_bonus = max(0.0, 1.0 - delta / max(max_track_jump_hz, 1e-6))

        score = (
            0.38 * min(1.0, energy_ratio / 0.20)
            + 0.18 * min(1.0, prominence_ratio / 8.0)
            + 0.18 * min(1.0, max(snr_db, 0.0) / 18.0)
            + 0.10 * min(1.0, local_concentration / 0.70)
            + 0.16 * continuity_bonus
        )

        candidates.append({
            "freq": pf,
            "raw_freq": pf,
            "strength": amp,
            "energy_ratio": energy_ratio,
            "prominence_ratio": prominence_ratio,
            "snr_db": snr_db,
            "local_concentration": local_concentration,
            "valid_gate": valid_gate,
            "score": score,
        })

    candidates.sort(key=lambda d: d["score"], reverse=True)
    best = candidates[0]

    if prev_freq is not None and np.isfinite(prev_freq):
        near = [c for c in candidates if abs(c["freq"] - prev_freq) <= max_track_jump_hz]
        best_near = max(near, key=lambda d: d["score"]) if near else None
        jump = abs(best["freq"] - prev_freq)
        if best_near is not None and jump > max_track_jump_hz:
            if best["score"] < best_near["score"] * switch_margin_score_ratio:
                best = best_near
            elif best["strength"] < prev_strength * big_jump_strength_ratio:
                best = best_near

    out = dict(best)
    out["valid"] = 1.0 if best["valid_gate"] else 0.0
    return out