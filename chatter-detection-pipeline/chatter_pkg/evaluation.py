from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG
from .control import apply_simple_advisory
from .detection import extract_events, extract_features
from .io_utils import save_json
from .paths import mkdir
from .reporting import compute_kpis, save_plots, summary

def _as_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _event_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=["start_sec", "end_sec", "duration_sec", "peak_state", "dominant_band"])

    out = events.copy()

    rename_map = {}
    if "start_t_sec" in out.columns:
        rename_map["start_t_sec"] = "start_sec"
    if "end_t_sec" in out.columns:
        rename_map["end_t_sec"] = "end_sec"
    if rename_map:
        out = out.rename(columns=rename_map)

    for c in ["start_sec", "end_sec", "duration_sec"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["start_sec", "end_sec"]).copy()
    out = out[out["end_sec"] > out["start_sec"]].sort_values("start_sec").reset_index(drop=True)

    if "duration_sec" not in out.columns:
        out["duration_sec"] = out["end_sec"] - out["start_sec"]

    return out


def load_labels(path: str | Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"run_name", "start_sec", "end_sec", "label"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")

    out = labels.copy()
    out["run_name"] = out["run_name"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip().str.lower()
    out["start_sec"] = pd.to_numeric(out["start_sec"], errors="coerce")
    out["end_sec"] = pd.to_numeric(out["end_sec"], errors="coerce")

    if "notes" not in out.columns:
        out["notes"] = ""

    out = out.dropna(subset=["start_sec", "end_sec"])
    out = out[out["end_sec"] > out["start_sec"]].sort_values(["run_name", "start_sec"]).reset_index(drop=True)
    return out


def match_events(
    truth_events: pd.DataFrame,
    pred_events: pd.DataFrame,
    pre_event_tolerance_sec: float = 2.0,
) -> pd.DataFrame:
    truth = _normalize_events(truth_events)
    pred = _normalize_events(pred_events)

    rows: list[dict[str, Any]] = []
    used_pred: set[int] = set()

    for true_idx, t in truth.iterrows():
        t0 = float(t["start_sec"])
        t1 = float(t["end_sec"])

        best_idx = None
        best_overlap = -1.0
        best_start_gap = np.inf

        for pred_idx, p in pred.iterrows():
            if pred_idx in used_pred:
                continue

            p0 = float(p["start_sec"])
            p1 = float(p["end_sec"])

            eligible = (p1 >= t0 - pre_event_tolerance_sec) and (p0 <= t1)
            if not eligible:
                continue

            overlap = _event_overlap(t0, t1, p0, p1)
            start_gap = abs(p0 - t0)

            if (overlap > best_overlap) or (overlap == best_overlap and start_gap < best_start_gap):
                best_idx = pred_idx
                best_overlap = overlap
                best_start_gap = start_gap

        if best_idx is not None:
            used_pred.add(best_idx)
            p = pred.loc[best_idx]

            p0 = float(p["start_sec"])
            p1 = float(p["end_sec"])

            watch_t = pd.to_numeric(pd.Series([p.get("watch_start_t_sec", np.nan)]), errors="coerce").iloc[0]
            warning_t = pd.to_numeric(pd.Series([p.get("warning_start_t_sec", np.nan)]), errors="coerce").iloc[0]

            if np.isfinite(watch_t):
                pred_early_sec = float(watch_t)
            elif np.isfinite(warning_t):
                pred_early_sec = float(warning_t)
            else:
                pred_early_sec = p0

            pred_warning_sec = float(warning_t) if np.isfinite(warning_t) else np.nan

            alarm_t = pd.to_numeric(pd.Series([p.get("alarm_start_t_sec", np.nan)]), errors="coerce").iloc[0]
            pred_alarm_sec = float(alarm_t) if np.isfinite(alarm_t) else np.nan

            rows.append(
                {
                    "true_idx": int(true_idx),
                    "pred_idx": int(best_idx),
                    "true_start_sec": t0,
                    "true_end_sec": t1,
                    "pred_start_sec": p0,
                    "pred_end_sec": p1,
                    "pred_early_start_sec": pred_early_sec,
                    "pred_warning_start_sec": pred_warning_sec,
                    "overlap_sec": float(_event_overlap(t0, t1, p0, p1)),
                    "early_lead_time_sec": float(t0 - pred_early_sec),
                    "warning_lead_time_sec": float(t0 - pred_warning_sec) if np.isfinite(pred_warning_sec) else np.nan,
                    "pred_peak_state": str(p.get("peak_state", "")),
                    "pred_dominant_band": str(p.get("dominant_band", "")),
                }
            )

    return pd.DataFrame(rows)

def _pair_onset_intervals(
    truth_onset: pd.DataFrame,
    truth_chatter: pd.DataFrame,
    max_gap_sec: float = 10.0,
) -> pd.DataFrame:
    """Pair onset intervals to chatter intervals in time order.

    The usual manual-labeling pattern is stable -> onset -> chatter. This helper
    associates each chatter interval with the latest onset interval that occurs
    immediately before it (or lightly overlaps it), so validated early-warning KPIs
    can be reported against onset as well as chatter.
    """
    if truth_onset is None or truth_onset.empty or truth_chatter is None or truth_chatter.empty:
        return pd.DataFrame(columns=[
            "true_idx", "onset_idx", "onset_start_sec", "onset_end_sec", "onset_gap_to_chatter_sec"
        ])

    onset = truth_onset.reset_index(drop=True).copy()
    chatter = truth_chatter.reset_index(drop=True).copy()

    rows: list[dict[str, Any]] = []
    used_onset: set[int] = set()

    for true_idx, c in chatter.iterrows():
        c0 = float(c["start_sec"])
        c1 = float(c["end_sec"])

        best_idx = None
        best_gap = np.inf
        best_start = -np.inf

        for onset_idx, o in onset.iterrows():
            if onset_idx in used_onset:
                continue

            o0 = float(o["start_sec"])
            o1 = float(o["end_sec"])
            if o0 > c0:
                continue

            overlap = _event_overlap(o0, o1, c0, c1)
            gap = max(0.0, c0 - o1)
            eligible = (overlap > 0.0) or (gap <= max_gap_sec)
            if not eligible:
                continue

            if (gap < best_gap) or (gap == best_gap and o0 > best_start):
                best_idx = int(onset_idx)
                best_gap = float(gap)
                best_start = float(o0)

        if best_idx is not None:
            used_onset.add(best_idx)
            o = onset.loc[best_idx]
            rows.append({
                "true_idx": int(true_idx),
                "onset_idx": int(best_idx),
                "onset_start_sec": float(o["start_sec"]),
                "onset_end_sec": float(o["end_sec"]),
                "onset_gap_to_chatter_sec": float(best_gap),
            })

    return pd.DataFrame(rows)


def evaluate_against_labels(
    df: pd.DataFrame,
    pred_events: pd.DataFrame,
    labels: pd.DataFrame,
    run_name: str,
    pre_event_tolerance_sec: float = 2.0,
) -> dict[str, Any]:
    pred = _normalize_events(pred_events)
    lab = labels[labels["run_name"].astype(str) == str(run_name)].copy()

    truth_chatter = lab[lab["label"] == "chatter"][["start_sec", "end_sec", "label", "notes"]].reset_index(drop=True)
    truth_onset = lab[lab["label"] == "onset"][["start_sec", "end_sec", "label", "notes"]].reset_index(drop=True)
    truth_stable = lab[lab["label"] == "stable"][["start_sec", "end_sec", "label", "notes"]].reset_index(drop=True)

    matches = match_events(truth_chatter, pred, pre_event_tolerance_sec=pre_event_tolerance_sec)
    onset_pairs = _pair_onset_intervals(truth_onset, truth_chatter)
    if not matches.empty and not onset_pairs.empty:
        matches = matches.merge(onset_pairs, on="true_idx", how="left")

    matched_true = set(matches["true_idx"].tolist()) if not matches.empty else set()
    matched_pred = set(matches["pred_idx"].tolist()) if not matches.empty else set()

    tp = len(matched_true)
    fn = max(0, len(truth_chatter) - tp)
    fp = max(0, len(pred) - len(matched_pred))

    recall = 100.0 * tp / len(truth_chatter) if len(truth_chatter) else np.nan
    precision = 100.0 * tp / len(pred) if len(pred) else np.nan

    early_leads = matches["early_lead_time_sec"].dropna().tolist() if not matches.empty else []
    warning_leads = matches["warning_lead_time_sec"].dropna().tolist() if not matches.empty else []

    early_to_onset = []
    warning_to_onset = []
    if not matches.empty and "onset_start_sec" in matches.columns:
        onset_start = pd.to_numeric(matches["onset_start_sec"], errors="coerce")
        pred_early = pd.to_numeric(matches["pred_early_start_sec"], errors="coerce")
        pred_warning = pd.to_numeric(matches["pred_warning_start_sec"], errors="coerce")

        early_to_onset = (onset_start - pred_early).replace([np.inf, -np.inf], np.nan).dropna().tolist()
        warning_to_onset = (onset_start - pred_warning).replace([np.inf, -np.inf], np.nan).dropna().tolist()

    # Count unmatched predicted events overlapping stable intervals, but only once per event.
    stable_fp_pred_indices: set[int] = set()
    stable_hours = 0.0
    if not truth_stable.empty and not pred.empty:
        for _, s in truth_stable.iterrows():
            s0 = float(s["start_sec"])
            s1 = float(s["end_sec"])
            stable_hours += max(0.0, s1 - s0) / 3600.0

            for pred_idx, p in pred.iterrows():
                if pred_idx in matched_pred:
                    continue
                p0 = float(p["start_sec"])
                p1 = float(p["end_sec"])
                if _event_overlap(s0, s1, p0, p1) > 0:
                    stable_fp_pred_indices.add(int(pred_idx))

    false_events_per_hour_stable = (
        len(stable_fp_pred_indices) / stable_hours if stable_hours > 0 else np.nan
    )

    stable_total_frames = 0
    stable_watch_or_worse_frames = 0
    stable_warning_or_worse_frames = 0
    stable_nonstable_frames = 0

    if not truth_stable.empty and not df.empty:
        t_series = pd.to_numeric(df["t_sec"], errors="coerce")
        state_series = df["state"].astype(str)

        for _, s in truth_stable.iterrows():
            s0 = float(s["start_sec"])
            s1 = float(s["end_sec"])
            mask = (t_series >= s0) & (t_series <= s1)
            stable_total_frames += int(mask.sum())
            stable_watch_or_worse_frames += int((state_series.isin(["WATCH", "WARNING", "ALARM", "SAFETY"]) & mask).sum())
            stable_warning_or_worse_frames += int((state_series.isin(["WARNING", "ALARM", "SAFETY"]) & mask).sum())
            stable_nonstable_frames += int(((state_series != "STABLE") & mask).sum())

    stable_warning_frame_fraction = (
        stable_warning_or_worse_frames / stable_total_frames if stable_total_frames > 0 else np.nan
    )
    stable_watch_or_worse_frame_fraction = (
        stable_watch_or_worse_frames / stable_total_frames if stable_total_frames > 0 else np.nan
    )
    stable_nonstable_frame_fraction = (
        stable_nonstable_frames / stable_total_frames if stable_total_frames > 0 else np.nan
    )

    stable_warning_frames_per_hour = (
        stable_warning_or_worse_frames / stable_hours if stable_hours > 0 else np.nan
    )
    stable_watch_or_worse_frames_per_hour = (
        stable_watch_or_worse_frames / stable_hours if stable_hours > 0 else np.nan
    )
    stable_nonstable_frames_per_hour = (
        stable_nonstable_frames / stable_hours if stable_hours > 0 else np.nan
    )

    primary_leads = warning_leads if len(warning_leads) else early_leads
    early_warning_rate = float(np.mean(np.asarray(primary_leads, dtype=float) > 0.0)) if len(primary_leads) else np.nan

    out = {
        "validated_kpi_source": "manual_labels",
        "run_name": str(run_name),
        "n_true_onset_events": int(len(truth_onset)),
        "n_true_chatter_events": int(len(truth_chatter)),
        "n_predicted_events": int(len(pred)),
        "true_positive_events": int(tp),
        "false_negative_events": int(fn),
        "false_positive_events": int(fp),
        "event_recall_pct": float(recall) if np.isfinite(recall) else np.nan,
        "event_precision_pct": float(precision) if np.isfinite(precision) else np.nan,
        "mean_early_lead_time_sec": _safe_mean(early_leads),
        "median_early_lead_time_sec": _safe_median(early_leads),
        "mean_warning_lead_time_sec": _safe_mean(warning_leads),
        "median_warning_lead_time_sec": _safe_median(warning_leads),
        "mean_watch_lead_to_onset_sec": _safe_mean(early_to_onset),
        "median_watch_lead_to_onset_sec": _safe_median(early_to_onset),
        "mean_warning_lead_to_onset_sec": _safe_mean(warning_to_onset),
        "median_warning_lead_to_onset_sec": _safe_median(warning_to_onset),
        "false_events_per_hour_in_stable_regions": float(false_events_per_hour_stable)
        if np.isfinite(false_events_per_hour_stable) else np.nan,
        "stable_warning_frame_fraction": float(stable_warning_frame_fraction)
        if np.isfinite(stable_warning_frame_fraction) else np.nan,
        "stable_watch_or_worse_frame_fraction": float(stable_watch_or_worse_frame_fraction)
        if np.isfinite(stable_watch_or_worse_frame_fraction) else np.nan,
        "stable_nonstable_frame_fraction": float(stable_nonstable_frame_fraction)
        if np.isfinite(stable_nonstable_frame_fraction) else np.nan,
        "stable_warning_frames_per_hour": float(stable_warning_frames_per_hour)
        if np.isfinite(stable_warning_frames_per_hour) else np.nan,
        "stable_watch_or_worse_frames_per_hour": float(stable_watch_or_worse_frames_per_hour)
        if np.isfinite(stable_watch_or_worse_frames_per_hour) else np.nan,
        "stable_nonstable_frames_per_hour": float(stable_nonstable_frames_per_hour)
        if np.isfinite(stable_nonstable_frames_per_hour) else np.nan,
        # Legacy headline fields, now mapped to validated metrics so existing scripts
        # and tables can use stronger label-based KPIs without further changes.
        "early_warning_rate": early_warning_rate,
        "mean_lead_time_sec": _safe_mean(primary_leads),
        "median_lead_time_sec": _safe_median(primary_leads),
        "avg_lead_time_sec": _safe_mean(primary_leads),
        "detection_rate_pct": float(recall) if np.isfinite(recall) else np.nan,
        "false_warnings_per_hour": float(stable_warning_frames_per_hour)
        if np.isfinite(stable_warning_frames_per_hour) else np.nan,
    }

    return out


def score_validation_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    out = summary_table.copy()

    recall = pd.to_numeric(out.get("event_recall_pct", np.nan), errors="coerce").fillna(0.0)
    precision = pd.to_numeric(out.get("event_precision_pct", np.nan), errors="coerce").fillna(0.0)
    lead = pd.to_numeric(out.get("mean_lead_time_sec", np.nan), errors="coerce").fillna(0.0)
    false_evt = pd.to_numeric(out.get("false_events_per_hour_in_stable_regions", np.nan), errors="coerce").fillna(1e6)
    stable_warn = pd.to_numeric(out.get("stable_warning_frames_per_hour", np.nan), errors="coerce").fillna(1e6)

    # Cap lead-time contribution so absurd values do not dominate ranking.
    lead_capped = lead.clip(lower=0.0, upper=10.0)
    stable_warn_capped = stable_warn.clip(lower=0.0, upper=500.0)

    out["validated_score"] = (
        0.45 * recall
        + 0.25 * precision
        + 1.50 * lead_capped
        - 10.0 * false_evt
        - 0.05 * stable_warn_capped
    )

    if "n_true_chatter_events" in out.columns:
        true_events = pd.to_numeric(out["n_true_chatter_events"], errors="coerce").fillna(0.0)
        out.loc[(true_events > 0) & (recall <= 0), "validated_score"] = -1e9

    return out.sort_values("validated_score", ascending=False).reset_index(drop=True)


def add_simple_rank_score(summary_table: pd.DataFrame) -> pd.DataFrame:
    out = summary_table.copy()

    false_warn = pd.to_numeric(
        out.get("false_warnings_per_hour", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(1e6)

    n_warning = pd.to_numeric(
        out.get("overall_n_warning_frames", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(1e6)

    n_alarm = pd.to_numeric(
        out.get("overall_n_alarm_frames", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(1e6)

    lead = pd.to_numeric(
        out.get("mean_lead_time_sec", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    peak_risk = pd.to_numeric(
        out.get("overall_max_risk_score", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)

    out["simple_rank_score"] = (
        -1.0 * false_warn
        -0.02 * n_warning
        -0.02 * n_alarm
        +0.50 * lead
        +0.20 * peak_risk
    )

    return out.sort_values("simple_rank_score", ascending=False).reset_index(drop=True)

def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def _run_once_with_current_cfg(data: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    df = extract_features(data)
    df = apply_simple_advisory(df)
    events = extract_events(df)
    kpi = compute_kpis(df, events)
    summ = summary(df, events)
    return df, events, kpi, summ


def run_validated_pipeline(
    data: dict[str, np.ndarray],
    labels_csv: str | Path,
    run_name: str = "run_001",
    results_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df, events, kpi, summ = _run_once_with_current_cfg(data)
    labels = load_labels(labels_csv)
    valid = evaluate_against_labels(df, events, labels, run_name=run_name)

    metrics = {}
    metrics.update(kpi)
    metrics.update(summ)
    metrics.update(valid)

    if results_dir is not None:
        out_dir = Path(results_dir)
        mkdir(out_dir)
        df.to_csv(out_dir / "timeline.csv", index=False)
        df.to_csv(out_dir / "features.csv", index=False)
        events.to_csv(out_dir / "events.csv", index=False)
        save_json(metrics, out_dir / "early_warning_kpi.json")
        save_json(summ, out_dir / "summary.json")
        save_plots(df, out_dir)
        if not labels.empty:
            labels.to_csv(out_dir / "labels_used.csv", index=False)

    return df, events, metrics


def sweep_parameters(
    data: dict[str, np.ndarray],
    output_root: str | Path,
    grid: dict[str, list[Any]],
    base_name: str = "sweep",
    labels_csv: str | Path | None = None,
    run_name: str = "run_001",
) -> pd.DataFrame:
    output_root = Path(output_root)
    mkdir(output_root)

    grid_keys = list(grid.keys())
    grid_values = [grid[k] for k in grid_keys]

    labels = load_labels(labels_csv) if labels_csv is not None else None

    original_cfg = deepcopy(CFG)
    rows: list[dict[str, Any]] = []

    try:
        for idx, values in enumerate(product(*grid_values), start=1):
            params = dict(zip(grid_keys, values))
            run_id = f"{base_name}_{idx:03d}"
            run_dir = output_root / run_id
            mkdir(run_dir)

            # apply parameter overrides
            for k, v in params.items():
                setattr(CFG, k, v)

            df, events, kpi, summ = _run_once_with_current_cfg(data)

            row: dict[str, Any] = {"run_name": run_id}
            row.update({f"param__{k}": v for k, v in params.items()})
            row.update(kpi)
            row.update(_flatten_metrics("overall_", summ))

            if labels is not None:
                valid = evaluate_against_labels(df, events, labels, run_name=run_name)
                valid = {f"validation_{k}" if k == "run_name" else k: v for k, v in valid.items()}
                row.update(valid)
                save_json(valid, run_dir / "validation_metrics.json")
            else:
                valid = {}

            df.to_csv(run_dir / "timeline.csv", index=False)
            events.to_csv(run_dir / "events.csv", index=False)
            save_json(kpi, run_dir / "early_warning_kpi.json")
            save_json(summ, run_dir / "summary.json")
            save_plots(df, run_dir)

            rows.append(row)

    finally:
        # restore original config object values
        for name, value in vars(original_cfg).items():
            setattr(CFG, name, value)

    out = pd.DataFrame(rows)
    out.to_csv(output_root / f"{base_name}_summary.csv", index=False)

    if labels is not None and not out.empty:
        ranked = score_validation_table(out)
        ranked.to_csv(output_root / f"{base_name}_ranked_validated.csv", index=False)

    return out