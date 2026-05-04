"""
  roll1    : roll1 1.xlsx + roll1 2.xlsx + roll1 3.xlsx   
  roll2    : roll2 1.xlsx + roll2 2.xlsx + roll2 3.xlsx   
  roll3    : roll 3 1.xlsx + roll 3 2.xlsx

  1. Reads source files and concatenates into one in-memory signal.
  2. Injects synthetic chatter at well-defined sample indices.
  3. Splits output into Excel-safe chunks (EXCEL_MAX_DATA_ROWS rows each).
     Named: synthetic_injected_{run_name}_chunk{NN}.xlsx
  4. Writes labels CSV that matches the actual injection location.
  5. Writes a verification JSON with measured vs intended timings.
  6. Runs detector on all chunks as one combined run.
  7. Trains exploratory RF with proper injection-based labelling.
"""

from __future__ import annotations
import json, sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAND_MODE           : str = "third"     
STRENGTH            : str = "strong"    
RANDOM_SEED         : int = 42
EXCEL_MAX_DATA_ROWS : int = 1_000_000   


RUN_GROUPS: dict[str, list[str]] = {
    "roll1":    ["roll1 1", "roll1 2", "roll1 3"],
    "roll2":    ["roll2 1", "roll2 2", "roll2 3"],
    "roll3":    ["roll 3 1", "roll 3 2"],
}


#  Project discovery 

def detect_project_root() -> Path:
    return Path(__file__).resolve().parent


def synthetic_output_dirs(root: Path) -> dict[str, Path]:
    base = root / "outputs" / "synthetic"
    dirs = {
        "base": base,
        "inputs": base / "files",
        "labels": base / "labels",
        "metadata": base / "metadata",
        "plots": base / "plots",
        "validated": base / "validated_results",
        "models": base / "synthetic_models",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def find_data_dir(root: Path) -> Path:
    for folder in [root/"data", root/"Data", root,
                   root.parent/"data", root.parent/"Data"]:
        if folder.is_dir() and list(folder.glob("*.xlsx")):
            return folder
    raise FileNotFoundError("No .xlsx files found near project root.")


def resolve_run_groups(data_dir: Path) -> dict[str, list[Path]]:
    available = {fp.stem.lower().strip(): fp
                 for fp in sorted(data_dir.glob("*.xlsx"))
                 if not fp.stem.lower().startswith("synthetic_injected")}
    out: dict[str, list[Path]] = {}
    for run_name, stems in RUN_GROUPS.items():
        missing = [s for s in stems if s.lower() not in available]
        if missing:
            print(f"  skip {run_name}: missing {missing}")
            continue
        out[run_name] = [available[s.lower()] for s in stems]
        print(f"  found {run_name}: {[p.name for p in out[run_name]]}")
    return out


#  Reading +  concatenation 

PIPE_COLS = ["time", "Std_Housing_OS", "Std_Housing_DS",
             "GA_speed", "StrM_fce", "U_thick"]


def read_pipeline_excel(path: Path) -> pd.DataFrame:
    raw   = pd.read_excel(path, header=None, usecols=[0], engine="openpyxl")
    lines = raw.iloc[:, 0].dropna().astype(str).tolist()
    rows  = [x.split(";") for x in lines[3:]]
    rows  = [r[:6] for r in rows if len(r) >= 6]
    df    = pd.DataFrame(rows, columns=PIPE_COLS)
    for c in PIPE_COLS[1:]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )
    df["time"] = pd.to_datetime(
        df["time"].astype(str).str.strip(),
        format="%d.%m.%Y %H:%M:%S.%f", errors="coerce",
    )
    return df.dropna().reset_index(drop=True)


def read_and_concat(files: list[Path]) -> pd.DataFrame:
    parts = []
    for fp in files:
        print(f"    reading {fp.name} …")
        parts.append(read_pipeline_excel(fp))
    return pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)


def infer_fs(df: pd.DataFrame) -> float:
    dt = df["time"].diff().dropna().dt.total_seconds().to_numpy()
    dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]
    return float(1.0 / np.median(dt)) if len(dt) else 5000.0


#  Injection 

@dataclass
class Episode:
    label:        str
    start_sec:    float
    onset_sec:    float
    steady_sec:   float
    freq_hz:      float
    amplitude:    float
    growth_rate:  float


def build_episodes(dur_detector_sec: float, band_mode: str, strength: str) -> list[Episode]:
    # Amplitude is given as a multiplicative scale on the raw signal (g or m/s²).
    amp = {"subtle": 0.010, "medium": 0.020, "strong": 0.035}[strength]
    # Growth rate in 1/s controls how fast the onset envelope reaches amplitude.
    gr  = {"subtle": 0.80,  "medium": 1.20,  "strong": 1.60} [strength]

    t3 = max(20.0, 0.25 * dur_detector_sec)  
    t5 = max(35.0, 0.55 * dur_detector_sec)  

    if band_mode == "third":
        return [Episode("third", t3, 3.0, 6.0, 165.0, amp,        gr)]
    if band_mode == "fifth":
        # Fifth band uses the same amplitude now — v1 scaled it down by 0.9 but
        # the narrow-band energy was already reduced by the sample/timestamp
        # mismatch. With continuous-sample-space sinusoid generation (below)
        # the fifth injection now produces comparable energy to the third.
        return [Episode("fifth", t5, 2.0, 5.0, 720.0, amp,        gr)]
    return [Episode("third", t3, 3.0, 6.0, 165.0, amp,            gr),
            Episode("fifth", t5, 2.0, 5.0, 720.0, amp,            gr)]


def smooth_gate(n: int) -> np.ndarray:
    """
    Raised-cosine windowing at the start and end of the steady segment so
    the injection doesn't introduce click artifacts at the boundaries.
    """
    x = np.ones(n)
    r = max(1, int(0.05 * n))
    f = max(1, int(0.10 * n))
    x[:r]  = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
    x[-f:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, f)))
    return x


def inject_episode(os_: np.ndarray, ds_: np.ndarray,
                   fs: float, ep: Episode,
                   rng: np.random.Generator) -> tuple[int, int]:
    """
    Add one chatter episode to the OS and DS arrays in-place.
    """
    n = len(os_)

    start_idx = int(round(ep.start_sec * fs))
    onset_n   = int(round(ep.onset_sec * fs))
    steady_n  = int(round(ep.steady_sec * fs))
    end_idx   = min(n, start_idx + onset_n + steady_n)
    start_idx = max(0, start_idx)

    if end_idx - start_idx < 10:
        return start_idx, end_idx

    idx = np.arange(start_idx, end_idx)
    lt  = (idx - start_idx) / fs        

    # Envelope: exponential rise during onset
    env = np.zeros(len(lt))
    om  = lt < ep.onset_sec                     
    sm  = ~om                                   

    if om.any():
        ot   = lt[om]
        raw  = np.exp(ep.growth_rate * ot) - 1
        env[om] = ep.amplitude * raw / max(np.exp(ep.growth_rate * ep.onset_sec) - 1, 1e-12)

    if sm.any():
        n_steady = int(sm.sum())
        env[sm] = ep.amplitude * smooth_gate(n_steady)

    ph   = rng.uniform(0, 2 * np.pi)
    sh_o = np.sin(2 * np.pi * ep.freq_hz * lt + ph)
    sh_d = np.sin(2 * np.pi * ep.freq_hz * lt + ph + 0.08)     
    mod  = 1 + 0.08 * np.sin(2 * np.pi * 1.1 * lt + rng.uniform(0, 2 * np.pi))

    noise_os = rng.normal(scale=0.10, size=len(idx))
    noise_ds = rng.normal(scale=0.10, size=len(idx))

    os_[idx] += env * mod * (0.92 * sh_o + 0.08 * noise_os)
    ds_[idx] += env * mod * (0.88 * sh_d + 0.12 * noise_ds)

    return start_idx, end_idx


def apply_injection(df: pd.DataFrame, episodes: list[Episode],
                    fs: float, seed: int = 42) -> tuple[pd.DataFrame, list[dict]]:

    out = df.copy()
    rng = np.random.default_rng(seed)

    os_ = out["Std_Housing_OS"].to_numpy(float).copy()
    ds_ = out["Std_Housing_DS"].to_numpy(float).copy()

    verify: list[dict] = []
    for ep in episodes:
        start_idx, end_idx = inject_episode(os_, ds_, fs, ep, rng)
        verify.append({
            "label":              ep.label,
            "intended_start_sec": float(ep.start_sec),
            "intended_end_sec":   float(ep.start_sec + ep.onset_sec + ep.steady_sec),
            "actual_start_idx":   int(start_idx),
            "actual_end_idx":     int(end_idx),
            "actual_start_sec":   float(start_idx / fs),
            "actual_end_sec":     float(end_idx   / fs),
            "freq_hz":            float(ep.freq_hz),
            "amplitude":          float(ep.amplitude),
        })

    out["Std_Housing_OS"] = os_
    out["Std_Housing_DS"] = ds_
    return out, verify


#  Labels 

def build_labels(run_name: str, episodes: list[Episode],
                 dur_detector_sec: float) -> pd.DataFrame:

    rows: list[dict] = []
    occ:  list[tuple[float, float]] = []

    for ep in sorted(episodes, key=lambda e: e.start_sec):
        onset_start, onset_end   = ep.start_sec, ep.start_sec + ep.onset_sec
        steady_start, steady_end = onset_end, ep.start_sec + ep.onset_sec + ep.steady_sec
        occ.append((onset_start, steady_end))

        rows += [
            {"run_name": run_name,
             "start_sec": round(onset_start, 6), "end_sec": round(onset_end, 6),
             "label": "onset",
             "notes": f"synthetic_{ep.label}_onset_{ep.freq_hz:.0f}Hz"},
            {"run_name": run_name,
             "start_sec": round(steady_start, 6), "end_sec": round(steady_end, 6),
             "label": "chatter",
             "notes": f"synthetic_{ep.label}_steady_{ep.freq_hz:.0f}Hz"},
        ]

    cur = 0.0
    for s, e in sorted(occ):
        if s > cur:
            rows.append({"run_name": run_name,
                         "start_sec": round(cur, 6), "end_sec": round(s, 6),
                         "label": "stable", "notes": "synthetic_stable_region"})
        cur = max(cur, e)
    if dur_detector_sec > cur:
        rows.append({"run_name": run_name,
                     "start_sec": round(cur, 6),
                     "end_sec": round(dur_detector_sec, 6),
                     "label": "stable", "notes": "synthetic_stable_region"})

    return (pd.DataFrame(rows)
              .sort_values(["start_sec", "end_sec", "label"])
              .reset_index(drop=True))



HEADER = [
    "Synthetic injected dataset",
    "Generated for existing pipeline compatibility",
    "time;Std_Housing_OS;Std_Housing_DS;GA_speed;StrM_fce;U_thick",
]


def _fmt_time(ts): return pd.Timestamp(ts).strftime("%d.%m.%Y %H:%M:%S.%f")
def _fmt_num(x):   return "" if pd.isna(x) else f"{float(x):.6f}".replace(".", ",")


def _df_to_lines(df: pd.DataFrame) -> list[str]:
    return [";".join([_fmt_time(r["time"]),
                      _fmt_num(r["Std_Housing_OS"]),
                      _fmt_num(r["Std_Housing_DS"]),
                      _fmt_num(r["GA_speed"]),
                      _fmt_num(r["StrM_fce"]),
                      _fmt_num(r["U_thick"])])
            for _, r in df.iterrows()]


def write_chunks(df: pd.DataFrame, run_name: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = _df_to_lines(df)
    total = len(lines)

    if total <= EXCEL_MAX_DATA_ROWS:
        p = out_dir / f"synthetic_injected_{run_name}_chunk01.xlsx"
        pd.DataFrame(HEADER + lines).to_excel(p, header=False, index=False, engine="openpyxl")
        print(f"  wrote: {p.name}  ({total:,} rows)")
        return [p]

    n   = int(np.ceil(total / EXCEL_MAX_DATA_ROWS))
    csz = int(np.ceil(total / n))
    ovl = max(0, int(0.05 * csz))
    print(f"  {total:,} rows → {n} chunks ({csz:,} rows, {ovl} overlap)")
    written: list[Path] = []
    for i in range(n):
        s = max(0, i * csz - (ovl if i > 0 else 0))
        e = min(total, (i + 1) * csz)
        p = out_dir / f"synthetic_injected_{run_name}_chunk{i+1:02d}.xlsx"
        pd.DataFrame(HEADER + lines[s:e]).to_excel(p, header=False, index=False, engine="openpyxl")
        print(f"  wrote: {p.name}  ({e-s:,} rows, source {s}–{e})")
        written.append(p)
    return written


#  Plots 

def _dec(n): return 20 if n > 500_000 else 10 if n > 200_000 else 4 if n > 50_000 else 1


def _shade(ax, episodes):
    c = {"third": "steelblue", "fifth": "darkorange"}
    for ep in episodes:
        col = c.get(ep.label, "green")
        ax.axvspan(ep.start_sec, ep.start_sec + ep.onset_sec,
                   alpha=0.20, color=col)
        ax.axvspan(ep.start_sec + ep.onset_sec,
                   ep.start_sec + ep.onset_sec + ep.steady_sec,
                   alpha=0.10, color=col)


def save_plots(orig: pd.DataFrame, inj: pd.DataFrame,
               episodes: list[Episode], fs: float,
               plot_dir: Path) -> list[Path]:
    """
    Save a 4-panel overview of the injected vs original data. The x-axis is
    in detector-frame seconds (sample_index / fs) so the episode shading
    matches what the detector will see.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    d = _dec(len(inj))

    # x axis in detector-frame seconds
    n = len(inj)
    x = (np.arange(n) / fs)[::d]

    oi = orig["Std_Housing_OS"].to_numpy(float)[::d]
    ii = inj ["Std_Housing_OS"].to_numpy(float)[::d]
    od = orig["Std_Housing_DS"].to_numpy(float)[::d]
    id_= inj ["Std_Housing_DS"].to_numpy(float)[::d]
    sp = inj ["GA_speed"].to_numpy(float)[::d]
    fo = inj ["StrM_fce"].to_numpy(float)[::d]
    th = inj ["U_thick"].to_numpy(float)[::d]

    saved: list[Path] = []
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Injected dataset overview (detector-frame seconds)",
                 fontsize=13, fontweight="bold")

    axes[0].plot(x, ii, lw=0.6, label="Injected OS", color="#2E75B6")
    axes[0].plot(x, oi, lw=0.6, label="Original OS", color="#A9CCE3", alpha=0.7)
    axes[0].set_ylabel("OS accel"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(x, id_, lw=0.6, label="Injected DS", color="#ED7D31")
    axes[1].plot(x, od,  lw=0.6, label="Original DS", color="#FAD7A0", alpha=0.7)
    axes[1].set_ylabel("DS accel"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].plot(x, fo, lw=0.6, color="#70AD47")
    axes[2].set_ylabel("Force"); axes[2].grid(alpha=0.3)

    axes[3].plot(x, sp, lw=0.6, label="Speed",     color="#2E75B6")
    axes[3].plot(x, th, lw=0.6, label="Thickness", color="#7030A0")
    axes[3].set_ylabel("Speed / Thickness")
    axes[3].set_xlabel("Time [s]  (detector-frame)")
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    for ax in axes:
        _shade(ax, episodes)

    plt.tight_layout()
    p = plot_dir / "00_overview.png"
    plt.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)
    return saved


#  Detector integration 

def run_detector(project_root: Path, validated_root: Path, chunks: list[Path],
                 labels: Path, run_name: str) -> tuple[Path, Path]:
    """
    Run the full detector pipeline on the injected chunks as a combined run.
    Output goes to validated_results/{run_name}/.
    """
    try:
        from chatter_pkg.io_utils   import load_data, save_json
        from chatter_pkg.detection  import extract_features
        from chatter_pkg.control    import apply_simple_advisory
        from chatter_pkg.events     import extract_events
        from chatter_pkg.reporting  import compute_kpis, summary, save_plots as rplots
    except ImportError as e:
        raise RuntimeError("Cannot import pipeline modules.") from e

    run_dir = validated_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for i, fp in enumerate(chunks):
        print(f"    chunk {i+1}/{len(chunks)}: {fp.name}")

    # Load all chunks together so overlapping boundary rows are de-duplicated by timestamp.
    # This avoids double-counting overlap regions and keeps the validated synthetic run
    # in the same time base as the original injected signal.
    merged = extract_features(load_data(chunks))
    merged = apply_simple_advisory(merged)
    events = extract_events(merged)
    kpi    = compute_kpis(merged, events)
    summ   = summary(merged, events)

    merged.to_csv(run_dir / "features.csv", index=False)
    merged.to_csv(run_dir / "timeline.csv", index=False)
    events.to_csv(run_dir / "events.csv",   index=False)
    save_json(kpi,  run_dir / "early_warning_kpi.json")
    save_json(summ, run_dir / "summary.json")
    rplots(merged, run_dir)

    print(f"  events detected: {len(events)}")
    if len(events):
        for _, ev in events.iterrows():
            print(f"    t={ev['start_t_sec']:.1f}–{ev['end_t_sec']:.1f}s  "
                  f"state={ev.get('peak_state','?')}  "
                  f"risk={ev.get('peak_risk_score',0):.3f}")

    return run_dir, run_dir / "features.csv"


#  Verification 

def verify_injection_vs_labels(features_path: Path, labels_path: Path,
                               verify_records: list[dict]) -> dict:
    """
    Cross-check: does the detector actually see the injection where the
    labels say it should? Reports per-episode, for the report and for
    debugging future injection framework changes.
    """
    try:
        df = pd.read_csv(features_path)
    except Exception:
        return {"status": "skip", "reason": "no features.csv"}

    if "t_sec" not in df.columns or "e_third" not in df.columns:
        return {"status": "skip", "reason": "missing expected columns"}

    results: list[dict] = []
    overall_status = "ok"
    for rec in verify_records:
        band_col = "e_third" if rec["label"] == "third" else "e_fifth"

        # Look inside the injection window for the peak band energy.
        mask = (df["t_sec"] >= rec["actual_start_sec"]) & \
               (df["t_sec"] <= rec["actual_end_sec"])
        inside = df.loc[mask, band_col]
        outside = df.loc[~mask, band_col]

        peak_inside  = float(inside.max())  if len(inside)  else float("nan")
        peak_outside = float(outside.max()) if len(outside) else float("nan")

        results.append({
            "label":            rec["label"],
            "freq_hz":          rec["freq_hz"],
            "window_start_sec": rec["actual_start_sec"],
            "window_end_sec":   rec["actual_end_sec"],
            "peak_energy_inside_window":  peak_inside,
            "peak_energy_outside_window": peak_outside,
            "inside_over_outside_ratio":  (peak_inside / max(peak_outside, 1e-12)),
        })

        ratio = peak_inside / max(peak_outside, 1e-12)
        if ratio < 1.0:
            overall_status = "weak"
            print(f"  WARN [{rec['label']}] injection peak inside window ({peak_inside:.2f}) "
                  f"is weaker than outside ({peak_outside:.2f}). "
                  f"Consider increasing amplitude or moving the episode to a quieter region.")
        elif ratio < 3.0:
            if overall_status == "ok":
                overall_status = "borderline"
            print(f"  WARN [{rec['label']}] injection peak inside window ({peak_inside:.2f}) "
                  f"is not clearly stronger than outside ({peak_outside:.2f}). "
                  f"Consider STRENGTH='strong' or check the injection amplitude.")
        else:
            print(f"  OK [{rec['label']}] inside/outside energy ratio = {ratio:.1f}x")

    return {"status": overall_status, "per_episode": results}


#  Per-run orchestration 

def process_run(run_name: str, source_files: list[Path],
                data_dir: Path, root: Path, synth_dirs: dict[str, Path]) -> None:
    print(f"\n{'='*60}\n  run: {run_name}\n  files: {[f.name for f in source_files]}\n{'='*60}")
    print("  reading and concatenating …")
    df = read_and_concat(source_files)

    # Detector-frame duration = n_samples / fs. This is the coordinate the
    # pipeline uses internally, and the one the labels will be written in.
    fs = infer_fs(df)
    n  = len(df)
    dur_detector = float(n) / fs
    dur_stamp    = float((df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds())
    gap_sec      = dur_stamp - dur_detector
    print(f"  rows: {n:,}  |  fs: {fs:.0f} Hz  |  "
          f"detector-frame dur: {dur_detector:.1f}s  |  "
          f"timestamp dur: {dur_stamp:.1f}s  |  "
          f"timestamp gaps: {gap_sec:.1f}s")

    episodes = build_episodes(dur_detector, BAND_MODE, STRENGTH)
    for ep in episodes:
        print(f"  [{ep.label}] start={ep.start_sec:.1f}s  freq={ep.freq_hz:.0f}Hz  "
              f"onset={ep.onset_sec:.1f}s  steady={ep.steady_sec:.1f}s  "
              f"amp={ep.amplitude:.4f}")

    print("  injecting …")
    inj, verify = apply_injection(df, episodes, fs, seed=RANDOM_SEED)

    print("  writing Excel chunks …")
    chunks = write_chunks(inj, run_name, synth_dirs["inputs"])

    labels_csv = synth_dirs["labels"] / f"validation_labels_synthetic_{run_name}.csv"
    build_labels(f"synthetic_injected_{run_name}", episodes, dur_detector).to_csv(
        labels_csv, index=False,
    )
    print(f"  labels → {labels_csv.name}")

    metadata = {
        "run_name":        run_name,
        "source_files":    [str(x) for x in source_files],
        "chunks":          [str(x) for x in chunks],
        "labels":          str(labels_csv),
        "fs":              fs,
        "n_samples":       n,
        "duration_detector_sec": dur_detector,
        "duration_timestamp_sec": dur_stamp,
        "timestamp_gap_sec":     gap_sec,
        "band_mode":       BAND_MODE,
        "strength":        STRENGTH,
        "seed":            RANDOM_SEED,
        "n_chunks":        len(chunks),
        "episodes":        [asdict(ep) for ep in episodes],
        "verification":    verify,
    }
    with open(synth_dirs["metadata"] / f"synthetic_injected_{run_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    plot_dir = synth_dirs["plots"] / f"synthetic_injected_{run_name}"
    saved    = save_plots(df, inj, episodes, fs, plot_dir)
    print(f"  {len(saved)} plot(s) → {plot_dir}")

    try:
        run_dir, fp = run_detector(root, synth_dirs["validated"], chunks, labels_csv,
                                   f"synthetic_injected_{run_name}")

        # Post-injection verification
        vres = verify_injection_vs_labels(fp, labels_csv, verify)
        with open(run_dir / "injection_verification.json", "w") as f:
            json.dump(vres, f, indent=2)

        # RF training now uses the verified label CSV
        train_rf_on_injection(fp, labels_csv, synth_dirs["models"], f"synthetic_injected_{run_name}")
    except Exception as exc:
        print(f"  WARNING: detector / RF step failed: {exc}")

    print(f"  done: {run_name}")


#  RF training (injection-label-based, not detector-state-based) 

def train_rf_on_injection(features_path: Path, labels_path: Path,
                          models_root: Path, run_name: str) -> None:
    """
    Train an exploratory Random Forest with labels derived from the
    injection ground truth (onset and chatter windows labelled positive,
    stable labelled negative), NOT from the rule-based detector's state.

    This is the principled sensitivity-style RF that replaces v1's
    circular-label setup on synthetic runs.

    The more thorough training logic (stratified-group CV, Platt calibration,
    ROC / PR curves) lives in train_random_forest.py. This function is a
    quick in-line RF so each injection run produces a result without a
    separate training step.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute   import SimpleImputer
    from sklearn.metrics  import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    from sklearn.model_selection import StratifiedGroupKFold

    df     = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    # Apply labels to every frame based on its t_sec falling inside a labelled
    # window. onset + chatter = positive, stable = negative.
    if "t_sec" not in df.columns:
        print("  skip RF: no t_sec column in features.csv")
        return

    df = df.copy()
    df["label"] = 0
    positive_types = {"onset", "chatter"}
    for _, lab in labels.iterrows():
        if str(lab["label"]).lower() in positive_types:
            mask = (df["t_sec"] >= float(lab["start_sec"])) & \
                   (df["t_sec"] <  float(lab["end_sec"]))
            df.loc[mask, "label"] = 1

    n_pos = int(df["label"].sum())
    n_neg = int(len(df) - n_pos)
    print(f"  RF labels from injection: {n_pos} positive, {n_neg} negative")
    if n_pos < 5 or n_neg < 5:
        print("  skip RF: too few positives or negatives after labelling")
        return

    # Feature selection
    feature_cols = [c for c in [
        "vib_rms_os","vib_rms_ds","vib_rms_mean","rms_ratio_os_ds","corr_os_ds",
        "coh_third","coh_fifth","spec_flatness","kurtosis","skewness",
        "e_wide","e_third","e_fifth",
        "alpha_third","alpha_fifth","growth_third","growth_fifth",
        "dom_third_hz","dom_fifth_hz","third_valid","fifth_valid",
        "third_energy_ratio","fifth_energy_ratio",
        "third_prominence_ratio","fifth_prominence_ratio",
        "third_snr_db","fifth_snr_db",
    ] if c in df.columns]

    X = df[feature_cols]
    y = df["label"].astype(int)

    # Group by 10-second blocks to prevent adjacent-frame leakage across
    # the train / test split.
    groups = (pd.to_numeric(df["t_sec"], errors="coerce").fillna(0) // 10).astype(int)

    # Simple 75/25 stratified-group split (CV lives in train_random_forest.py)
    try:
        sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
        tr, te = next(sgkf.split(X, y, groups=groups))
    except ValueError:
        # Fallback to a plain split if stratified-group fails (very small labels)
        from sklearn.model_selection import GroupShuffleSplit
        sp = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
        tr, te = next(sp.split(X, y, groups=groups))

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X.iloc[tr])
    Xte = imp.transform(X.iloc[te])
    ytr, yte = y.iloc[tr], y.iloc[te]

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12,
        min_samples_split=8, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(Xtr, ytr)
    yp  = rf.predict(Xte)
    ypr = rf.predict_proba(Xte)[:, 1]

    metrics = {
        "accuracy":     float(accuracy_score(yte, yp)),
        "precision":    float(precision_score(yte, yp, zero_division=0)),
        "recall":       float(recall_score(yte, yp, zero_division=0)),
        "f1":           float(f1_score(yte, yp, zero_division=0)),
        "roc_auc":      (float(roc_auc_score(yte, ypr))
                         if len(np.unique(yte)) > 1 else None),
        "pr_auc":       (float(average_precision_score(yte, ypr))
                         if len(np.unique(yte)) > 1 else None),
        "confusion_matrix": confusion_matrix(yte, yp).tolist(),
        "n_train":      int(len(Xtr)),
        "n_test":       int(len(Xte)),
        "n_pos_train":  int(ytr.sum()),
        "n_pos_test":   int(yte.sum()),
        "label_source": "injection_windows",
        "note":         "Labels from injection ground truth (onset + chatter). "
                        "Measures sensitivity to synthetic chatter, not "
                        "agreement with the rule-based detector.",
    }

    imps = (pd.DataFrame({"feature": feature_cols,
                          "importance": rf.feature_importances_})
              .sort_values("importance", ascending=False))

    md = models_root / run_name
    md.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf,  md / "random_forest_chatter.pkl")
    joblib.dump(imp, md / "rf_imputer.pkl")
    imps.to_csv(md / "rf_feature_importance.csv", index=False)
    with open(md / "rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    auc_str = f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else "—"
    print(f"  RF (injection labels): acc={metrics['accuracy']:.3f}  "
          f"rec={metrics['recall']:.3f}  f1={metrics['f1']:.3f}  "
          f"roc_auc={auc_str}")
    print(f"  top 5 features: {', '.join(imps['feature'].head(5).tolist())}")


#  Main 

def main() -> None:
    root       = detect_project_root()
    data_dir   = find_data_dir(root)
    synth_dirs = synthetic_output_dirs(root)
    print(f"Chatter Injection — group-aware automatic\n"
          f"  band: {BAND_MODE}  strength: {STRENGTH}  seed: {RANDOM_SEED}\n"
          f"  data: {data_dir}\n\nresolving run groups …")
    groups = resolve_run_groups(data_dir)
    if not groups:
        print("ERROR: no complete run groups found.")
        sys.exit(1)

    failed: list[str] = []
    for rn, files in groups.items():
        try:
            process_run(rn, files, data_dir, root, synth_dirs)
        except Exception as exc:
            print(f"\nERROR {rn}: {exc}")
            import traceback
            traceback.print_exc()
            failed.append(rn)

    print(f"\n{'='*60}")
    print(f"done. {len(groups)-len(failed)}/{len(groups)} runs processed.")
    if failed:
        print(f"failed: {', '.join(failed)}")
    print(f"outputs → {synth_dirs['base']}")
    print(f"source files were NOT modified.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ncancelled.")
        sys.exit(1)