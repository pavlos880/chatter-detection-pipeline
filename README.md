[README.md](https://github.com/user-attachments/files/27377534/README.md)
# Chatter Detection Pipeline

Offline Python prototype for vibration-based chatter monitoring on a 6-high cold rolling mill. Industrial internship at Elval / Viohalco (Oinofyta plant), supervised by Petros Levakos, January–April 2026.

The system processes accelerometer recordings from operator-side and drive-side roll-stack chocks, extracts narrowband spectral features in the third- and fifth-octave chatter bands, and runs a five-state persistence machine that maps the result to a supervisory speed-reduction recommendation. A separate Random Forest stage was trained on synthetic-injection labels as an exploratory feature-validation benchmark.

## Status

Validated for **specificity** — zero false alarms across 1,221 s of confirmed-stable production data. **Sensitivity** is not yet validated against confirmed real chatter events; rule-based decision-layer tuning on synthetic injections is ongoing. Treat this as an offline prototype, not a deployed detector.

## Pipeline

```
.xlsx recordings  ──►  preprocess  ──►  per-frame features  ──►  state machine  ──►  advisory
   (5 kHz, OS+DS)        (HP 20 Hz,        (FFT, OS↔DS              (5 states,        (×1.00 / ×0.98 /
                          Hann window,      coherence, α growth,    persistence +     ×0.95 / stop)
                          2 s frames,       narrowband peak SNR,    hysteresis)
                          0.25 s hop)       frequency stability)
```

## Repository layout

```
main.py                       Top-level entry — runs the full offline pipeline
dashboard_live.py             Live replay dashboard (machine state, risk trend, advisory)
inject_noise.py               Synthetic chatter injection for sensitivity benchmarks
train_random_forest.py        Exploratory RF trained on injection-window labels
validate_run.py               Per-run KPIs (event recall, lead time, FEW/h)

chatter_pkg/
    config.py                 All thresholds, band definitions, persistence rules
    io_utils.py               Excel loading, timestamp gap handling, channel alignment
    signal_utils.py           Detrending, high-pass, Hann windowing, FFT
    detection.py              Per-frame features: SNR, prominence, coherence
    tracking.py               EMA frequency tracker for the dominant peak
    candidate_logic.py        Per-band vote aggregation
    state_logic.py            Five-state persistence machine with hysteresis
    control.py                State → advisory mapping (speed factor, operator text)
    events.py                 Onset/chatter event extraction from the state trace
    evaluation.py             Labelled-window KPIs
    reporting.py              Timeline CSV, summary JSON, dashboard summary PNG
    paths.py                  Filesystem helpers for run discovery
    pipeline.py               End-to-end orchestrator
```

## Running it

```bash
pip install -r requirements.txt
mkdir data && cp /path/to/recordings/*.xlsx data/
python main.py
```

The pipeline auto-discovers runs in `data/` (or any of the candidate folders listed in `paths.py`), processes each one, and writes per-run outputs to `results_by_run/<run_id>/`:
- `timeline.csv` — per-frame state, risk score, dominant frequencies
- `summary.json` — run-level statistics
- `dashboard_summary.png` — five-row diagnostic plot

Then to launch the live replay dashboard for a processed run:

```bash
python dashboard_live.py
```

## Synthetic benchmarking

`inject_noise.py` reads an existing real recording, embeds a configurable chatter episode (third- or fifth-band, with controllable amplitude, growth rate, onset duration, steady duration), and writes a new Excel file in the same format plus a CSV of ground-truth interval labels. Running `train_random_forest.py` against the labelled set returns held-out test metrics.

The reported balanced RF benchmark used a 5:1 negative:positive ratio, 800 trees, `class_weight='balanced_subsample'`. On the held-out test set it achieved F1 = 0.800, ROC-AUC = 0.968, PR-AUC = 0.823 — confirming the engineered features carry information, while the rule-based decision layer remains the bottleneck.

## Configuration

All tunable parameters live in `chatter_pkg/config.py` as fields of a single `Config` dataclass:

| Group | Key fields |
|---|---|
| Sampling / windowing | `fs_fallback`, `window_sec`, `hop_sec`, `highpass_hz` |
| Frequency search bands | `third_search`, `fifth_search`, `narrow_band_halfwidth_hz` |
| Peak-quality gates | `min_peak_snr_db`, `min_local_concentration` |
| Growth-rate gate | `alpha_threshold`, `alpha_horizon_sec` |
| State machine | `watch_persistence_*`, `warning_persistence_*`, `release_frames_*` |
| Scoring | band weights, composite-score weights, risk thresholds |

A grid-search driver covering 2,187 combinations is included (`tune.py` inside `chatter_pkg`); the current defaults are the top-ranked combination on the available stable runs.

## Data

No production data is committed to this repository. The pipeline expects `.xlsx` files with the columns `Std_Housing_OS`, `Std_Housing_DS`, `GA_speed`, `StrM_fce`, `U_thick`. Any compatible recording at 5 kHz sampling will run end-to-end.

## Report

The full report — dynamic model, signal-processing pipeline, state machine derivation, RF benchmark methodology, limitations and future work — is on the [project page](https://pavloskotronaros.github.io/projects/chatter-detection.html) of my portfolio.
