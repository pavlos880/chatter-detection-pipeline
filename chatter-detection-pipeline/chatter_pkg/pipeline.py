"""
Main orchestration for running the detector on one or more discovered runs.

Ties together: loading → feature extraction → event extraction →
advisory layer → reporting → optional dashboard launch.
"""

import logging
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from .config import CFG
from .control import apply_simple_advisory
from .detection import extract_features
from .events import extract_events
from .io_utils import load_data, save_json
from .paths import RESULTS_DIR, mkdir, resolve_all_run_files
from .reporting import compute_kpis, save_plots, summary

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# Dashboard helpers (unchanged)

def _dashboard_script_candidates(results_dir: Path) -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    return [
        project_root / "dashboard.py",
        project_root / "dashboard_live.py",
        project_root / "dashboard-2.py",
        results_dir.parent / "dashboard.py",
        results_dir.parent / "dashboard_live.py",
        results_dir.parent / "dashboard-2.py",
    ]


def launch_dashboard_process(
    results_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8877,
    open_browser: bool = True,
):
    script = next((p for p in _dashboard_script_candidates(results_dir) if p.exists()), None)
    if script is None:
        log.warning("Dashboard script not found.")
        return None

    env = dict(**__import__("os").environ)
    env["CHATTER_RESULTS_DIR"] = str(results_dir.resolve())
    env["CHATTER_DASHBOARD_HOST"] = host
    env["CHATTER_DASHBOARD_PORT"] = str(port)

    proc = subprocess.Popen([sys.executable, str(script)], env=env)
    log.info("Dashboard started with PID %s", proc.pid)

    if open_browser:
        time.sleep(1.0)
        try:
            webbrowser.open(f"http://{host}:{port}")
        except Exception:
            log.warning("Could not open browser automatically.")

    return proc


# Run summary printer

def _print_run_summary(
    run_name: str,
    df,
    events,
    kpi: dict,
) -> None:
    """
    Print a human-readable one-page summary of what the detector found.
    This is useful when running from the command line.
    """
    import numpy as np

    dur   = float(df["t_sec"].max()) if len(df) else 0.0
    n_w   = int((df["state"] == "WATCH").sum())
    pct_w = 100.0 * n_w / max(len(df), 1)

    coh3  = float(df["coh_third"].mean()) if "coh_third" in df.columns else float("nan")
    coh5  = float(df["coh_fifth"].mean()) if "coh_fifth" in df.columns else float("nan")
    dom3  = float(df["dom_third_hz"].dropna().median()) if "dom_third_hz" in df.columns else float("nan")
    dom5  = float(df["dom_fifth_hz"].dropna().median()) if "dom_fifth_hz" in df.columns else float("nan")

    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  RUN SUMMARY: {run_name}")
    print(SEP)
    print(f"  Duration             : {dur:.1f} s  ({dur/60:.1f} min)")
    print(f"  Frames analysed      : {len(df)}")
    print(f"  WARNING events       : {kpi.get('n_events', 0)}")
    print(f"  False warnings/hour  : {kpi.get('false_warnings_per_hour', 0.0):.2f}")
    print(f"  WATCH frames         : {n_w}  ({pct_w:.1f}%)")
    print(f"  Max risk score       : {df['risk_score'].max():.3f}")
    print(f"  Mean risk score      : {df['risk_score'].mean():.3f}")
    print(f"  3rd coherence (mean) : {coh3:.3f}  "
          f"[warning gate: {CFG.min_coherence_third}]")
    print(f"  5th coherence (mean) : {coh5:.3f}  "
          f"[warning gate: {CFG.min_coherence_fifth}]")
    if not (dom3 != dom3):   # not NaN
        print(f"  3rd dominant freq    : {dom3:.1f} Hz")
    if not (dom5 != dom5):
        print(f"  5th dominant freq    : {dom5:.1f} Hz")
    print(SEP + "\n")


# Main pipeline

def run_pipeline(
    results_dir: Path = RESULTS_DIR,
    launch_dashboard: bool = False,
    open_browser: bool = False,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 8765,
) -> None:
    """
    Detect chatter on every discovered Excel run and save outputs.

    Output files per run:
      timeline.csv           — frame-level features + states + advisory
      features.csv           — same as timeline (alias for RF training)
      events.csv             — event-level chatter episode summaries
      early_warning_kpi.json — internal KPIs (no ground truth required)
      summary.json           — one-line snapshot of the run
      *.png                  — diagnostic plots including dashboard_summary.png
    """
    runs = resolve_all_run_files()

    single_run_mode = len(runs) == 1
    base_output_dir = (
        results_dir if single_run_mode
        else results_dir.parent / "results_by_run"
    )
    mkdir(base_output_dir)

    print("\nDetected runs:")
    for run_name, files in runs.items():
        print(f"  {run_name}: {[f.name for f in files]}")

    last_run_dir = None

    for run_name, data_files in runs.items():
        print(f"\nProcessing: {run_name} …")

        data   = load_data(data_files)
        df     = extract_features(data)
        df     = apply_simple_advisory(df)
        events = extract_events(df)
        kpi    = compute_kpis(df, events)
        summ   = summary(df, events)

        run_out_dir = base_output_dir if single_run_mode else base_output_dir / run_name
        mkdir(run_out_dir)

        df.to_csv(run_out_dir / "timeline.csv", index=False)
        df.to_csv(run_out_dir / "features.csv", index=False)
        events.to_csv(run_out_dir / "events.csv", index=False)
        save_json(kpi,  run_out_dir / "early_warning_kpi.json")
        save_json(summ, run_out_dir / "summary.json")
        save_plots(df,  run_out_dir)

        _print_run_summary(run_name, df, events, kpi)

        print(f"  Outputs → {run_out_dir}")
        last_run_dir = run_out_dir

    if launch_dashboard and last_run_dir is not None:
        launch_dashboard_process(
            results_dir=last_run_dir,
            host=dashboard_host,
            port=dashboard_port,
            open_browser=open_browser,
        )
