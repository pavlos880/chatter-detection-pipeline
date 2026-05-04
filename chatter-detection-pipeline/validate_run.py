"""
Command-line entry point for validating one or more detected runs against labels.

It loads the selected runs, applies the validated pipeline, and saves per-run metrics
plus a combined summary file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chatter_pkg.evaluation import run_validated_pipeline
from chatter_pkg.io_utils import load_data
from chatter_pkg.paths import resolve_all_run_files, mkdir


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the validation script.
    """
    parser = argparse.ArgumentParser(
        description="Run validated chatter detection for one or all auto-detected runs."
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="validation_labels.csv",
        help="Path to validation labels CSV. Relative paths are resolved from the project folder.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Exact run name to validate, e.g. Book1 or run_main.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Validate all detected runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="validated_results",
        help="Directory for validated outputs. Relative paths are resolved from the project folder.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    """
    Resolve a user path relative to the validation script location.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def choose_run_names(
    all_runs: dict[str, list[Path]],
    requested: str | None,
    all_runs_flag: bool,
) -> list[str]:
    """
    Decide which detected runs should be validated in the current invocation.
    """
    if all_runs_flag:
        return list(all_runs.keys())

    if requested is not None:
        if requested not in all_runs:
            available = ", ".join(sorted(all_runs.keys()))
            raise ValueError(
                f"Run '{requested}' was not found. Available runs: {available}"
            )
        return [requested]

    if len(all_runs) == 1:
        return [next(iter(all_runs.keys()))]

    print("Multiple runs detected. Defaulting to all runs.")
    return list(all_runs.keys())


def main() -> None:
    """
    Load the requested runs, validate them, and save a combined metrics summary.
    """
    args = parse_args()

    labels_csv = resolve_project_path(args.labels)
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {labels_csv}\n"
            f"Put validation_labels.csv inside:\n{PROJECT_ROOT}\n"
            f"or pass an absolute path with --labels"
        )

    all_runs = resolve_all_run_files()
    run_names = choose_run_names(all_runs, args.run_name, args.all_runs)

    print("\nDetected runs:")
    for name, files in all_runs.items():
        print(f"  {name}: {[f.name for f in files]}")

    print(f"\nUsing labels: {labels_csv}")

    output_root = resolve_project_path(args.output_dir)
    mkdir(output_root)

    all_metrics: list[dict] = []

    for run_name in run_names:
        print(f"\n--- Validating run: {run_name} ---")
        data_files = all_runs[run_name]
        data = load_data(data_files)

        output_dir = output_root / run_name
        mkdir(output_dir)

        _, _, metrics = run_validated_pipeline(
            data=data,
            labels_csv=labels_csv,
            run_name=run_name,
            results_dir=output_dir,
        )

        all_metrics.append(metrics)

        print(json.dumps(metrics, indent=2))

    summary_path = output_root / "validated_metrics_all_runs.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved combined metrics to: {summary_path}")


if __name__ == "__main__":
    main()