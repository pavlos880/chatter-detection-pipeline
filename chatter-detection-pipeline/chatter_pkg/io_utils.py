"""
Input/output helpers for reading raw data and saving project artifacts.

The focus here is convenience and reproducibility: cached Excel parsing, consistent
JSON saving, and a single loader that returns the arrays used by the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CFG
from .paths import CACHE_DIR, mkdir

log = logging.getLogger("chatter")


def save_json(obj: Any, path: Path) -> None:
    """
    Save Python objects to JSON while handling NumPy and pandas types gracefully.
    """
    def _fix(o):
        """
        Convert non-standard numeric and timestamp types into JSON-friendly values.
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return o

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_fix)


def _cache_path(path: Path) -> Path:
    """
    Return the pickle-cache path associated with one source Excel file.
    """
    mkdir(CACHE_DIR)
    return CACHE_DIR / f"{path.stem}.pkl"


def _read_segment(path: Path) -> pd.DataFrame:
    """
    Read one semicolon-packed Excel segment and convert it into a clean DataFrame.
    """
    cache = _cache_path(path)
    try:
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            return pd.read_pickle(cache)
    except Exception:
        pass

    raw = pd.read_excel(path, header=None, usecols=[0], engine="openpyxl")
    lines = raw.iloc[:, 0].dropna().astype(str).tolist()
    rows = [x.split(";") for x in lines[3:]]
    rows = [r[:6] for r in rows if len(r) >= 6]

    df = pd.DataFrame(
        rows,
        columns=["time", "Std_Housing_OS", "Std_Housing_DS", "GA_speed", "StrM_fce", "U_thick"],
    )

    for col in ["Std_Housing_OS", "Std_Housing_DS", "GA_speed", "StrM_fce", "U_thick"]:
        s = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
        df[col] = pd.to_numeric(s, errors="coerce")

    df["time"] = pd.to_datetime(
        df["time"].astype(str).str.strip(),
        format="%d.%m.%Y %H:%M:%S.%f",
        errors="coerce",
    )
    df = df.dropna().reset_index(drop=True)

    try:
        df.to_pickle(cache)
    except Exception:
        pass

    return df


def load_data(paths: list[Path]) -> dict[str, np.ndarray]:
    """
    Load one or more data files and return the arrays expected by the detector.
    """
    parts = []
    for p in paths:
        if p.exists():
            log.info("Reading %s", p.name)
            parts.append(_read_segment(p))

    if not parts:
        raise FileNotFoundError("No valid Excel files found.")

    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    dt = full["time"].diff().dropna().dt.total_seconds().to_numpy()
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = 1.0 / np.median(dt) if len(dt) else CFG.fs_fallback
    log.info("Inferred sampling rate: %.2f Hz", fs)

    return {
        "time": full["time"].to_numpy(),
        "os": full["Std_Housing_OS"].to_numpy(dtype=np.float32),
        "ds": full["Std_Housing_DS"].to_numpy(dtype=np.float32),
        "speed": full["GA_speed"].to_numpy(dtype=np.float32),
        "tension": full["StrM_fce"].to_numpy(dtype=np.float32),
        "thickness": full["U_thick"].to_numpy(dtype=np.float32),
        "fs": np.array(fs, dtype=np.float64),
    }
