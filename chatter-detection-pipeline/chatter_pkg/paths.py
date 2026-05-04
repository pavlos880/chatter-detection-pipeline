"""
Shared filesystem helpers for locating data and output folders.


Run grouping rules
------------------
Real source files (current naming):
  roll1 1 / roll1 2 / roll1 3           → roll1
  roll2 1 / roll2 2 / roll2 3           → roll2
  roll 3 1 / roll 3 2                   → roll3



"""

from pathlib import Path
import re

SCRIPT_DIR  = Path(__file__).resolve().parent.parent
RESULTS_DIR = SCRIPT_DIR / "results"
CACHE_DIR   = SCRIPT_DIR / "cache"


def mkdir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _find_data_root() -> Path:
    """Search likely folders and return the first one that contains .xlsx files."""
    candidates = [
        SCRIPT_DIR / "data",
        SCRIPT_DIR / "Data",
        SCRIPT_DIR,
        SCRIPT_DIR.parent / "data",
        SCRIPT_DIR.parent / "Data",
        Path("/mnt/data"),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            if list(p.glob("*.xlsx")):
                return p
    raise FileNotFoundError(
        "Could not find a data folder with .xlsx files.\n"
        f"Tried: {[str(p) for p in candidates]}"
    )


def _run_key_for(stem: str) -> str:
    """
    Map a file stem to a logical run key.

    Rules are evaluated in order; the first match wins. Comparison is done on a
    normalised lowercase form so 'roll1 1', 'Roll1_1' and 'roll1-1' all match.
    """
    # Normalise: lowercase, strip, collapse any whitespace/underscore/dash to single space
    name = stem.lower().strip()
    norm = re.sub(r"[\s_\-]+", " ", name).strip()

    # ── Injected chunk files (written by the injection script) ────────────
    # Pattern: synthetic_injected_{run_name}_chunk{NN}
    # Note: compare against the ORIGINAL `name` here, not `norm`, because
    # underscores in "synthetic_injected_" are meaningful.
    if name.startswith("synthetic_injected_"):
        remainder = name[len("synthetic_injected_"):]
        if "_chunk" in remainder:
            remainder = remainder.rsplit("_chunk", 1)[0]
        return f"synthetic_{remainder}"

    #  Current real source runs 
    # roll1 1 / roll1 2 / roll1 3  
    if re.fullmatch(r"roll1\s*\d+", norm):
        return "roll1"

    # roll2 1 / roll2 2 / roll2 3  
    if re.fullmatch(r"roll2\s*\d+", norm):
        return "roll2"

    # roll 3 1 / roll 3 2  
    if re.fullmatch(r"roll\s*3\s*\d+", norm):
        return "roll3"

    #  Fallback
    return stem


def resolve_all_run_files() -> dict[str, list[Path]]:
    """
    Group discovered Excel files into logical run names used by the pipeline.

    Returns a dict mapping run_name → sorted list of file paths. Files within
    each run are sorted by name so multi-part runs are processed in
    chronological order (part 1 before part 2, chunk01 before chunk02, etc.).
    """
    data_root  = _find_data_root()
    xlsx_files = sorted(data_root.glob("*.xlsx"))

    runs: dict[str, list[Path]] = {}
    for fp in xlsx_files:
        key = _run_key_for(fp.stem)
        runs.setdefault(key, []).append(fp)

    for key in runs:
        runs[key] = sorted(runs[key], key=lambda p: p.stem.lower())

    return runs