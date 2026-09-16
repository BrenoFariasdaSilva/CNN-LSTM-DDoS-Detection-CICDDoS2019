"""Locate direct day-level CICDDoS2019 source CSVs, excluding nested artifacts."""

from pathlib import Path
from typing import List


def discover_csv_files(root: Path) -> List[Path]:
    """Return sorted raw CSV paths from 01-12 and 03-11."""
    files = sorted(
        path for day in ("01-12", "03-11") for path in (root / day).glob("*.csv") if path.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No .csv files were found directly under 01-12 or 03-11 in {root}")
    return files
