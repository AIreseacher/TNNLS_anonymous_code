# -*- coding: utf-8 -*-
"""
Batch script to find all Excel files containing '_results' in their name
under a given directory, read the last column (containing checkpoint paths),
and copy all files from the corresponding directories to another root directory
(checkpoints2/) while preserving relative paths.

Usage:
    python copy_checkpoints.py
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from typing import List, Iterable


def find_results_excels(results_dir: Path, name_substring: str = "_results") -> List[Path]:
    """Recursively find all .xlsx files containing the substring in their filename."""
    excels = []
    for p in results_dir.rglob("*.xlsx"):
        if name_substring in p.name:
            excels.append(p)
    return sorted(excels)


def read_last_column_paths(xlsx_path: Path) -> List[str]:
    """Read the last column from a .xlsx file and return non-empty checkpoint paths."""
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(xlsx_path)

    if df.shape[1] == 0:
        return []

    col = df.iloc[:, -1].dropna().astype(str).str.strip()
    # Keep only rows containing 'checkpoints/'
    col = col[col.str.contains(r"checkpoints/")]
    return col.tolist()


def ensure_under_checkpoints(path_str: str):
    """
    Convert a string path to Path and ensure it contains 'checkpoints'.
    Extract the relative directory after 'checkpoints/'.
    Example:
        checkpoints/Brain/ResNet101_ResNet18/nosra_nogcr/seed_1234/best.pth
        -> rel_dir = Brain/ResNet101_ResNet18/nosra_nogcr/seed_1234
    """
    p = Path(path_str)
    parts = list(p.parts)
    if "checkpoints" not in parts:
        raise ValueError(f"'checkpoints' not found in path: {path_str}")
    idx = parts.index("checkpoints")
    after = parts[idx + 1:]
    if not after:
        raise ValueError(f"No relative path after 'checkpoints': {path_str}")
    rel_dir = Path(*after).parent
    src_dir = Path("checkpoints") / rel_dir
    return rel_dir, src_dir


def copy_dir_flat(src_dir: Path, dst_root: Path) -> int:
    """
    Copy all files recursively from src_dir to dst_root, preserving structure
    relative to 'checkpoints/'.
    Returns the number of copied files.
    """
    if not src_dir.exists():
        return 0

    copied = 0
    for item in src_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(Path("checkpoints"))
            dst_path = dst_root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_path)
            copied += 1
    return copied


def collect_unique_dirs(paths: Iterable[str]) -> List[Path]:
    """Collect unique seed directories (relative to checkpoints/) from all paths."""
    uniq = []
    seen = set()
    for s in paths:
        try:
            rel_dir, _src_dir = ensure_under_checkpoints(s)
        except Exception:
            continue
        if rel_dir not in seen:
            uniq.append(rel_dir)
            seen.add(rel_dir)
    return uniq


def main(
    results_dir: str = "/nas/unas15/chentao/TNNLS/results/result",
    dst_root: str = "checkpoints3",
    name_substring: str = "_results",
    dry_run: bool = False,
):
    """Main routine for collecting checkpoint directories and copying files."""
    results_dir = Path(results_dir).expanduser()
    dst_root = Path(dst_root)

    print(f"[INFO] Scanning directory: {results_dir}")
    excels = find_results_excels(results_dir, name_substring=name_substring)
    if not excels:
        print("[WARN] No Excel files found containing '_results'.")
        return

    all_paths = []
    for x in excels:
        paths = read_last_column_paths(x)
        print(f"[INFO] {x.name}: found {len(paths)} candidate paths")
        all_paths.extend(paths)

    uniq_rel_dirs = collect_unique_dirs(all_paths)
    print(f"[INFO] Parsed {len(uniq_rel_dirs)} unique seed directories")
    if not uniq_rel_dirs:
        print("[WARN] No valid checkpoint directories found.")
        return

    total_copied = 0
    for rel_dir in uniq_rel_dirs:
        src_dir = Path("checkpoints") / rel_dir
        if dry_run:
            print(f"[DRY-RUN] Would copy: {src_dir} -> {dst_root / rel_dir}")
            continue
        copied = copy_dir_flat(src_dir, dst_root)
        print(f"[INFO] Copied {copied} files from {src_dir} -> {dst_root / rel_dir}")
        total_copied += copied

    if dry_run:
        print("[DRY-RUN] Completed simulation (no files actually copied).")
    else:
        print(f"[DONE] All complete. {total_copied} files copied to {dst_root}/")


if __name__ == "__main__":
    # You can adjust the parameters here if needed
    main(
        results_dir="/nas/unas15/chentao/TNNLS/results/result",
        dst_root="checkpoints3",
        name_substring="_results",
        dry_run=False,  # Set True for test run
    )
