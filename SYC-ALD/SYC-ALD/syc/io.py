"""File I/O: input CSVs, CIF writing/reading, and the relaxation file handshake.

Atoms are represented throughout as a list of tuples ``(element, x, y, z)`` in Å.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

Atom = Tuple[str, float, float, float]


# ----------------------------------------------------------------------------
# Input files
# ----------------------------------------------------------------------------
def load_xyz_csv(filename: str) -> List[Atom]:
    """Load a molecule from a CSV with columns ``name, 1, 2, 3`` (element, x, y, z).

    Rows with a missing element or coordinate are dropped.
    """
    df = pd.read_csv(filename)
    df["name"] = df["name"].astype(str).str.strip()
    for col in ["1", "2", "3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["name", "1", "2", "3"])
    return [(row["name"], float(row["1"]), float(row["2"]), float(row["3"]))
            for _, row in df.iterrows()]


def load_xyz_excel(filename: str) -> List[Atom]:
    """Same as :func:`load_xyz_csv` for .xlsx files (requires openpyxl)."""
    df = pd.read_excel(filename)
    df["name"] = df["name"].astype(str).str.strip()
    for col in ["1", "2", "3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["name", "1", "2", "3"])
    return [(row["name"], float(row["1"]), float(row["2"]), float(row["3"]))
            for _, row in df.iterrows()]


def parse_lattice_csv(filepath: str):
    """Read a substrate unit cell.

    Format (no header):
        lattice, a1x, a1y, a1z
        lattice, a2x, a2y, a2z
        Ru, x, y, z
        ...
    Empty rows and rows whose first field is ``0`` are skipped.
    Returns ``(lattice_vectors, atoms)``.
    """
    lattice_vectors, atoms = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0] == "0":
                continue
            if row[0].lower() == "lattice":
                lattice_vectors.append(list(map(float, row[1:])))
            else:
                x, y, z = map(float, row[1:])
                atoms.append((row[0], x, y, z))
    return lattice_vectors, atoms


# ----------------------------------------------------------------------------
# CIF writing
# ----------------------------------------------------------------------------
def _atomic_replace(tmp_path: str, final_path: str) -> None:
    """Move a finished file into place so watchers never see a half-written file."""
    os.replace(tmp_path, final_path)


def write_cif_cartesian(filename: str, atoms_cart: List[Atom], cell_a: float, cell_b: float, cell_c: float,
                        alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0,
                        data_name: str = "snapshot") -> None:
    """Write a P1 CIF using Cartesian ``_atom_site_Cartn_x/y/z`` columns (Å).

    The file is written to ``<filename>.tmp`` and renamed when complete.
    """
    tmp = filename + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"data_{data_name}\n")
        f.write(f"_cell_length_a    {cell_a:.6f}\n")
        f.write(f"_cell_length_b    {cell_b:.6f}\n")
        f.write(f"_cell_length_c    {cell_c:.6f}\n")
        f.write(f"_cell_angle_alpha {alpha:.6f}\n")
        f.write(f"_cell_angle_beta  {beta:.6f}\n")
        f.write(f"_cell_angle_gamma {gamma:.6f}\n")
        f.write("_symmetry_space_group_name_H-M   'P 1'\n")
        f.write("loop_\n")
        f.write("  _atom_site_label\n")
        f.write("  _atom_site_type_symbol\n")
        f.write("  _atom_site_Cartn_x\n")
        f.write("  _atom_site_Cartn_y\n")
        f.write("  _atom_site_Cartn_z\n")
        for i, (el, x, y, z) in enumerate(atoms_cart, start=1):
            f.write(f"  {el}{i} {el} {x:.6f} {y:.6f} {z:.6f}\n")
    _atomic_replace(tmp, filename)


def make_cell_and_shift(atoms: List[Atom], padding_xy: float = 5.0, padding_z: float = 10.0):
    """Shift atoms into a positive orthorhombic box = bounding box + padding.

    Returns ``(shifted_atoms, (a, b, c), shift_vector)``.
    """
    coords = np.array([[x, y, z] for (_, x, y, z) in atoms], float)
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    shift = np.array([padding_xy, padding_xy, padding_z], float) - xyz_min
    shifted = [(el, x + shift[0], y + shift[1], z + shift[2]) for (el, x, y, z) in atoms]
    span = xyz_max - xyz_min
    cell = (span[0] + 2 * padding_xy, span[1] + 2 * padding_xy, span[2] + 2 * padding_z)
    return shifted, cell, shift


def export_atoms_cif(path: str, atoms: List[Atom], data_name: str = "snapshot",
                     padding_xy: float = 8.0, padding_z: float = 12.0) -> None:
    """Write a snapshot CIF (for inspection only; no meta file)."""
    shifted, (ca, cb, cc), _ = make_cell_and_shift(atoms, padding_xy, padding_z)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_cif_cartesian(path, shifted, ca, cb, cc, data_name=data_name)


# ----------------------------------------------------------------------------
# CIF reading
# ----------------------------------------------------------------------------
def read_cif_positions_any(cif_path: str):
    """Read a CIF with either fractional or Cartesian coordinates.

    Assumes an orthorhombic cell (alpha = beta = gamma = 90°), which holds for every
    CIF produced by this package and by ASE from those inputs.
    Returns ``(atoms_cartesian, (a, b, c, alpha, beta, gamma))``.
    """
    with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    def get_float(tag, default=None):
        for line in lines:
            if line.strip().startswith(tag):
                return float(line.split()[-1].strip().strip("'").strip('"'))
        return default

    a = get_float("_cell_length_a")
    b = get_float("_cell_length_b")
    c = get_float("_cell_length_c")
    alpha = get_float("_cell_angle_alpha", 90.0)
    beta = get_float("_cell_angle_beta", 90.0)
    gamma = get_float("_cell_angle_gamma", 90.0)

    colnames, data_start = None, None
    for i, line in enumerate(lines):
        if line.strip().lower() == "loop_":
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            lower = [x.lower() for x in cols]
            if any("_atom_site_fract_x" in x for x in lower) or any("_atom_site_cartn_x" in x for x in lower):
                colnames, data_start = cols, j
                break
    if data_start is None:
        raise ValueError(f"No atom_site coordinate loop found in {cif_path}")

    lower = [x.lower() for x in colnames]

    def idx_of(key):
        return lower.index(key) if key in lower else None

    ix_fx, ix_fy, ix_fz = idx_of("_atom_site_fract_x"), idx_of("_atom_site_fract_y"), idx_of("_atom_site_fract_z")
    ix_cx, ix_cy, ix_cz = idx_of("_atom_site_cartn_x"), idx_of("_atom_site_cartn_y"), idx_of("_atom_site_cartn_z")
    ix_type = idx_of("_atom_site_type_symbol")
    use_fract = ix_fx is not None and ix_fy is not None and ix_fz is not None

    atoms = []
    for k in range(data_start, len(lines)):
        line = lines[k].strip()
        if (not line) or line.startswith("#") or line.startswith("_") or line.lower().startswith("loop_"):
            continue
        parts = line.split()
        if len(parts) < len(colnames):
            continue
        elem = parts[ix_type] if ix_type is not None else parts[0]
        if use_fract:
            x, y, z = float(parts[ix_fx]) * a, float(parts[ix_fy]) * b, float(parts[ix_fz]) * c
        else:
            x, y, z = float(parts[ix_cx]), float(parts[ix_cy]), float(parts[ix_cz])
        atoms.append((elem, x, y, z))
    return atoms, (a, b, c, alpha, beta, gamma)


# ----------------------------------------------------------------------------
# Relaxation handshake
# ----------------------------------------------------------------------------
def export_for_relaxation_with_meta(atoms: List[Atom], out_cif: str, out_meta_json: str,
                                    padding_xy: float = 8.0, padding_z: float = 12.0,
                                    data_name: str = "before_oxidant") -> None:
    """Export a structure for the external relaxer.

    The CIF is shifted into a positive box; the shift is stored in ``out_meta_json``
    so the relaxed structure can be mapped back to simulation coordinates.
    The meta file is written first, then the CIF (atomically), because the relaxer
    starts as soon as the CIF appears.
    """
    shifted, (ca, cb, cc), shift = make_cell_and_shift(atoms, padding_xy, padding_z)
    meta = {"shift": shift.tolist(), "cell": [ca, cb, cc, 90.0, 90.0, 90.0]}
    with open(out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    write_cif_cartesian(out_cif, shifted, ca, cb, cc, data_name=data_name)


def import_relaxed_and_unshift(relaxed_cif: str, meta_json: str) -> List[Atom]:
    """Read a relaxed CIF and undo the export shift."""
    atoms_cart, _ = read_cif_positions_any(relaxed_cif)
    with open(meta_json, "r", encoding="utf-8") as f:
        shift = np.array(json.load(f)["shift"], float)
    return [(el, x - shift[0], y - shift[1], z - shift[2]) for (el, x, y, z) in atoms_cart]


def wait_for_file(path: str, poll_sec: float = 60, timeout_sec=None, log_sec: float = 3600) -> None:
    """Block until ``path`` exists. Prints a status line every ``log_sec`` seconds."""
    t0 = time.time()
    last_log = 0.0
    while not os.path.exists(path):
        elapsed = time.time() - t0
        if elapsed - last_log >= log_sec:
            print(f"[WAIT] relaxed CIF not found yet: {path} (elapsed={elapsed:.0f}s)")
            last_log = elapsed
        if timeout_sec is not None and elapsed > timeout_sec:
            raise TimeoutError(f"Timeout waiting for: {path}")
        time.sleep(poll_sec)
