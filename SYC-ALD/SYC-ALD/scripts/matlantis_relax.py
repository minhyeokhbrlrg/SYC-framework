#!/usr/bin/env python
"""Matlantis (PFP) relaxation watcher for the SYC framework.

Runs on Matlantis. It watches an input folder for structures exported by the SYC
Monte Carlo driver, relaxes each one with the PFP machine-learning potential, and
writes the relaxed CIF to an output folder, where the MC driver picks it up.

Two stages exist per ALD cycle; run one watcher per stage (e.g. two notebook
terminals):

    python matlantis_relax.py --stage pre  --in-dir runs/x/before --out-dir runs/x/relaxation
    python matlantis_relax.py --stage post --in-dir runs/x/after  --out-dir runs/x/relaxation2

=========  ==========================================  ==========================  ==============
stage      input file                                  output file                 default mode
=========  ==========================================  ==========================  ==============
pre        cycle{N}_before_oxidant_for_relax.cif       cycle{N}_before_oxidant_    PBE_PLUS_D3
                                                       relaxed.cif
post       cycle{N}_after_oxidant_for_relax.cif        cycle{N}_after_oxidant_     PBE_U
                                                       relaxed.cif
=========  ==========================================  ==========================  ==============

Each structure is relaxed in two passes:
1. FIRE pre-relaxation (fmax >= 0.05 eV/Å, <= 500 steps) with the z coordinate of
   near-surface adsorbate atoms frozen, to remove bad contacts without letting
   adsorbates sink into the slab;
2. LBFGS fine relaxation to the target ``fmax`` with only the bottom layer fixed.
In both passes the topmost Ru atom is pinned in x, y to suppress lateral drift.

This file is self-contained (only numpy, ase and pfp_api_client are needed) so it
can be copied to Matlantis on its own.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
from ase import Atoms
from ase.build import add_vacuum
from ase.constraints import FixAtoms, FixCartesian
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS

# Stage-specific defaults. ``tag`` must match the file names written by the MC driver.
STAGES = {
    "pre": {"tag": "before_oxidant", "calc_mode": "PBE_PLUS_D3"},
    "post": {"tag": "after_oxidant", "calc_mode": "PBE_U"},
}
DEFAULT_MODEL_VERSION = "v8.0.0"


# ==========================================================================
# Calculator
# ==========================================================================
def make_pfp_calculator(calc_mode: str, model_version: str = DEFAULT_MODEL_VERSION):
    """Create a PFP ASE calculator (imported lazily so the module loads without Matlantis)."""
    import pfp_api_client
    from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
    from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode

    mode = getattr(EstimatorCalcMode, calc_mode)
    estimator = Estimator(calc_mode=mode, model_version=model_version)
    print(f"pfp_api_client {pfp_api_client.__version__} | calc_mode={mode} | model_version={model_version}")
    return ASECalculator(estimator), str(mode)


# ==========================================================================
# Utilities
# ==========================================================================
def safe_read_cif_as_atoms(path: str) -> Atoms:
    """Read a CIF with ASE; fall back to parsing ``_atom_site_Cartn_x/y/z`` columns,
    which some ASE versions do not support."""
    try:
        return read(path)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] ase.io.read failed ({e}); using Cartesian CIF fallback parser")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    def get_float(tag):
        for ln in lines:
            if ln.startswith(tag):
                return float(ln.split()[-1].strip("'").strip('"'))
        return None

    a, b, c = (get_float(t) for t in ("_cell_length_a", "_cell_length_b", "_cell_length_c"))
    if a is None or b is None or c is None:
        raise ValueError(f"_cell_length_a/b/c not found in {path}")

    cols, data_start = [], None
    for i, ln in enumerate(lines):
        if ln.lower() == "loop_":
            tmp, j = [], i + 1
            while j < len(lines) and lines[j].startswith("_"):
                tmp.append(lines[j])
                j += 1
            lower = [x.lower() for x in tmp]
            if all(k in lower for k in ("_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z")):
                cols, data_start = tmp, j
                break
    if data_start is None:
        raise ValueError(f"No Cartn_x/y/z loop found in {path}")

    lower = [x.lower() for x in cols]
    ix_type = lower.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in lower else None
    ix_x, ix_y, ix_z = (lower.index(k) for k in ("_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z"))

    symbols, positions = [], []
    for ln in lines[data_start:]:
        if ln.lower().startswith("loop_") or ln.startswith("_"):
            break
        parts = ln.split()
        if len(parts) < len(cols):
            continue
        symbols.append(parts[ix_type] if ix_type is not None else parts[0])
        positions.append([float(parts[ix_x]), float(parts[ix_y]), float(parts[ix_z])])
    return Atoms(symbols=symbols, positions=positions, cell=[a, b, c], pbc=False)


def ensure_minimum_z_vacuum(atoms: Atoms, min_vacuum: float = 20.0) -> None:
    """Add vacuum along z only if the existing vacuum is smaller than ``min_vacuum``."""
    z = atoms.get_positions()[:, 2]
    z_span = float(z.max() - z.min())
    cell_z = float(atoms.cell.lengths()[2])
    if cell_z < 1e-6:
        add_vacuum(atoms, min_vacuum)
        return
    current = cell_z - z_span
    if current < min_vacuum:
        add_vacuum(atoms, min_vacuum - current)
        print(f"[VACUUM] added {min_vacuum - current:.3f} Å (was {current:.3f} Å)")
    else:
        print(f"[VACUUM] existing vacuum is sufficient: {current:.3f} Å")


def build_constraint_free_mask(natoms: int, constraints) -> np.ndarray:
    """Boolean (natoms, 3) mask: True where a Cartesian degree of freedom is free.
    Handles FixAtoms and FixCartesian across ASE versions."""
    free = np.ones((natoms, 3), dtype=bool)
    if constraints is None:
        return free
    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]

    for c in constraints:
        if isinstance(c, FixAtoms):
            try:
                free[np.array(c.get_indices(), dtype=int), :] = False
            except Exception:  # noqa: BLE001
                pass
        elif isinstance(c, FixCartesian):
            idxs = None
            try:
                idxs = np.array(c.get_indices(), dtype=int)
            except Exception:  # noqa: BLE001
                for attr in ("index", "a", "indices"):
                    if hasattr(c, attr):
                        idxs = np.atleast_1d(np.array(getattr(c, attr), dtype=int))
                        break
            fixed_xyz = None
            if hasattr(c, "mask"):
                fixed_xyz = np.array(c.mask, dtype=bool)
            elif hasattr(c, "fix"):
                fixed_xyz = np.array(c.fix, dtype=bool)
            if idxs is not None and fixed_xyz is not None:
                for idx in idxs:
                    free[int(idx), fixed_xyz] = False
    return free


def get_effective_max_force(atoms: Atoms) -> float:
    """Largest force norm over the *free* degrees of freedom only (eV/Å)."""
    forces = atoms.get_forces().copy()
    free = build_constraint_free_mask(len(atoms), getattr(atoms, "constraints", None))
    forces[~free] = 0.0
    norms = np.linalg.norm(forces, axis=1)
    return float(norms.max()) if len(norms) else 0.0


def estimate_slab_top_z(atoms: Atoms, freeze_mask: np.ndarray, percentile: float = 85.0) -> float:
    """Approximate slab top as a high percentile of mobile-atom heights, so that a few
    protruding adsorbates do not define it."""
    z = atoms.get_positions()[:, 2]
    mobile = np.where(~freeze_mask)[0]
    return float(np.max(z)) if len(mobile) == 0 else float(np.percentile(z[mobile], percentile))


def _make_optimizer(name: str, atoms: Atoms, traj: str, log: str, maxstep: float):
    cls = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}.get(name.upper(), LBFGS)
    try:
        return cls(atoms, trajectory=traj, logfile=log, maxstep=maxstep), maxstep
    except TypeError:  # very old ASE without maxstep
        return cls(atoms, trajectory=traj, logfile=log), None


# ==========================================================================
# Relax one structure
# ==========================================================================
def relax_cif_one(in_cif: str, out_cif: str, calculator, calc_label: str = "",
                  model_version: str = DEFAULT_MODEL_VERSION, optimizer: str = "LBFGS",
                  fmax: float = 0.001, steps: int = 4000, maxstep: float = 0.02,
                  freeze_bottom_thickness: float = 3.0, pbc=(True, True, False),
                  min_vacuum_z: float = 20.0, constrain_ads_z: bool = False,
                  constrain_anchor_xy: bool = True, adsorbate_elements=("Ru",),
                  anchor_element: str = "Ru", stage: str = "", write_energy_json: bool = True,
                  print_interval: int = 50) -> dict:
    """Relax ``in_cif`` and write ``out_cif`` (+ .traj, .log, .energy.json). Returns the energy record."""
    if not os.path.exists(in_cif):
        raise FileNotFoundError(f"Input CIF not found: {in_cif}")

    atoms = safe_read_cif_as_atoms(in_cif)
    atoms.set_pbc(pbc)
    ensure_minimum_z_vacuum(atoms, min_vacuum=min_vacuum_z)
    atoms.calc = calculator

    # ---- constraints ----
    z_all = atoms.get_positions()[:, 2]
    freeze_mask = z_all < (float(z_all.min()) + freeze_bottom_thickness)   # fixed bottom layer
    constraints = [FixAtoms(mask=freeze_mask)] if freeze_mask.any() else []
    symbols = np.array(atoms.get_chemical_symbols())

    if constrain_ads_z:  # freeze z of near-surface adsorbate atoms (FIRE pass only)
        z_slab_top = estimate_slab_top_z(atoms, freeze_mask)
        z_cut = z_slab_top - 0.5
        cand = np.where(np.isin(symbols, list(adsorbate_elements)))[0]
        ads = cand[z_all[cand] > z_cut]
        print(f"[ADS-Z] slab_top={z_slab_top:.3f} z_cut={z_cut:.3f} n_fixed_z={len(ads)}")
        constraints += [FixCartesian(int(i), mask=(False, False, True)) for i in ads]

    if constrain_anchor_xy:  # pin x, y of the topmost anchor atom against lateral drift
        idx = np.where(symbols == anchor_element)[0]
        if len(idx):
            anchor = int(idx[np.argmax(z_all[idx])])
            constraints.append(FixCartesian(anchor, mask=(True, True, False)))
            print(f"[ANCHOR] {anchor_element} anchor index = {anchor}")

    if constraints:
        atoms.set_constraint(constraints)

    # ---- optimise ----
    os.makedirs(os.path.dirname(out_cif) or ".", exist_ok=True)
    root = out_cif[:-4] if out_cif.endswith(".cif") else out_cif
    opt, applied_maxstep = _make_optimizer(optimizer, atoms, root + ".traj", root + ".log", maxstep)

    print(f">>> {optimizer} start: {os.path.basename(in_cif)}")
    print("      Step      Time            Energy     eff_fmax")
    t0, counter = time.time(), {"i": 0}

    def _report():
        counter["i"] += 1
        print(f"{counter['i'] * print_interval:10d} {time.time() - t0:9.1f} "
              f"{atoms.get_potential_energy():17.6f} {get_effective_max_force(atoms):12.6f}")

    opt.attach(_report, interval=print_interval)
    opt.run(fmax=fmax, steps=steps)

    nsteps = opt.get_number_of_steps()
    max_force = get_effective_max_force(atoms)
    converged = max_force <= fmax
    energy = float(atoms.get_potential_energy())
    print(f"[{stage}] steps={nsteps} converged={converged} eff_max_force={max_force:.6f} eV/Å "
          f"(target {fmax}) E={energy:.6f} eV")

    record = {
        "stage": stage, "in_cif": os.path.basename(in_cif), "out_cif": os.path.basename(out_cif),
        "optimizer": optimizer, "fmax": float(fmax), "steps": int(steps), "nsteps": int(nsteps),
        "max_force_eV_per_A": max_force, "converged": bool(converged), "maxstep": applied_maxstep,
        "freeze_bottom_thickness": float(freeze_bottom_thickness), "pbc": [bool(p) for p in pbc],
        "min_vacuum_z": float(min_vacuum_z), "constrain_ads_z": bool(constrain_ads_z),
        "constrain_anchor_xy": bool(constrain_anchor_xy), "adsorbate_elements": list(adsorbate_elements),
        "energy_eV": energy, "calc_mode": calc_label, "model_version": model_version,
    }
    if write_energy_json:
        with open(root + ".energy.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    # Write to a temp file, then rename: the MC driver must never read a half-written CIF.
    tmp = out_cif + ".tmp"
    try:
        write(tmp, atoms, format="cif", wrap=False)
    except TypeError:
        write(tmp, atoms, format="cif")
    os.replace(tmp, out_cif)
    return record


def relax_two_pass(in_cif: str, out_cif: str, calculator, calc_label: str, tag: str, args) -> dict:
    """FIRE pre-relaxation followed by LBFGS fine relaxation."""
    common = dict(calculator=calculator, calc_label=calc_label, model_version=args.model_version,
                  maxstep=args.maxstep, freeze_bottom_thickness=args.freeze_bottom,
                  pbc=tuple(args.pbc), min_vacuum_z=args.min_vacuum, constrain_anchor_xy=True,
                  adsorbate_elements=tuple(args.adsorbate_elements), anchor_element=args.anchor_element)
    prefire = out_cif.replace("_relaxed.cif", "_prefire.cif")
    relax_cif_one(in_cif, prefire, optimizer="FIRE", constrain_ads_z=True,
                  fmax=max(0.05, args.fmax), steps=min(500, args.steps), stage=f"{tag}_prefire", **common)
    return relax_cif_one(prefire, out_cif, optimizer="LBFGS", constrain_ads_z=False,
                         fmax=args.fmax, steps=args.steps, stage=tag, **common)


# ==========================================================================
# Watch loop
# ==========================================================================
def run_watch_loop(args) -> None:
    stage = STAGES[args.stage]
    tag = stage["tag"]
    calc_mode = args.calc_mode or stage["calc_mode"]
    calculator, calc_label = make_pfp_calculator(calc_mode, args.model_version)

    os.makedirs(args.in_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    pat = re.compile(rf"cycle(\d+)_{tag}_for_relax\.cif$")

    done, gave_up, failures, results = set(), set(), {}, []
    print(f"[WATCH-{args.stage.upper()}] {args.in_dir} -> {args.out_dir} (max_cycles={args.max_cycles})")

    while True:
        for in_cif in sorted(glob.glob(os.path.join(args.in_dir, f"cycle*_{tag}_for_relax.cif"))):
            m = pat.search(os.path.basename(in_cif))
            if not m:
                continue
            cyc = int(m.group(1))
            if cyc in done or cyc in gave_up:
                continue

            out_cif = os.path.join(args.out_dir, f"cycle{cyc}_{tag}_relaxed.cif")
            if os.path.exists(out_cif):
                print(f"[SKIP] already relaxed: {out_cif}")
                done.add(cyc)
                continue

            print(f"[RELAX-{args.stage.upper()}] cycle {cyc}: {in_cif} -> {out_cif}")
            try:
                results.append(relax_two_pass(in_cif, out_cif, calculator, calc_label, tag, args))
                print(f"[DONE] {out_cif}")
                done.add(cyc)
            except Exception as e:  # noqa: BLE001
                failures[cyc] = failures.get(cyc, 0) + 1
                print(f"[ERROR] cycle {cyc} failed ({failures[cyc]}/{args.max_retries}): {e}")
                if failures[cyc] >= args.max_retries:
                    print(f"[ERROR] giving up on cycle {cyc}; the MC driver will keep waiting for it")
                    gave_up.add(cyc)

        if len(done) + len(gave_up) >= args.max_cycles:
            print(f"[FINISH-{args.stage.upper()}] processed {len(done)} cycle(s), {len(gave_up)} failed")
            break
        time.sleep(args.poll_sec)

    if results:
        print("\n" + "=" * 100)
        print(f"{'Cycle':>5} | {'Stage':<22} | {'E (eV)':>14} | {'maxF (eV/A)':>12} | {'conv':>5} | Out CIF")
        print("-" * 100)
        for r in results:
            mm = re.search(r"cycle(\d+)_", r["out_cif"])
            print(f"{int(mm.group(1)) if mm else -1:5d} | {r['stage']:<22} | {r['energy_eV']:14.6f} | "
                  f"{r['max_force_eV_per_A']:12.6f} | {str(r['converged']):>5} | {r['out_cif']}")
        print("=" * 100)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Matlantis/PFP relaxation watcher for SYC")
    p.add_argument("--stage", choices=sorted(STAGES), required=True,
                   help="pre = before co-reactant, post = after co-reactant")
    p.add_argument("--in-dir", required=True, help="folder the MC driver writes *_for_relax.cif to")
    p.add_argument("--out-dir", required=True, help="folder the relaxed CIFs are written to")
    p.add_argument("--max-cycles", type=int, default=50, help="stop after this many cycles are processed")
    p.add_argument("--poll-sec", type=float, default=10.0, help="folder polling interval (s)")
    p.add_argument("--fmax", type=float, default=0.001, help="LBFGS force criterion (eV/Å)")
    p.add_argument("--steps", type=int, default=4000, help="max LBFGS steps")
    p.add_argument("--maxstep", type=float, default=0.02, help="optimizer max step (Å)")
    p.add_argument("--freeze-bottom", type=float, default=3.0, help="thickness of the fixed bottom layer (Å)")
    p.add_argument("--pbc", type=lambda s: [c.strip().lower() in ("1", "t", "true") for c in s.split(",")],
                   default=[True, True, False], help="periodicity, e.g. 1,1,0")
    p.add_argument("--min-vacuum", type=float, default=20.0, help="minimum vacuum along z (Å)")
    p.add_argument("--calc-mode", default=None,
                   help="EstimatorCalcMode name (default: PBE_PLUS_D3 for pre, PBE_U for post)")
    p.add_argument("--model-version", default=DEFAULT_MODEL_VERSION, help="PFP model version")
    p.add_argument("--adsorbate-elements", nargs="+", default=["Ru"],
                   help="elements whose z is frozen during the FIRE pass")
    p.add_argument("--anchor-element", default="Ru", help="element of the x,y-pinned anchor atom")
    p.add_argument("--max-retries", type=int, default=3, help="attempts per cycle before giving up")
    return p


if __name__ == "__main__":
    run_watch_loop(build_parser().parse_args())
