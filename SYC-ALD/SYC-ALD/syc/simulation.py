"""SYC (Stochastic Yielding and Correction) Monte Carlo driver for ALD growth.

One ALD cycle consists of

1. **Precursor exposures** (repeated up to ``max_exposures_per_cycle`` times)
   a. *Gas layer*: precursor molecules are inserted at random positions and
      orientations in a slab ``z_gas_min..z_gas_max`` until ``max_failures``
      consecutive insertions overlap.
   b. *Physisorption*: each gas molecule descends in steps of ``dz`` above one of
      its nearest free on-sites. It physisorbs (with probability ``p_phys``) once
      its lowest atom is within ``phys_gap_max`` of the site and it does not
      overlap the film, already-chemisorbed precursors (with ligands), or
      molecules physisorbed earlier in the same exposure.
   c. *Chemisorption*: each newly physisorbed molecule is converted into the
      chemisorbed template on its on-site, trying ``n_rotations`` in-plane
      rotations until one fits. Ligands are kept and block later molecules for
      the rest of the cycle (steric hindrance). Molecules that fail to
      chemisorb are purged.
   The exposure loop ends early after ``max_no_new_phy_streak`` consecutive
   exposures with no new physisorption.

2. **Relaxation before the co-reactant** (external, Matlantis/PFP).
3. **Co-reactant step**: all ligand atoms (``ligand_elements``) are removed.
4. **Relaxation after the co-reactant** (external, Matlantis/PFP).

The relaxed film is the starting surface of the next cycle.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

from .analysis import film_thickness
from .config import SYCConfig
from .geometry import (build_kdtree, build_tiled_substrate, check_overlap, generate_random_rotation,
                       get_precursor_center, get_surface_onsites, place_template_on_onsite,
                       transform_precursor)
from .io import Atom, export_atoms_cif, load_xyz_csv, parse_lattice_csv
from .relaxation_bridge import STAGE_POST, STAGE_PRE, load_restart_state, run_relaxation_stage


# ----------------------------------------------------------------------------
# Event probabilities
# ----------------------------------------------------------------------------
def event_occurs(p: float) -> bool:
    """Bernoulli trial with probability p (always draws one random number)."""
    return np.random.rand() < p


def physisorption_probability(cfg: SYCConfig, gap: float) -> float:
    if not cfg.use_dynamic_prob:
        return cfg.p_phys
    # Example dynamic rule: the closer to the surface, the more likely.
    return float(np.clip(cfg.p_phys * max(0.0, 1.0 - gap / cfg.phys_gap_max), 0.0, 1.0))


def chemisorption_probability(cfg: SYCConfig) -> float:
    return cfg.p_chem if not cfg.use_dynamic_prob else float(np.clip(cfg.p_chem, 0.0, 1.0))


def chem_desorption_probability(cfg: SYCConfig) -> float:
    return cfg.p_des_chem if not cfg.use_dynamic_prob else float(np.clip(cfg.p_des_chem, 0.0, 1.0))


# ----------------------------------------------------------------------------
# Per-cycle bookkeeping
# ----------------------------------------------------------------------------
@dataclass
class CycleCounters:
    gas_trials: int = 0
    phys_attempts: int = 0
    phys_success: int = 0
    chem_attempts: int = 0
    chem_success: int = 0
    chem_desorption: int = 0


@dataclass
class OnSites:
    """Exposed adsorption sites at the start of a cycle."""
    sites: list
    xy: np.ndarray = field(init=False)
    z: np.ndarray = field(init=False)
    film_idx: np.ndarray = field(init=False)
    tree: object = field(init=False)

    def __post_init__(self):
        if self.sites:
            self.xy = np.array([[x, y] for (_, x, y, _) in self.sites], float)
            self.z = np.array([s[3] for s in self.sites], float)
            self.film_idx = np.array([s[0] for s in self.sites], int)
            self.tree = cKDTree(self.xy)
        else:
            self.xy = np.zeros((0, 2))
            self.z = np.zeros((0,))
            self.film_idx = np.zeros((0,), int)
            self.tree = None

    def __len__(self):
        return len(self.sites)


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------
def generate_gas_layer(precursor: List[Atom], cfg: SYCConfig):
    """Random sequential insertion of non-overlapping precursor molecules.
    Returns the list of insertion points ``(x0, y0, z0)``."""
    gas_atoms: List[Atom] = []
    centers = []
    tree, radii = None, []
    failures = 0
    while failures < cfg.max_failures:
        x0, y0 = np.random.uniform(0, cfg.area_size, 2)
        z0 = np.random.uniform(cfg.z_gas_min, cfg.z_gas_max)
        R = generate_random_rotation()
        new_atoms = transform_precursor(precursor, x0, y0, z0, R)
        if not check_overlap(new_atoms, tree, radii, scale=cfg.overlap_scale):
            gas_atoms.extend(new_atoms)
            centers.append((x0, y0, z0))
            failures = 0
            tree, _, radii = build_kdtree(gas_atoms)  # obstacles only change on acceptance
        else:
            failures += 1
    return centers


def physisorb(gas_centers, precursor: List[Atom], onsites: OnSites, used_onsites: set,
              blocking_atoms: List[Atom], cfg: SYCConfig, counters: CycleCounters):
    """Let each gas molecule descend onto a nearby free on-site.

    ``blocking_atoms`` = current film + ligands of precursors chemisorbed earlier
    in this cycle. Returns a list of ``(atoms, center, onsite_index)``.
    """
    adsorbed = []
    for x0, y0, z0 in gas_centers:
        R = generate_random_rotation()  # new random orientation for the descent
        xy0 = get_precursor_center(transform_precursor(precursor, x0, y0, z0, R), cfg.center_element)[:2]
        if onsites.tree is None:
            continue

        k_cand = min(cfg.n_onsite_candidates, len(onsites))
        _, idxs = onsites.tree.query(xy0, k=k_cand)
        idxs = np.atleast_1d(idxs)

        # Obstacles are fixed until this molecule adsorbs, so build the tree once (lazily).
        obstacle_tree, obstacle_radii, tree_built = None, [], False
        picked = False

        for k in idxs:
            if np.linalg.norm(onsites.xy[k] - xy0) > cfg.phys_xy_cutoff:
                continue
            if int(onsites.film_idx[k]) in used_onsites:
                continue
            z_surf = float(onsites.z[k])

            for step in range(int((z0 - z_surf) / cfg.dz)):
                z_try = z0 - step * cfg.dz
                new_atoms = transform_precursor(precursor, x0, y0, z_try, R)
                center = get_precursor_center(new_atoms, cfg.center_element)
                if center[2] < z_surf - cfg.phys_descent_below_site:
                    break

                if not tree_built:
                    obstacles = blocking_atoms + [a for mol, _, _ in adsorbed for a in mol]
                    obstacle_tree, _, obstacle_radii = build_kdtree(obstacles)
                    tree_built = True
                if check_overlap(new_atoms, obstacle_tree, obstacle_radii, scale=cfg.overlap_scale):
                    continue

                gap = min(a[3] for a in new_atoms) - z_surf
                if 0.0 <= gap <= cfg.phys_gap_max:
                    counters.phys_attempts += 1
                    if event_occurs(physisorption_probability(cfg, gap)):
                        counters.phys_success += 1
                        adsorbed.append((new_atoms, tuple(center), int(k)))
                        picked = True
                        break
            if picked:
                break
    return adsorbed


def chemisorb(physisorbed, template: List[Atom], onsites: OnSites, used_onsites: set,
              film_atoms: List[Atom], ligand_blocking_atoms: List[Atom], cfg: SYCConfig,
              counters: CycleCounters):
    """Convert physisorbed molecules into the chemisorbed template on their on-site.

    Successful structures are appended to ``ligand_blocking_atoms`` (in place) and
    returned as ``[(placed_atoms, film_index, onsite_xyz), ...]``. Failures are purged.
    """
    results = []
    n_failed = 0
    for _mol, _center, k in physisorbed:
        film_idx, ox, oy, oz = onsites.sites[k]
        onsite_xyz = (ox, oy, oz)
        succeeded = False

        if film_idx not in used_onsites:  # one chemisorbed precursor per site per cycle
            z_cut = oz - cfg.chem_local_z_margin
            local = [a for a in (film_atoms + ligand_blocking_atoms) if a[3] >= z_cut]
            local_tree, _, local_radii = build_kdtree(local)

            for theta in np.linspace(0, 2 * np.pi, cfg.n_rotations, endpoint=False):
                placed = place_template_on_onsite(template, onsite_xyz, theta)
                if check_overlap(placed, local_tree, local_radii, scale=cfg.overlap_scale):
                    continue
                counters.chem_attempts += 1
                if not event_occurs(chemisorption_probability(cfg)):
                    continue
                if event_occurs(chem_desorption_probability(cfg)):
                    counters.chem_desorption += 1
                    continue
                counters.chem_success += 1
                ligand_blocking_atoms.extend(placed)
                results.append((placed, film_idx, onsite_xyz))
                used_onsites.add(film_idx)
                succeeded = True
                break

        if not succeeded:
            n_failed += 1
    return results, n_failed


# ----------------------------------------------------------------------------
# One ALD cycle
# ----------------------------------------------------------------------------
def run_cycle(cycle: int, film_atoms: List[Atom], substrate: List[Atom], substrate_top_z: float,
              precursor: List[Atom], template: List[Atom], cfg: SYCConfig) -> Dict:
    print(f"\n========== Cycle {cycle} ==========")
    ligand_elements = set(cfg.ligand_elements)
    counters = CycleCounters()
    used_onsites: set = set()
    ligand_blocking_atoms: List[Atom] = []   # chemisorbed precursors incl. ligands (cleared at cycle end)
    chemisorbed = []                         # [(placed_atoms, film_idx, onsite_xyz)]
    phys_success_atoms: List[Atom] = []      # every physisorbed molecule of this cycle (for inspection)
    physisorbed_total = 0

    onsites = OnSites(get_surface_onsites(film_atoms, element=cfg.site_element, r_xy=cfg.onsite_r_xy,
                                          dz=cfg.onsite_dz, z_min=substrate_top_z - cfg.onsite_z_tolerance))
    print(f"[Cycle {cycle}] exposed on-sites ({cfg.site_element}) = {len(onsites)}")

    # ---------------- precursor exposures ----------------
    no_new_phy_streak = 0
    for exposure in range(1, cfg.max_exposures_per_cycle + 1):
        gas_centers = generate_gas_layer(precursor, cfg)
        counters.gas_trials += len(gas_centers)

        new_phys = physisorb(gas_centers, precursor, onsites, used_onsites,
                             film_atoms + ligand_blocking_atoms, cfg, counters)
        for mol, _, _ in new_phys:
            phys_success_atoms.extend(mol)
        physisorbed_total += len(new_phys)

        new_chem, n_failed = chemisorb(new_phys, template, onsites, used_onsites, film_atoms,
                                       ligand_blocking_atoms, cfg, counters)
        chemisorbed.extend(new_chem)

        print(f"[Cycle {cycle}] exposure {exposure}: gas={len(gas_centers)}, new_phy={len(new_phys)}, "
              f"chem={len(new_chem)}, failed_chem(purged)={n_failed}, phy_total={physisorbed_total}")

        if len(new_phys) < cfg.min_new_physisorbed:
            no_new_phy_streak += 1
            print(f"[Cycle {cycle}] exposure {exposure}: no new physisorption "
                  f"({no_new_phy_streak}/{cfg.max_no_new_phy_streak})")
        else:
            no_new_phy_streak = 0
        if no_new_phy_streak >= cfg.max_no_new_phy_streak:
            print(f"[Cycle {cycle}] {cfg.max_no_new_phy_streak} consecutive empty exposures -> end of exposure")
            break

    print(f"[Cycle {cycle}] gas molecules tried = {counters.gas_trials}, physisorbed = {physisorbed_total}, "
          f"chemisorbed = {len(chemisorbed)}")

    # Inspection snapshot: bare substrate + every molecule that physisorbed this cycle.
    phy_cif = os.path.join(cfg.output_dir, "phy", f"cycle{cycle}_substrate_plus_phy.cif")
    export_atoms_cif(phy_cif, substrate + phys_success_atoms, data_name=f"cycle{cycle}_sub_plus_phy")
    print(f"[SAVE] substrate + physisorbed CIF: {phy_cif}")

    snapshots = {}

    # ---------------- relaxation before the co-reactant ----------------
    pre_in = list(film_atoms) + [a for placed, _, _ in chemisorbed for a in placed]
    snapshots["pre_in"] = pre_in
    pre_out = pre_in
    if cfg.use_relaxation_pre:
        pre_out = run_relaxation_stage(pre_in, cycle, STAGE_PRE, cfg.relax_before_dir, cfg.relax_out_dir_pre,
                                       padding_xy=cfg.relax_padding_xy, padding_z=cfg.relax_padding_z,
                                       poll_sec=cfg.poll_sec_pre, timeout_sec=cfg.wait_timeout_sec,
                                       log_sec=cfg.wait_log_sec)
    snapshots["pre_out"] = list(pre_out)

    # ---------------- co-reactant step: remove ligands ----------------
    # (Ru metal ALD: ligand atoms are removed; no new atoms are added.)
    film_atoms = [a for a in pre_out if a[0] not in ligand_elements]
    snapshots["post_in"] = list(film_atoms)

    # ---------------- relaxation after the co-reactant ----------------
    if cfg.use_relaxation_post:
        relaxed = run_relaxation_stage(film_atoms, cycle, STAGE_POST, cfg.relax_after_dir, cfg.relax_out_dir_post,
                                       padding_xy=cfg.relax_padding_xy, padding_z=cfg.relax_padding_z,
                                       poll_sec=cfg.poll_sec_post, timeout_sec=cfg.wait_timeout_sec,
                                       log_sec=cfg.wait_log_sec)
        film_atoms = [a for a in relaxed if a[0] not in ligand_elements]
    snapshots["post_out"] = list(film_atoms)

    thickness = film_thickness(film_atoms, substrate_top_z)
    print(f"[Cycle {cycle}] thickness = {thickness:.2f} Å ({thickness * 0.1:.3f} nm)")
    print(f"[Cycle {cycle}] chemisorbed = {len(chemisorbed)}")

    stats = {
        "cycle": cycle,
        "onsites": len(onsites),
        "gas_trials": counters.gas_trials,
        "phys_attempts": counters.phys_attempts,
        "phys_success": counters.phys_success,
        "physisorbed_total": physisorbed_total,
        "chem_attempts": counters.chem_attempts,
        "chem_success": counters.chem_success,
        "chem_desorption": counters.chem_desorption,
        "chemisorbed": len(chemisorbed),
        "thickness_A": round(thickness, 4),
    }
    return {
        "film_atoms": film_atoms,
        "stats": stats,
        "snapshots": snapshots,
        "chemisorbed_sites": [xyz for _, _, xyz in chemisorbed],
    }


def _append_stats_row(path: str, row: Dict) -> None:
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------------------------------------------------------
# Full run
# ----------------------------------------------------------------------------
def run_simulation(cfg: SYCConfig) -> Dict:
    """Run (or resume) the SYC simulation up to ``cfg.max_cycles``.

    Returns a dict with per-cycle ``stats``, structure ``snapshots`` and the final
    ``film_atoms``. Per-cycle stats are also appended to ``<output_dir>/cycle_stats.csv``.
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    precursor = load_xyz_csv(cfg.precursor_gas_csv)
    template = load_xyz_csv(cfg.precursor_adsorbed_csv)
    lattice_vectors, unit_atoms = parse_lattice_csv(cfg.substrate_csv)
    substrate = build_tiled_substrate(unit_atoms, lattice_vectors, cfg.area_size)
    substrate_top_z = max(a[3] for a in substrate)
    print(f"[INIT] substrate atoms = {len(substrate)}, top z = {substrate_top_z:.3f} Å")

    for d in (cfg.output_dir, cfg.relax_before_dir, cfg.relax_after_dir,
              cfg.relax_out_dir_pre, cfg.relax_out_dir_post):
        os.makedirs(d, exist_ok=True)
    stats_csv = os.path.join(cfg.output_dir, "cycle_stats.csv")

    film_atoms, start_cycle, last_done = load_restart_state(substrate, cfg.relax_out_dir_post, cfg.relax_after_dir)
    results = {"stats": [], "snapshots": {}, "chemisorbed_sites": {}, "film_atoms": film_atoms,
               "substrate": substrate, "substrate_top_z": substrate_top_z}
    if start_cycle > cfg.max_cycles:
        print(f"[RESTART] Already completed up to cycle {last_done} (max_cycles={cfg.max_cycles})")
        return results

    no_chem_streak = 0
    no_phys_streak = 0
    for cycle in range(start_cycle, cfg.max_cycles + 1):
        out = run_cycle(cycle, film_atoms, substrate, substrate_top_z, precursor, template, cfg)
        film_atoms = out["film_atoms"]
        results["stats"].append(out["stats"])
        results["snapshots"][cycle] = out["snapshots"]
        results["chemisorbed_sites"][cycle] = out["chemisorbed_sites"]
        results["film_atoms"] = film_atoms
        _append_stats_row(stats_csv, out["stats"])

        no_chem_streak = no_chem_streak + 1 if out["stats"]["chemisorbed"] == 0 else 0
        no_phys_streak = no_phys_streak + 1 if out["stats"]["physisorbed_total"] == 0 else 0

        if cfg.saturation_limit_chem > 0 and no_chem_streak >= cfg.saturation_limit_chem:
            print(f"[Cycle {cycle}] no chemisorption for {no_chem_streak} cycles -> stop")
            break
        if cfg.saturation_limit_phys > 0 and no_phys_streak >= cfg.saturation_limit_phys:
            print(f"[Cycle {cycle}] no physisorption for {no_phys_streak} cycles -> stop")
            break

    return results
