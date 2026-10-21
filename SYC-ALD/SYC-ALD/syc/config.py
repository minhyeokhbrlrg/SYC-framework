"""Run configuration for the SYC Monte Carlo driver.

All parameters that were hard-coded globals in the original script live here.
Defaults reproduce the values used for the TRuST-on-Ru runs. Override any field
from a JSON file (see ``config/example_trust_on_ru.json``); relative paths in the
JSON are resolved against the JSON file's own folder.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

_PATH_FIELDS = (
    "precursor_gas_csv",
    "precursor_adsorbed_csv",
    "substrate_csv",
    "output_dir",
    "relax_before_dir",
    "relax_after_dir",
    "relax_out_dir_pre",
    "relax_out_dir_post",
)


@dataclass
class SYCConfig:
    # ---------------- input files ----------------
    precursor_gas_csv: str = "data/precursor_gas.csv"            # gas-phase precursor geometry
    precursor_adsorbed_csv: str = "data/precursor_chemisorbed.csv"  # chemisorbed template, relative to the on-site atom
    substrate_csv: str = "data/substrate.csv"                    # unit cell with 'lattice' rows

    # ---------------- output / exchange folders ----------------
    output_dir: str = "runs/example"            # stats CSV, physisorption snapshots
    relax_before_dir: str = "runs/example/before"       # MC -> relaxer (pre-co-reactant input)
    relax_out_dir_pre: str = "runs/example/relaxation"  # relaxer -> MC (pre-co-reactant output)
    relax_after_dir: str = "runs/example/after"         # MC -> relaxer (post-co-reactant input)
    relax_out_dir_post: str = "runs/example/relaxation2"  # relaxer -> MC (post-co-reactant output)

    # ---------------- chemistry ----------------
    center_element: str = "Ru"          # central metal atom of the precursor
    site_element: str = "Ru"            # surface atoms that act as adsorption on-sites
    ligand_elements: List[str] = field(default_factory=lambda: ["C", "H"])  # removed by the co-reactant step

    # ---------------- geometry ----------------
    area_size: float = 50.0             # lateral simulation box (Å), square
    z_gas_min: float = 50.0             # gas layer bottom (Å)
    z_gas_max: float = 60.0             # gas layer top (Å)
    dz: float = 0.5                     # descent step of a gas molecule (Å)
    phys_gap_max: float = 3.0           # max gap between lowest precursor atom and on-site for physisorption (Å)
    phys_xy_cutoff: float = 5.0         # max lateral distance precursor centre <-> on-site (Å)
    phys_descent_below_site: float = 2.0  # stop descending once the centre is this far below the site (Å)
    overlap_scale: float = 0.2          # overlap if d < scale * (r1 + r2)
    onsite_r_xy: float = 1.5            # lateral radius for the "buried site" test (Å)
    onsite_dz: float = 0.4              # a site is buried if a neighbour is higher by more than this (Å)
    onsite_z_tolerance: float = 0.2     # sites must lie above (substrate top - tolerance) (Å)
    chem_local_z_margin: float = 0.8    # only atoms above (site z - margin) are used for the chem overlap test (Å)

    # ---------------- MC loop ----------------
    max_cycles: int = 1                 # last ALD cycle to run (absolute cycle number, restart-aware)
    max_failures: int = 2000            # consecutive rejected insertions that end gas-layer generation
    max_exposures_per_cycle: int = 30   # gas exposures (gas layer + phys + chem) per cycle
    min_new_physisorbed: int = 1        # an exposure with fewer new physisorbed molecules counts as "empty"
    max_no_new_phy_streak: int = 5      # consecutive empty exposures that end the cycle
    n_onsite_candidates: int = 10       # nearest on-sites tried per gas molecule
    n_rotations: int = 72               # in-plane rotations tried for the chemisorbed template
    saturation_limit_chem: int = 100    # stop after this many consecutive cycles with no chemisorption (<=0 disables)
    saturation_limit_phys: int = 100    # stop after this many consecutive cycles with no physisorption (<=0 disables)
    seed: Optional[int] = None          # RNG seed (None = non-deterministic)

    # ---------------- event probabilities ----------------
    p_phys: float = 1.0                 # physisorption probability
    p_chem: float = 1.0                 # chemisorption probability
    p_des_chem: float = 0.0             # desorption probability right after chemisorption
    use_dynamic_prob: bool = False      # True: gap-dependent physisorption probability

    # ---------------- relaxation handshake ----------------
    use_relaxation_pre: bool = True     # relax before the co-reactant step
    use_relaxation_post: bool = True    # relax after the co-reactant step
    relax_padding_xy: float = 8.0       # CIF cell padding around the structure (Å)
    relax_padding_z: float = 12.0
    poll_sec_pre: float = 60.0          # how often to check for the relaxed CIF (s)
    poll_sec_post: float = 30.0
    wait_log_sec: float = 3600.0        # how often to print a "still waiting" line (s)
    wait_timeout_sec: Optional[float] = None  # None = wait forever

    @classmethod
    def from_json(cls, path: str) -> "SYCConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        known = {f.name for f in fields(cls)}
        unknown = [k for k in raw if k not in known and not k.startswith("_")]
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {unknown}")
        cfg = cls(**{k: v for k, v in raw.items() if k in known})
        base = os.path.dirname(os.path.abspath(path))
        for name in _PATH_FIELDS:
            value = getattr(cfg, name)
            if value and not os.path.isabs(value):
                setattr(cfg, name, os.path.normpath(os.path.join(base, value)))
        return cfg
