# SYC-ALD

**SYC (Stochastic Yielding and Correction)** is a simulation framework for the initial growth of
atomic layer deposition (ALD) films. It couples multi-layer Monte Carlo (MC) sampling of
precursor adsorption with structure relaxation by the PFP machine-learning potential on
[Matlantis](https://matlantis.com/).

- **Stochastic yielding** — precursor molecules are sampled from a gas layer, physisorb and
  chemisorb on exposed surface sites, with steric hindrance from ligands already on the surface.
- **Correction** — after the precursor pulse and after the co-reactant pulse, the whole structure
  is relaxed with PFP, so the next cycle starts from a physically relaxed surface.

This repository contains the code used in:

> M. Lee, W. Kim, K. Min, B. Lee, Y. Choi, B. Gu, H.-B.-R. Lee,
> *Investigation of Surface-Dependent Initial Growth Behavior in Atomic Layer Deposition Using SYC Framework* (in preparation).

---

## Repository layout

```
SYC-ALD/
├── syc/                        # MC driver (runs on your local machine)
│   ├── config.py               #   all parameters + JSON loading
│   ├── simulation.py           #   ALD cycle: gas layer → physisorption → chemisorption → relax → co-reactant → relax
│   ├── geometry.py             #   substrate tiling, rigid-body placement, overlap tests, on-site detection
│   ├── io.py                   #   CSV/CIF I/O and the relaxation file handshake
│   ├── relaxation_bridge.py    #   export → wait for relaxed CIF → import; restart logic
│   ├── analysis.py             #   thickness, height maps, post-relaxation atom matching
│   └── constants.py            #   element radii / colours
├── scripts/
│   ├── run_syc.py              # entry point for the MC driver
│   └── matlantis_relax.py      # PFP relaxation watcher (runs on Matlantis, self-contained)
├── config/
│   └── example_trust_on_ru.json
├── data/                       # input geometries (see "Input files")
└── requirements.txt
```

## How it works

```
 local machine (MC driver)                                   Matlantis (relaxation watchers)
 ─────────────────────────                                   ───────────────────────────────
 cycle N
  ├─ exposures × up to 30
  │    ├─ gas layer (random sequential insertion)
  │    ├─ physisorption (descent onto free on-sites)
  │    └─ chemisorption (template, 72 rotations, ligands kept)
  ├─ write  before/cycleN_before_oxidant_for_relax.cif ──────▶  matlantis_relax.py --stage pre
  │  wait   relaxation/cycleN_before_oxidant_relaxed.cif ◀──────  FIRE → LBFGS (PBE+D3)
  ├─ co-reactant step: remove ligand atoms (C, H)
  ├─ write  after/cycleN_after_oxidant_for_relax.cif ────────▶  matlantis_relax.py --stage post
  │  wait   relaxation2/cycleN_after_oxidant_relaxed.cif ◀─────  FIRE → LBFGS (PBE+U)
  └─ relaxed film = starting surface of cycle N+1
```

The two programs communicate only through files. Each exported CIF has a matching
`*.meta.json` holding the coordinate shift applied on export, which is undone when the relaxed
structure is read back. Files are written to a temporary name and renamed when complete, so
neither side reads a partial file.

If the MC driver and the watchers run on different machines, the four exchange folders
(`before/`, `relaxation/`, `after/`, `relaxation2/`) must be kept in sync (shared drive, rsync,
or manual upload/download).

### One ALD cycle in detail

| Step | What happens | Key parameters |
|---|---|---|
| On-site detection | Exposed `site_element` atoms: no atom of the same element within `onsite_r_xy` laterally sits more than `onsite_dz` higher. | `onsite_r_xy`, `onsite_dz` |
| Gas layer | Molecules inserted at random positions/orientations in `z_gas_min..z_gas_max` until `max_failures` consecutive insertions overlap. | `max_failures`, `overlap_scale` |
| Physisorption | Each molecule descends by `dz` above one of its `n_onsite_candidates` nearest free on-sites (within `phys_xy_cutoff`). It physisorbs with probability `p_phys` when its lowest atom is within `phys_gap_max` of the site and it does not overlap the film, chemisorbed precursors (with ligands), or molecules physisorbed earlier in the same exposure. | `dz`, `phys_gap_max`, `phys_xy_cutoff`, `p_phys` |
| Chemisorption | The physisorbed molecule is replaced by the chemisorbed template on its site; `n_rotations` in-plane rotations are tried until one fits. One precursor per site per cycle. Ligands stay and block later molecules (steric hindrance). Molecules that fail are purged. | `n_rotations`, `p_chem`, `p_des_chem` |
| End of exposure | After `max_no_new_phy_streak` consecutive exposures without new physisorption (or `max_exposures_per_cycle`). | |
| Pre-co-reactant relaxation | Film + chemisorbed precursors (with ligands) relaxed by PFP. | `use_relaxation_pre` |
| Co-reactant step | All `ligand_elements` atoms are removed (metal ALD: no atoms added). | `ligand_elements` |
| Post-co-reactant relaxation | Ligand-free film relaxed by PFP. | `use_relaxation_post` |

## Installation

```bash
git clone https://github.com/<your-account>/SYC-ALD.git
cd SYC-ALD
pip install -r requirements.txt
```

On Matlantis, `ase` and `pfp-api-client` are preinstalled. `scripts/matlantis_relax.py`
depends on nothing else in this repository, so you can copy that single file if you prefer.

## Input files

Place these in `data/` and point the config at them.

**Gas-phase precursor** and **chemisorbed template** — CSV with a header row `name,1,2,3`
(element, x, y, z in Å):

```
name,1,2,3
Ru,0.000,0.000,0.000
C,1.234,0.000,1.050
...
```

- The gas-phase geometry is rotated about its own origin and translated, so place the molecule
  near the origin.
- The chemisorbed template must be given **relative to the surface on-site atom at (0, 0, 0)**;
  it is placed on each site and rotated about z.

**Substrate** — CSV without header: two `lattice` rows (in-plane lattice vectors a1, a2) followed
by the atoms of the unit cell. The cell is tiled to fill `area_size × area_size`.

```
lattice,2.706,0.000,0.000
lattice,-1.353,2.343,0.000
Ru,0.000,0.000,0.000
Ru,1.353,0.781,2.142
...
```

Rows that are empty or start with `0` are ignored.

## Configuration

All parameters are defined, with comments, in [`syc/config.py`](syc/config.py). A JSON file only
needs the values you want to change; relative paths are resolved against the JSON file's folder.
See [`config/example_trust_on_ru.json`](config/example_trust_on_ru.json).

## Running

**1. Start the watchers on Matlantis** (one terminal or notebook cell each):

```bash
python scripts/matlantis_relax.py --stage pre  --in-dir runs/trust_on_ru/before --out-dir runs/trust_on_ru/relaxation  --max-cycles 50
python scripts/matlantis_relax.py --stage post --in-dir runs/trust_on_ru/after  --out-dir runs/trust_on_ru/relaxation2 --max-cycles 50
```

Defaults reproduce the published settings: FIRE pre-relaxation (fmax ≥ 0.05 eV/Å, ≤ 500 steps,
adsorbate z frozen) followed by LBFGS (fmax = 0.001 eV/Å, ≤ 4000 steps, maxstep 0.02 Å),
bottom 3 Å fixed, PBC = (1, 1, 0), ≥ 20 Å vacuum, PFP model v8.0.0,
`PBE_PLUS_D3` for `pre` and `PBE_U` for `post`. Run `python scripts/matlantis_relax.py -h` for all options.

**2. Start the MC driver:**

```bash
python scripts/run_syc.py --config config/example_trust_on_ru.json
```

Options: `--max-cycles N`, `--seed S` (reproducible sampling), `--no-relax` (MC only, no Matlantis;
useful for testing inputs).

From Python / Jupyter:

```python
from syc import SYCConfig, run_simulation

cfg = SYCConfig.from_json("config/example_trust_on_ru.json")
cfg.max_cycles = 10
res = run_simulation(cfg)
res["stats"]                 # per-cycle counters
res["snapshots"][3]["post_out"]  # relaxed film after cycle 3, list of (element, x, y, z)
```

### Restarting

On start-up the driver looks for the highest cycle N with both
`relaxation2/cycleN_after_oxidant_relaxed.cif` and `after/cycleN_after_oxidant_for_relax.meta.json`,
loads that structure, and continues from cycle N+1. `max_cycles` is the **absolute** last cycle.

Relaxed CIFs that already exist are reused without waiting. **Use a fresh run folder for each new
simulation**, otherwise old relaxed structures will be picked up.

## Outputs

| Path (under `output_dir`) | Content |
|---|---|
| `cycle_stats.csv` | Per cycle: on-sites, gas molecules tried, physisorption attempts/successes, chemisorption attempts/successes/desorptions, chemisorbed count, film thickness (Å) |
| `phy/cycleN_substrate_plus_phy.cif` | Bare substrate + every molecule that physisorbed in cycle N (inspection only) |
| `before/`, `after/` | Structures sent for relaxation (+ `.meta.json`) |
| `relaxation/`, `relaxation2/` | Relaxed structures, FIRE intermediates (`*_prefire.cif`), `.traj`, `.log`, `.energy.json` |

Film thickness is the height of the highest film atom above the substrate top.
`syc/analysis.py` provides height-map helpers for surface plots.

## Notes and limitations

- **Cell used for relaxation.** Structures are exported in an orthorhombic box equal to their
  bounding box plus `relax_padding_xy` (8 Å) on each side. With `--pbc 1,1,0` the slab is therefore
  a finite island separated from its periodic images by ~16 Å of vacuum, not an infinite slab.
- **Different functionals per stage.** `pre` uses PBE+D3 and `post` uses PBE+U by default, so
  energies are not comparable between the two stages.
- **Steric overlap test.** Two atoms overlap when their distance is below
  `overlap_scale × (r1 + r2)`, with `overlap_scale = 0.2` and radii from `syc/constants.py`. Only
  the nearest existing atom is checked for each new atom.
- **Desorption.** When a chemisorption event is followed by desorption (`p_des_chem > 0`), the
  next in-plane rotation is tried on the same site. With the default `p_des_chem = 0` this has no
  effect.

## Citation

If you use this code, please cite the paper above (details will be added on publication).

## License

<!-- TODO: choose a license (e.g. MIT, BSD-3-Clause) with your PI and add a LICENSE file. -->

## Contact

Prof. Han-Bo-Ram Lee's group, Department of Materials Science and Engineering, Incheon National University.
