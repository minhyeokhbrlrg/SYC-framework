"""MC side of the relaxation handshake and restart logic.

The MC driver never calls the MLP directly. For each relaxation it

1. writes ``cycle{N}_{stage}_for_relax.cif`` (+ ``.meta.json`` with the coordinate
   shift) to the stage's *input* folder,
2. waits until ``cycle{N}_{stage}_relaxed.cif`` appears in the stage's *output*
   folder (written by ``scripts/matlantis_relax.py`` on Matlantis), and
3. reads it back and undoes the shift.

``stage`` is ``before_oxidant`` (pre-co-reactant) or ``after_oxidant``
(post-co-reactant). The file names are kept identical to the original workflow so
existing run folders remain restartable.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List

from .io import Atom, export_for_relaxation_with_meta, import_relaxed_and_unshift, wait_for_file

STAGE_PRE = "before_oxidant"
STAGE_POST = "after_oxidant"


def run_relaxation_stage(atoms_in: List[Atom], cycle: int, stage_tag: str, in_dir: str, out_dir: str,
                         padding_xy: float = 8.0, padding_z: float = 12.0, poll_sec: float = 60,
                         timeout_sec=None, log_sec: float = 3600) -> List[Atom]:
    """Export -> wait for the external relaxer -> import. Returns relaxed atoms in simulation coordinates.

    If the relaxed CIF already exists (e.g. after a restart) it is used directly.
    """
    in_cif = os.path.join(in_dir, f"cycle{cycle}_{stage_tag}_for_relax.cif")
    in_meta = os.path.join(in_dir, f"cycle{cycle}_{stage_tag}_for_relax.meta.json")
    export_for_relaxation_with_meta(atoms_in, in_cif, in_meta, padding_xy=padding_xy,
                                    padding_z=padding_z, data_name=f"cycle{cycle}_{stage_tag}")
    print(f"[SAVE] Relax input ({stage_tag}): {in_cif}")

    relaxed_cif = os.path.join(out_dir, f"cycle{cycle}_{stage_tag}_relaxed.cif")
    if not os.path.exists(relaxed_cif):
        wait_for_file(relaxed_cif, poll_sec=poll_sec, timeout_sec=timeout_sec, log_sec=log_sec)

    relaxed = import_relaxed_and_unshift(relaxed_cif, in_meta)
    print(f"[LOAD] Using relaxed CIF ({stage_tag}): {relaxed_cif}")
    return relaxed


def find_latest_completed_cycle(relax_out_dir_post: str, relax_after_dir: str):
    """Highest cycle N with both ``cycleN_after_oxidant_relaxed.cif`` (output folder)
    and the matching ``.meta.json`` (input folder). Returns ``(N, cif, meta)``; N = 0 if none."""
    latest, latest_cif, latest_meta = 0, None, None
    for cif_path in glob.glob(os.path.join(relax_out_dir_post, f"cycle*_{STAGE_POST}_relaxed.cif")):
        m = re.match(rf"cycle(\d+)_{STAGE_POST}_relaxed\.cif$", os.path.basename(cif_path))
        if not m:
            continue
        cyc = int(m.group(1))
        meta = os.path.join(relax_after_dir, f"cycle{cyc}_{STAGE_POST}_for_relax.meta.json")
        if os.path.exists(meta) and cyc > latest:
            latest, latest_cif, latest_meta = cyc, cif_path, meta
    return latest, latest_cif, latest_meta


def load_restart_state(substrate_atoms: List[Atom], relax_out_dir_post: str, relax_after_dir: str):
    """Return ``(film_atoms, start_cycle, last_completed_cycle)``."""
    latest, cif, meta = find_latest_completed_cycle(relax_out_dir_post, relax_after_dir)
    if latest == 0:
        print("[RESTART] No completed post-relaxation found -> starting from cycle 1")
        return list(substrate_atoms), 1, 0
    film = import_relaxed_and_unshift(cif, meta)
    print(f"[RESTART] Cycles up to {latest} are complete")
    print(f"[RESTART] Restart structure: {cif}")
    print(f"[RESTART] Resuming at cycle {latest + 1}")
    return film, latest + 1, latest
