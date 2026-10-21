"""Post-processing helpers (not used by the MC loop itself)."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .io import Atom


def film_thickness(film_atoms: List[Atom], substrate_top_z: float, threshold: float = 0.5) -> float:
    """Maximum height of film atoms above the substrate top (Å). Atoms within
    ``threshold`` of the substrate top are treated as substrate."""
    z = [a[3] for a in film_atoms if a[3] > substrate_top_z + threshold]
    return max(z) - substrate_top_z if z else 0.0


def make_height_map(atoms: List[Atom], area_size: float, substrate_top_z: float, bin_size: float = 1.0):
    """Surface height map from the maximum z in each XY bin. Returns ``(x_grid, y_grid, z_grid)``."""
    n = int(np.ceil(area_size / bin_size))
    z_grid = np.full((n, n), substrate_top_z, dtype=float)
    for (_, x, y, z) in atoms:
        if z <= substrate_top_z + 0.5:
            continue
        ix = min(int(x / bin_size), n - 1)
        iy = min(int(y / bin_size), n - 1)
        if z > z_grid[iy, ix]:
            z_grid[iy, ix] = z
    grid = np.linspace(0, area_size, n)
    return grid, grid.copy(), z_grid


def make_height_map_continuous(atoms: List[Atom], area_size: float, substrate_top_z: float,
                               bin_size: float = 1.0, sigma: float = 1.2):
    """Height map with empty bins filled by nearest neighbour, then Gaussian-smoothed
    (``sigma = 0`` disables smoothing)."""
    n = int(np.ceil(area_size / bin_size))
    z_grid = np.full((n, n), np.nan, dtype=float)
    for (_, x, y, z) in atoms:
        if z <= substrate_top_z + 0.5:
            continue
        ix = min(int(x / bin_size), n - 1)
        iy = min(int(y / bin_size), n - 1)
        if np.isnan(z_grid[iy, ix]) or z > z_grid[iy, ix]:
            z_grid[iy, ix] = z

    yy, xx = np.mgrid[0:n, 0:n]
    known = np.isfinite(z_grid)
    if known.any():
        tree = cKDTree(np.column_stack([xx[known], yy[known]]))
        missing = ~known
        _, idx = tree.query(np.column_stack([xx[missing], yy[missing]]), k=1)
        z_grid[missing] = z_grid[known][idx]
    else:
        z_grid[:] = substrate_top_z

    if sigma and sigma > 0:
        z_grid = gaussian_filter(z_grid, sigma=sigma)
    z_grid = np.maximum(z_grid, substrate_top_z)
    grid = np.linspace(0, area_size, n)
    return grid, grid.copy(), z_grid


def match_atoms_after_relax(raw_atoms: List[Atom], relaxed_film: List[Atom], element: str = "O",
                            match_cutoff: float = 1.5) -> List[Atom]:
    """Map atoms of ``element`` added in a cycle (pre-relaxation coordinates) to their
    nearest counterpart in the relaxed film (one-to-one, within ``match_cutoff`` Å).
    Useful for oxide variants of the workflow where the co-reactant step adds O."""
    post = [a for a in relaxed_film if a[0] == element]
    if not raw_atoms or not post:
        return []
    post_xyz = np.array([a[1:] for a in post], float)
    tree = cKDTree(post_xyz)
    used, matched = set(), []
    for a in raw_atoms:
        dist, idx = tree.query(a[1:], k=1)
        if dist <= match_cutoff and idx not in used:
            used.add(idx)
            matched.append((element, *map(float, post_xyz[idx])))
    return matched
