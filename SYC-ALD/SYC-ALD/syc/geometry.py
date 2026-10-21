"""Geometry helpers: substrate tiling, rigid-body placement, overlap tests, on-sites."""

from __future__ import annotations

from typing import List

import numpy as np
from scipy.spatial import cKDTree

from .constants import DEFAULT_RADIUS_EXISTING, DEFAULT_RADIUS_NEW, ELEMENT_RADII
from .io import Atom


# ----------------------------------------------------------------------------
# Substrate
# ----------------------------------------------------------------------------
def build_tiled_substrate(unit_atoms: List[Atom], lattice_vectors, area_size: float) -> List[Atom]:
    """Tile the unit cell along a1, a2 and keep atoms with 0 <= x, y <= area_size."""
    a1 = np.array(lattice_vectors[0])
    a2 = np.array(lattice_vectors[1])
    nx = int(np.ceil(2 * area_size / np.linalg.norm(a1)))
    ny = int(np.ceil(2 * area_size / np.linalg.norm(a2)))

    tiled = []
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            shift = i * a1 + j * a2
            for el, x, y, z in unit_atoms:
                pos = np.array([x, y, z]) + shift
                if 0 <= pos[0] <= area_size and 0 <= pos[1] <= area_size:
                    tiled.append((el, *pos))
    return tiled


# ----------------------------------------------------------------------------
# Rigid-body placement
# ----------------------------------------------------------------------------
def generate_random_rotation() -> np.ndarray:
    """Random rotation matrix from three uniform Euler angles (z-y-z)."""
    theta, phi, psi = np.random.uniform(0, 2 * np.pi, 3)
    rz1 = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    rz2 = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return rz1 @ ry @ rz2


def transform_precursor(base_atoms: List[Atom], x0: float, y0: float, z0: float, R: np.ndarray) -> List[Atom]:
    """Rotate a molecule by R about the origin of its input frame, then translate to (x0, y0, z0)."""
    t = np.array([x0, y0, z0])
    return [(el, *(R @ np.array([x, y, z]) + t)) for el, x, y, z in base_atoms]


def place_template_on_onsite(template_atoms: List[Atom], onsite_xyz, angle_rad: float) -> List[Atom]:
    """Place the chemisorbed template (defined relative to the on-site atom at the origin)
    on a surface site, rotated by ``angle_rad`` about z."""
    ox, oy, oz = onsite_xyz
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return [(el, x * c - y * s + ox, x * s + y * c + oy, z + oz) for el, x, y, z in template_atoms]


def get_precursor_center(atoms: List[Atom], center_element: str = "Ru") -> np.ndarray:
    """Position of the first ``center_element`` atom; falls back to the geometric centre."""
    for el, x, y, z in atoms:
        if el == center_element:
            return np.array([x, y, z])
    return np.array([[x, y, z] for (_, x, y, z) in atoms]).mean(axis=0)


# ----------------------------------------------------------------------------
# Overlap tests
# ----------------------------------------------------------------------------
def build_kdtree(atom_list: List[Atom]):
    """Return ``(tree, coords, radii)``; ``tree`` is None for an empty list."""
    if not atom_list:
        return None, [], []
    coords = np.array([[x, y, z] for (_, x, y, z) in atom_list])
    radii = np.array([ELEMENT_RADII.get(a, DEFAULT_RADIUS_EXISTING) for (a, _, _, _) in atom_list])
    return cKDTree(coords), coords, radii


def check_overlap(new_atoms: List[Atom], tree, radii, scale: float = 0.2) -> bool:
    """True if any new atom is closer than ``scale * (r_new + r_old)`` to its nearest existing atom.

    scale < 1 is permissive (allows bonding distances), scale = 1 is a hard-sphere test.
    Only the single nearest neighbour of each new atom is checked.
    """
    if tree is None or len(radii) == 0:
        return False
    for a, x, y, z in new_atoms:
        r_new = ELEMENT_RADII.get(a, DEFAULT_RADIUS_NEW)
        dist, idx = tree.query([x, y, z], k=1)
        if dist < scale * (r_new + radii[idx]):
            return True
    return False


# ----------------------------------------------------------------------------
# Adsorption sites
# ----------------------------------------------------------------------------
def get_surface_onsites(film_atoms: List[Atom], element: str = "Ru", r_xy: float = 1.5,
                        dz: float = 0.4, z_min=None):
    """Exposed surface atoms of ``element`` that can host a chemisorbed precursor.

    An atom is *buried* if another atom of the same element within lateral radius
    ``r_xy`` sits more than ``dz`` above it. Atoms below ``z_min`` are ignored.
    Returns a list of ``(film_index, x, y, z)``.
    """
    cand = [(i, x, y, z) for i, (a, x, y, z) in enumerate(film_atoms) if a == element]
    if z_min is not None:
        cand = [c for c in cand if c[3] >= z_min]
    if not cand:
        return []

    xy = np.array([[x, y] for (_, x, y, _) in cand], float)
    z = np.array([c[3] for c in cand], float)
    tree = cKDTree(xy)

    onsites = []
    for k, (film_idx, x, y, zk) in enumerate(cand):
        neigh = tree.query_ball_point(xy[k], r_xy)
        if not any(z[j] > zk + dz for j in neigh if j != k):
            onsites.append((film_idx, x, y, zk))
    return onsites
