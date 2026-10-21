"""Element radii (Å) used for steric-overlap checks, and colours for plotting.

Note: the original script defined 'Ru' twice (1.97 and 2.12). Python keeps the
last value, so 2.12 Å was the radius actually used in all published runs; it is
kept here unchanged.
"""

ELEMENT_RADII = {
    'H': 1.2, 'He': 1.4, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.1, 'P': 1.8, 'S': 1.8, 'Cl': 1.75, 'Ar': 1.88,
    'K': 2.75, 'Ca': 2.31, 'Sc': 2.11, 'Ti': 2.0, 'V': 1.92, 'Cr': 1.88, 'Mn': 1.83, 'Fe': 1.94, 'Co': 1.92, 'Ni': 1.63,
    'Cu': 1.4, 'Zn': 1.39, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.9, 'Br': 1.85, 'Kr': 2.02,
    'Rb': 3.03, 'Sr': 2.49, 'Y': 2.32, 'Zr': 2.23, 'Nb': 2.18, 'Mo': 2.08, 'Tc': 2.0, 'Ru': 2.12, 'Rh': 1.95,
    'Pd': 1.86, 'Ag': 1.63, 'Cd': 1.58, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
    'Cs': 3.43, 'Ba': 2.68, 'La': 2.43, 'Ce': 2.42, 'Pr': 2.4, 'Nd': 2.38, 'Pm': 2.37, 'Sm': 2.29, 'Eu': 2.32,
    'Gd': 2.37, 'Tb': 2.21, 'Dy': 2.29, 'Ho': 2.16, 'Er': 2.35, 'Tm': 2.27, 'Yb': 2.42, 'Lu': 2.21,
    'Ta': 2.09, 'W': 2.08, 'Re': 2.05, 'Os': 2.02, 'Ir': 2.0, 'Pt': 1.93, 'Au': 1.72, 'Hg': 1.55,
    'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Th': 2.45, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8,
    'Cm': 1.69, 'Bk': 1.68, 'Cf': 1.68, 'Es': 1.65, 'Fm': 1.67, 'Md': 1.73, 'No': 1.76, 'Lr': 1.61,
    'Rf': 2.0, 'Db': 2.0, 'Sg': 2.0, 'Bh': 2.0, 'Hs': 2.0, 'Mt': 2.0, 'Ds': 2.0, 'Rg': 2.0, 'Cn': 2.0,
    'Nh': 2.0, 'Fl': 2.0, 'Mc': 2.0, 'Lv': 2.0, 'Ts': 2.0, 'Og': 2.0,
}

# Fallback radii for elements missing from ELEMENT_RADII.
DEFAULT_RADIUS_EXISTING = 1.5  # atoms already in the KD-tree
DEFAULT_RADIUS_NEW = 0.5       # atoms being tested for overlap

ELEMENT_COLORS = {
    'H': 'white', 'He': 'cyan', 'Li': 'purple', 'Be': 'olive', 'B': 'brown', 'C': 'black', 'N': 'green', 'O': 'red',
    'F': 'lime', 'Ne': 'cyan', 'Na': 'blue', 'Mg': 'green', 'Al': 'gray', 'Si': 'orange', 'P': 'violet', 'S': 'yellow',
    'Cl': 'green', 'Ar': 'cyan', 'K': 'violet', 'Ca': 'green', 'Sc': 'gray', 'Ti': 'silver', 'V': 'darkgray',
    'Cr': 'gray', 'Mn': 'darkorange', 'Fe': 'brown', 'Co': 'blue', 'Ni': 'green', 'Cu': 'orange', 'Zn': 'gray',
    'Ga': 'purple', 'Ge': 'orange', 'As': 'gray', 'Se': 'red', 'Br': 'brown', 'Kr': 'cyan', 'Rb': 'violet',
    'Sr': 'green', 'Y': 'gray', 'Zr': 'silver', 'Nb': 'gray', 'Mo': 'gray', 'Tc': 'gray', 'Ru': 'gray', 'Rh': 'gray',
    'Pd': 'gray', 'Ag': 'gray', 'Cd': 'gray', 'In': 'gray', 'Sn': 'gray', 'Sb': 'gray', 'Te': 'gray', 'I': 'purple',
    'Xe': 'cyan', 'Cs': 'violet', 'Ba': 'green', 'Au': 'gold',
}
