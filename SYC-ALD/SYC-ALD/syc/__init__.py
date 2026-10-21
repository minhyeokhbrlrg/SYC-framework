"""SYC (Stochastic Yielding and Correction) framework for atomic layer deposition growth.

Multi-layer Monte Carlo sampling of precursor adsorption coupled with machine-learning
potential (Matlantis/PFP) structure relaxation.
"""

from .config import SYCConfig
from .simulation import run_cycle, run_simulation

__version__ = "1.0.0"
__all__ = ["SYCConfig", "run_simulation", "run_cycle", "__version__"]
