"""
PERMM: Permeability Prediction for Membrane-Mediated Transport.

A physically-based model for predicting membrane permeability of small
molecules using lipid bilayer structure and solvation thermodynamics.

Example usage:

    from pypermm import run_permm

    symbols = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    coords = [[0, 0, 0], [1.5, 0, 0], [2.3, 1.1, 0], ...]
    results = run_permm(symbols, coords)
    print(f"logP PAMPA: {results['logP_PAMPA']:.2f}")
"""

from .atoms import Atom, ReferenceSelection
from .dipoles import DipoleTable
from .membrane import MembraneProfile, clear_profile_cache
from .orientation import AtomArrays
from .pypermm import run_permm

__all__ = [
    # Main function
    "run_permm",
    # Data structures
    "Atom",
    "AtomArrays",
    "DipoleTable",
    "MembraneProfile",
    "ReferenceSelection",
    # Utilities
    "clear_profile_cache",
]
