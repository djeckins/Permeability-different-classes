"""
Physical constants, atom type parameters, and membrane composition data.

This module contains all the empirical parameters used in the PERMM
membrane permeability calculation, including van der Waals radii,
atomic solvation parameters, dielectric coefficients, and DOPC
membrane geometry parameters.
"""

import math

import numpy as np

LN10 = math.log(10.0)
"""Natural logarithm of 10, used for pH/pKa calculations."""

ASAREF_CUT = 0.1
"""Cutoff for reference accessible surface area (Å²)."""


VDW_RADII: dict[int, float] = {
    1: 1.88,  # sp3 carbon (aliphatic)
    2: 1.76,  # sp2 carbon (aromatic/alkene)
    3: 1.64,  # nitrogen with H (amine)
    4: 1.64,  # nitrogen without H
    5: 1.46,  # oxygen with H (hydroxyl)
    6: 1.42,  # oxygen without H (carbonyl/ether)
    7: 1.77,  # sulfur
    8: 1.88,  # sp3 carbon adjacent to polar
    9: 1.42,  # carboxylate oxygen (COO-)
    10: 1.64,  # ammonium nitrogen (NH3+)
    11: 1.75,  # sp carbon (alkyne)
    12: 1.61,  # nitrile nitrogen
    13: 1.44,  # fluorine
    14: 1.74,  # chlorine
    15: 1.85,  # bromine
    16: 2.00,  # iodine
    17: 1.53,  # nitro group (N or O)
}
"""Van der Waals radii (Å) indexed by atom type."""

ASP_LIP: dict[int, float] = {
    1: -0.022,  # sp3 carbon
    2: -0.019,  # sp2 carbon
    3: 0.053,  # amine nitrogen
    4: 0.053,  # other nitrogen
    5: 0.057,  # hydroxyl oxygen
    6: 0.057,  # carbonyl/ether oxygen
    7: 0.010,  # sulfur
    8: -0.022,  # polar-adjacent carbon
    9: 0.129,  # carboxylate
    10: 0.118,  # ammonium
    11: -0.002,  # alkyne carbon
    12: 0.008,  # nitrile nitrogen
    13: -0.007,  # fluorine
    14: -0.010,  # chlorine
    15: -0.012,  # bromine
    16: -0.012,  # iodine
    17: 0.016,  # nitro group
}
"""Atomic solvation parameters for lipid phase (kcal/mol/Å²)."""


CFSIGMA = np.array(
    [
        -0.017,  # 0: C(sp3) constant
        0.008,  # 1: C(sp3) dielectric term
        -0.013,  # 2: C(sp2) constant
        0.010,  # 3: C(sp2) alpha term
        -0.088,  # 4: N-H dielectric term
        0.000,  # 5: N-H beta term
        -0.124,  # 6: N (no H) dielectric term
        0.000,  # 7: N (no H) alpha term
        -0.028,  # 8: O-H dielectric term
        -0.027,  # 9: O-H alpha term
        -0.063,  # 10: O-H beta term
        -0.044,  # 11: O (no H) dielectric term
        -0.019,  # 12: O (no H) alpha term
        0.002,  # 13: S constant
        0.010,  # 14: S alpha term
        -0.221,  # 15: COO-/NH3+ alpha term
        -0.023,  # 16: NH3+ beta term
        -0.002,  # 17: sp carbon
        0.008,  # 18: nitrile
        -0.007,  # 19: F constant
        0.010,  # 20: F alpha term
        -0.010,  # 21: Cl constant
        0.010,  # 22: Cl alpha term
        -0.013,  # 23: Br constant
        0.010,  # 24: Br alpha term
        -0.012,  # 25: I constant
        0.010,  # 26: I alpha term
        -0.016,  # 27: nitro constant
        0.010,  # 28: nitro alpha term
    ]
)
"""Coefficients for computing atom-type-specific solvation energy contributions."""

CFDIEL = np.array([1.865, 0.198])
"""Dielectric-dependent energy coefficients (dipole and ionic)."""

CFPOL = 0.001
"""Polarity correction coefficient for polar-adjacent carbons."""


PKA_VALUES: dict[int, float] = {
    9: 4.5,  # Carboxylate (COO-)
    10: 9.5,  # Ammonium (NH3+)
}
"""Reference pKa values for ionizable groups."""


DOPC: dict[str, np.ndarray | float] = {
    # Gaussian parameters for membrane regions
    "sigma0": np.array([3.05, 2.05, 2.41, 2.98]),
    "Zc0": np.array([9.6, 14.8, 19.1, 20.6]),
    "vmol0": np.array([44.0, 139.0, 86.0, 106.0]),
    "deltaz": np.array([-4.8, 0.4, 4.7, 6.2]),
    # Hydrogen bonding parameters by region
    "alp": np.array([0.00, 0.00, 0.00, 0.82, 0.83, 0.82]),
    "bet": np.array([0.07, 0.00, 0.88, 1.74, 1.74, 0.35]),
    "pia": np.array([0.34, 0.00, 0.60, 0.73, 0.14, 1.09]),
    # Dielectric and volume parameters
    "eps": np.array([2.00, 2.24, 6.94, 78.4, 78.4, 78.4]),
    "vmol": np.array([44.0, 108.0, 139.0, 86.0, 106.0, 18.0]),
    # Membrane geometry
    "area": 67.4,
    "si": 2.48,
    # Water penetration parameters
    "awat": 0.066,
    "bwat": 0.010,
    "z0wat_offset": -6.0,
    "alam_wat": 1.10,
}
"""DOPC (dioleoylphosphatidylcholine) membrane model parameters."""
