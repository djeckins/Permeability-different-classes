"""
Mathematical utility functions for membrane permeability calculations.

This module provides dielectric response functions for computing solvation
energies, geometric helpers for distance calculations, and element
identification from atom names.
"""

import math

from numpy.typing import NDArray


def dielectric_born_factor(eps1: float, eps2: float) -> float:
    """
    Compute the Born solvation energy difference between two dielectric media.

    Represents the change in electrostatic self-energy when moving a charged
    sphere from medium with dielectric eps1 to one with eps2.
    :param eps1: dielectric constant of the initial medium
    :param eps2: dielectric constant of the final medium
    :returns: Born factor difference (1/eps1 - 1/eps2)
    """
    return 1.0 / eps1 - 1.0 / eps2


def dielectric_dipole_factor(eps1: float, eps2: float) -> float:
    """
    Compute the dipole solvation energy difference between two dielectric media.

    Calculates the difference in dipole-medium interaction energy when transferring
    a molecular dipole between media of different dielectric constants.
    :param eps1: dielectric constant of the initial medium
    :param eps2: dielectric constant of the final medium
    :returns: dipole solvation factor difference f(eps2) - f(eps1)
    """

    def _response(eps: float) -> float:
        if eps <= 1.05:
            return (1.0 / 6.0) * math.log(eps)
        ln = math.log(eps)
        return 3.0 * eps * ln / (eps * ln - eps + 1.0) - 6.0 / ln - 2.0

    return _response(eps2) - _response(eps1)


def dielectric_ionic_factor(eps1: float, eps2: float) -> float:
    """
    Compute the ionic solvation contribution for charge-dielectric coupling.

    Captures the entropy-like contribution to ionic solvation from reorganization

    of the medium around a charge.
    :param eps1: dielectric constant of the initial medium
    :param eps2: dielectric constant of the final medium
    :returns: ionic reorganization factor difference f(eps1) - f(eps2)
    """

    def _response(eps: float) -> float:
        ln = math.log(eps)
        return 1.0 / ln - 1.0 / (eps * ln) - 1.0

    return _response(eps1) - _response(eps2)


def distance(xyz: NDArray, i: int, j: int) -> float:
    """
    Calculate Euclidean distance between two atoms.

    :param xyz: coordinate array of shape (3, n_atoms) where xyz[:, i] gives (x, y, z) of atom i
    :param i: index of the first atom
    :param j: index of the second atom
    :returns: distance between atoms i and j in the same units as xyz
    """
    d = xyz[:, i] - xyz[:, j]
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


def get_element(atom_name: str) -> str:
    """
    Extract the element symbol from an atom name.

    Handles standard PDB format, numbered hydrogens (e.g., "1HB" -> "H"),
    and two-letter elements (CL, BR).

    :param atom_name: atom name string, typically from PDB or similar format
    :returns: one or two character element symbol in uppercase
    """
    name = atom_name.strip()
    if not name:
        return ""

    # Handle digit-prefixed names (common for hydrogens)
    if name[0].isdigit():
        return name[1].upper() if len(name) > 1 else ""

    # Check for two-letter elements
    if len(name) >= 2:
        two = name[:2].upper()
        if two in ("CL", "BR"):
            return two

    return name[0].upper()


def normalize_atom_name(name: str) -> str:
    """
    Normalize an atom name for comparison by stripping whitespace and uppercasing.

    :param name: raw atom name string
    :returns: normalized atom name
    """
    return name.strip().upper()
