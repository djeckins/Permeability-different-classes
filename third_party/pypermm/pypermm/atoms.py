"""
Atom representation and atom-level operations.

This module provides the Atom dataclass for storing atomic properties,
atom type assignment based on connectivity, ionization state assignment
based on pH, and reference atom selection for orientation.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .constants import ASP_LIP, LN10, PKA_VALUES, VDW_RADII
from .math_utils import distance


@dataclass
class Atom:
    """
    Representation of an atom with properties for permeability calculation.

    :param element: element symbol (e.g., 'C', 'N', 'O', 'CL')
    :param x: x-coordinate in Angstroms
    :param y: y-coordinate in Angstroms
    :param z: z-coordinate in Angstroms
    :param name: atom name (e.g., 'CA', 'N', 'O1')
    :param atom_type: internal atom type code (1-21)
    :param radius: van der Waals radius in Angstroms
    :param asa: accessible surface area in Angstroms squared
    :param dipole: local dipole moment contribution
    :param charge: formal charge for ionizable groups
    :param charge2: secondary charge parameter for electrostatics
    :param eioniz: ionization energy correction based on pH
    :param asaref: reference ASA for scaling dipole/charge contributions
    :param hbond: intramolecular hydrogen bond flag (1.0 if H-bonded)
    :param solv: atomic solvation parameter for lipid phase
    """

    element: str
    x: float
    y: float
    z: float
    name: str = ""
    atom_type: int = 0
    radius: float = 0.0
    asa: float = 0.0
    dipole: float = 0.0
    charge: float = 0.0
    charge2: float = 0.0
    eioniz: float = 0.0
    asaref: float = 0.0
    hbond: float = 0.0
    solv: float = 0.0


@dataclass
class ReferenceSelection:
    """
    Result of reference atom selection for molecule orientation.

    :param indices: indices of selected reference atoms in the heavy atom list
    :param labels: human-readable labels for reference atoms
    """

    indices: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def assign_atom_types(atoms: list[Atom]) -> None:
    """
    Assign internal atom types based on element and connectivity.

    Analyzes molecular connectivity to determine the chemical environment
    of each atom. Types are assigned based on element identity, coordination
    number, presence of hydrogen neighbors, and presence of polar neighbors.
    Modifies atoms in place, setting atom_type, radius, and solv attributes.

    :param atoms: list of atoms to type (modified in place)
    """
    nat = len(atoms)
    xyz = np.array([[a.x, a.y, a.z] for a in atoms]).T

    for i, atom in enumerate(atoms):
        elem = atom.element
        nbr = nbrh = nbrpol = 0

        for j in range(nat):
            if i == j:
                continue
            dist = distance(xyz, i, j)
            elem_j = atoms[j].element

            # Skip H-H pairs for neighbor counting
            if elem == "H" and elem_j == "H":
                continue

            # Determine bond distance cutoff based on elements
            if elem == "H" or elem_j == "H":
                cutoff = 1.4
            elif elem in ("BR", "S") or elem_j in ("BR", "S"):
                cutoff = 1.95
            elif elem == "I" or elem_j == "I":
                cutoff = 2.3
            else:
                cutoff = 1.84

            if dist <= cutoff:
                nbr += 1
                if elem_j == "H":
                    nbrh += 1
                if elem_j in ("N", "O"):
                    nbrpol += 1

        # Assign type based on element and neighbors
        if elem == "C":
            if nbr >= 4:
                atom.atom_type = 1  # sp3
            elif nbr == 3:
                atom.atom_type = 2  # sp2
            elif nbr == 2:
                atom.atom_type = 11  # sp
            else:
                atom.atom_type = 1
            # sp3 carbon adjacent to polar becomes type 8
            if atom.atom_type == 1 and nbrpol >= 1:
                atom.atom_type = 8
        elif elem == "N":
            atom.atom_type = 4  # default N
            if nbrh >= 1:
                atom.atom_type = 3  # N-H
            if nbr == 1:
                atom.atom_type = 12  # nitrile
            if nbrpol >= 1:
                atom.atom_type = 17  # nitro
        elif elem == "O":
            atom.atom_type = 6  # default O
            if nbrh >= 1:
                atom.atom_type = 5  # O-H
            if nbrpol >= 1:
                atom.atom_type = 17  # nitro
        elif elem == "S":
            atom.atom_type = 7
        elif elem == "F":
            atom.atom_type = 13
        elif elem == "CL":
            atom.atom_type = 14
        elif elem == "BR":
            atom.atom_type = 15
        elif elem == "I":
            atom.atom_type = 16
        elif elem == "H":
            atom.atom_type = 21
        else:
            atom.atom_type = 1  # default to sp3 carbon

        # Assign radius and solvation parameter
        if atom.atom_type in VDW_RADII:
            atom.radius = VDW_RADII[atom.atom_type]
        if atom.atom_type in ASP_LIP:
            atom.solv = ASP_LIP[atom.atom_type]


def assign_ionization(
    atoms: list[Atom],
    pH: float,
    temperature: float,
    enabled: bool = True,
) -> None:
    """
    Assign ionization states and energies based on pH.

    For ionizable groups (carboxylates and amines), calculates the free energy
    cost of maintaining the charged state at the given pH relative to the pKa.
    Modifies atoms in place, setting charge and eioniz attributes.

    :param atoms: list of atoms (modified in place)
    :param pH: solution pH for ionization calculations
    :param temperature: temperature in Kelvin
    :param enabled: if False, all ionization properties are set to zero
    """
    rt = (8.314 * 0.239 / 1000.0) * temperature  # kcal/mol

    for atom in atoms:
        atom.charge = 0.0
        atom.eioniz = 0.0
        atom.asaref = 0.0

    if not enabled:
        return

    for atom in atoms:
        if atom.atom_type not in PKA_VALUES:
            continue

        pka = PKA_VALUES[atom.atom_type]

        if atom.atom_type == 9:  # Carboxylate
            atom.eioniz = rt * LN10 * (pH - pka)
            atom.charge = -1.0
        elif atom.atom_type == 10:  # Ammonium
            atom.eioniz = rt * LN10 * (pka - pH)
            atom.charge = +1.0


def determine_reference_atoms(atoms: list[Atom]) -> ReferenceSelection:
    """
    Select reference atoms for defining molecular orientation.

    The reference atoms define the center point for rotating the molecule
    during membrane insertion optimization. Selection prioritizes atoms
    near polar groups (likely membrane-interacting).

    :param atoms: list of all atoms in the molecule
    :returns: ReferenceSelection containing indices and labels for 1-4 reference atoms
    """
    heavy = [(idx, atom) for idx, atom in enumerate(atoms) if atom.atom_type != 21]
    if not heavy:
        return ReferenceSelection()

    coords = np.array([[atom.x, atom.y, atom.z] for _, atom in heavy], dtype=np.float64).T

    preferred = _select_reference_by_polar_neighbors(heavy, coords)
    ref_indices = _build_reference_set(heavy, coords, preferred or 0)
    labels = [heavy[i][1].element + str(i) for i in ref_indices]

    return ReferenceSelection(indices=ref_indices, labels=labels)


def _select_reference_by_polar_neighbors(heavy: list[tuple[int, Atom]], coords: NDArray) -> int:
    """
    Find the atom with the most polar neighbors.

    :param heavy: list of (original_index, atom) for heavy atoms
    :param coords: coordinates array of shape (3, n_heavy)
    :returns: index (in heavy list) of atom with most polar neighbors,
              or atom closest to center of mass if none have polar neighbors
    """
    polar_types = {3, 4, 5, 6, 7, 9, 10, 12, 17}
    distcut = 3.0
    num_polar = np.zeros(len(heavy), dtype=int)

    for i, (_, _atom_i) in enumerate(heavy):
        for j, (_, atom_j) in enumerate(heavy):
            if i == j:
                continue
            if atom_j.atom_type in polar_types:
                if distance(coords, i, j) <= distcut:
                    num_polar[i] += 1

    if num_polar.max() > 0:
        return int(num_polar.argmax())

    # Fallback: atom closest to center of mass
    com = coords.mean(axis=1, keepdims=True)
    dists = np.sqrt(((coords - com) ** 2).sum(axis=0))
    return int(np.argmin(dists))


def _build_reference_set(
    heavy: list[tuple[int, Atom]],
    coords: NDArray,
    start_idx: int,
    target_len: int = 4,
) -> list[int]:
    """
    Build a connected set of reference atoms starting from a given atom.

    :param heavy: list of (original_index, atom) for heavy atoms
    :param coords: coordinates array of shape (3, n_heavy)
    :param start_idx: index to start building the reference set from
    :param target_len: target number of reference atoms
    :returns: list of indices of selected reference atoms
    """
    selected = [start_idx]
    used = {start_idx}
    dcut = 1.8  # Initial bond distance cutoff

    while len(selected) < target_len:
        added = False
        for idx in range(len(heavy)):
            if idx in used:
                continue
            dist = distance(coords, idx, selected[-1])
            if dist <= dcut:
                selected.append(idx)
                used.add(idx)
                added = True
                if len(selected) == target_len:
                    break
        if not added:
            dcut += 0.5
            if dcut > 6.0:
                break

    # Fill remaining with nearest atoms if needed
    if len(selected) < target_len:
        remaining = [idx for idx in range(len(heavy)) if idx not in used]
        remaining.sort(key=lambda idx: distance(coords, idx, start_idx))
        for idx in remaining:
            selected.append(idx)
            if len(selected) == target_len:
                break

    return selected
