"""
Dipole assignment and intramolecular hydrogen bond detection.

This module handles loading dipole libraries from JSON format, assigning
dipole moments based on local chemical environment, and detecting
intramolecular hydrogen bonds.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .atoms import Atom
from .constants import ASAREF_CUT
from .math_utils import distance


@dataclass
class DipoleTable:
    """
    Dipole assignment lookup tables.

    Contains two tables for dipole assignment: table1 for pattern-based
    assignment where the center atom and specific neighbors define the
    dipole distribution, and table2 for simple neighbor-based assignment.

    :param table1_center: atom types for center atoms in pattern matching
    :param table1_neighbors: required neighbor types for each pattern (4 slots, 0=empty)
    :param table1_dipoles: dipole values assigned to matched neighbors
    :param table2_center: atom types for simple matching
    :param table2_neighbors: required neighbor types for simple matching
    :param table2_dipole: dipole value for the center atom
    """

    table1_center: list[int]
    table1_neighbors: list[list[int]]
    table1_dipoles: list[list[float]]
    table2_center: list[int]
    table2_neighbors: list[list[int]]
    table2_dipole: list[float]


def read_dipole_lib_json(filepath: str | Path) -> DipoleTable:
    """
    Load dipole library from JSON format.

    :param filepath: path to the JSON dipole library file
    :returns: DipoleTable with loaded dipole assignment parameters
    """
    with open(filepath) as f:
        data = json.load(f)

    table1_center = []
    table1_neighbors = []
    table1_dipoles = []

    for entry in data["tab1"]:
        table1_center.append(entry["center"])
        table1_neighbors.append(entry["neighbors"])
        # Pad dipoles to 4 values (original format had implicit 4th value of 0.0)
        dipoles = entry["dipoles"]
        if len(dipoles) < 4:
            dipoles = dipoles + [0.0] * (4 - len(dipoles))
        table1_dipoles.append(dipoles)

    table2_center = []
    table2_neighbors = []
    table2_dipole = []

    for entry in data["tab2"]:
        table2_center.append(entry["center"])
        table2_neighbors.append(entry["neighbors"])
        table2_dipole.append(entry["dipole"])

    return DipoleTable(
        table1_center=table1_center,
        table1_neighbors=table1_neighbors,
        table1_dipoles=table1_dipoles,
        table2_center=table2_center,
        table2_neighbors=table2_neighbors,
        table2_dipole=table2_dipole,
    )


def assign_dipoles(atoms: list[Atom], dipole_table: DipoleTable) -> None:
    """
    Assign dipole moments to atoms based on local chemical environment.

    Uses pattern matching against the dipole table to identify functional
    groups and assign appropriate dipole contributions. Only heavy atoms
    (non-hydrogen) are assigned dipoles.

    :param atoms: list of atoms (modified in place to set dipole and asaref)
    :param dipole_table: dipole assignment lookup tables
    """
    heavy = [a for a in atoms if a.atom_type != 21]
    nat = len(heavy)
    xyz = np.array([[a.x, a.y, a.z] for a in heavy]).T

    def remap(atom: Atom) -> int:
        """Map atom types to simplified categories for pattern matching."""
        t = atom.atom_type
        if t == 10:
            return 3
        if t == 17:
            # NO2 type depends on element
            if atom.element == "N":
                return 4
            elif atom.element == "O":
                return 6
            return 4  # fallback
        if t == 3:
            return 4
        if t == 8:
            return 1
        if t == 9:
            return 6
        return t

    iatdi = [remap(a) for a in heavy]
    nbdi: list[list[int]] = [[] for _ in range(nat)]

    # Build neighbor lists
    for i in range(nat):
        for j in range(nat):
            if i == j:
                continue
            dist = distance(xyz, i, j)
            ti, tj = iatdi[i], iatdi[j]

            # Adjust cutoff for heavier elements
            if ti in (7, 15) or tj in (7, 15):
                cutoff = 1.93
            elif ti == 16 or tj == 16:
                cutoff = 2.3
            else:
                cutoff = 1.84

            if dist <= cutoff and len(nbdi[i]) < 4:
                nbdi[i].append(j)

    # table1 matching: functional group patterns
    for center, req_nb, dip_vals in zip(
        dipole_table.table1_center,
        dipole_table.table1_neighbors,
        dipole_table.table1_dipoles,
        strict=True,
    ):
        for i in range(nat):
            if iatdi[i] != center or heavy[i].dipole != 0.0:
                continue

            required = [(req_nb[k], dip_vals[k]) for k in range(4) if req_nb[k] != 0]
            if not required:
                continue

            used = [False] * len(nbdi[i])
            matched = []
            for req_type, dip_val in required:
                for k, nb_idx in enumerate(nbdi[i]):
                    if not used[k] and iatdi[nb_idx] == req_type:
                        used[k] = True
                        matched.append((nb_idx, dip_val))
                        break

            if len(matched) == len(required):
                heavy[i].dipole = 0.001  # Mark center as processed
                for nb_idx, dip_val in matched:
                    if heavy[nb_idx].dipole == 0.0:
                        heavy[nb_idx].dipole = dip_val
                        heavy[nb_idx].asaref = ASAREF_CUT

    # table2 matching: simple coordination patterns
    for i in range(nat):
        if heavy[i].dipole != 0.0:
            continue

        for center, req_nb, dip_val in zip(
            dipole_table.table2_center,
            dipole_table.table2_neighbors,
            dipole_table.table2_dipole,
            strict=True,
        ):
            if iatdi[i] != center:
                continue

            actual_neighbors = sorted([iatdi[k] for k in nbdi[i]])
            required_neighbors = sorted([t for t in req_nb if t != 0])

            if actual_neighbors == required_neighbors:
                heavy[i].dipole = dip_val
                heavy[i].asaref = ASAREF_CUT
                break

    # Final pass: ensure asaref is set for ALL atoms with non-zero dipole
    for i in range(nat):
        if heavy[i].dipole != 0.0:
            heavy[i].asaref = ASAREF_CUT


def assign_hbonds(atoms: list[Atom], enabled: bool = True) -> None:
    """
    Detect intramolecular hydrogen bonds.

    Identifies H-bond patterns where H is bonded to a donor (atom types
    3=NH, 5=OH, 9=COO-, 10=NH3+), acceptor is within 2.3A of H (atom types
    4=N, 5=OH, 6=O, 9=COO-), and angle Acceptor-H-Donor > 90 degrees.
    Sets hbond=1.0 on the donor heavy atom (not the hydrogen).

    :param atoms: list of atoms (modified in place to set hbond attribute)
    :param enabled: if False, all hbond values are set to 0
    """
    nat = len(atoms)
    xyz = np.array([[a.x, a.y, a.z] for a in atoms]).T

    angcut = 90.0  # degrees
    dcut = 2.3  # Angstroms, max distance from H to acceptor

    # Initialize all hbond to 0
    for a in atoms:
        a.hbond = 0.0

    if not enabled:
        return

    donor_types = {3, 5, 9, 10}
    acceptor_types = {4, 5, 6, 9}

    # Find hydrogens and check for H-bond patterns
    for i in range(nat):
        if atoms[i].atom_type != 21:  # Must be H
            continue

        # Find the donor atom that H is bonded to
        for j in range(nat):
            if j == i:
                continue

            r_hi = distance(xyz, i, j)

            # H must be bonded to donor (< 1.2 A)
            if r_hi < 1.2 and atoms[j].atom_type in donor_types:
                # Found H bonded to donor j, now look for acceptor k
                for k in range(nat):
                    if k in {i, j}:
                        continue

                    if atoms[k].atom_type not in acceptor_types:
                        continue

                    r_hk = distance(xyz, i, k)

                    if r_hk < dcut:
                        # Check angle: Acceptor-H-Donor (k-i-j)
                        r_jk_sq = sum((xyz[d, j] - xyz[d, k]) ** 2 for d in range(3))

                        cos_angle = (r_hi**2 + r_hk**2 - r_jk_sq) / (2 * r_hi * r_hk)
                        cos_angle = max(-1.0, min(1.0, cos_angle))
                        angle = math.acos(cos_angle) * 180.0 / math.pi

                        if angle > angcut:
                            atoms[j].hbond = 1.0
