"""
Main entry point for PERMM membrane permeability predictions.

This module contains the run_permm() function which orchestrates
the complete permeability calculation workflow.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .asa import calculate_asa
from .atoms import Atom, assign_atom_types, assign_ionization, determine_reference_atoms
from .dipoles import assign_dipoles, assign_hbonds, read_dipole_lib_json
from .membrane import build_membrane_profile
from .orientation import (
    N_ORIENT,
    compute_energy_profile,
    find_best_orientation_batch,
    prepare_atom_arrays,
)
from .permeability import calculate_permeability

logger = logging.getLogger(__name__)


def run_permm(
    atomic_symbols: list[str],
    coordinates: list[list[float]],
    dipole_lib_path: str | None = None,
    dmembr: float = 30.0,
    reference_atoms: Sequence[str] | None = None,
    ph: float = 7.4,
    temperature: float = 298.0,
    swpka: bool = True,
) -> dict:
    """
    Run the PERMM membrane permeability prediction.

    This is the main entry point for permeability calculations. It takes
    molecular coordinates and returns predicted permeability coefficients
    for various membrane systems.

    :param atomic_symbols: element symbols for each atom (e.g., ["C", "C", "O", "H", ...])
    :param coordinates: cartesian coordinates for each atom in Angstroms,
                        as [[x1, y1, z1], [x2, y2, z2], ...]
    :param dipole_lib_path: path to JSON dipole library file; if None, uses bundled default
    :param dmembr: membrane thickness in Angstroms
    :param reference_atoms: names of atoms to use as orientation reference;
                            if None, automatically selects atoms near polar groups
    :param ph: solution pH for ionization calculations
    :param temperature: temperature in Kelvin
    :param swpka: if True, consider ionization states; if False, treat all groups as neutral
    :returns: dictionary containing atoms, arrays, profile, z, energies, asatot,
              E_bind, logP_BLM, logP_plasma, logP_BBB, logP_Caco2, and logP_PAMPA
    """
    # Create atom list
    atoms = []
    for symbol, xyz in zip(atomic_symbols, coordinates, strict=True):
        assert len(xyz) == 3
        atoms.append(Atom(element=symbol, x=xyz[0], y=xyz[1], z=xyz[2]))

    # Assign atom types based on connectivity
    assign_atom_types(atoms)
    assign_ionization(atoms, ph, temperature, swpka)

    heavy = [a for a in atoms if a.atom_type != 21]
    logger.info(f"{len(heavy)} heavy atoms")
    types: dict[int, int] = {}
    for a in heavy:
        types[a.atom_type] = types.get(a.atom_type, 0) + 1
    logger.debug(f"Atom types: {types}")

    # Load and apply dipole library
    if dipole_lib_path is None:
        default_lib = Path(__file__).parent / "dipole_lib.json"
        if default_lib.exists():
            dipole_lib_path = str(default_lib)

    if dipole_lib_path:
        dipole_table = read_dipole_lib_json(dipole_lib_path)
        assign_dipoles(atoms, dipole_table)
        logger.debug(f"{sum(1 for a in heavy if a.dipole > 0)} atoms with dipoles")

    # Detect intramolecular H-bonds (disabled if using pKa)
    assign_hbonds(atoms, enabled=not swpka)

    nhbond = sum(1 for a in heavy if a.hbond != 0)
    if nhbond > 0:
        logger.debug(f"{nhbond} intramolecular H-bonds detected")

    # Calculate accessible surface area
    calculate_asa(atoms)
    asatot = sum(a.asa for a in heavy)
    logger.info(f"Total ASA: {asatot:.2f} Å²")

    # Determine reference atoms for orientation
    selection = None
    if reference_atoms is None:
        selection = determine_reference_atoms(atoms)

    logger.info(f"Building membrane profile (thickness={dmembr:.1f} Å)...")
    profile = build_membrane_profile(dmembr)

    logger.debug("Pre-computing atom arrays...")
    arrays = prepare_atom_arrays(
        atoms,
        reference_atom_indices=selection.indices if selection else None,
        reference_atom_names=reference_atoms,
    )
    if selection and selection.labels:
        logger.debug(f"Reference atoms: {', '.join(selection.labels[:4])}")
    logger.debug(f"{len(arrays.active_indices)} atoms with non-zero ASA")
    logger.debug(f"{N_ORIENT} orientations per z-position")

    logger.info("Finding optimal binding position...")

    # Coarse search for best binding depth
    best_shift = 15.0
    best_energy = float("inf")
    for shift in np.arange(10, 20.2, 0.2):
        _, _, e = find_best_orientation_batch(arrays, profile, shift)
        if e < best_energy:
            best_energy = e
            best_shift = shift

    logger.info(f"Binding: E={best_energy:.2f} kcal/mol at z={best_shift:.1f} Å")
    logger.info("Computing energy profile...")

    # Full energy profile
    z_vals, energies = compute_energy_profile(arrays, profile)
    perm = calculate_permeability(z_vals, energies, asatot)

    logger.info(f"ΔGbind: {perm['E_bind']:.2f} kcal/mol")
    logger.info(f"logP BLM: {perm['logP_BLM']:.2f}")
    logger.info(f"logP PAMPA: {perm['logP_PAMPA']:.2f}")
    logger.info(f"logP Plasma: {perm['logP_plasma']:.2f}")
    logger.info(f"logP Caco-2: {perm['logP_Caco2']:.2f}")
    logger.info(f"logP BBB: {perm['logP_BBB']:.2f}")

    return {
        "atoms": atoms,
        "arrays": arrays,
        "profile": profile,
        "z": z_vals,
        "energies": energies,
        "asatot": asatot,
        **perm,
    }
