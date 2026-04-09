"""
Molecular orientation optimization for membrane insertion.

This module handles pre-computing rotation matrices for orientation sampling,
converting atomic properties to efficient array format, finding optimal
molecular orientation at each membrane depth, and computing energy profiles.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .atoms import Atom, _build_reference_set, _select_reference_by_polar_neighbors
from .math_utils import normalize_atom_name
from .membrane import MembraneProfile


def build_rotation_matrices(
    step: float = 2.0,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Pre-compute rotation matrix z-row for all orientations.

    Generates a grid of molecular orientations by sampling rotation
    angles (phi, theta) uniformly in spherical coordinates.

    :param step: angular step size in degrees for both phi and theta
    :returns: tuple of (rot_z, phi_flat, theta_flat) where rot_z is the
              z-row of rotation matrix for each orientation with shape
              (n_orient, 3), and phi_flat/theta_flat are the angles in radians
    """
    phi_deg = np.arange(0, 180, step)
    theta_deg = np.arange(0, 360, step)

    phi_vals = np.radians(phi_deg)
    theta_vals = np.radians(theta_deg)

    phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals, indexing="ij")
    phi_flat = phi_grid.ravel()
    theta_flat = theta_grid.ravel()

    a = np.sin(phi_flat)
    b = np.cos(phi_flat)
    si = np.sin(theta_flat)
    co = np.cos(theta_flat)

    # Z-row of rotation matrix: [-b*si, a*si, co]
    rot_z = np.column_stack([-b * si, a * si, co])

    return rot_z, phi_flat, theta_flat


# Pre-compute at module load
ROT_Z, PHI_FLAT, THETA_FLAT = build_rotation_matrices(2.0)
N_ORIENT = len(PHI_FLAT)


@dataclass
class AtomArrays:
    """
    Atom properties in array format for vectorized energy calculations.

    :param xyz_centered: centered coordinates, shape (3, n_atoms)
    :param rotated_z: pre-computed z-coordinates for all orientations, shape (n_orient, n_atoms)
    :param asa: accessible surface area for each atom
    :param atom_type: atom type codes
    :param dipole: dipole contributions
    :param asaref: reference ASA for scaling
    :param charge: formal charges
    :param charge2: secondary charge parameters
    :param eioniz: ionization energies
    :param hbond: hydrogen bond flags
    :param active_indices: indices of atoms with ASA > 0
    :param reference_indices: indices of reference atoms used for centering
    """

    xyz_centered: NDArray[np.floating]
    rotated_z: NDArray[np.floating]
    asa: NDArray[np.floating]
    atom_type: NDArray[np.int32]
    dipole: NDArray[np.floating]
    asaref: NDArray[np.floating]
    charge: NDArray[np.floating]
    charge2: NDArray[np.floating]
    eioniz: NDArray[np.floating]
    hbond: NDArray[np.floating]
    active_indices: NDArray[np.int32]
    reference_indices: list[int]


def prepare_atom_arrays(
    atoms: list[Atom],
    reference_atom_indices: Sequence[int] | None = None,
    reference_atom_names: Sequence[str] | None = None,
) -> AtomArrays:
    """
    Convert atom list to array format for efficient computation.

    Extracts properties from heavy atoms and pre-computes rotated
    z-coordinates for all orientations in the sampling grid.

    :param atoms: list of all atoms in the molecule
    :param reference_atom_indices: indices of atoms to use as reference for centering
    :param reference_atom_names: names of atoms to use as reference (alternative to indices)
    :returns: pre-computed arrays for energy calculation
    :raises ValueError: if no heavy atoms are found
    """
    heavy = [(idx, atom) for idx, atom in enumerate(atoms) if atom.atom_type != 21]
    if not heavy:
        raise ValueError("No heavy atoms found for array preparation.")

    xyz = np.array([[atom.x, atom.y, atom.z] for _, atom in heavy], dtype=np.float64).T

    # Determine reference atoms
    reference_indices: list[int] = []
    if reference_atom_indices:
        reference_indices = [idx for idx in reference_atom_indices if 0 <= idx < len(heavy)]
    elif reference_atom_names:
        normalized = [
            normalize_atom_name(name) for name in reference_atom_names if isinstance(name, str) and name.strip()
        ]
        for target in normalized:
            for heavy_idx, (_, atom) in enumerate(heavy):
                if normalize_atom_name(atom.name) == target:
                    reference_indices.append(heavy_idx)
                    break

    if not reference_indices:
        start = _select_reference_by_polar_neighbors(heavy, xyz)
        reference_indices = _build_reference_set(heavy, xyz, start)

    # Center coordinates on first reference atom
    center_idx = reference_indices[0] if reference_indices else 0
    center = xyz[:, center_idx : center_idx + 1]
    xyz_centered = xyz - center

    # Find atoms with non-zero ASA
    asa = np.array([atom.asa for _, atom in heavy], dtype=np.float64)
    active_indices = np.where(asa > 0)[0].astype(np.int32)

    # Pre-compute rotated z-coordinates for all orientations
    xyz_centered = np.ascontiguousarray(xyz_centered)
    rotated_z = ROT_Z @ xyz_centered

    return AtomArrays(
        xyz_centered=xyz_centered,
        rotated_z=rotated_z,
        asa=asa,
        atom_type=np.array([atom.atom_type for _, atom in heavy], dtype=np.int32),
        dipole=np.array([atom.dipole for _, atom in heavy], dtype=np.float64),
        asaref=np.array([atom.asaref for _, atom in heavy], dtype=np.float64),
        charge=np.array([atom.charge for _, atom in heavy], dtype=np.float64),
        charge2=np.array([atom.charge2 for _, atom in heavy], dtype=np.float64),
        eioniz=np.array([atom.eioniz for _, atom in heavy], dtype=np.float64),
        hbond=np.array([atom.hbond for _, atom in heavy], dtype=np.float64),
        active_indices=active_indices,
        reference_indices=reference_indices,
    )


def find_best_orientation_batch(
    arrays: AtomArrays,
    profile: MembraneProfile,
    shift: float,
) -> tuple[float, float, float]:
    """
    Find optimal molecular orientation at a given membrane depth.

    Evaluates the insertion energy for all pre-computed orientations
    and returns the angles and energy of the best one.

    :param arrays: pre-computed atom arrays from prepare_atom_arrays()
    :param profile: membrane property profile from build_membrane_profile()
    :param shift: vertical shift (z-coordinate) of the molecular center
    :returns: tuple of (phi, theta, energy) where phi and theta are optimal
              angles in radians and energy is insertion energy in kcal/mol
    """
    active = arrays.active_indices
    if active.size == 0:
        return 0.0, 0.0, 0.0

    nz = len(profile.z)

    # Compute z-coordinates and membrane profile indices
    z_all = arrays.rotated_z[:, active] + shift
    m_all = ((z_all + 50.0) * 100.0).astype(np.int32)
    np.clip(m_all, 0, nz - 1, out=m_all)
    valid_mask = (z_all >= -50.0) & (z_all <= 50.0)

    # Extract atom properties for active atoms
    asa = arrays.asa[active]
    atom_types = arrays.atom_type[active]
    dipoles = arrays.dipole[active]
    asaref = arrays.asaref[active]
    charge = arrays.charge[active]
    charge2 = arrays.charge2[active]
    eioniz = arrays.eioniz[active]
    hbonds = arrays.hbond[active]

    # Profile arrays
    sig = profile.sig
    sig_coo = profile.sig_coo
    ehbond = profile.ehbond
    echarge = profile.echarge
    edip = profile.edip
    eclm = profile.eclm

    energies = np.zeros(N_ORIENT, dtype=np.float64)

    # Hydrogen bond contributions
    hb_idx = np.nonzero(hbonds)[0]
    if hb_idx.size:
        mask = valid_mask[:, hb_idx]
        m = m_all[:, hb_idx]
        energies += (ehbond[m] * mask).sum(axis=1)

    # Non-ionizable atoms (types <=8 or >=11)
    nonion_mask = (atom_types <= 8) | (atom_types >= 11)
    nonion_idx = np.nonzero(nonion_mask)[0]
    if nonion_idx.size:
        nonion_types = atom_types[nonion_idx]
        for t in np.unique(nonion_types):
            t_idx = nonion_idx[nonion_types == t]
            if t_idx.size == 0:
                continue
            mask = valid_mask[:, t_idx]
            if not mask.any():
                continue
            m = m_all[:, t_idx]
            sig_t = sig[t]
            contrib = sig_t[m] * asa[t_idx]
            energies += (contrib * mask).sum(axis=1)

    # Ionizable atoms (types 9=COO- and 10=NH3+)
    for t in (9, 10):
        ion_idx = np.nonzero(atom_types == t)[0]
        if ion_idx.size == 0:
            continue
        mask = valid_mask[:, ion_idx]
        if not mask.any():
            continue
        m = m_all[:, ion_idx]
        sig_t = sig[t]
        asa_vals = asa[ion_idx]
        asaref_vals = asaref[ion_idx]
        charge_vals = charge[ion_idx]
        eioniz_vals = eioniz[ion_idx]

        # Ionized form energy
        eion = sig_t[m] * asa_vals
        has_ref = asaref_vals > 0
        if np.any(has_ref):
            scale = np.ones_like(asa_vals)
            less = (asa_vals < asaref_vals) & has_ref
            scale[less] = asa_vals[less] / asaref_vals[less]
            ref_idx = np.nonzero(has_ref)[0]
            if ref_idx.size:
                m_ref = m[:, ref_idx]
                charge_term = charge_vals[has_ref] * scale[has_ref]
                eion[:, ref_idx] += echarge[m_ref] * charge_term

        # Neutral form energy
        if t == 9:
            eneutr = sig_coo[m] * asa_vals
        else:
            eneutr = sig[3, m] * asa_vals
        eneutr += eioniz_vals

        # Take minimum of ionized and neutral
        energies += (np.minimum(eion, eneutr) * mask).sum(axis=1)

    # Dipole contributions
    dip_idx = np.nonzero((dipoles != 0.0) & (asaref > 0.0))[0]
    if dip_idx.size:
        mask = valid_mask[:, dip_idx]
        if mask.any():
            m = m_all[:, dip_idx]
            asa_vals = asa[dip_idx]
            asaref_vals = asaref[dip_idx]
            dip_vals = dipoles[dip_idx]
            scale = np.ones_like(asa_vals)
            less = asa_vals < asaref_vals
            scale[less] = asa_vals[less] / asaref_vals[less]
            contrib = edip[m] * (dip_vals * scale)
            energies += (contrib * mask).sum(axis=1)

    # Additional electrostatic term (charge2)
    charge2_idx = np.nonzero(charge2 != 0.0)[0]
    if charge2_idx.size:
        mask = valid_mask[:, charge2_idx]
        if mask.any():
            m = m_all[:, charge2_idx]
            contrib = eclm[m] * charge2[charge2_idx]
            energies += (contrib * mask).sum(axis=1)

    best = np.argmin(energies)
    return PHI_FLAT[best], THETA_FLAT[best], energies[best]


def compute_energy_profile(
    arrays: AtomArrays,
    profile: MembraneProfile,
    z_range: tuple[float, float] = (-45, 45),
) -> tuple[NDArray, NDArray]:
    """
    Compute optimal insertion energy across all membrane depths.

    :param arrays: pre-computed atom arrays
    :param profile: membrane property profile
    :param z_range: range of z-positions to evaluate
    :returns: tuple of (z_vals, energies) arrays
    """
    z_vals = np.arange(z_range[0], z_range[1] + 1, 1.0)
    energies = np.zeros(len(z_vals))

    for i, z in enumerate(z_vals):
        _, _, e = find_best_orientation_batch(arrays, profile, z)
        energies[i] = e

    return z_vals, energies
