"""
Accessible Surface Area (ASA) calculation.

This module implements the Shrake-Rupley algorithm for computing
solvent-accessible surface area of atoms in a molecule.
"""

import math

import numpy as np

from .atoms import Atom


def calculate_asa(
    atoms: list[Atom],
    probe: float = 1.4,
    zslice: float = 0.05,
) -> None:
    """
    Calculate accessible surface area for each atom using Shrake-Rupley algorithm.

    Only heavy atoms (non-hydrogen) are considered. Atoms with ASA < 0.02 Å**2
    are set to zero to remove numerical noise.

    :param atoms: list of atoms (modified in place to set asa attribute)
    :param probe: probe sphere radius in Angstroms (water = 1.4 A)
    :param zslice: z-slice thickness in Angstroms for numerical integration
    """
    heavy = [a for a in atoms if a.atom_type != 21]
    nat = len(heavy)

    if nat == 0:
        return

    xyz = np.array([[a.x, a.y, a.z] for a in heavy])
    rad = np.array([a.radius + probe for a in heavy])
    radsq = rad**2

    # Bounding box for spatial hashing
    xmin, ymin, zmin = xyz.min(axis=0) - rad.max()
    rmax = rad.max() * 2

    # Spatial hash for neighbor lookup
    cubes: dict[tuple[int, int, int], list[int]] = {}
    for i in range(nat):
        key = (
            int((xyz[i, 0] - xmin) / rmax),
            int((xyz[i, 1] - ymin) / rmax),
            int((xyz[i, 2] - zmin) / rmax),
        )
        if key not in cubes:
            cubes[key] = []
        cubes[key].append(i)

    pi2 = 2 * math.pi

    for ir in range(nat):
        xr, yr, zr = xyz[ir]
        rr, rrsq = rad[ir], radsq[ir]

        # Find cube containing this atom
        ix = int((xr - xmin) / rmax)
        iy = int((yr - ymin) / rmax)
        iz = int((zr - zmin) / rmax)

        # Collect neighbors from adjacent cubes
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    key = (ix + dx, iy + dy, iz + dz)
                    if key in cubes:
                        for j in cubes[key]:
                            if j != ir:
                                dxy = math.sqrt((xr - xyz[j, 0]) ** 2 + (yr - xyz[j, 1]) ** 2)
                                neighbors.append((j, xr - xyz[j, 0], yr - xyz[j, 1], dxy))

        # No neighbors = fully exposed
        if not neighbors:
            heavy[ir].asa = pi2 * 2 * rr * rr
            continue

        # Slice through z-axis
        nzp = max(1, int(2 * rr / zslice + 0.5))
        zres = 2 * rr / nzp
        area = 0.0

        for izz in range(nzp):
            zgrid = zr - rr + zres * (izz + 0.5)
            rsec2r = rrsq - (zgrid - zr) ** 2
            if rsec2r <= 0:
                continue
            rsecr = math.sqrt(rsec2r)

            # Collect arcs blocked by neighbors
            arcs: list[tuple[float, float]] = []

            for j, dx, dy, dxy in neighbors:
                rsec2n = radsq[j] - (zgrid - xyz[j, 2]) ** 2
                if rsec2n <= 0:
                    continue
                rsecn = math.sqrt(rsec2n)

                # No overlap
                if dxy >= rsecr + rsecn:
                    continue

                # Completely buried
                if dxy <= abs(rsecr - rsecn):
                    if rsecr <= rsecn:
                        arcs = [(0.0, pi2)]
                        break
                    continue

                # Compute blocked arc
                cos_alpha = max(-1, min(1, (dxy**2 + rsec2r - rsec2n) / (2 * dxy * rsecr)))
                alpha = math.acos(cos_alpha)
                beta = math.atan2(dy, dx) + math.pi
                ti, tf = beta - alpha, beta + alpha

                # Wrap angles to [0, 2pi]
                if ti < 0:
                    ti += pi2
                if tf > pi2:
                    tf -= pi2

                if tf >= ti:
                    arcs.append((ti, tf))
                else:
                    arcs.append((ti, pi2))
                    arcs.append((0.0, tf))

            # Compute exposed arc length
            if not arcs:
                arcsum = pi2
            else:
                arcs.sort()
                arcsum = arcs[0][0]  # Gap before first arc
                t = arcs[0][1]
                for ti, tf in arcs[1:]:
                    if t < ti:
                        arcsum += ti - t  # Gap between arcs
                    t = max(t, tf)
                arcsum += pi2 - t  # Gap after last arc

            area += arcsum * zres

        heavy[ir].asa = area * rr

    # Clean up numerical noise
    for a in heavy:
        if a.asa < 0.02:
            a.asa = 0.0
