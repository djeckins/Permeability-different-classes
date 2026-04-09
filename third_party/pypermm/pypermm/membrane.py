"""
Membrane profile construction and caching.

This module builds the position-dependent physicochemical properties
of the lipid bilayer membrane, including atom-type-specific solvation
energies, dielectric environment, and hydrogen bonding capacity profiles.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import CFDIEL, CFPOL, CFSIGMA, DOPC
from .math_utils import (
    dielectric_born_factor,
    dielectric_dipole_factor,
    dielectric_ionic_factor,
)


@dataclass
class MembraneProfile:
    """
    Pre-computed membrane property profiles.

    :param z: z-positions in Angstroms, typically from -50 to +50
    :param sig: solvation parameters by atom type, shape (18, n_z)
    :param sig_coo: special solvation profile for neutral carboxylic acid
    :param edip: dipole energy contribution at each z-position
    :param echarge: ionic charge energy contribution at each z-position
    :param ehbond: hydrogen bond energy contribution at each z-position
    :param eclm: additional electrostatic contribution (currently unused)
    """

    z: NDArray[np.floating]
    sig: NDArray[np.floating]
    sig_coo: NDArray[np.floating]
    edip: NDArray[np.floating]
    echarge: NDArray[np.floating]
    ehbond: NDArray[np.floating]
    eclm: NDArray[np.floating]


# Module-level cache for membrane profiles
_PROFILE_CACHE: dict[tuple[float, float], MembraneProfile] = {}


def build_membrane_profile(dmembr: float = 30.0) -> MembraneProfile:
    """
    Build or retrieve a cached membrane property profile.

    Constructs the complete z-dependent profile of membrane properties
    for computing solvation free energies. The profile spans from
    z=-50A to z=+50A with 0.01A resolution based on the DOPC membrane model.

    :param dmembr: membrane thickness in Angstroms
    :returns: pre-computed membrane properties at each z-position
    """
    key = (dmembr, 0.01)
    cached = _PROFILE_CACHE.get(key)
    if cached is not None:
        return cached

    z = np.arange(-50.0, 50.01, 0.01)
    nz = len(z)

    sig = np.zeros((18, nz))
    sig_coo = np.zeros(nz)
    edip = np.zeros(nz)
    echarge = np.zeros(nz)
    ehbond = np.zeros(nz)
    eclm = np.zeros(nz)

    p = DOPC
    zi0 = dmembr / 2.0
    Zc = zi0 + p["deltaz"]
    z0wat = zi0 + p["z0wat_offset"]
    p0 = math.sqrt(2 * math.pi)

    for m, zcur in enumerate(z):
        # Compute amplitude of each membrane region at this z
        amp0 = np.zeros(6)
        for i in range(4):
            gauss = (p["vmol0"][i] / (p0 * p["area"] * p["sigma0"][i])) * (
                math.exp(-0.5 * ((zcur - Zc[i]) / p["sigma0"][i]) ** 2)
                + math.exp(-0.5 * ((zcur + Zc[i]) / p["sigma0"][i]) ** 2)
            )
            if i == 0:
                gauss *= 2
            amp0[i] = gauss

        # Hydrocarbon region (error function profile)
        a = (zcur + zi0) / math.sqrt(2 * p["si"])
        b = (zcur - zi0) / math.sqrt(2 * p["si"])
        amphdc = 0.5 * (math.erf(a) - math.erf(b))
        amp0[4] = max(0, amphdc - amp0[0])
        amptot = sum(amp0[:5])

        # Water region
        ampwat = max(0, 1 - min(1, amptot))
        cwat = (abs(zcur) - z0wat) / p["alam_wat"]
        wat_pen = p["awat"] - (p["awat"] - p["bwat"]) / (1 + math.exp(cwat))
        if ampwat < wat_pen:
            amp0[4] -= wat_pen - ampwat
            ampwat = wat_pen
        amp0[5] = ampwat

        # Reorder and normalize
        amp = np.array([amp0[0], amp0[4], amp0[1], amp0[2], amp0[3], amp0[5]])
        fm = amp / p["vmol"]
        fm = fm / fm.sum()

        amp_nw = amp[:5]
        amplip = amp_nw / amp_nw.sum() if amp_nw.sum() > 0 else np.zeros(5)

        fm_aq = fm[3:6]
        xaq = fm_aq / fm_aq.sum() if fm_aq.sum() > 0 else np.zeros(3)

        # Compute effective dielectric
        eps_inv = sum(amplip[i] / p["eps"][i] for i in range(5) if p["eps"][i] > 0)
        epsbil = 1 / eps_inv if eps_inv > 0 else 2.0

        fmwat = fm[5]
        if fmwat > 0.10:
            epsbil += (p["eps"][5] - epsbil) * (fmwat - 0.10) / 0.90

        # Compute effective hydrogen bonding parameters
        alpbil = sum(fm[i] * p["alp"][i] for i in range(6))
        alplip = sum(amplip[i] * p["alp"][i] for i in range(3))
        betlip = sum(amplip[i] * p["bet"][i] for i in range(3))
        alpaq = sum(xaq[i] * p["alp"][i + 3] for i in range(3))
        betaq = sum(xaq[i] * p["bet"][i + 3] for i in range(3))

        # Dielectric response factors
        f1 = dielectric_born_factor(epsbil, 78.4)
        f2 = dielectric_dipole_factor(epsbil, 78.4)
        f3 = dielectric_ionic_factor(epsbil, 78.4)

        # Reference parameter differences
        dbetlip, dbetaq = betlip - 0.35, betaq - 0.35
        dalplip, dalpaq = alplip - 0.82, alpaq - 0.82
        fmaq = fm[3] + fm[4] + fm[5]

        # Compute solvation contribution for each atom type
        for iat in range(1, 18):
            sig_lip = sig_aq = sig_lip1 = sig_aq1 = 0.0

            if iat == 1:  # sp3 carbon
                sig_lip = CFSIGMA[0] - CFSIGMA[1] * f1
            elif iat == 2:  # sp2 carbon
                sig_lip = CFSIGMA[2] + CFSIGMA[3] * dalplip
            elif iat == 3:  # N-H
                sig_lip = -CFSIGMA[4] * f1 + CFSIGMA[5] * dbetlip
                sig_aq = CFSIGMA[5] * dbetaq
            elif iat == 4:  # N (no H)
                sig_lip = -CFSIGMA[6] * f1 + CFSIGMA[7] * dalplip
                sig_aq = CFSIGMA[7] * dalpaq
            elif iat == 5:  # O-H
                sig_lip = -CFSIGMA[8] * f1 + CFSIGMA[9] * dalplip + CFSIGMA[10] * dbetlip
                sig_aq = CFSIGMA[9] * dalpaq + CFSIGMA[10] * dbetaq
            elif iat == 6:  # O (no H)
                sig_lip = -CFSIGMA[11] * f1 + CFSIGMA[12] * dalplip
                sig_aq = CFSIGMA[12] * dalpaq
            elif iat == 7:  # S
                sig_lip = CFSIGMA[13] + CFSIGMA[14] * dalplip
            elif iat == 8:  # polar-adjacent C
                sig_lip = CFPOL
            elif iat == 9:  # COO-
                sig_lip = CFSIGMA[15] * dalplip
                sig_aq = CFSIGMA[15] * dalpaq
                sig_lip1 = 0.5 * (
                    -CFSIGMA[8] * f1
                    + CFSIGMA[9] * dalplip
                    + CFSIGMA[10] * dbetlip
                    - CFSIGMA[11] * f1
                    + CFSIGMA[12] * dalplip
                )
                sig_aq1 = 0.5 * (CFSIGMA[9] * dalpaq + CFSIGMA[10] * dbetaq + CFSIGMA[12] * dalpaq)
            elif iat == 10:  # NH3+
                sig_lip = CFSIGMA[16] * dbetlip
                sig_aq = CFSIGMA[16] * dbetaq
                sig_lip1 = -CFSIGMA[4] * f1 + CFSIGMA[5] * dbetlip
                sig_aq1 = CFSIGMA[5] * dbetaq
            elif iat == 11:  # sp carbon
                sig_lip = CFSIGMA[17]
            elif iat == 12:  # nitrile
                sig_lip = CFSIGMA[18]
            elif iat == 13:  # F
                sig_lip = CFSIGMA[19] + CFSIGMA[20] * dalplip
            elif iat == 14:  # Cl
                sig_lip = CFSIGMA[21] + CFSIGMA[22] * dalplip
            elif iat == 15:  # Br
                sig_lip = CFSIGMA[23] + CFSIGMA[24] * dalplip
            elif iat == 16:  # I
                sig_lip = CFSIGMA[25] + CFSIGMA[26] * dalplip
            elif iat == 17:  # nitro
                sig_lip = CFSIGMA[27] + CFSIGMA[28] * dalplip

            # Mix lipid and aqueous contributions
            if fmaq > 0.9999:
                sig_bil = sig_aq
            else:
                rt, surf = 0.592, 14.0
                dg = max(-50, min(50, (sig_lip - sig_aq) * surf / rt))
                xsl = 1 / (math.exp(dg) * fmaq / (1 - fmaq) + 1)
                sig_bil = sig_aq * (1 - xsl) + sig_lip * xsl
            sig[iat, m] = sig_bil

            # Special handling for neutral carboxylic acid
            if iat == 9:
                if fmaq > 0.9999:
                    sig_coo[m] = sig_aq1
                else:
                    dg = max(-50, min(50, (sig_lip1 - sig_aq1) * surf / rt))
                    xsl = 1 / (math.exp(dg) * fmaq / (1 - fmaq) + 1)
                    sig_coo[m] = sig_aq1 * (1 - xsl) + sig_lip1 * xsl

        # Hydrogen bond correction
        ehbond[m] = min(0, 1.50 * (alpbil - p["alp"][5]) / p["alp"][5])

        # Dielectric-dependent contributions
        edip[m] = CFDIEL[0] * f2
        echarge[m] = f3 * CFDIEL[1]

    profile = MembraneProfile(
        z=z,
        sig=sig,
        sig_coo=sig_coo,
        edip=edip,
        echarge=echarge,
        ehbond=ehbond,
        eclm=eclm,
    )
    _PROFILE_CACHE[key] = profile
    return profile


def clear_profile_cache() -> None:
    """Clear the membrane profile cache."""
    _PROFILE_CACHE.clear()
