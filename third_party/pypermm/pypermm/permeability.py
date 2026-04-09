"""
Permeability coefficient calculations.

This module converts energy profiles to predicted permeability
coefficients for various membrane systems.
"""

import math

import numpy as np
from numpy.typing import NDArray


def calculate_permeability(
    z: NDArray,
    energies: NDArray,
    asatot: float,
    temp: float = 298.0,
) -> dict[str, float]:
    """
    Calculate permeability coefficients from the energy profile.

    Uses the energy barrier profile to predict membrane permeability

    using empirically calibrated relationships.
    :param z: z-positions through the membrane
    :param energies: insertion energy at each z-position (kcal/mol)
    :param asatot: total accessible surface area of the molecule (A^2)
    :param temp: temperature in Kelvin
    :returns: dictionary containing E_bind (kcal/mol), logP_BLM, logP_plasma,
              logP_BBB, logP_Caco2, and logP_PAMPA (all log cm/s)
    """
    rt = (8.314 * 0.239 / 1000) * temp  # kcal/mol

    # Minimum binding energy
    emin = energies.min()

    # Integration over membrane core region
    mask = (z >= -15) & (z <= 16)
    estat = np.sum(np.exp(energies[mask] / rt) * asatot)
    estat = -math.log10(estat)

    # Empirical corrections for different membrane systems
    estat_cor = 1.063 * estat + 3.669

    return {
        "E_bind": emin,
        "logP_BLM": estat_cor,
        "logP_plasma": 0.81 * estat_cor - 1.88,
        "logP_BBB": 0.375 * estat - 1.60,
        "logP_Caco2": 0.272 * estat - 2.541,
        "logP_PAMPA": 0.981 * estat + 2.159,
    }
