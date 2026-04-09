"""Application-specific prediction modules.

Each application type has its own module with independent screening logic
(criteria weights, classification thresholds, decision rules) that can be
modified without affecting other application types.

Supported application types:
    - Skin care
    - Oral care
    - Hair care
    - Supplement (Transbuccal)
    - Supplement (Oral)
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from epidermal_barrier_screen.applications.skin_care import predict_skin_care
from epidermal_barrier_screen.applications.oral_care import predict_oral_care
from epidermal_barrier_screen.applications.hair_care import predict_hair_care
from epidermal_barrier_screen.applications.supplement_transbuccal import predict_supplement_transbuccal
from epidermal_barrier_screen.applications.supplement_oral import predict_supplement_oral

# All valid application type labels (used by the UI selector)
APPLICATION_TYPES: list[str] = [
    "Skin care",
    "Oral care",
    "Hair care",
    "Supplement (Transbuccal)",
    "Supplement (Oral)",
]

# Maps each application type label to its prediction function
_DISPATCH: dict[str, Any] = {
    "Skin care":                 predict_skin_care,
    "Oral care":                 predict_oral_care,
    "Hair care":                 predict_hair_care,
    "Supplement (Transbuccal)":  predict_supplement_transbuccal,
    "Supplement (Oral)":         predict_supplement_oral,
}


def predict(application_type: str, records: list[dict[str, Any]], ph: float) -> pd.DataFrame:
    """Dispatch to the correct application-specific prediction function.

    Parameters
    ----------
    application_type:
        One of the labels in :data:`APPLICATION_TYPES`.
    records:
        List of molecule records from :func:`~epidermal_barrier_screen.io.parse_input`.
    ph:
        Target pH for ionization and logD calculations.

    Returns
    -------
    pandas.DataFrame with screening results from the selected application module.

    Raises
    ------
    ValueError
        If *application_type* is not a recognised type.
    """
    fn = _DISPATCH.get(application_type)
    if fn is None:
        raise ValueError(
            f"Unknown application type: {application_type!r}. "
            f"Valid types: {APPLICATION_TYPES}"
        )
    return fn(records, ph=ph)
