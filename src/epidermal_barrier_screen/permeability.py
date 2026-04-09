"""Permeability prediction via pypermm (PerMM).

Converts SMILES → 3-D coordinates (RDKit) → pypermm membrane permeability
predictions. Returns the five logP values: plasma, PAMPA, Caco2, BLM, BBB.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the vendored pypermm is importable
_PYPERMM_ROOT = str(Path(__file__).resolve().parents[2] / "third_party" / "pypermm")
if _PYPERMM_ROOT not in sys.path:
    sys.path.insert(0, _PYPERMM_ROOT)

from pypermm.pypermm import run_permm  # noqa: E402

logger = logging.getLogger(__name__)

# The five permeability columns we extract
PERMM_COLUMNS: list[str] = [
    "logP_plasma",
    "logP_PAMPA",
    "logP_Caco2",
    "logP_BLM",
    "logP_BBB",
]


def compute_permeability(smiles: str, ph: float = 7.4) -> dict[str, float | None]:
    """Compute membrane permeability for a SMILES string.

    Parameters
    ----------
    smiles:
        Canonical SMILES of the molecule.
    ph:
        Solution pH passed to pypermm for ionization calculations.

    Returns
    -------
    Dict with keys ``logP_plasma``, ``logP_PAMPA``, ``logP_Caco2``,
    ``logP_BLM``, ``logP_BBB``.  All values are ``None`` when the
    calculation fails (e.g. bad SMILES, embedding failure).
    """
    empty: dict[str, float | None] = {k: None for k in PERMM_COLUMNS}

    if not smiles:
        return empty

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return empty

        # Add explicit hydrogens (required for 3-D embedding)
        mol_h = Chem.AddHs(mol)

        # Generate 3-D coordinates
        result_code = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
        if result_code != 0:
            # Retry with random coordinates
            result_code = AllChem.EmbedMolecule(
                mol_h, AllChem.ETKDGv3(), randomSeed=42
            )
            if result_code != 0:
                logger.warning("3-D embedding failed for %s", smiles)
                return empty

        # Optimise geometry
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)

        # Extract atomic symbols and coordinates
        conf = mol_h.GetConformer()
        symbols: list[str] = [atom.GetSymbol() for atom in mol_h.GetAtoms()]
        coords: list[list[float]] = [
            list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())
        ]

        # Run pypermm
        permm_result: dict[str, Any] = run_permm(
            atomic_symbols=symbols,
            coordinates=coords,
            ph=ph,
        )

        return {
            k: round(float(permm_result[k]), 2) if permm_result.get(k) is not None else None
            for k in PERMM_COLUMNS
        }

    except Exception:
        logger.exception("pypermm failed for %s", smiles)
        return empty
