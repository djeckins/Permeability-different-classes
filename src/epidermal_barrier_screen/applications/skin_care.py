"""Skin care prediction module — passive skin penetration scoring.

Rule-based scoring system (0–100) for passive skin entry / skin delivery.
Uses 9 criteria: MW, TPSA, HBD, HBA, RotB, FormalCharge, LogD, logP_BLM,
logP_plasma.  PAINS and BRENK remain as separate alert flags (not scored).

All criteria weights, classification thresholds, and decision rules
can be independently modified for this application type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from epidermal_barrier_screen.descriptors import calculate
from epidermal_barrier_screen.ionization import (
    PH_SC,
    _hhb_acid,
    _hhb_base,
    predict_pka,
)
from epidermal_barrier_screen.screen import _check_pains, _check_brenk
from epidermal_barrier_screen.permeability import compute_permeability, PERMM_COLUMNS

# ---------------------------------------------------------------------------
# Scoring configuration  —  Skin care  (passive skin penetration)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CriterionCfg:
    weight: float
    is_core: bool


_CRITERIA: dict[str, _CriterionCfg] = {
    "MW":                _CriterionCfg(weight=12, is_core=True),
    "TPSA":              _CriterionCfg(weight=16, is_core=True),
    "HBD":               _CriterionCfg(weight=14, is_core=True),
    "HBA":               _CriterionCfg(weight=6,  is_core=False),
    "RotB":              _CriterionCfg(weight=5,  is_core=False),
    "FormalCharge":      _CriterionCfg(weight=10, is_core=True),
    "LogD":              _CriterionCfg(weight=20, is_core=True),
    "logP_BLM":          _CriterionCfg(weight=12, is_core=True),
    "logP_plasma":       _CriterionCfg(weight=5,  is_core=False),
}

_MAX_RAW_SCORE: float = sum(c.weight for c in _CRITERIA.values())  # 100


# ---------------------------------------------------------------------------
# Per-criterion classification  →  "optimal" / "acceptable" / "poor"
# ---------------------------------------------------------------------------

def _classify_mw(v: float) -> str:
    """MW (Da): PASS ≤400, BORDERLINE 400–500, FAIL >500."""
    if v <= 400:
        return "optimal"
    if v <= 500:
        return "acceptable"
    return "poor"


def _classify_tpsa(v: float) -> str:
    """TPSA (Å²): PASS ≤90, BORDERLINE 90–140, FAIL >140."""
    if v <= 90:
        return "optimal"
    if v <= 140:
        return "acceptable"
    return "poor"


def _classify_hbd(v: int) -> str:
    """HBD: PASS ≤4, BORDERLINE 5–6, FAIL >6."""
    if v <= 4:
        return "optimal"
    if v <= 6:
        return "acceptable"
    return "poor"


def _classify_hba(v: int) -> str:
    """HBA: PASS ≤8, BORDERLINE 9–10, FAIL >10."""
    if v <= 8:
        return "optimal"
    if v <= 10:
        return "acceptable"
    return "poor"


def _classify_rotb(v: int) -> str:
    """RotB: PASS ≤5, BORDERLINE 6–8, FAIL >8."""
    if v <= 5:
        return "optimal"
    if v <= 8:
        return "acceptable"
    return "poor"


def _classify_formal_charge(v: int) -> str:
    """Formal charge: PASS 0, BORDERLINE ±1, FAIL ≥±2."""
    if v == 0:
        return "optimal"
    if abs(v) == 1:
        return "acceptable"
    return "poor"


def _classify_logd(v: float | None) -> str:
    """LogD: PASS 1.0–3.0, BORDERLINE 0.0–1.0 or 3.0–4.0, FAIL <0 or >4."""
    if v is None:
        return "poor"
    if 1.0 <= v <= 3.0:
        return "optimal"
    if (0.0 <= v < 1.0) or (3.0 < v <= 4.0):
        return "acceptable"
    return "poor"


def _classify_logp_blm(v: float | None) -> str:
    """logP_BLM: PASS ≥-5.5, BORDERLINE -10.5 to -5.5, FAIL <-10.5."""
    if v is None:
        return "poor"
    if v >= -5.5:
        return "optimal"
    if v >= -10.5:
        return "acceptable"
    return "poor"


def _classify_logp_plasma(v: float | None) -> str:
    """logP_plasma: PASS ≥-6.5, BORDERLINE -11.5 to -6.5, FAIL <-11.5."""
    if v is None:
        return "poor"
    if v >= -6.5:
        return "optimal"
    if v >= -11.5:
        return "acceptable"
    return "poor"


def _compute_unionized_pct(logd: float | None, logp: float | None) -> float | None:
    if logd is None or logp is None:
        return None
    try:
        ld = float(logd)
        lp = float(logp)
    except (TypeError, ValueError):
        return None
    if math.isnan(ld) or math.isnan(lp):
        return None
    return round(10.0 ** (ld - lp) * 100.0, 2)


# ---------------------------------------------------------------------------
# Legacy per-criterion status functions (backward-compat colour-coding)
# Thresholds mirror the new classification; terminology kept as
# "suboptimal" so existing CSS keys in app.py continue to work.
# ---------------------------------------------------------------------------

def _mw_status(v: float) -> str:
    if v <= 400:   return "optimal"
    if v <= 500:   return "suboptimal"
    return "poor"


def _logd_status(v: float | None) -> str:
    if v is None:  return "poor"
    if 1.0 <= v <= 3.0:  return "optimal"
    if (0.0 <= v < 1.0) or (3.0 < v <= 4.0):  return "suboptimal"
    return "poor"


def _tpsa_status(v: float) -> str:
    if v <= 90:    return "optimal"
    if v <= 140:   return "suboptimal"
    return "poor"


def _hbd_status(v: int) -> str:
    if v <= 4:     return "optimal"
    if v <= 6:     return "suboptimal"
    return "poor"


def _hba_status(v: int) -> str:
    if v <= 8:     return "optimal"
    if v <= 10:    return "suboptimal"
    return "poor"


def _rotb_status(v: int) -> str:
    if v <= 5:     return "optimal"
    if v <= 8:     return "suboptimal"
    return "poor"


def _hac_status(v: int) -> str:
    """Informational only — not used in weighted scoring."""
    if v < 30:     return "optimal"
    if v <= 50:    return "suboptimal"
    return "poor"


def _charge_status(v: int) -> str:
    if v == 0:         return "optimal"
    if abs(v) == 1:    return "suboptimal"
    return "poor"


def _ionization_status(pct_unionized: float | None) -> str:
    if pct_unionized is None:  return "poor"
    if pct_unionized >= 40.0:  return "optimal"
    if pct_unionized >= 10.0:  return "suboptimal"
    return "poor"


def _logp_blm_status(v: float | None) -> str:
    if v is None:  return "poor"
    if v >= -5.5:  return "optimal"
    if v >= -10.5: return "suboptimal"
    return "poor"


def _logp_plasma_status(v: float | None) -> str:
    if v is None:  return "poor"
    if v >= -6.5:  return "optimal"
    if v >= -11.5: return "suboptimal"
    return "poor"


# ---------------------------------------------------------------------------
# Weighted scoring helpers
# ---------------------------------------------------------------------------

def _criterion_score(cls: str, weight: float) -> float:
    """Convert a classification to its numeric score contribution."""
    if cls == "optimal":    return weight
    if cls == "acceptable": return weight * 0.5
    return 0.0


def _compute_weighted_score(classes: dict[str, str]) -> float:
    raw = sum(_criterion_score(classes[k], _CRITERIA[k].weight) for k in _CRITERIA)
    return round(raw, 1)


def _count_core_poor(classes: dict[str, str]) -> int:
    return sum(
        1 for k, cfg in _CRITERIA.items()
        if cfg.is_core and classes[k] == "poor"
    )


def _final_decision(weighted_score: float, core_poor: int) -> str:
    """
    PASS        WeightedScore >= 75  AND  CorePoorCount == 0
    BORDERLINE  55 <= score < 75  OR  (score >= 75 and CorePoorCount == 1)
    FAIL        score < 55  OR  CorePoorCount >= 2
    """
    if core_poor >= 2 or weighted_score < 55:
        return "FAIL"
    if weighted_score >= 75 and core_poor == 0:
        return "PASS"
    return "BORDERLINE"


# ---------------------------------------------------------------------------
# Main prediction function  —  Skin care
# ---------------------------------------------------------------------------

def predict_skin_care(records: list[dict[str, Any]], ph: float = PH_SC) -> pd.DataFrame:
    """Screen molecules for passive skin penetration.

    Parameters
    ----------
    records:
        List of molecule records from parse_input().
    ph:
        Target pH for ionization and logD calculations.

    Returns
    -------
    pandas.DataFrame with screening results.
    """
    rows = []
    for rec in records:
        row: dict[str, Any] = {
            "name":             rec.get("name"),
            "input_smiles":     rec.get("input_smiles"),
            "canonical_smiles": rec.get("canonical_smiles"),
            "parse_status":     rec.get("parse_status", "invalid"),
        }

        if rec.get("parse_status") != "ok" or rec.get("mol") is None:
            row["FinalDecision"] = "invalid_input"
            row["final_result"]  = "invalid_input"
            rows.append(row)
            continue

        mol  = rec["mol"]
        desc = calculate(mol)
        row.update(desc)

        # ── PAINS and Brenk structural alerts (unchanged) ────────────────────
        pains_hit = _check_pains(mol)
        brenk_hit = _check_brenk(mol)
        row["PAINS"]            = f"{pains_hit} \U0001f6a9" if pains_hit else ""
        row["Toxicity (BRENK)"] = f"{brenk_hit} \U0001f6a9" if brenk_hit else ""

        # ── pKa / ionization ─────────────────────────────────────────────────
        smiles    = rec.get("canonical_smiles") or ""
        input_pka = rec.get("input_pka")

        if input_pka is not None:
            _, ion_type = predict_pka(smiles, name=rec.get("name", ""), ph=ph)
            if ion_type == "non_ionizable":
                ion_type = "acid"
            row["ionization_class"] = ion_type
            pka_val: float = input_pka
            if ion_type == "base":
                f_neutral, charge = _hhb_base(pka_val, ph)
            else:
                f_neutral, charge = _hhb_acid(pka_val, ph)
            row["predicted_pka"]       = round(pka_val, 2)
            row["fraction_unionized"]  = round(f_neutral * 100.0, 2)
            row["fraction_ionized"]    = round((1.0 - f_neutral) * 100.0, 2)
            row["expected_net_charge"] = round(charge, 4)
            row["logd"]        = round(desc["clogp"] + math.log10(max(f_neutral, 1e-10)), 4)
            row["logd_method"] = "pKa-corrected (input)"

        else:
            pka_val, ion_type = predict_pka(smiles, name=rec.get("name", ""), ph=ph)
            row["ionization_class"] = ion_type

            if ion_type == "non_ionizable" or pka_val is None:
                row["predicted_pka"]       = None
                row["fraction_unionized"]  = 100.0
                row["fraction_ionized"]    = 0.0
                row["expected_net_charge"] = 0.0
                row["logd"]        = desc["clogp"]
                row["logd_method"] = "neutral (= cLogP)"
            else:
                if ion_type == "base":
                    f_neutral, charge = _hhb_base(pka_val, ph)
                else:
                    f_neutral, charge = _hhb_acid(pka_val, ph)
                row["predicted_pka"]       = round(pka_val, 2)
                row["fraction_unionized"]  = round(f_neutral * 100.0, 2)
                row["fraction_ionized"]    = round((1.0 - f_neutral) * 100.0, 2)
                row["expected_net_charge"] = round(charge, 4)
                row["logd"]        = round(desc["clogp"] + math.log10(max(f_neutral, 1e-10)), 4)
                row["logd_method"] = "pKa-corrected (pKaLearn)"

        # ── LogD override from SDF input ─────────────────────────────────────
        input_logd = rec.get("input_logd_7_4")
        if input_logd is not None:
            row["logd"]        = input_logd
            row["logd_method"] = "input (experimental)"

        # ── unionized % ──────────────────────────────────────────────────────
        row["unionized"] = _compute_unionized_pct(row.get("logd"), desc.get("clogp"))

        # ── Membrane permeability (pypermm) ──────────────────────────────────
        permm = compute_permeability(smiles, ph=ph)
        row.update(permm)

        # ── Legacy status columns (backward-compat colour-coding) ────────────
        row["mw_status"]            = _mw_status(desc["mw"])
        row["logd_status"]          = _logd_status(row["logd"])
        row["tpsa_status"]          = _tpsa_status(desc["tpsa"])
        row["hbd_status"]           = _hbd_status(desc["hbd"])
        row["hba_status"]           = _hba_status(desc["hba"])
        row["rotb_status"]          = _rotb_status(desc["rotb"])
        row["hac_status"]           = _hac_status(desc["hac"])
        row["formal_charge_status"] = _charge_status(desc["formal_charge"])
        row["ionization_status"]    = _ionization_status(row["unionized"])
        row["logP_BLM_status"]      = _logp_blm_status(row.get("logP_BLM"))
        row["logP_plasma_status"]   = _logp_plasma_status(row.get("logP_plasma"))

        # ── Classification columns (optimal / acceptable / poor) ─────────────
        classes: dict[str, str] = {
            "MW":                _classify_mw(desc["mw"]),
            "TPSA":              _classify_tpsa(desc["tpsa"]),
            "HBD":               _classify_hbd(desc["hbd"]),
            "HBA":               _classify_hba(desc["hba"]),
            "RotB":              _classify_rotb(desc["rotb"]),
            "FormalCharge":      _classify_formal_charge(desc["formal_charge"]),
            "LogD":              _classify_logd(row["logd"]),
            "logP_BLM":          _classify_logp_blm(row.get("logP_BLM")),
            "logP_plasma":       _classify_logp_plasma(row.get("logP_plasma")),
        }

        row["MW_class"]                = classes["MW"]
        row["TPSA_class"]              = classes["TPSA"]
        row["HBD_class"]               = classes["HBD"]
        row["HBA_class"]               = classes["HBA"]
        row["RotB_class"]              = classes["RotB"]
        row["FormalCharge_class"]       = classes["FormalCharge"]
        row["LogD_class"]              = classes["LogD"]
        row["logP_BLM_class"]          = classes["logP_BLM"]
        row["logP_plasma_class"]       = classes["logP_plasma"]

        # ── Per-criterion score contributions ────────────────────────────────
        row["MW_score"]                = _criterion_score(classes["MW"],           _CRITERIA["MW"].weight)
        row["TPSA_score"]              = _criterion_score(classes["TPSA"],         _CRITERIA["TPSA"].weight)
        row["HBD_score"]               = _criterion_score(classes["HBD"],          _CRITERIA["HBD"].weight)
        row["HBA_score"]               = _criterion_score(classes["HBA"],          _CRITERIA["HBA"].weight)
        row["RotB_score"]              = _criterion_score(classes["RotB"],         _CRITERIA["RotB"].weight)
        row["FormalCharge_score"]       = _criterion_score(classes["FormalCharge"], _CRITERIA["FormalCharge"].weight)
        row["LogD_score"]              = _criterion_score(classes["LogD"],         _CRITERIA["LogD"].weight)
        row["logP_BLM_score"]          = _criterion_score(classes["logP_BLM"],     _CRITERIA["logP_BLM"].weight)
        row["logP_plasma_score"]       = _criterion_score(classes["logP_plasma"],  _CRITERIA["logP_plasma"].weight)

        # ── Final weighted score and decision ────────────────────────────────
        weighted_score = _compute_weighted_score(classes)
        core_poor      = _count_core_poor(classes)

        row["WeightedScore"]  = weighted_score
        row["CorePoorCount"]  = core_poor
        row["FinalDecision"]  = _final_decision(weighted_score, core_poor)
        row["final_result"]   = row["FinalDecision"]

        rows.append(row)

    col_order = [
        "name",
        "parse_status",
        # ── Raw descriptors ──────────────────────────────────────────────────
        "mw", "tpsa", "hbd", "hba", "rotb", "hac",
        "predicted_pka", "ionization_class", "clogp", "logd",
        "unionized", "logd_method",
        "fraction_unionized", "fraction_ionized", "expected_net_charge",
        "formal_charge",
        # ── Permeability (pypermm) ───────────────────────────────────────────
        "logP_plasma", "logP_PAMPA", "logP_Caco2", "logP_BLM", "logP_BBB",
        # ── Legacy status columns ────────────────────────────────────────────
        "mw_status", "logd_status", "tpsa_status", "hbd_status",
        "hba_status", "rotb_status", "hac_status", "formal_charge_status",
        "ionization_status", "logP_BLM_status", "logP_plasma_status",
        # ── Classification columns ───────────────────────────────────────────
        "MW_class", "TPSA_class", "HBD_class", "HBA_class", "RotB_class",
        "FormalCharge_class", "LogD_class",
        "logP_BLM_class", "logP_plasma_class",
        # ── Per-criterion score contributions ────────────────────────────────
        "MW_score", "TPSA_score", "HBD_score", "HBA_score", "RotB_score",
        "FormalCharge_score", "LogD_score",
        "logP_BLM_score", "logP_plasma_score",
        # ── Summary ──────────────────────────────────────────────────────────
        "WeightedScore", "CorePoorCount", "FinalDecision", "final_result",
        # ── Structural alerts ────────────────────────────────────────────────
        "PAINS", "Toxicity (BRENK)",
        # ── SMILES ───────────────────────────────────────────────────────────
        "input_smiles", "canonical_smiles",
    ]

    df = pd.DataFrame(rows)
    for col in col_order:
        if col not in df.columns:
            df[col] = None
    if "unionized" in df.columns:
        df["unionized"] = pd.to_numeric(df["unionized"], errors="coerce")
    return df[col_order]
