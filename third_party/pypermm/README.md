# PyPermm

*permeability modeling made easy*

![License](https://img.shields.io/badge/license-MIT-black)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rowansci/pypermm/test.yml?branch=master&logo=github-actions)](https://github.com/rowansci/pypermm/actions)

## About
PyPermm is a Python-based reimplementation of the PerMM library written by Andrei Lomize and Alexey Kovalenko. The original code can be found [here](https://cggit.cc.lehigh.edu/biomembhub/permm_server_code). This code has been relicensed with the permission of the original authors.

If you use PyPermm, please cite [the original PerMM publication](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00225).

## Installation

Simply run `uv pip install pypermm`, `pixi add --pypi pypermm`, or `pip install pypermm`.

## Usage 

PyPermm primarily exposes a single function, `run_permm`, which takes a list of atomic symbols and a list of atomic coordinates (in Å). 

`run_permm` returns a dictionary with a lot of useful computed properties, including:
- `asatot`, the total accessible surface area of the heavy atoms (in Å**2).
- `E_bind`, the minimum insertion energy (in kcal/mol), representing optimal binding to the membrane interface.
- `logP_BLM`, the predicted intrinsic permeability for black lipid membrances
- `logP_plasma`, the predicted intrinsic permeability for plasma membranes
- `logP_BBB`, the predicted intrinsic permeability for the blood–brain barrier
- `logP_Caco2`, the predicted intrinsic permeability for Caco-2 membranes
- `logP_PAMPA`, the predicted intrinsic permeability for PAMPA membranes
- `z`, an array of z-positions (in Å) through the membrane where energies were calculated
- `energies`, the insertion energy (in kcal/mol) at each z-position after optimizing orientation

Here's what running PyPermm looks like:

```python
from pytest import approx
from pypermm.pypermm import run_permm

symbols = ["N", "C", "C", "C", ...]

xyz = [
    [5.04, 1.944, -8.324],
    [6.469, 2.092, -7.915],
    [7.431, 0.865, -8.072],
    [6.916, -0.391, -8.544],
    ...
]
       
result = run_permm(symbols, xyz)

assert result["asatot"] == approx(366.19, abs=0.01)
assert result["logP_BLM"] == approx(-6.85, abs=0.01)
assert result["logP_plasma"] == approx(-7.43, abs=0.01)
assert result["logP_BBB"] == approx(-5.31, abs=0.01)
assert result["logP_Caco2"] == approx(-5.23, abs=0.01)
assert result["logP_PAMPA"] == approx(-7.55, abs=0.01)
assert result["E_bind"] == approx(-0.297, abs=0.01)

```


## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) project template.
