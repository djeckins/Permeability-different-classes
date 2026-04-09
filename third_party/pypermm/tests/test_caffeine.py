"""Tests for the pypermm package."""

from pytest import approx

from pypermm.pypermm import run_permm


def test_caffeine() -> None:
    # fmt: off
    symbols = [
        "N", "C", "C", "C", "N", "C", "C", "H", "H", "H",
        "O", "C", "H", "H", "H", "O", "N", "N", "C", "H",
        "C", "H", "H", "H",
    ]

    xyz = [
        [5.04, 1.944, -8.324],
        [6.469, 2.092, -7.915],
        [7.431, 0.865, -8.072],
        [6.916, -0.391, -8.544],
        [5.532, -0.541, -8.901],
        [4.59, 0.523, -8.394],
        [4.045, 3.041, -8.005],
        [4.453, 4.038, -8.264],
        [3.101, 2.907, -8.57],
        [3.795, 3.05, -6.926],
        [6.879, 3.181, -7.503],
        [4.907, -1.659, -9.696],
        [4.397, -1.273, -10.599],
        [5.669, -2.391, -10.028],
        [4.161, -2.209, -9.089],
        [3.47, 0.208, -7.986],
        [8.807, 0.809, -7.799],
        [7.982, -1.285, -8.604],
        [9.015, -0.5, -8.152],
        [10.007, -0.926, -8.079],
        [9.756, 1.835, -7.299],
        [10.776, 1.419, -7.199],
        [9.437, 2.207, -6.309],
        [9.801, 2.693, -7.994],
    ]

    # fmt: on

    result = run_permm(symbols, xyz)

    assert result["asatot"] == approx(366.19, abs=0.01)
    assert result["logP_BLM"] == approx(-6.85, abs=0.01)
    assert result["logP_plasma"] == approx(-7.43, abs=0.01)
    assert result["logP_BBB"] == approx(-5.31, abs=0.01)
    assert result["logP_Caco2"] == approx(-5.23, abs=0.01)
    assert result["logP_PAMPA"] == approx(-7.55, abs=0.01)
    assert result["E_bind"] == approx(-0.297, abs=0.01)
