"""
Tests of functions to interpolate across geography.

"""

import numpy as np
from numpy.testing import assert_allclose

import solar_energy

def test_bilinear_returns_nodes_at_nodes():
    x = np.arange(10)
    y = np.arange(10)
    z = np.ones((10, 10))
    f_interpolated = solar_energy.bilinear_interpolation(x, y, z)
    assert_allclose(f_interpolated, np.ones(len(x)))

def test_bilinear_returns_normal():
    x_grid = np.arange(3)
    y_grid = np.arange(3)
    z = np.ones((len(x_grid), len(y_grid)))
    z = (z*x_grid).T*y_grid
    x = np.arange(0.5, 2.5)
    y = np.arange(0.5, 2.5)
    f_expected = np.outer(x, y).ravel()
    f_interpolated = solar_energy.bilinear_interpolation(x, y, z)
    assert_allclose(f_interpolated, f_expected)
