"""
Tests of functions to interpolate across geography.

"""

import numpy as np
from numpy.testing import assert_allclose

import solar_energy

def test_bilinear_returns_nodes_at_nodes():
    x = np.arange(10.)
    y = np.arange(10.)
    z = np.ones((10, 10))
    z_interpolated = solar_energy.bilinear_interpolation(x, y, z)
    assert_allclose(z_interpolated, np.ones(len(x)))

def test_bilinear_returns_normal():
    x_grid = np.arange(3)
    y_grid = np.arange(3)
    z = np.ones((len(x_grid), len(y_grid)))
    z = (z*x_grid).T*y_grid
    x = np.array([0.5, 0.5, 1.5, 1.5])
    y = np.array([0.5, 1.5, 0.5, 1.5])
    z_expected = np.array([0.25, 0.75, 0.75, 2.25])
    z_interpolated = solar_energy.bilinear_interpolation(x, y, z)
    assert_allclose(z_interpolated, z_expected)

def test_bilinear_x_at_nodes():
    x_grid = np.arange(3)
    y_grid = np.arange(3)
    z = np.ones((len(x_grid), len(y_grid)))
    z = (z*x_grid).T*y_grid
    x = np.array([0., 1., 1., 2.])
    y = np.array([0.5, 1.5, 0.5, 1.5])
    z_expected = np.array([0., 1.5, 0.5, 3.])
    z_interpolated = solar_energy.bilinear_interpolation(x, y, z)
    assert_allclose(z_interpolated, z_expected)

