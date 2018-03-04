"""
Tests of functions to interpolate across geography.

"""

import numpy as np
from numpy.testing import assert_allclose

import solar_energy

def test_bilinear():
    x = np.arange(10)
    y = np.arange(10)
    z = np.ones((10, 10))
    f_interpolated = solar_energy.bilinear_interpolation(x, y, z)
    assert_allclose(f_interpolated, np.ones((10, 10)))
