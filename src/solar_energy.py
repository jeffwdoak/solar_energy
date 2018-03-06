"""
Functions to aid the analysis of solar energy data from the kaggle competition:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest

"""

import numpy as np

def bilinear_interpolation(x, y, z):
    """
    Linearly interpolates between points on a regularly-spaced 2D grid.
    
    Assumes the grid spacing in both x and y is 1 and that z[0, 0] corresponds
    to x=0, y=0. The grid in x and y do not need to be the same length.

    Parameters
    ----------
    x, y : array-like
        1-D arrays of x and y coordinates. len(x) == len(y)
    z : array-like
        2-D array of z values defined on a grid.
    
    Returns
    -------
    z_interpolated : array-like
        1-D array of interpolated z values at the points x, y. len(z) == len(x)
    """
    x_lower_indices = np.floor(x).astype(int)
    x_upper_indices = np.ceil(x).astype(int)
    y_lower_indices = np.floor(y).astype(int)
    y_upper_indices = np.ceil(y).astype(int)

    delta_x = x_upper_indices - x_lower_indices
    delta_y = y_upper_indices - y_lower_indices

    z_x_at_y_lower = (x_upper_indices - x)/delta_x*z[x_lower_indices, y_lower_indices]
    z_x_at_y_lower += (x - x_lower_indices)/delta_x*z[x_upper_indices, y_lower_indices]
    z_x_at_y_lower = np.where(np.isnan(z_x_at_y_lower),
                              z[x_lower_indices, y_lower_indices],
                              z_x_at_y_lower,
                             )
    
    z_x_at_y_upper = (x_upper_indices - x)/delta_x*z[x_lower_indices, y_upper_indices]
    z_x_at_y_upper += (x - x_lower_indices)/delta_x*z[x_upper_indices, y_upper_indices]
    z_x_at_y_upper = np.where(np.isnan(z_x_at_y_upper),
                              z[x_upper_indices, y_upper_indices],
                              z_x_at_y_upper,
                             )
    
    z_interpolated = (y_upper_indices - y)/delta_y*z_x_at_y_lower
    z_interpolated += (y - y_lower_indices)/delta_y*z_x_at_y_upper
    z_interpolated = np.where(np.isnan(z_interpolated),
                              z_x_at_y_lower,
                              z_interpolated,
                             )

    return z_interpolated


    


