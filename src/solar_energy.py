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
    f_interpolated : array-like
        1-D array of interpolated z values at the points x, y. len(z) == len(x)
    """
    print(x)
    print(y)
    x_lower_indices = np.floor(x).astype(int)
    x_upper_indices = np.ceil(x).astype(int)
    y_lower_indices = np.floor(y).astype(int)
    y_upper_indices = np.ceil(y).astype(int)
    print (x_lower_indices)
    print (x_upper_indices)
    print (y_lower_indices)
    print (y_upper_indices)

    delta_x = x_upper_indices - x_lower_indices
    delta_y = y_upper_indices - y_lower_indices
    print(delta_x, delta_y)

    f_x_at_y_lower = (x_upper_indices - x)/delta_x*z[x_lower_indices, y_lower_indices]
    f_x_at_y_lower += (x - x_lower_indices)/delta_x*z[x_upper_indices, y_lower_indices]
    
    f_x_at_y_upper = (x_upper_indices - x)/delta_x*z[x_lower_indices, y_upper_indices]
    f_x_at_y_upper += (x - x_lower_indices)/delta_x*z[x_upper_indices, y_upper_indices]
    
    f_interpolated = (y_upper_indices - y)/delta_y*f_x_at_y_lower
    f_interpolated += (y - y_lower_indices)/delta_y*f_x_at_y_upper
    return f_interpolated


    


