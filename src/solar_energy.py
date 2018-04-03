"""
Functions to aid the analysis of solar energy data from the kaggle competition:
https://www.kaggle.com/c/ams-2014-solar-energy-prediction-contest

"""

import numpy as np
import pandas as pd
import netCDF4
from scipy.interpolate import SmoothBivariateSpline

def todatetime(intdate):
    """
    Convert integer timestamp to datetime object.

    """
    strdate = str(intdate)
    year = int(strdate[0:4])
    month = int(strdate[4:6])
    day = int(strdate[6:8])
    return pd.datetime(year, month, day)

def netcdf4_to_dataframe(netcdf):
    """
    Convert entire netcdf object to dataframe. 
    
    Only works well for small netcdf objects.
    
    """
    data_as_array = np.array([np.array(i).flatten() 
                             for i in netcdf.variables.values()]
                            ).T
    data_as_dataframe = pd.DataFrame(data_as_array, 
                                     columns=netcdf.variables.keys()
                                    )
    return data_as_dataframe

def interpolate_simulation_data(x,
                                y,
                                filename,
                                location='../data/train/',
                                ensemble_aggregator=np.mean,
                                forecast_aggregator=np.mean,
                               ):
    """
    Interpolate simulation data at all the mesonet x, y coordinates.
    
    Parameters
    ----------
    x, y : numpy arrays
        1-D arrays of latitudes and longitudes respectively.
    filename : str
        Name of the file from which to load and interpolate data.
    location : str, optional
        Path to file. Defaults to '../data/train/'
    
    Returns
    -------
    averaged_data : numpy array
        
    
    """
    data_array = load_netcdf4_data(filename, location)
    interpolated_data = bilinear_interpolation(x, y, data_array)
    # Average over ensembles. This could be converted 
    # to an elevation-weighted average.
    averaged_data = ensemble_aggregator(interpolated_data, axis=1)
    # Average over forecast hours.
    # This could be converted to a sum, or integral.
    averaged_data = forecast_aggregator(averaged_data, axis=1)
    return averaged_data

def load_netcdf4_data(filename, location='../data/train/'):
    """
    Load last entry in a netcdf4 file into a numpy array.

    """
    dataset = netCDF4.Dataset(location+filename, 'r')
    data_array = np.array(list(dataset.variables.values())[-1])
    return data_array

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

    z_x_at_y_lower = (x_upper_indices - x)/delta_x*z[..., x_lower_indices, y_lower_indices]
    z_x_at_y_lower += (x - x_lower_indices)/delta_x*z[..., x_upper_indices, y_lower_indices]
    z_x_at_y_lower = np.where(np.isnan(z_x_at_y_lower),
                              z[..., x_lower_indices, y_lower_indices],
                              z_x_at_y_lower,
                             )
    
    z_x_at_y_upper = (x_upper_indices - x)/delta_x*z[..., x_lower_indices, y_upper_indices]
    z_x_at_y_upper += (x - x_lower_indices)/delta_x*z[..., x_upper_indices, y_upper_indices]
    z_x_at_y_upper = np.where(np.isnan(z_x_at_y_upper),
                              z[..., x_upper_indices, y_upper_indices],
                              z_x_at_y_upper,
                             )
    
    z_interpolated = (y_upper_indices - y)/delta_y*z_x_at_y_lower
    z_interpolated += (y - y_lower_indices)/delta_y*z_x_at_y_upper
    z_interpolated = np.where(np.isnan(z_interpolated),
                              z_x_at_y_lower,
                              z_interpolated,
                             )

    return z_interpolated

def spline_interpolation(x, y, z, **kwargs):
    """
    """
    x_grid, y_grid = np.meshgrid(np.arange(np.shape(z)[0]),
                                 np.arange(np.shape(z)[1]),
                                )
    x_grid = x_grid.ravel()
    y_grid = y_grid.ravel()
    z = z.T.ravel()

    z_spline = SmoothBivariateSpline(x_grid, y_grid, z, **kwargs)
    z_interpolated = z_spline(x, y, grid=False)
    return z_interpolated

