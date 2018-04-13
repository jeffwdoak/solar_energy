"""
Load data, 
clean data,
transform data,
run gradient boosting on each sensor's data

"""

import numpy as np
import pandas as pd
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
#from sklearn.model_selection import TimeSeriesSplit, cross_val_score

import netCDF4

from solar_energy import *

## Load Data

# load the simulation coords and elevations
print('loading elevation data')
elevation = netCDF4.Dataset('../data/gefs_elevations.nc', 'r')
elevation_pd = netcdf4_to_dataframe(elevation)
# shift simulation longitudes to same scale as mesonet longitudes
elevation_pd['longitude'] = elevation_pd['longitude'] - 360.

# load the mesonet coords
print('loading mesonet coords')
station_info = pd.read_csv('../data/station_info.csv')
mesonet_lats = np.array(station_info['nlat']) - elevation_pd['latitude'].min()
mesonet_lons = np.array(station_info['elon']) - elevation_pd['longitude'].min()

# load mesonet training data (response)
print('loading training data - response')
training_y = pd.read_csv('../data/train.csv')
training_y.set_index('Date', inplace=True)
training_y.index = training_y.index.map(todatetime)

# load simulation training data (features)
print('loading training data - features')
training_location = '../data/train/'
training_sim_filenames = os.listdir(training_location)
training_sim_names = ['_'.join(f.split('_')[0:2]) for f in training_sim_filenames]
training_sim_data = {n:interpolate_simulation_data(mesonet_lats, 
                                                   mesonet_lons, 
                                                   f, 
                                                   training_location, 
                                                   forecast_aggregator=np.sum,
                                                  ) for n, f in 
                     zip(training_sim_names, training_sim_filenames)}
training_sim_dfs = {k:pd.DataFrame(v, index=training_y.index, columns=training_y.columns) 
                    for k, v in training_sim_data.items()}

# load simulation testing data
print('loading testing data - features')
testing_location = '../data/test/'
testing_sim_filenames = os.listdir(testing_location)
testing_sim_names = ['_'.join(f.split('_')[0:2]) for f in testing_sim_filenames]
testing_sim_data = {n:interpolate_simulation_data(mesonet_lats, mesonet_lons, f, testing_location) for n, f in 
                     zip(testing_sim_names, testing_sim_filenames)}
testing_sim_dfs = {k:pd.DataFrame(v, columns=training_y.columns) 
                    for k, v in testing_sim_data.items()}
sample_submission = pd.read_csv('../data/sampleSubmission.csv', index_col=0)

## Data Cleaning

# combine simulations and mesonet data into one multilevel dataframe
print('cleaning data')
list(training_sim_dfs.values())[0].columns.name = 'Name'
training_df = pd.concat([v.T.stack() for v in training_sim_dfs.values()], axis=1)
training_df = pd.concat([training_df, training_y.T.stack()], axis=1)
names = list(training_sim_dfs.keys())
names.append('mesonets')
training_df.columns = names
cleaned_training_df = training_df.loc[training_df['mesonets'].diff() != 0,]

# combine simulations on testing set in the same way we did for the training set
testing_df = pd.concat([v.T.stack() for v in testing_sim_dfs.values()], axis=1)
names = list(testing_sim_dfs.keys())
testing_df.columns = names


## Run Gradient Boosting on all sensor's data

# CV-based parameter tuning results:
# num_estimators = 150
# max_depth = 3
# learning_rate = 0.1
# min_samples_leaf = 20
model = GradientBoostingRegressor(loss='ls',
                              	  learning_rate=0.1,
                                  n_estimators=150,
                                  max_depth=3,
                                  min_samples_leaf=20,
                                  criterion='mae',
                                 )

# Loop over each sensor, create a dataframe, and fit a GBRegressor to it.
sensors = cleaned_training_df.index.levels[0]
simulations = list(set(cleaned_training_df.columns) - {'mesonets'})
prediction_df = pd.DataFrame(index=sample_submission.index)
for sensor in sensors:
    print('starting work on %s' % (sensor))
    # create dataframe for testing/training features and response
    training_x = cleaned_training_df.loc[sensor, simulations]
    training_y = cleaned_training_df.loc[sensor, 'mesonets']
    testing_x = testing_df.loc[sensor]
    print('shape of testing_x', np.shape(testing_x))
    print('shape of sample index', np.shape(sample_submission.index))
    # Fit Gradientboosted regressor
    model.fit(training_x, training_y)
    print(model.feature_importances_)
    # Make predictions
    sensor_predictions = model.predict(testing_x)
    print('shape of sensor_predictions', sensor_predictions)
    prediction_df[sensor] = sensor_predictions
# write predictions to file
prediction_df.to_csv('../results/gradient_boosting_cleaned_all_estimators-150_depth-3_rate-0-1_min-samples-20.csv')
    


