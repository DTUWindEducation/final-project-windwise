#%% Importing libraries
import Wind_Re as wr
from pathlib import Path
import pandas as pd

#%% DATA INPUTS

# Directory of the current file
directory = Path(__file__).resolve().parent

# Hub heights
hub_heights = [90, 150] # Write them in the same order as the turbines files are foun in the directoy

# Data for testing the functions, selected by user
year = 1998 # Options: 1997-2008
turbine_name = "NREL_Reference_15MW_240" # Options: NREL_Reference_5MW_126, NREL_Reference_15MW_240
latitude = 55.65 # Options: 55.5-55.75
longitude = 7.98 # Options: 7.75-8
height = 72 # Options: any positive number

#%% 1. Load and parse multiple provided netCDF files

# Obtain dictionary with u,v time series data for the four provided locations
wind_data = wr.get_data(directory)


#%% 2. Compute wind speed and wind direction time series at 10 m and 100 m heights for the four provided locations

# Dictionary to save the time series dataframes
wind_data_ts = {}

# Create the object, specifying heihgts and u,v components columns 
wind_data_time_series_object = wr.time_series(wind_data=wind_data, u_1='u10', v_1='v10', height_1=10, u_2='u100', v_2='v100', height_2=100)

# Iterate over the keys of the wind_data dictionary and compute the wind speed and wind direction time series for each location
for label in wind_data.keys():

    # Create a new dataframe for each location
    wind_data_ts[label] = pd.DataFrame()

    # Adding columns for wind speed and wind direction
    wind_data_ts[label] = wind_data_time_series_object.compute_ws_time_series(label)
    wind_data_ts[label] = wind_data_time_series_object.compute_wdir_time_series(label)


#%% 3. Compute wind speed and wind direction time series at 10 m and 100 m heights for a given location inside the box bounded by the four locations,
#  such as the Horns Rev 1 site, using interpolation.

# Create the object for wind interpolation, input is the wind speed and direction time series
wind_data_interpolation_object = wr.wind_interpolation(wind_data_ts)

# Using the coordinates stated above, compute the wind speed time series
wind_speed_at_location = wind_data_interpolation_object.speed_interpolator(latitude, longitude)

# Using the coordinates stated above, compute the wind direction time series
wind_direction_at_location = wind_data_interpolation_object.direction_interpolator(latitude, longitude)

#%% 4. Compute wind speed time series at height z for the four provided locations using power law profile.

# Iterate over all locations
for label in wind_data_ts.keys():

    # Add a column in each DataFrame for the wind speed at the specified height
    wind_data_ts[label] = wr.compute_wind_speed_power_law(wind_data=wind_data_ts[label], height=height, wd_ts_f=wind_data_time_series_object)


#%% 5. Fit Weibull distribution for wind speed at a given location (inside the box) and a given height.

# Create the object for the weibull distribution, inpute is the coordinates, height, wind time series and object
weibull_object = wr.weibull(x=latitude, y=longitude, height=height, wind_data_processed=wind_data_ts, wd_ts_f=wind_data_time_series_object)

# Optional: Obtain the parameters of the Weibull distribution
A, k = weibull_object.obtain_parameters()

print(f"Weibull parameters for {latitude}, {longitude}, {height} m: A = {A}, k = {k}")

# Obtain weibull curve values
u_weibull = weibull_object.get_pdf()


#%% 6. Plot wind speed distribution (histogram vs. fitted Weibull distribution) at a given location (inside the box) and a given height.

# Plot the weibull PDF and histogram of the wind speed time series, saves figure in output directory
weibull_object.plot_pdf()

#%% 7. Plot wind rose diagram that shows the frequencies of different wind direction at a given location (inside the box) and a given height.

# Plot the wind rose diagram, saves figure in output directory. Default number of bins is 12
wr.obtain_wind_rose(wd = wind_data_ts, x=latitude, y=longitude, height=height, wd_ts_f=wind_data_time_series_object, n_sector=12)


#%% 8. Compute AEP of a specifed wind turbine (NREL 5 MW or NREL 15 MW) at a given location inside the box for a given year in the period
#  we have provided the wind data.

# Load turbine files from input directory
turbines = wr.load_turbines(filepath=directory)

# Create the object for the turbines, input is the turbines dictionary containing the power curve and hub heights
turbines_object = wr.turbine(turbine_data=turbines, hub_heights=hub_heights)

# Compute the AEP for the specified turbine, location and year
AEP = turbines_object.compute_AEP(turbine_name=turbine_name, lat=latitude, lon=longitude, wind_data=wind_data_ts, year=year, ts_object=wind_data_time_series_object) 

print(f"AEP for {turbine_name} at {latitude}, {longitude}, {height} m in {year}: {AEP/1e3} MWh")

# %% Extra function: plot power curve

# Plot the power curve of the specified turbine, saves figure in output directory
turbines_object.plot_power_curve(turbine_name=turbine_name)  

# %% Extra funtion: plot wind resource time series of a given year, location and height

wr.plot_wind_speed_year(wind_data=wind_data, wd_ts_f=wind_data_time_series_object, year=year, lat=latitude, lon=longitude, height=height)
# %%
