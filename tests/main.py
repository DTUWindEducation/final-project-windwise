#%% Data
import src
from pathlib import Path
import pandas as pd

# Directory of the current file
directory = Path(__file__).resolve().parent

# Hub heights
hub_heights = [90, 150] # Write them in the same order as the turbines files are foun in the directoy

# Loading wind data for all 4 locations for all years available
wind_data = src.get_data(directory)

#%% Creating dictionay where processed data will be stored

wind_data_ts = {}

wd_ts = src.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)

for label in wind_data.keys():

    wind_data_ts[label] = pd.DataFrame()

    # Adding columns for wind speed and wind direction

    wind_data_ts[label] = wd_ts.compute_ws_time_series(label)
    wind_data_ts[label] = wd_ts.compute_wdir_time_series(label)


#%% Obtaining weibull parameters for any location and height 

weibull_obj = src.weibull(55.65, 8, 60, wind_data_ts, wd_ts)

A, k = weibull_obj.obtain_parameters()

u_weibull = weibull_obj.plot_pdf()

wind_rose = src.obtain_wind_rose(wind_data_ts, 55.65, 8, 60, wd_ts, 12)

# %% Turbines

turbines = src.load_turbines(directory)

turbine_obj  = src.turbine(turbines, hub_heights)




# %%
