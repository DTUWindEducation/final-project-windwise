#%% Data
import src
from pathlib import Path
import pandas as pd

# Directory of the current file
directory = Path(__file__).resolve().parent

# Loading wind data for all 4 locations for all years available
wind_data = src.get_data(directory)

#%% Creating dictionay where processed data will be stored

wind_data_ts = {}

wd_ts = src.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)

for label in wind_data.keys():

    wind_data_ts[label] = pd.DataFrame()

    # Adding columns for wind speed and wind direction

    wind_data_ts[label] = wd_ts.compute_ws_time_series(label)






# #%% Creating dictionay where processed data will be stored
# wind_data_processed = {}

# for label in wind_data.keys():

#     wind_data_processed[label] = pd.DataFrame()

#     # Adding columns for wind speed and wind direction

#     # Deberíamos u y v como argumento opcional?
#     wind_data_processed[label] = src.compute_ws_time_series(wind_data[label], u='u10', v='v10', height=10)
#     wind_data_processed[label] = src.compute_wdir_time_series(wind_data_processed[label], u='u10', v='v10', height=10)
#     wind_data_processed[label] = src.compute_ws_time_series(wind_data_processed[label], u='u100', v='v100', height=100)
#     wind_data_processed[label] = src.compute_wdir_time_series(wind_data_processed[label], u='u100', v='v100', height=100)


#%% Interpolation wind speed and direction for location within square
interpolator = src.wind_interpolation(wind_data_processed)

test1 = interpolator.speed_interpolator(x=8, y=55.625)


test2 = interpolator.direction_interpolator(x=8, y=55.625)

#%%  Wind speed at given height using power law

#wind_data_target_U_at_z = src.compute_wind_speed_power_law(wind_data_processed[(8.0, 55.5)], U1 = "wind_speed_10m", U2="wind_speed_100m", z1=10, z2=100, height=50)


# %%
 