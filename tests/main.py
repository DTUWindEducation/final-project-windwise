import src
from pathlib import Path
import pandas as pd

# Directory of the current file
directory = Path(__file__).resolve().parent

# Loading wind data for all 4 locations for all years available
wind_data = src.get_data(directory)

# Creating dictionay where processed data will be stored
wind_data_processed = {}

for label in wind_data.keys():

    wind_data_processed[label] = pd.DataFrame()

    # Adding columns for wind speed and wind direction
    wind_data_processed[label] = src.compute_ws_time_series(wind_data[label], u='u10', v='v10', height=10)
    wind_data_processed[label] = src.compute_wdir_time_series(wind_data_processed[label], u='u10', v='v10', height=10)


interpolator = src.wind_interpolation(wind_data_processed)

test1 = interpolator.speed_interpolator(x=8, y=55.625)


test2 = interpolator.direction_interpolator(x=8, y=55.625)


