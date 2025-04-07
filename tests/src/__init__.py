
import xarray as xr
from pathlib import Path
import pandas as pd
import numpy as np

def load_netcdf_to_dataframe(file_path):
    """
    Load a NetCDF file and convert it to a Pandas DataFrame.

    Parameters:
    file_path (str): Path to the NetCDF file.

    Returns:
    pd.DataFrame: DataFrame containing the data from the NetCDF file.
    """
    # Open the .nc file using xarray
    nc_data = xr.open_dataset(file_path)

    # Convert the dataset to a Pandas DataFrame
    df = nc_data.to_dataframe()
    
    return df

def collect_nc_files(folder_path):
    """
    Collect all NetCDF (.nc) files from a given folder and its subfolders.

    Parameters:
    folder_path (str): Path to the folder where the search for .nc files will be performed.

    Returns:
    list: A list of Path objects representing the .nc files found in the folder and its subfolders.
    """
    # Create a Path object for the folder
    folder = Path(folder_path)

    # Use the rglob method to recursively find all .nc files
    nc_files = list(folder.rglob('*.nc'))

    # Return the list of Path objects
    return nc_files


def get_data(directory):
    """
    Load and process all NetCDF (.nc) files from the 'inputs' folder located in the parent directory.

    Parameters:
    directory (Path): A Path object representing the current directory.

    Returns:
    dict: A dictionary where the keys are the file names (without the .nc extension) 
          and the values are Pandas DataFrames containing the data from the corresponding NetCDF files.
    """
    # Navigate to the parent directory and then to the 'inputs' folder
    inputs_path = directory.parent / 'inputs'

    # Collect all .nc files from the 'inputs' folder and its subfolders
    l_nc_files = collect_nc_files(inputs_path)

    # Initialize an empty dictionary to store the data
    wind_data = {}
    df = load_netcdf_to_dataframe(l_nc_files[0])
    df = df.reset_index()
    locations = df[['latitude', 'longitude']].drop_duplicates()

    # Initialize empty DataFrames for each location using coordinates as keys
    for index, location in locations.iterrows():
        lat, lon = location['latitude'], location['longitude']
        location_key = tuple((lat,lon))
        wind_data[location_key] = pd.DataFrame()

    # Iterate through each .nc file
    for file in l_nc_files:
        # Load the NetCDF file into a Pandas DataFrame
        df = load_netcdf_to_dataframe(file)

        # Convert to DataFrame
        df = df.reset_index()

        for index, location in locations.iterrows():
            lat, lon = location['latitude'], location['longitude']
            location_key = f"Location_{lat}_{lon}"
            location_df = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
            wind_data[(lat,lon)]=pd.concat([wind_data[(lat,lon)],location_df], ignore_index=True)
    
    return wind_data

def compute_ws_time_series(wind_data, u, v, height):
    """
    Compute wind speed time series from the wind components.

    Parameters:
    wind_data (DataFrame): A DataFrame withe the wind data for one location.
                           The DataFrame should contain columns for the u-component and v-component of wind speed.
    u (str): The name of the column representing the u-component of wind speed.
    v (str): The name of the column representing the v-component of wind speed.
    height (int): The height at which the wind speed is measured.

    Returns:
    dict: A DataFrame equivalent to the input one, with a new column wind the wind speed timeseries.
    """
    
    u_data = wind_data[u]
    v_data = wind_data[v]

    # Compute wind speed using the formula: ws = sqrt(u^2 + v^2)
    ws_data = (u_data**2 + v_data**2)**0.5

    wind_data[f'wind_speed_{height}m'] = ws_data

    return wind_data

def compute_wdir_time_series(wind_data, u, v, height):
    """
    Compute wind direction time series from the wind components.

    Parameters:
    wind_data (DataFrame): A DataFrame withe the wind data for one location.
                           The DataFrame should contain columns for the u-component and v-component of wind speed.
    u (str): The name of the column representing the u-component of wind speed.
    v (str): The name of the column representing the v-component of wind speed.
    height (int): The height at which the wind speed is measured.

    Returns:
    dict: A DataFrame equivalent to the input one, with a new column wind the wind direction timeseries.
    """
    
    u_data = wind_data[u]
    v_data = wind_data[v]

    wdir_data = np.mod(np.degrees(np.arctan2(-u_data, -v_data)),360)

    wind_data[f'wind_direction_{height}m'] = wdir_data

    return wind_data


class wind_interpolation(object): 

    def __init__(self, wind_data):
        """
        Initialize the wind_interpolation class.
        """        
        self.wind_data = wind_data
        self.locations= list(wind_data.keys())

    def speed_interpolator(self, x, y, t):

        Q11 = self.wind_data[self.locations[0]].iloc[t]['wind_speed_10m']
        Q12 = self.wind_data[self.locations[1]].iloc[t]['wind_speed_10m']
        Q21 = self.wind_data[self.locations[2]].iloc[t]['wind_speed_10m']
        Q22 = self.wind_data[self.locations[3]].iloc[t]['wind_speed_10m']

        x1 = self.locations[0][0]
        x2 = self.locations[2][0]
        y1 = self.locations[0][1]
        y2 = self.locations[1][1]

        value = (Q11 * (x2 - x) * (y2 - y) +
            Q21 * (x - x1) * (y2 - y) +
            Q12 * (x2 - x) * (y - y1) +
            Q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
        
        return value

