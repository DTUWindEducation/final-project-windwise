
import xarray as xr
from pathlib import Path

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

    # Iterate through each .nc file
    for file in l_nc_files:
        # Load the NetCDF file into a Pandas DataFrame
        df = load_netcdf_to_dataframe(file)
        # Extract the variable name from the file name (without the .nc extension)
        variable_name = file.stem
        # Store the DataFrame in the dictionary with the variable name as the key
        wind_data[variable_name] = df
        break

    # Convert to DataFrame
    df = wind_data[variable_name].reset_index()

    # Assuming 'latitude' and 'longitude' are the coordinates in your dataset
    locations = df[['latitude', 'longitude']].drop_duplicates()

    # Dictionary to hold DataFrames for each location
    location_dfs = {}

    for index, location in locations.iterrows():
        lat, lon = location['latitude'], location['longitude']
        location_key = f"Location_{lat}_{lon}"
        location_df = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
        location_dfs[location_key] = location_df
    
    return location_dfs