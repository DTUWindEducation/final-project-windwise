from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma
import matplotlib.pyplot as plt

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
    Load and process all NetCDF (.nc) files from the 'inputs' folder
    located in the parent directory.

    Parameters:
    directory (Path): A Path object representing the current directory.

    Returns:
    dict: A dictionary where the keys are tuples of latitude and longitude
          coordinates, and the values are DataFrames containing wind data
          for those coordinates.
    """
    # Navigate to the parent directory and then to the 'inputs' folder
    inputs_path = directory.parent / 'inputs'

    # Collect all .nc files from the 'inputs' folder and its subfolders
    l_nc_files = collect_nc_files(inputs_path)

    # Initialize an empty dictionary to store the data
    wind_data = {}

    # Load the first .nc file to get the coordinates
    df = load_netcdf_to_dataframe(l_nc_files[0])
    df = df.reset_index()
    locations = df[['latitude', 'longitude']].drop_duplicates()

    # Initialize empty DataFrames for each location using coordinates as keys
    for location in locations.itertuples():
        lat, lon = location.latitude, location.longitude
        location_key = (lat, lon)
        wind_data[location_key] = pd.DataFrame()

    # Iterate through each .nc file
    for file in l_nc_files:
        # Load the NetCDF file into a Pandas DataFrame
        df = load_netcdf_to_dataframe(file)

        # Convert to DataFrame
        df = df.reset_index()

        # Create a dataframe for each of the locations
        for location in locations.itertuples():
            lat, lon = location.latitude, location.longitude
            location_key = f"Location_{lat}_{lon}"
            location_df = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
            wind_data[(lat, lon)] = pd.concat([wind_data[(lat, lon)], location_df], ignore_index=True)
    
    return wind_data


class time_series(object): 

    def __init__(self, wind_data, u_1, v_1, height_1, u_2, v_2, height_2):
        """
        Initialize the wind_interpolation class.

        Parameters:
        wind_data (dict): Dictionary containing wind data for different locations.
        u_1 (str): Name of the u-component of wind at height 1.
        v_1 (str): Name of the v-component of wind at height 1.
        height_1 (int): Height 1 in meters.
        u_2 (str): Name of the u-component of wind at height 2.
        v_2 (str): Name of the v-component of wind at height 2.
        height_2 (int): Height 2 in meters.
        """        
        self.wind_data = wind_data
        self.height_1 = height_1
        self.height_2 = height_2
        self.u_1 = u_1
        self.u_2 = u_2
        self.v_1 = v_1
        self.v_2 = v_2

    def compute_ws_time_series(self, cords):
        """
        Compute wind speed time series from the wind components at the specified position.

        Parameters:
        cords (tuple): Tuple containing the latitude and longitude coordinates.

        Returns:
        pd.DataFrame: DataFrame containing the wind speed time series for the specified coordinates.
        """
        wind_data_df = self.wind_data[cords]

        # Extract the u and v components of wind at the specified heights
        u_data = wind_data_df[self.u_1]
        v_data = wind_data_df[self.v_1]

        # Compute wind speed using the formula: ws = sqrt(u^2 + v^2)
        ws_data = (u_data**2 + v_data**2)**0.5

        wind_data_df[f'wind_speed_{self.height_1}m'] = ws_data

        # Extract the u and v components of wind at the specified heights
        u_data = wind_data_df[self.u_2]
        v_data = wind_data_df[self.v_2]

        # Compute wind speed using the formula: ws = sqrt(u^2 + v^2)
        ws_data = (u_data**2 + v_data**2)**0.5

        wind_data_df[f'wind_speed_{self.height_2}m'] = ws_data

        return wind_data_df
    
    def compute_wdir_time_series(self, cords):
        """
        Compute wind direction time series from the wind components.

        Parameters:
        cords (tuple): Tuple containing the latitude and longitude coordinates.

        Returns:
        pd.DataFrame: DataFrame containing the wind direction time series for the specified coordinates.   
        """

        wind_data_df = self.wind_data[cords]

        # Extract the u and v components of wind at the specified heights
        u_data = wind_data_df[self.u_1]
        v_data = wind_data_df[self.v_1]

        # Compute wind direction using the formula: wdir = atan2(-u, -v)
        wdir_data = np.mod(np.degrees(np.arctan2(-u_data, -v_data)),360)

        wind_data_df[f'wind_direction_{self.height_1}m'] = wdir_data

        # Extract the u and v components of wind at the specified heights
        u_data = wind_data_df[self.u_2]
        v_data = wind_data_df[self.v_2]

        # Compute wind direction using the formula: wdir = atan2(-u, -v)
        wdir_data = np.mod(np.degrees(np.arctan2(-u_data, -v_data)),360)

        wind_data_df[f'wind_direction_{self.height_2}m'] = wdir_data

        return wind_data_df


class wind_interpolation(object): 

    def __init__(self, wind_data):
        """
        Initialize the wind_interpolation class.

        Parameters:
        wind_data (dict): Dictionary containing wind data for different locations.
        """

        self.wind_data = wind_data
        self.locations= list(wind_data.keys())

    def speed_interpolator(self, x_coord, y_coord):

        """
        Interpolate wind speed at the point (x, y) within the square.

        Parameters:
        x_coord: X-coordinate of the point
        y_coord: Y-coordinate of the point

        Returns:
        pd.DataFrame: DataFrame with interpolated wind speeds over time
        """

        lat = x_coord
        lon = y_coord

        # Extract wind speed data for the four corners of the square
        Q11 = self.wind_data[self.locations[0]]['wind_speed_10m']
        Q12 = self.wind_data[self.locations[1]]['wind_speed_10m']
        Q21 = self.wind_data[self.locations[2]]['wind_speed_10m']
        Q22 = self.wind_data[self.locations[3]]['wind_speed_10m']

        # Create arrays for the x and y coordinates of the corners
        x1 = np.array([self.locations[0][0]] * len(Q11))
        x2 = np.array([self.locations[2][0]] * len(Q12))
        y1 = np.array([self.locations[0][1]] * len(Q21))
        y2 = np.array([self.locations[1][1]] * len(Q22))

        # Create arrays for the x and y coordinates of the point to be interpolated
        x = np.array([lat] * len(Q11))
        y = np.array([lon] * len(Q12))        

        # Perform bilinear interpolation
        value = (Q11 * (x2 - x) * (y2 - y) +
            Q21 * (x - x1) * (y2 - y) +
            Q12 * (x2 - x) * (y - y1) +
            Q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
        
        # Create a DataFrame to store the interpolated values
        location = pd.DataFrame()
        location['time'] = self.wind_data[self.locations[0]]['valid_time']
        location['longitude'] = x
        location['latitude'] = y
        location['wind_speed_10m'] = value

        # Extract wind speed data for the four corners of the square
        Q11 = self.wind_data[self.locations[0]]['wind_speed_100m']
        Q12 = self.wind_data[self.locations[1]]['wind_speed_100m']
        Q21 = self.wind_data[self.locations[2]]['wind_speed_100m']
        Q22 = self.wind_data[self.locations[3]]['wind_speed_100m']

        # Create arrays for the x and y coordinates of the corners
        x1 = np.array([self.locations[0][0]] * len(Q11))
        x2 = np.array([self.locations[2][0]] * len(Q12))
        y1 = np.array([self.locations[0][1]] * len(Q21))
        y2 = np.array([self.locations[1][1]] * len(Q22))

        # Create arrays for the x and y coordinates of the point to be interpolated
        x = np.array([lat] * len(Q11))
        y = np.array([lon] * len(Q12))        

        # Perform bilinear interpolation
        value = (Q11 * (x2 - x) * (y2 - y) +
            Q21 * (x - x1) * (y2 - y) +
            Q12 * (x2 - x) * (y - y1) +
            Q22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
        
        # Create a DataFrame to store the interpolated values
        location['wind_speed_100m'] = value
        
        return location

    def direction_interpolator(self, x_coord, y_coord):
        """
        Interpolate wind direction at the point (x, y) within the square.
        
        Parameters:
        x_coord: X-coordinate of the point
        y_coord: Y-coordinate of the point

        Returns:
        pd.DataFrame: DataFrame with interpolated wind directions over time
        """

        lat = x_coord
        lon = y_coord

        # Extract wind direction data for the four corners of the square
        D11 = self.wind_data[self.locations[0]]['wind_direction_10m']
        D12 = self.wind_data[self.locations[1]]['wind_direction_10m']
        D21 = self.wind_data[self.locations[2]]['wind_direction_10m']
        D22 = self.wind_data[self.locations[3]]['wind_direction_10m']

        # Create arrays for the x and y coordinates of the corners
        x1 = np.array([self.locations[0][0]] * len(D11))
        x2 = np.array([self.locations[2][0]] * len(D12))
        y1 = np.array([self.locations[0][1]] * len(D21))
        y2 = np.array([self.locations[1][1]] * len(D22))

        # Create arrays for the x and y coordinates of the point to be interpolated
        x = np.array([lat] * len(D11))
        y = np.array([lon] * len(D12))

        # Convert directions to Cartesian coordinates
        sin_D11, cos_D11 = np.sin(np.radians(D11)), np.cos(np.radians(D11))
        sin_D21, cos_D21 = np.sin(np.radians(D21)), np.cos(np.radians(D21))
        sin_D12, cos_D12 = np.sin(np.radians(D12)), np.cos(np.radians(D12))
        sin_D22, cos_D22 = np.sin(np.radians(D22)), np.cos(np.radians(D22))

        # Interpolate the Cartesian components
        sin_interp = (
            sin_D11 * (x2 - x) * (y2 - y) +
            sin_D21 * (x - x1) * (y2 - y) +
            sin_D12 * (x2 - x) * (y - y1) +
            sin_D22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

        cos_interp = (
            cos_D11 * (x2 - x) * (y2 - y) +
            cos_D21 * (x - x1) * (y2 - y) +
            cos_D12 * (x2 - x) * (y - y1) +
            cos_D22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

        # Convert back to polar coordinates
        interpolated_direction = np.degrees(np.arctan2(sin_interp, cos_interp))
        interpolated_direction[interpolated_direction < 0] += 360

        # Create a DataFrame to store the interpolated values
        location = pd.DataFrame()
        location['time'] = self.wind_data[self.locations[0]]['valid_time']
        location['longitude'] = x
        location['latitude'] = y
        location['wind_direction_10m'] = interpolated_direction

        # Extract wind direction data for the four corners of the square
        D11 = self.wind_data[self.locations[0]]['wind_direction_100m']
        D12 = self.wind_data[self.locations[1]]['wind_direction_100m']
        D21 = self.wind_data[self.locations[2]]['wind_direction_100m']
        D22 = self.wind_data[self.locations[3]]['wind_direction_100m']

        # Create arrays for the x and y coordinates of the corners
        x1 = np.array([self.locations[0][0]] * len(D11))
        x2 = np.array([self.locations[2][0]] * len(D12))
        y1 = np.array([self.locations[0][1]] * len(D21))
        y2 = np.array([self.locations[1][1]] * len(D22))

        # Create arrays for the x and y coordinates of the point to be interpolated
        x = np.array([lat] * len(D11))
        y = np.array([lon] * len(D12))

        # Convert directions to Cartesian coordinates
        sin_D11, cos_D11 = np.sin(np.radians(D11)), np.cos(np.radians(D11))
        sin_D21, cos_D21 = np.sin(np.radians(D21)), np.cos(np.radians(D21))
        sin_D12, cos_D12 = np.sin(np.radians(D12)), np.cos(np.radians(D12))
        sin_D22, cos_D22 = np.sin(np.radians(D22)), np.cos(np.radians(D22))

        # Interpolate the Cartesian components
        sin_interp = (
            sin_D11 * (x2 - x) * (y2 - y) +
            sin_D21 * (x - x1) * (y2 - y) +
            sin_D12 * (x2 - x) * (y - y1) +
            sin_D22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

        cos_interp = (
            cos_D11 * (x2 - x) * (y2 - y) +
            cos_D21 * (x - x1) * (y2 - y) +
            cos_D12 * (x2 - x) * (y - y1) +
            cos_D22 * (x - x1) * (y - y1)
        ) / ((x2 - x1) * (y2 - y1))

        # Convert back to polar coordinates
        interpolated_direction = np.degrees(np.arctan2(sin_interp, cos_interp))
        interpolated_direction[interpolated_direction < 0] += 360

        location['wind_direction_100m'] = interpolated_direction

        return location

def compute_wind_speed_power_law(wind_data, height, wd_ts_f):

    """

    """

    U1 = wind_data["wind_speed_10m"]
    U2 = wind_data["wind_speed_100m"]
    z = height
    z1 = wd_ts_f.height_1
    z2 = wd_ts_f.height_2
    
    wind_data['alpha'] = (np.log(U2/U1))/(np.log(z2/z1))

    wind_data[f'ws_pl_{height}m'] = U2*((z/z2)**wind_data['alpha'])

    return wind_data


def gamma_func(k, mu_1, mu_2):

    return gamma(1+1/k)**2/gamma(1+2/k) - (mu_1**2)/mu_2


class weibull(object):

    def __init__(self, x, y, height, wind_data_processed, wd_ts_f):
        """
        Initialize the Weibull class.
        """        
        self.x = x
        self.y = y
        self.height = height
        self.wind_data_processed = wind_data_processed
        self.wd_ts_f = wd_ts_f

    def obtain_parameters(self):

        #obtain time series at the guven location
        inter = wind_interpolation(self.wind_data_processed)

        ts = inter.speed_interpolator(self.x, self.y)

        ts_at_height = compute_wind_speed_power_law(ts, self.height, self.wd_ts_f)

        fst_moment = np.mean(ts_at_height[f"ws_pl_{self.height}m"])
        snd_moment = np.mean(ts_at_height[f"ws_pl_{self.height}m"]**2)

        k = fsolve(gamma_func, 2, args=(fst_moment, snd_moment))

        A = fst_moment/gamma(1+1/k)

        return A, k
    

    def get_pdf(self, u_min=0, u_max=25, delta_u=0.1):
        """

        """

        A, k = self.obtain_parameters()
        wind_speed = np.arange(u_min, u_max, delta_u)  # Wind speed range from 0 to 30 m/s

        pdf = (k / A) * (wind_speed / A) ** (k-1) * np.exp(-(wind_speed / A)**k)

        return pdf
    
    
    def plot_pdf(self, u_min=0, u_max=25, delta_u=0.1):

        #obtain time series at the guven location
        inter = wind_interpolation(self.wind_data_processed)

        ts = inter.speed_interpolator(self.x, self.y)

        ts_at_height = compute_wind_speed_power_law(ts, self.height, self.wd_ts_f)

        p = self.get_pdf()
        wind_speed = np.arange(u_min, u_max, delta_u)

        plt.figure(figsize=(10, 6))
        plt.plot(wind_speed, p, label='Weibull PDF', color='blue')
        plt.hist(ts_at_height[f"ws_pl_{self.height}m"], bins=int(u_max-u_min), density=True, alpha=0.5, color='gray', label='Wind Speed Histogram')
        plt.title('Weibull Probability Density Function')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Probability Density')
        plt.grid()
        plt.legend()
        plt.savefig("weibull_pdf.png")

        return


# def obtain_wind_rose(wd, x, y, height, wd_ts_f, n_sector=12):

#     inter = wind_interpolation(wd)

#     ts = inter.direction_interpolator(x, y)

#     ts_at_height = [np.interp(height, [wd_ts_f.height_1, wd_ts_f.height_2], [ts['wind_direction_10m'].iloc[i], ts['wind_direction_100m'].iloc[i]]) for i in range(len(ts))]

#     ts[f"wd_at_{height}m"] = ts_at_height

#     # Divide the wind direction into n_sector bins
#     sector_size = 360 / n_sector
#     bins = np.arange(0, 360 + sector_size, sector_size)
#     labels = np.arange(1, n_sector+1)

#     # Assign each wind direction to a sector
#     ts['sector'] = pd.cut(ts_at_height, bins=bins, labels=labels, right=False)

#     probability = ts_at_height['sector'].value_counts(normalize=True).sort_index()

#     ax = plt.subplot(projection='polar')
#     ax.set_theta_zero_location('N')
#     ax.set_theta_direction(-1)  
#     ax.bar(np.radians(probability.index * sector_size), probability.values, width=np.radians(sector_size), alpha=0.5, color='blue', edgecolor='black')
#     ax.xaxis.set_ticks(np.arange(0, 2*np.pi, np.pi/(6)))

#     return ts_at_height

def obtain_wind_rose(wd, x, y, height, wd_ts_f, n_sector=12):

    inter = wind_interpolation(wd)

    ts = inter.direction_interpolator(x, y)

    # ts_at_height = [np.interp(height, [wd_ts_f.height_1, wd_ts_f.height_2], [ts['wind_direction_10m'].iloc[i], ts['wind_direction_100m'].iloc[i]]) for i in range(len(ts))]
    ts_at_height = (ts[f"wind_direction_{wd_ts_f.height_2}m"] - ts[f"wind_direction_{wd_ts_f.height_1}m"])/(wd_ts_f.height_2 - wd_ts_f.height_1) * (height - wd_ts_f.height_1) + ts["wind_direction_10m"]

    ts[f"wd_at_{height}m"] = ts_at_height

    # Divide the wind direction into n_sector bins
    sectors = {}

    nd = n_sector

    ts_at_height_series = pd.Series(ts_at_height)

    for i in range(1, nd+1):

        sectors[f"Sector_{i}"] = pd.DataFrame(columns=["wd"])
        sectors["prob"] = []

        sectors[f"Sector_{i}"] = ts_at_height_series[(ts_at_height_series > (i-1)*360/nd) & (ts_at_height_series <= i*360/nd)]

    plt.figure(figsize=(10, 6))
    plt.title(f"Wind Rose at {height}m", pad=20)
    sectors["prob"] = [len(sectors[f"Sector_{i}"])/len(ts_at_height) for i in range(1, nd+1)]
    ax = plt.subplot(projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  
    ax.bar(np.radians(np.arange(1, 361, 360/nd)), sectors["prob"], width=np.pi/(nd/2))
    ax.xaxis.set_ticks(np.arange(0, 2*np.pi, np.pi/(6)))

    return ts_at_height

def load_turbines(filepath):

    filepath_turb = filepath.parent / 'inputs'

    # Open the .nc file using xarray
    directory = Path(filepath_turb)

    # Initialize an empty dictionary to store DataFrames
    turbines = {}

    # Iterate through all CSV files in the directory
    for csv_file in directory.glob("*.csv"):
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        # Use the file name (without extension) as the key
        file_name = csv_file.stem
        turbines[file_name] = df

    return turbines


class turbine(object):
    def __init__(self, turbine_data, hub_heights):

        self.turbine_data = turbine_data
        self.hub_heights = hub_heights

        for i, key in enumerate(turbine_data.keys()):

            turbine_ = {}
            turbine_['hub_height'] = self.hub_heights[i]
            turbine_['power_curve'] = self.turbine_data[key]
            
            setattr(self, key, turbine_.copy())

    def compute_AEP(self, turbine, lat, lon, wind_data, year, ts_object):
        """
        Compute the Annual Energy Production (AEP) for a given turbine and location.
        """
        
        inter = wind_interpolation(wind_data)

        ts = inter.speed_interpolator(lat, lon)

        turbine_ = getattr(self, turbine)

        ts = compute_wind_speed_power_law(ts, turbine_['hub_height'], ts_object)

        print(ts.head())

        ts.set_index('time', inplace=True)
        ts_year = ts.loc[f'{year}']

        return ts_year
            

   
        

