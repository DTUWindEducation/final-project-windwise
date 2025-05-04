# pylint: disable=C0103
# pylint: disable=R0913
# pylint: disable=R0914
# pylint: disable=R0917
# pylint: disable=C0301

from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma
import matplotlib.pyplot as plt

# Outputs direcoty
current_directory = Path(__file__).resolve().parent
outputs_dir= current_directory.parent.parent / 'outputs'


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
            wind_data[(lat, lon)] = pd.concat([wind_data[(lat, lon)], 
                                               location_df], ignore_index=True)
            
    return wind_data


class time_series(object): 

    def __init__(self, wind_data, u_1, v_1, height_1, u_2, v_2, height_2):
        """
        Initialize the time_series class.

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
    Compute the wind speed at a given height using power law method.

    Parameters:
    wind_data (pd.DataFrame): DataFrame containing wind speed data at two heights.
    height (int): The height (in meters) at which to compute the wind speed.
    wd_ts_f (object): An object containing attributes `height_1` and `height_2`, 
                      representing the heights (in meters) of the wind speed measurements.

    Returns:
    pd.DataFrame: The input DataFrame with two additional columns:
                  - 'alpha': The power law exponent.
                  - 'ws_pl_{height}m': The computed wind speed at the specified height.
    """

    # Extract wind speed at bith heights
    U1 = wind_data["wind_speed_10m"]
    U2 = wind_data["wind_speed_100m"]

    # Define the target height and the reference heights
    z = height # Target heigh
    z1 = wd_ts_f.height_1
    z2 = wd_ts_f.height_2
    
    # Compute the power law exponent (alpha)
    wind_data['alpha'] = (np.log(U2/U1))/(np.log(z2/z1))

    # Compute the wind speed at the specified height using the power law formula
    wind_data[f'ws_pl_{height}m'] = U2*((z/z2)**wind_data['alpha'])

    return wind_data

def plot_wind_speed_year(wind_data, wd_ts_f, year, lat, lon, height):

    """
    Plot the weekly mean wind speed at a given height for a specific year and location.

    Parameters:
    wind_data (dict): Dictionary containing wind data for different locations.
    wd_ts_f (object): An object containing attributes for wind data time series.
    year (int): The year for which the wind speed data is to be plotted.
    lat (float): Latitude of the location at which the wind speed is to be plotted..
    lon (float): Longitude of the location at which the wind speed is to be plotted..
    height (int): The height (in meters) at which the wind speed is to be plotted.

    Returns:
    None: The function saves the plot as a PNG file in the outputs directory.
    """

    # Initialize the wind interpolation object with the wind data
    inter = wind_interpolation(wind_data)

    # Interpolate wind speed at the specified latitude and longitude
    ts = inter.speed_interpolator(lat, lon)

    # Compute wind speed at the specified height using the power law
    ts_at_height = compute_wind_speed_power_law(ts, height, wd_ts_f)

    # Filter the data for the specified year
    ts_at_height['time'] = pd.to_datetime(ts_at_height['time'])
    ts_at_height.set_index('time', inplace=True)
    ts_at_height = ts_at_height.loc[f'{year}']

    # Compute the weekly mean wind speed at the specified height
    weekly_mean = ts_at_height[f'ws_pl_{height}m'].resample('W').mean()

    # Plot the weekly mean wind speed
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_mean.index, weekly_mean)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.xlabel('Month')
    plt.ylabel('Wind Speed [m/s]')
    plt.title(f'Wind Speed at {height}m at {lat}, {lon} for {year}')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{outputs_dir}/wind_speed_{lat}_{lon}_{height}m_{year}.png")

    return

def gamma_func(k, mu_1, mu_2):

    """
    Compute the difference between the squared mean and the second moment 
    of a Weibull distribution.

    Parameters:
    k (float): Shape parameter of the Weibull distribution.
    mu_1 (float): First moment (mean) of the wind speed data.
    mu_2 (float): Second moment of the wind speed data.

    Returns:
    float: The difference between the squared mean and the second moment 
           of the Weibull distribution.
    """

    return gamma(1+1/k)**2/gamma(1+2/k) - (mu_1**2)/mu_2


class weibull(object):

    def __init__(self, x, y, height, wind_data_processed, wd_ts_f):

        """
        Initialize the Weibull class.

        This constructor initializes the Weibull class with the necessary parameters 
        to compute Weibull distribution properties for a specific location and height.

        Parameters:
        x (float): X-coordinate (latitude) of the location.
        y (float): Y-coordinate (longitude) of the location.
        height (int): The height (in meters) at which the Weibull distribution is to be computed.
        wind_data_processed (dict): Dictionary containing processed wind data for different locations.
        wd_ts_f (object): An object containing attributes for wind data time series.

        Returns:
        None
        """   

        self.x = x
        self.y = y
        self.height = height
        self.wind_data_processed = wind_data_processed
        self.wd_ts_f = wd_ts_f

    def obtain_parameters(self):

        """
        Obtain the Weibull distribution parameters (scale and shape) for the wind speed 
        at a given height and location.

        This method computes the scale parameter (A) and shape parameter (k) of the Weibull 
        distribution by analyzing the wind speed time series at the specified height and location.

        Returns:
        tuple: A tuple containing:
            - A (float): The scale parameter of the Weibull distribution.
            - k (float): The shape parameter of the Weibull distribution.
        """

        # Obtain time series at the given location
        inter = wind_interpolation(self.wind_data_processed)

        # Interpolate wind speed at the specified latitude and longitude
        ts = inter.speed_interpolator(self.x, self.y)

        # Compute wind speed at the specified height using the power law
        ts_at_height = compute_wind_speed_power_law(ts, self.height, self.wd_ts_f)

        # Compute the first and second moment at the specified height
        fst_moment = np.mean(ts_at_height[f"ws_pl_{self.height}m"])
        snd_moment = np.mean(ts_at_height[f"ws_pl_{self.height}m"]**2)

        # Solve for the shape parameter (k) using the gamma function and the moments
        k = fsolve(gamma_func, 2, args=(fst_moment, snd_moment))

        # Compute the scale parameter (A) using the first moment and the shape parameter
        A = fst_moment/gamma(1+1/k)

        return A, k
    

    def get_pdf(self, u_max=25, u_min=0):

        """
        Compute the Weibull probability density function (PDF) for a range of wind speeds.

        Parameters:
        u_max (int, optional): Maximum wind speed (in m/s) for the PDF range. Default is 25.
        u_min (int, optional): Minimum wind speed (in m/s) for the PDF range. Default is 0.

        Returns:
        np.ndarray: An array containing the Weibull PDF values for the specified wind speed range.
        """

        # Obtain the Weibull parameters (scale A and shape k)
        A, k = self.obtain_parameters()

        # Create a range of wind speeds from u_min to u_max with a step of 1 m/s
        ws_range = np.arange(u_min, u_max, 1)

        # Compute the Weibull PDF using the formula
        pdf = (k / A) * (ws_range / A) ** (k-1) * np.exp(-(ws_range / A)**k)

        return pdf
    
    
    def plot_pdf(self, u_max=25, u_min=0):

        """
        Plot the Weibull probability density function (PDF) and a histogram of wind speeds.

        This method generates a plot of the Weibull PDF for a specified range of wind speeds
        and overlays it with a histogram of the wind speed data at the specified height and location.

        Parameters:
        u_max (int, optional): Maximum wind speed (in m/s) for the PDF range. Default is 25.
        u_min (int, optional): Minimum wind speed (in m/s) for the PDF range. Default is 0.

        Returns:
        None: The function saves the plot as a PNG file in the outputs directory.
        """

        # Obtain time series at the given location using wind interpolation
        inter = wind_interpolation(self.wind_data_processed)

        # Interpolate wind speed at the specified latitude and longitude
        ts = inter.speed_interpolator(self.x, self.y)

        # Compute wind speed at the specified height using the power law
        ts_at_height = compute_wind_speed_power_law(ts, self.height, self.wd_ts_f)

        # Create a range of wind speeds from u_min to u_max with a step of 1 m/s
        ws_range = np.arange(u_min, u_max, 1)

        # Compute the Weibull PDF for the specified wind speed range
        p = self.get_pdf(u_max=u_max, u_min=u_min)

        # Plot the Weibull PDF and histogram of wind speeds
        plt.figure(figsize=(10, 6))
        plt.plot(ws_range, p, label='Weibull PDF', color='blue')
        plt.hist(ts_at_height[f"ws_pl_{self.height}m"], bins=len(ws_range), density=True, alpha=0.5, color='gray', label='Wind Speed Histogram')
        plt.title('Weibull Probability Density Function')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Probability Density')
        plt.grid()
        plt.legend()
        plt.savefig(f"{outputs_dir}/weibull_pdf.png")

        return


def obtain_wind_rose(wd, x, y, height, wd_ts_f, n_sector=12):

    """
    Generate a wind rose plot for a given location and height.

    This function calculates the wind direction at a specified height and location,
    divides the wind directions into sectors, and generates a wind rose plot.

    Parameters:
    wd (dict): Dictionary containing wind direction data for different locations.
    x (float): X-coordinate (latitude) of the location.
    y (float): Y-coordinate (longitude) of the location.
    height (int): The height (in meters) at which the wind rose is to be generated.
    wd_ts_f (object): An object containing attributes `height_1` and `height_2`, 
                      representing the heights (in meters) of the wind direction measurements.
    n_sector (int, optional): Number of sectors to divide the wind direction into. Default is 12.

    Returns:
    None: The function saves the plot as a PNG file in the outputs directory.
    """

    # Initialize the wind interpolation object with the wind direction data
    inter = wind_interpolation(wd)

    # Interpolate wind direction at the specified latitude and longitude
    ts = inter.direction_interpolator(x, y)

    # Compute wind direction at the specified height using linear interpolation
    ts_at_height = (ts[f"wind_direction_{wd_ts_f.height_2}m"] - ts[f"wind_direction_{wd_ts_f.height_1}m"])/(wd_ts_f.height_2 - wd_ts_f.height_1) * (height - wd_ts_f.height_1) + ts["wind_direction_10m"]

    ts[f"wd_at_{height}m"] = ts_at_height

    # Divide the wind direction into n_sector bins
    sectors = {}

    nd = n_sector

    ts_at_height_series = pd.Series(ts_at_height)

    for i in range(1, nd+1):

        sectors[f"Sector_{i}"] = pd.DataFrame(columns=["wd"])
        sectors["prob"] = []

        # Filter wind directions that fall within the current sector
        sectors[f"Sector_{i}"] = ts_at_height_series[(ts_at_height_series > (i-1)*360/nd) & (ts_at_height_series <= i*360/nd)]

    # Create a new figure for the wind rose plot
    plt.figure(figsize=(10, 6))
    plt.title(f"Wind Rose at {height}m", pad=20)

    # Compute the probability for each sector
    sectors["prob"] = [len(sectors[f"Sector_{i}"])/len(ts_at_height) for i in range(1, nd+1)]
    ax = plt.subplot(projection='polar')

    # Set the zero location to North and the direction to clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  

    # Plot the wind rose as a bar chart
    ax.bar(np.radians(np.arange(1, 361, 360/nd)), sectors["prob"], width=np.pi/(nd/2))
    ax.xaxis.set_ticks(np.arange(0, 2*np.pi, np.pi/(6)))
    plt.savefig(f"{outputs_dir}/windrose.png")

    return

def load_turbines(filepath):

    """
    Load turbine data from CSV files in the 'inputs' directory.

    This function reads all CSV files in the 'inputs' directory located relative to the given filepath
    and stores the data in a dictionary, where the keys are the file names (without extensions) and 
    the values are Pandas DataFrames containing the turbine data.

    Parameters:
    filepath (Path): Path to a file or directory. The function uses this to locate the 'inputs' directory.

    Returns:
    dict: A dictionary where keys are file names (without extensions) and values are Pandas DataFrames 
          containing the turbine data.
    """

    # Set the path to the 'inputs' directory relative to the given filepath
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

        """
        Initialize the turbine class.

        This constructor initializes the turbine class by associating turbine data and hub heights 
        with turbine objects. Each turbine object is created as an attribute of the class, 
        containing its hub height and power curve.

        Parameters:
        turbine_data (dict): A dictionary where keys are turbine names and values are Pandas DataFrames 
                            containing the power curve data for each turbine.
        hub_heights (list): A list of hub heights (in meters) corresponding to each turbine in the turbine_data.

        Returns:
        None
        """

        # Store the turbine data and hub heights as class attributes
        self.turbine_data = turbine_data
        self.hub_heights = hub_heights

        # Iterate through each turbine in the turbine_data dictionary 
        for i, key in enumerate(turbine_data.keys()):

            turbine_ = {}

            # Assign the hub height for the current turbine
            turbine_['hub_height'] = self.hub_heights[i]

            # Assign the power curve for the current turbine
            turbine_['power_curve'] = self.turbine_data[key]
            
            # Dynamically create an attribute for the turbine and assign the dictionary to it
            setattr(self, key, turbine_.copy())

    def compute_AEP(self, turbine_name, lat, lon, wind_data, year, ts_object):

        """
        Compute the Annual Energy Production (AEP) for a given turbine and location.

        This method calculates the AEP by combining the Weibull probability density function (PDF)
        with the turbine's power curve. It interpolates the power curve to match the wind speed range
        and integrates the Weibull PDF to estimate the energy production.

        Parameters:
        turbine_name (str): Name of the turbine for which the AEP is to be computed.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        wind_data (dict): Dictionary containing wind data for different locations.
        year (int): The year for which the AEP is to be computed.
        ts_object (object): An object containing attributes for wind data time series.

        Returns:
        float: The computed Annual Energy Production (AEP) in kilowatt-hours (kWh).
        """
        
        # Retrieve the turbine data for the specified turbine
        turbine_ = getattr(self, turbine_name)

        # Filter wind data for the specified year
        for location in wind_data.keys():

            wind_data[location].set_index('time', inplace=True)
            wind_data[location] = wind_data[location].loc[f'{year}']

        # Compute the Weibull PDF for the wind speed range
        weibull_obj = weibull(lat, lon, turbine_['hub_height'], wind_data, ts_object)
        u_weibull = weibull_obj.get_pdf(u_max=turbine_['power_curve']['Wind Speed [m/s]'].max(), u_min=turbine_['power_curve']['Wind Speed [m/s]'].min())

        # Round wind speeds to the nearest integer

        turbine_r = turbine_.copy()

        turbine_r['power_curve'] = turbine_r['power_curve'].copy()

        turbine_r['power_curve']['Wind Speed [m/s]'] = turbine_r['power_curve']['Wind Speed [m/s]'].round()

        # Group by the rounded wind speeds and calculate the mean power
        grouped_df = turbine_r['power_curve'].groupby('Wind Speed [m/s]')['Power [kW]'].mean().reset_index()

        # Create a range of wind speeds from the minimum to the maximum in the original data
        wind_speed_range = np.arange(int(turbine_r['power_curve']['Wind Speed [m/s]'].min()), 
                         int(turbine_r['power_curve']['Wind Speed [m/s]'].max()) + 1)

        # Interpolate the power values for the full range of wind speeds
        interpolated_df = pd.DataFrame({'Wind Speed [m/s]': wind_speed_range})
        interpolated_df = interpolated_df.merge(grouped_df, on='Wind Speed [m/s]', how='left')
        interpolated_df['Power [kW]'] = interpolated_df['Power [kW]'].interpolate()

        AEP = 8760 * np.sum(u_weibull * interpolated_df['Power [kW]'][:-1])

        return AEP
    
    def plot_power_curve(self, turbine_name):

        """
        Plot the power curve for a specified turbine.

        This method generates a plot of the turbine's power curve, showing the relationship 
        between wind speed and power output. The plot is saved as a PNG file in the outputs directory.

        Parameters:
        turbine (str): Name of the turbine for which the power curve is to be plotted.

        Returns:
        None: The function saves the plot as a PNG file in the outputs directory.
        """

        # Retrieve the turbine data for the specified turbine
        turbine_ = getattr(self, turbine_name)

        # Plot the power curve (wind speed vs. power output)
        plt.figure(figsize=(10, 6))
        plt.plot(turbine_['power_curve']['Wind Speed [m/s]'], turbine_['power_curve']['Power [kW]'], marker='o', label='Power Curve')
        plt.title(f'Power Curve for {turbine_name}')
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Power [kW]')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{outputs_dir}/power_curve_{turbine_name}.png")

        return
                    
        

