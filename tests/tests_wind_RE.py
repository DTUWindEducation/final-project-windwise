from pathlib import Path
import pandas as pd
import os
import Wind_Re

DATA_DIR = Path(__file__).resolve().parent

outputs_dir = Path(__file__).resolve().parent.parent / 'outputs'

def test_load_and_parse():

    """Test loading and parsing of wind data. Checks that all locations
    are loaded and differenciated"""

    # Given
    path_to_nc_files = DATA_DIR
    exp_num_locations = 4 # Number of locations in the dataset

    # When
    wind_data = Wind_Re.get_data(path_to_nc_files)

    num_locations = len(wind_data.keys())

    # Then
    assert num_locations == exp_num_locations


def test_ws_time_series():

    """Test wind speed time series calculation. Checks that the wind speed
    time series is calculated correctly and that the mean wind speed is
    greater than zero."""

    # Given
    path_to_nc_files = DATA_DIR
    expected_column_name = ["wind_speed_100m", "wind_speed_10m"]
    expected_average_min = 0

    # When
    wind_data = Wind_Re.get_data(path_to_nc_files)

    wind_data_ts = {}
    wd_ts = Wind_Re.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)
    labels = list(wind_data.keys())
    label_to_check = labels[0]

    wind_data_ts[label_to_check] = pd.DataFrame()

    # Adding columns for wind speed and wind direction

    wind_data_ts[label_to_check] = wd_ts.compute_ws_time_series(label_to_check)

    # Then
    # Check that the expected columns are present in the DataFrame
    assert expected_column_name[0] in wind_data_ts[label_to_check].columns
    assert expected_column_name[1] in wind_data_ts[label_to_check].columns

    # Check that values are within the expected range
    assert wind_data_ts[label_to_check].wind_speed_10m.mean() >= expected_average_min

def test_wd_time_series():

    """Test wind speed time series calculation. Checks that the wind speed
    time series is calculated correctly and that the mean wind speed is
    greater than zero."""

    # Given
    path_to_nc_files = DATA_DIR
    expected_column_name = ["wind_direction_100m", "wind_direction_10m"]
    expected_min = 0
    expected_max = 360

    # When
    wind_data = Wind_Re.get_data(path_to_nc_files)

    wind_data_ts = {}
    wd_ts = Wind_Re.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)
    labels = list(wind_data.keys())
    label_to_check = labels[0]

    wind_data_ts[label_to_check] = pd.DataFrame()

    # Adding columns for wind speed and wind direction

    wind_data_ts[label_to_check] = wd_ts.compute_wdir_time_series(label_to_check)

    # Then
    # Check that the expected columns are present in the DataFrame
    assert expected_column_name[0] in wind_data_ts[label_to_check].columns
    assert expected_column_name[1] in wind_data_ts[label_to_check].columns

    # Check that values are within the expected range
    assert wind_data_ts[label_to_check].wind_direction_10m.max() <= expected_max
    assert wind_data_ts[label_to_check].wind_direction_10m.min() >= expected_min
    assert wind_data_ts[label_to_check].wind_direction_100m.min() >= expected_min
    assert wind_data_ts[label_to_check].wind_direction_100m.max() <= expected_max

def test_power_law():

    """Test power law calculation. Checks that the power law is calculated
    correctly and that the wind speed at 100m is greater than zero."""

    # Given
    path_to_nc_files = DATA_DIR
    height = 65
    expected_column_name = f"ws_pl_{height}m"

    # When
    wind_data = Wind_Re.get_data(path_to_nc_files)

    wind_data_ts = {}

    wd_ts = Wind_Re.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)

    for label in wind_data.keys():

        wind_data_ts[label] = pd.DataFrame()

        # Adding columns for wind speed and wind direction

        wind_data_ts[label] = wd_ts.compute_ws_time_series(label)
        wind_data_ts[label] = Wind_Re.compute_wind_speed_power_law(wind_data_ts[label], height, wd_ts)

    # Then
    # Check that the expected columns are present in the DataFrame
    for label in wind_data.keys():
        assert expected_column_name in wind_data_ts[label].columns

        # Check that values are within the expected range
        assert wind_data_ts[label][expected_column_name].mean() > wind_data_ts[label].wind_speed_10m.mean()
        assert wind_data_ts[label][expected_column_name].mean() < wind_data_ts[label].wind_speed_100m.mean()


def test_weibull_distribution():

    """Test the weibull curve fitting, ensuring that the summation is equal to 1"""

    # Given
    path_to_nc_files =  DATA_DIR
    expected_max = 1.0
    expected_min = 0.9
    height = 65
    latitude = 55.65
    longitude = 7.9

    # When
    wind_data = Wind_Re.get_data(path_to_nc_files)

    wind_data_ts = {}

    wd_ts = Wind_Re.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)

    for label in wind_data.keys():

        wind_data_ts[label] = pd.DataFrame()

        # Adding columns for wind speed and wind direction

        wind_data_ts[label] = wd_ts.compute_ws_time_series(label)
        wind_data_ts[label] = wd_ts.compute_wdir_time_series(label)

    weibull_object = Wind_Re.weibull(latitude, longitude, height, wind_data_ts, wd_ts)
    U = weibull_object.get_pdf()

    # Then
    # Check that the area under the curve is equal to 1
    assert sum(U) >= expected_min
    assert sum(U) <= expected_max


def test_weibull_distribution_plot():

    """Test the weibull curve and histogram plotting, ensuring that there 
    is a figure saved"""

    # Given
    path_to_nc_files =  DATA_DIR
    height = 65
    latitude = 55.65
    longitude = 7.9
    expected_filename = outputs_dir / "weibull_pdf.png"

    # When
    wind_data = Wind_Re.get_data(path_to_nc_files)

    wind_data_ts = {}

    wd_ts = Wind_Re.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)

    for label in wind_data.keys():

        wind_data_ts[label] = pd.DataFrame()

        # Adding columns for wind speed and wind direction

        wind_data_ts[label] = wd_ts.compute_ws_time_series(label)
        wind_data_ts[label] = wd_ts.compute_wdir_time_series(label)

    weibull_object = Wind_Re.weibull(latitude, longitude, height, wind_data_ts, wd_ts)
    weibull_object.plot_pdf()

    # Then
    # Check that a figure is created
    assert expected_filename.exists(), f"Expected file {expected_filename} was not found."

def test_wind_rose():

    """Test the wind rose plotting, ensuring that there is a figure saved"""

    # Given
    path_to_nc_files =  DATA_DIR
    height = 65
    latitude = 55.65
    longitude = 7.9
    expected_filename = outputs_dir / "windrose.png"

    # When
    wind_data = Wind_Re.get_data(path_to_nc_files)

    wind_data_ts = {}

    wd_ts = Wind_Re.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)

    for label in wind_data.keys():

        wind_data_ts[label] = pd.DataFrame()

        # Adding columns for wind speed and wind direction

        wind_data_ts[label] = wd_ts.compute_ws_time_series(label)
        wind_data_ts[label] = wd_ts.compute_wdir_time_series(label)

    wind_rose = Wind_Re.obtain_wind_rose(wind_data_ts, latitude, longitude, height, wd_ts, 12)

    # Then
    # Check that a figure is sved
    assert expected_filename.exists(), f"Expected file {expected_filename} was not found."

def test_AEP_calculation():

    """Test the AEP calculatiom, ensuring that the wind turbine's capacity factor
    is not greater than 1"""

    # Given
    path_to_files =  DATA_DIR
    height = 65
    latitude = 55.65
    longitude = 7.9
    expected_max = 1.0
    turbine = "NREL_Reference_5MW_126"
    P_nom = 5
    year = 1998

    # When
    hub_heights = [90, 150]

    wind_data = Wind_Re.get_data(path_to_files)

    wind_data_ts = {}

    wd_ts = Wind_Re.time_series(wind_data, 'u10', 'v10', 10, 'u100', 'v100', 100)

    for label in wind_data.keys():

        wind_data_ts[label] = pd.DataFrame()

        # Adding columns for wind speed and wind direction

        wind_data_ts[label] = wd_ts.compute_ws_time_series(label)
        wind_data_ts[label] = wd_ts.compute_wdir_time_series(label)

    turbines = Wind_Re.load_turbines(path_to_files)

    turbine_object = Wind_Re.turbine(turbines, hub_heights)

    AEP = turbine_object.compute_AEP(turbine, latitude, longitude, wind_data_ts, year, wd_ts)

    capacity_factor = AEP / (P_nom * 8760 * 1000)

    # Then
    # Check that the capacity factor is not greater than 1
    assert capacity_factor <= expected_max