README

Team: WindWise

Package name: Wind_Re

# OVERVIEW

Wind_Re is a powerful package to carry out wind resource assessments. Being the input wind component (u,v) time series at two heights for 4 locations and the turbine's power curve, it is capable of, in addition to other intermediate steps, compute the AEP for any location within the area formed by the 4 locations, for any selected year given in the input.

These additional features include the computation of the wind speed and time series at any height and location, generating he Weibull fit, or plotting the wind turbine's power curve.

A step by step guide of the installation, as well as description and usage of the functions and classes is provided below.


# QUICK START GUIDE 

*******************INCLUDE INSTALLATION********************

# Functions and clasess descriptions

## Functions

**load_netcfd_to_dataframe(file_path)**

Takes *file_path* to the .nc file that you want to load into a DataFrame, which stores the data as it is in the netcfd file.

**collect_nc_files(folder_path)**

Given the *folder_path* to the folder where the .nc files are stored, the function retrieves all the paths to the .nc files in the desired folder. Returns list of paths.

**get_data(directory)**

Being *directory* the path to all .nc files, loads the data using *load_netcfd_to_datagrame()*, extracts the coordinates of the 4 locations, and creates a dictionary containing a DataFrame for each of the 4 locations. The stored time series is the result of the concatenation of all years for each location in each of the .nc files. Returns the mentioned dictionary.

**compute_wind_speed_power_law(wind_data, height, wd_ts_f)**

For the given wind speed time series in the *wind_data* DataFrame for a single location, inserts a column containing the wind speed at the given *height*, with the name *ws_pl_{height}m*. A column with the computed shear exponent is included as well, called *alpha*. The *wd_ts_f* is the *time_series* object, used to know which columns to look into, as well as obtaining the respective heights.


**plot_wind_speed_year(wind_data, wd_ts_f, year, lat, lon, height)**

[EXTRA FUNCTION]

Given the *wind_data* dictionary containing the time series for each location, some coordinates *lat* and *lon*, and the desired year and height, it returns a plot of the time series for the described condition using the *wind_interpolator* class and the *compute_wind_speed_power_law()* function.. The *wd_ts_f* is the *time_series* object, used to know which columns to look into, as well as obtaining the respective heights. The plot is saved under the name *wind_speed_{lat}_{lon}_{height}m_{year}.png*.

**gamma_func(k, mu_1, mu_2)**

Auxilliary function for weibull.

**obtain_wind_rose(wd, x, y, height, wd_ts_f, n_sector=12)**

Given the *wd* DataFrame containing wind data, coordinates *x* and *y*, and the desired *height*, this function generates a wind rose plot. The *wd_ts_f* is the *time_series* object, used to know which columns to look into, as well as obtaining the respective heights. The function first interpolates the wind direction data at the specified coordinates and height. It then divides the wind direction into *n_sector* bins (default is 12) and calculates the probability of wind occurrence in each sector. Finally, it plots the wind rose using a polar projection, with the north direction at the top, and saves the plot as an image file named "windrose.png" in the specified output directory.

**load_turbines(filepath)**

Given the path *file_path* to the .csv files containing the wind turbines data, it creates a dictionary containing the power and CT curve for each of the turbines, where each turbine is stored in a separate DataFrame. It returns the dictionary.

## Classess

### time_series(elf, wind_data, u_1, v_1, height_1, u_2, v_2, height_2)

This class is used to transform from the wind components u and v to wind speed and wind direction. The inputs are the dictionary containing the timeseries for each of the locations, as well as the name of the column names containing the components, and the heights at which they are measured.

**compute_ws_time_series(self, cords)**

For the given *cords*, it computes the windspeed time series at the two available and stated heights in the object declaration. It does so by means of the following equation.

$ws = \sqrt{u^2 + v^2}$

The coordinates must be one of the 4 initial coordinates. It returns the dictionary, in which the DataFrame corresponding to the coordinates has an additional column per height, containing the wind speed time series.

**compute_wdir_time_series(self, cords)**

For the given *cords*, it computes the wind direction time series at the two available and stated heights in the object declaration. It does so by means of the following equation, which is adjusted to give wind directions between 0 and 360 degrees.

$wdir = \arctan{(-u, -v)}$

The coordinates must be one of the 4 initial coordinates. It returns the dictionary, in which the DataFrame corresponding to the coordinates has an additional column per height, containing the wind direction time series.

### wind_interpolation(wind_data)

This class aims to compute the wind speed and direction time series for any of the locations wihtin the square formed by the 4 locations. The input is the wind data time series obtained by means of the time_series class.

**speed_interpolator(self, x_coord, y_coord)**

For the *x_coords* and *y_coord*, it obtains the interpolated wind speed time series for both available heights, by means of a bilinear interpolation. It returns a DataFrame, containing the time wind speed timeseries for the new location at the two given heights.

**direction_interpolator(slef, x_coord, y_coord)**

For the *x_coords* and *y_coord*, it obtains the interpolated wind direction time series for both available heights, by means of a bilinear interpolation. It returns a DataFrame, containing the time wind direction timeseries for the new location at the two given heights.

### weibull(self, x, y, height, wind_data_processed, wd_ts_f)

Class used to weibull distribution related actions. The input is the coordinates *x* and *y* and *height* where you will want the weibull fit to be carried out, as well as the *wind_data_processed* coming from the *time_series* object, and the *time_series* object itself, used to know which columns to look into, as well as obtaining the respective heights.

**obtain_parameters(self)**

From the data input into the object declaration, it obtains the weibull paramters A and K for the distributions, using the first and second moments method. The time series for the selected location and height is obtained by means of the *wind_interpolation* class and the *compute_wind_speed_power_law()* function It returns the both parameters.

**get_pdf(self, u_max=25, u_min=0)**

From the data input into the object declaration, after obtaining the parameters using *obtain_parameters()*, it obtains the weibull distribution and returns the probability density function, being the max and min wind speed that indicated in *u_min* and *u_max*.

**plot_pdf(self, u_max=25, u_min=0)**

From the data input into the object declaration, it first computes the time series for the desired location and height, for it to do the histogram. For the weibull curve, it uses the *get_pdf()* function. Both plots are made into one figure, and save in the outputs folder under the name *weibull_pdf.png*.

### turbine(turbine_data, hub_heights)

This class is used to obtain the AEP and the information related with the wind turbines. The input is the *turbine_data* dictionary coming form the input .csv and loaded by means of the *load_turbines()* function. Hub heights of the wind turines has to be input as well.

**compute_AEP(self, turbine_name, lat, lon, wind_data, year, ts_object)**

For the specified *turbine_name*, *lat*, *lon*, and *year*, it returns the AEP under those conditions. It first uses the *weibull* object to obtain the weibull distribution for the specified conditions, to then use the adjusted power curve to obtain the AEP in kWh.

**plot_power_curve(self, turbine_name)**

Plots the selected *turbine_name* power curve and saves it in the output folder under the name *ower_curve_{turbine}.png*.
