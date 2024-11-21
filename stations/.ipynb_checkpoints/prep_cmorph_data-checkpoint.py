import os
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib as mpl
import rioxarray as rio
from rasterio.warp import calculate_default_transform

cmorph_prd = '/cpc/fews/production/cmorph_cpcwork5/cmorph_RH6_BC_ADJ_EOD/output/bin/CMORPH_V1.0_ADJ_0.25deg-DLY_EOD_'
cmorph_clim = '/cpc/fews/production/cmorph_cpcwork5/cmorph_RH6_BC_ADJ_EOD/output/bin/clim_dly/clim_dly.'
figure_dir = '/cpc/int_desk/pac_isl/stations/images'

#from Ravi's ctl file for cmorph
xdim = 1440
ydim = 480
xmin = 0.125
xmax = 360
ymin = -59.875
ymax = 60

#anomaly max/mins on the cmorph anomaly figures for color bar
minanom = -25
maxanom = 25

# Define custom intervals for the colormap
#anomaly color bar breaks
anom_intervals = [-25, -20, -15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15, 20, 25]

# Define colors for the colormap (corresponding to your value intervals)
#these were found usuing find_dominant_colors function in prep_data.ipynb in this folder
anom_colors = [
    (204/255, 51/255, 51/255), #dark red,
    (254/255,  4/255, 0/255), #bright red
    (255/255,  158/255, 0/255), #orange
    (255/255, 232/255, 123/255), #yellow
    (119/255,  79/255,  70/255), #dark brown
    (180/255, 140/255, 128/255), #med brown
    (237/255, 221/255, 212/255), #light brown
    (254/255, 254/255, 254/255), # off white
    (201/255, 254/255, 192/255), #light green
    (119/255, 244/255, 114/255), #bright green
    (30/255, 180/255,  30/255), #dark green
    (152/255, 210/255, 250/255), #lightmed blue
    (39/255, 129/255, 240/255), #darker blue
    (220/255, 220/255, 254/255), #light purple
    (127/255, 111/255, 234/255) #med purple
]

########## functions
#get a string of dates that will match the cmorph binary file naming conventions
def get_date_str(date):
    if date.month <10:
        cmonth_str = '0' + str(date.month)
    else:
        cmonth_str = str(date.month)
    if date.day <10:
        cday_str = '0' + str(date.day)
    else: cday_str = str(date.day)
    current_str = str(date.year) + cmonth_str + cday_str
    return current_str, cmonth_str+cday_str

#read in a binary file to xarray
def read_in_binary(location, xdim, ydim, xmin, xmax, ymin, ymax):
    if os.path.exists(location):
        with open(location, 'rb') as binary_file:
            binary_data = binary_file.read()
        data = np.frombuffer(binary_data, dtype = np.float32)
        data = data.reshape(ydim, xdim)
        lat = np.linspace(ymin, ymax, data.shape[0])
        lon = np.linspace(xmin, xmax, data.shape[1])
        da = xr.DataArray(data, dims = ['lat', 'lon'], coords = {'lat': lat, 'lon':lon})
        return da
    
#convert lat/lon coords to mercator projection for tiles
def convert_to_mercator(ds, var):
    # ds = ds.where(~np.isnan(ds[var]), drop=True)
    ds_mercator = ds.rio.reproject("EPSG:3857")
    # ds_clip = ds_mercator.where(ds_mercator.notnull(), drop=True)
    return ds_mercator

#calculate the average anomaly over past days, e.g. last 90, 30, 7, etc...
#must give function a xarray dataset with last days and a dataset with climatology values
def calc_anom_pastdays(ds, ds_var, ds_clim, ds_clim_var, days):
    days_ds = ds.isel(time=slice(days, None)).mean(dim='time')
    days_clim = ds_clim.isel(time=slice(days, None)).mean(dim = 'time')
    days_anom = (days_ds[ds_var] - days_clim[ds_clim_var]).to_dataset(name = 'anom')
    return days_anom

#convert the data array to RGB values for image export using defined colorschemes
# Apply the colormap and norm to the data
def apply_colormap(da, colormap, norm):
    """
    Apply a custom colormap to the data array based on specified boundaries.

    Parameters:
        da: xarray.DataArray
            The data array to which the colormap will be applied.
        colormap: matplotlib.colors.Colormap
            The custom colormap to apply.
        norm: matplotlib.colors.BoundaryNorm
            The normalizer that defines the color intervals.

    Returns:
        xarray.DataArray
            DataArray with RGBA values.
    """
    # Clip the data to the specified range (you could also use np.clip if needed)
    da_clipped = np.clip(da, value_intervals[0], value_intervals[-1])

    # Map the data values to the colormap using BoundaryNorm
    colormap_values = colormap(norm(da_clipped))

    # Scale to 0-255 for RGB and add an alpha channel
    da_rgb = (colormap_values[:, :, :3] * 255).astype(np.uint8)
    da_alpha = (~np.isnan(da_clipped)) * 255  # Transparency: 0 for NaN, 255 otherwise
    da_rgba = np.dstack((da_rgb, da_alpha.astype(np.uint8)))  # Combine RGB + Alpha

    # Convert to xarray for exporting with spatial coordinates
    da_rgba_xarray = xr.DataArray(
        da_rgba,
        dims=("y", "x", "band"),
        coords={"y": da.y, "x": da.x, "band": [1, 2, 3, 4]},
    )
    da_rgba_xarray = da_rgba_xarray.transpose("band", "y", "x").rio.write_crs(da.rio.crs)

    return da_rgba_xarray


########## MAIN SCRIPT
# Get the current date
current_date = datetime.now().date()

# Get the date up to 185 days prior (because of lags in data availability, grab a few extra days to be safe
last180_days_ago = current_date - timedelta(days=185)

# Generate a list of dates from seven_days_ago to current_date
last180 = [last180_days_ago + timedelta(days=i) for i in range(186)]  # 8 to include both endpoints

date_strs = []
for date in last180:
    date_strs.append((get_date_str(date)[0], get_date_str(date)[1]))
    
#read in files 
panom_da = []
panomclim_da = []
for d, date in enumerate(date_strs):
    da = read_in_binary(cmorph_prd + date[0], xdim, ydim, xmin, xmax, ymin, ymax)
    da_clim = read_in_binary(cmorph_clim + date[1], xdim, ydim, xmin, xmax, ymin, ymax)
    if da is not None:
        da = da.expand_dims({'time': [last180[d]]})
        da_clim = da_clim.expand_dims({'time':[last180[d]]})
        panom_da.append(da)
        panomclim_da.append(da_clim)
allptotal = xr.concat(panom_da, dim = 'time').to_dataset(name = 'ptotal') #.sum(dim = 'date')
allptotalclim = xr.concat(panomclim_da, dim = 'time').to_dataset(name = 'ptotalclim')
allptotal = allptotal.rename({'lon':'x', 'lat':'y'})
allptotalclim = allptotalclim.rename({'lon':'x', 'lat':'y'})

#compute anomalies and convert to mercator for list of dates specified
last90_anom = calc_anom_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 90) 
last90_anom_crs = last90_anom.rio.write_crs('EPSG:4326', inplace = True)
last90_anom_clipped = last90_anom.sel(x=slice(0,359.999), y = slice(-60,60))
last90_anom_mc = convert_to_mercator(last90_anom_clipped, 'anom')
last90_anomnorm = np.clip(last90_anom_mc.anom, minanom, maxanom)

last30_anom = calc_anom_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 30) 
last30_anom_crs = last30_anom.rio.write_crs('EPSG:4326', inplace = True)
last30_anom_clipped = last30_anom.sel(x=slice(0,359.999), y = slice(-60,60))
last30_anom_mc = convert_to_mercator(last30_anom_clipped, 'anom')
last30_anomnorm = np.clip(last30_anom_mc.anom, minanom, maxanom)

last7_anom = calc_anom_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 7) 
last7_anom_crs = last7_anom.rio.write_crs('EPSG:4326', inplace = True)
last7_anom_clipped = last7_anom.sel(x=slice(0,359.999), y = slice(-60,60))
last7_anom_mc = convert_to_mercator(last7_anom_clipped, 'anom')
last7_anomnorm = np.clip(last7_anom_mc.anom, minanom, maxanom)

# Create a ListedColormap using your defined colors
anom_cmap = ListedColormap(anom_colors)

# Create a BoundaryNorm to map the data to the value intervals
anom_norm = BoundaryNorm(boundaries=anom_intervals, ncolors=len(anom_colors))

last90_rgb = apply_colormap(last90_anomnorm, anom_cmap, anom_norm)
last30_rgb = apply_colormap(last30_anomnorm, anom_cmap, anom_norm)
last7_rgb = apply_colormap(last7_anomnorm, anom_cmap, anom_norm)

#write to raster
last90_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph90anom.tif'), dtype="uint8")
last30_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph30anom.tif'), dtype="uint8")
last7_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph7anom.tif'), dtype="uint8")