import os
from datetime import datetime,timedelta
import cftime
import warnings
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import dask
import rioxarray
import glob

figure_dir = '/cpc/int_desk/pac_isl/stations/images'
climyear1 = 1991
climyear2 = 2020

current_year = datetime.now().year
# Define the file path pattern for the specific years you want
total_years = range(1991, current_year + 1)  # years between 1982 and 2020 inclusive

files = [f"/cpc/int_desk/data/oisstv2/sst.day.mean.{year}.nc" for year in total_years]

# Use xarray to open the files
sst_daily = xr.open_mfdataset(files, engine = "netcdf4",combine='by_coords', chunks = {"lat":720, "lon":1440, "time":100})

#compute sst anomalies for past 7 days, and diff
with dask.config.set(**{'array.slicing.split_large_chunks': False}):
   # Step 1: Select the climatology data for the fixed baseline period (1991-2020)
    sst_clim_baseline = sst_daily.sel(time=sst_daily.time.dt.year.isin(range(climyear1, climyear2 + 1)))

    # Step 2: Compute the mean climatology by day of the year for the baseline period (1991-2020)
    sst_clim_mean = sst_clim_baseline.groupby('time.dayofyear').mean(dim='time')

    # Step 3: Compute the day of year for each time in the original SST dataset
    sst_daily_dayofyear = sst_daily.time.dt.dayofyear

    # Step 4: Expand the climatology to match the time dimension of sst_daily, maintaining the same day-of-year values
    sst_climatology_expanded = sst_clim_mean.sel(dayofyear=sst_daily_dayofyear)

    # Ensure that climatology is expanded across all the time steps in sst_daily
    # sst_climatology_expanded = sst_climatology_expanded.expand_dims(time=sst_daily.time)

    # Step 5: Calculate the anomaly by subtracting the climatological baseline from the original SST values
    sst_anomaly = sst_daily - sst_climatology_expanded

    # Step 6: Calculate 7-day and 14-day anomalies (if needed)
    sst_anom_7 = sst_anomaly.isel(time=slice(-7, None)).mean(dim='time')
    sst_anom_814 = sst_anomaly.isel(time=slice(-14, -7)).mean(dim='time')
    sst_anom_diff = sst_anom_7 - sst_anom_814

    # Persist the results if necessary
    sst_anom_7 = sst_anom_7.persist()
    sst_anom_diff = sst_anom_diff.persist()

    #collapse all dask graph layers to one to conserve space
    sst_anom_7 = sst_anom_7.persist()
    sst_anom_diff = sst_anom_diff.persist()

# Define the colormap
colors = ["darkblue", "blue", "dodgerblue", "lightblue", "white", "lightyellow", "gold", "orange", "red", "darkred"]
custom_cmap = LinearSegmentedColormap.from_list("blue_white_yellow_red", colors)

# Write CRS to the datasets
sst_anom_7_crs = sst_anom_7.rio.write_crs('EPSG:4326', inplace=True)
sst_anom_diff_crs = sst_anom_diff.rio.write_crs('EPSG:4326', inplace=True)

# Rename latitude and longitude dimensions
sst_anom_7_crs = sst_anom_7_crs.rename({'lat': 'y', 'lon': 'x'})
sst_anom_diff_crs = sst_anom_diff_crs.rename({'lat': 'y', 'lon': 'x'})

# Clip to a specific geographic extent and reproject to EPSG:3857 (Mercator)
sst_anom_7_clipped = sst_anom_7_crs.rio.clip_box(minx=0, miny=-85, maxx=360, maxy=85)
sst_anom7_mercator = sst_anom_7_clipped.rio.reproject("EPSG:3857")

sst_anom_diff_clipped = sst_anom_diff_crs.rio.clip_box(minx=0, miny=-85, maxx=360, maxy=85)
sst_anomdiff_mercator = sst_anom_diff_clipped.rio.reproject("EPSG:3857")

# Normalize the data to the desired range [-2, 2] (e.g., SST anomalies)
vmin, vmax = -3, 3
vmindiff, vmaxdiff = -2, 2
sst_7norm = np.clip(sst_anom7_mercator.sst, vmin, vmax)
sst_7diffnorm = np.clip(sst_anomdiff_mercator.sst, vmindiff, vmaxdiff)

# Apply a colormap (e.g., "RdBu" or custom color map)
cmap = plt.get_cmap(custom_cmap)
sst_7rgb = cmap(sst_7norm)[:, :, :3]  # Extract RGB channels (ignore alpha channel)
sst_7rgb = (sst_7rgb * 255).astype(np.uint8)  # Scale to 0-255 for image representation

sst_7diffrgb = cmap(sst_7diffnorm)[:, :, :3]  # Extract RGB channels (ignore alpha channel)
sst_7diffrgb = (sst_7diffrgb * 255).astype(np.uint8)  # Scale to 0-255 for image representation

# Create an alpha channel for transparency (0 for NaN, 255 for valid data)
sst_7alpha = ~np.isnan(sst_7norm) * 255
sst_7rgba = np.dstack((sst_7rgb, sst_7alpha.astype(np.uint8)))  # Add alpha channel

sst_7diffalpha = ~np.isnan(sst_7diffnorm) * 255
sst_7diffrgba = np.dstack((sst_7diffrgb, sst_7diffalpha.astype(np.uint8)))  # Add alpha channel

# Convert to xarray for export
sst_7rgba_xarray = xr.DataArray(
    sst_7rgba,
    dims=("y", "x", "band"),
    coords={"y": sst_anom7_mercator.y, "x": sst_anom7_mercator.x, "band": [1, 2, 3, 4]},
)
sst_7rgba_xarray = sst_7rgba_xarray.transpose("band", "y", "x").rio.write_crs(sst_anom7_mercator.rio.crs)

sst_7diffrgba_xarray = xr.DataArray(
    sst_7diffrgba,
    dims=("y", "x", "band"),
    coords={"y": sst_anomdiff_mercator.y, "x": sst_anomdiff_mercator.x, "band": [1, 2, 3, 4]},
)
sst_7diffrgba_xarray = sst_7diffrgba_xarray.transpose("band", "y", "x").rio.write_crs(sst_anomdiff_mercator.rio.crs)

# Write to GeoTIFF with transparency
sst_7rgba_xarray.rio.to_raster(os.path.join(figure_dir, 'sst_mercator7.tif'), dtype="uint8")
sst_7diffrgba_xarray.rio.to_raster(os.path.join(figure_dir, 'sst_mercator7diff.tif'), dtype="uint8")