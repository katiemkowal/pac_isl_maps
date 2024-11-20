import os
from io import BytesIO
import requests
from datetime import datetime,timedelta
import warnings
import xarray as xr
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import pandas as pd
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.colors as mcolors
import helper_dicts as hdict
import helper_functions as helper
import cftime
import dask
import rioxarray
import glob

figure_dir = '/cpc/int_desk/pac_isl/stations/images'
climyear1 = 1991
climyear2 = 2020

def getYears(start, end):
    #start year, to end year inclusive
    # >get YearStrings ("1981", "1982"... "2016")
    #return [expression for var is iterable if condition]
    return [year for year in range(start, end+1)]

current_year = datetime.now().year
# Define the file path pattern for the specific years you want
total_years = range(1991, current_year + 1)  # years between 1982 and 2020 inclusive

files = [f"/cpc/int_desk/data/oisstv2/sst.day.mean.{year}.nc" for year in total_years]

# Use xarray to open the files
sst_daily = xr.open_mfdataset(files, engine = "netcdf4",combine='by_coords', chunks = {"lat":720, "lon":1440, "time":100})

#compute sst anomalies for past 7 days, and diff
with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    clim_years = getYears(climyear1, climyear2)
    sst_daily = sst_daily.sortby('time')
    sst_clim = sst_daily.sel(time=sst_daily.time.dt.year.isin(clim_years))

    sst_clim_mean = sst_clim.sortby('time').groupby('time.dayofyear').mean(dim = 'time')
    # Step 1: Compute the day of year for each time in the original dataset
    sst_daily_dayofyear = sst_daily.time.dt.dayofyear

    # Step 2: Select matching climatology values for each day of the year
    sst_climatology_expanded = sst_clim_mean.sortby('dayofyear').sel(dayofyear=sst_daily_dayofyear)

     # Step 3: Calculate the anomaly by subtracting the climatology from the original dataset
    sst_anomaly = sst_daily.sortby('time') - sst_climatology_expanded.sortby('time')
    sst_anomaly = sst_anomaly.sortby('time')
    sst_anom_7 = sst_anomaly.isel(time=slice(-7,None)).mean(dim='time')
    sst_anom_814 = sst_anomaly.isel(time=slice(-14,-7)).mean(dim='time')
    sst_anom_diff = sst_anom_7 - sst_anom_814

#collapse all dask graph layers to one to conserve space
sst_anom_7 = sst_anom_7.persist()
sst_anom_diff = sst_anom_diff.persist()

sst_anom_7_crs = sst_anom_7.rio.write_crs('EPSG:4326', inplace=True)
sst_anom_diff_crs = sst_anom_diff.rio.write_crs('EPSG:4326', inplace=True)

sst_anom_diff_crs.sst.plot()
sst_anom_7_crs = sst_anom_7_crs.rename({'lat':'y', 'lon':'x'})
sst_anom_diff_crs = sst_anom_diff_crs.rename({'lat':'y', 'lon':'x'})
# sst_anom7_mercator = sst_anom_7_crs.rio.reproject('EPSG:3857')
sst_anom_7_clipped = sst_anom_7_crs.rio.clip_box(minx=0, miny=-85, maxx=360, maxy=85)
sst_anom7_mercator = sst_anom_7_clipped.rio.reproject("EPSG:3857")
sst_anom_diff_clipped = sst_anom_diff_crs.rio.clip_box(minx=0, miny=-85, maxx=360, maxy=85)
sst_anomdiff_mercator = sst_anom_diff_clipped.rio.reproject("EPSG:3857")

# Normalize the data to the desired range [-2, 2] (for example, SST anomalies)
vmin, vmax = -2, 2
# Clip values outside of the desired range (optional, but useful for extreme values)
sst_7norm = np.clip(sst_anom7_mercator.sst, vmin, vmax)
sst_7diffnorm = np.clip(sst_anomdiff_mercator.sst, vmin, vmax)

# Normalize the data so that it fits between 0 and 1
sst_7normalized = (sst_7norm - vmin) / (vmax - vmin)
sst_7diffnormalized = (sst_7diffnorm - vmin) / (vmax - vmin)

# Apply a color map (e.g., "RdBu" for anomalies)
cmap = plt.get_cmap("RdBu")
# Apply the color map to the normalized data
sst_7rgb = cmap(sst_7normalized)[:, :, :3]  # Extract RGB channels (ignore alpha channel)
sst_7rgb = (sst_7rgb * 255).astype(np.uint8)  # Scale to 0-255 for image representation
sst_7diffrgb = cmap(sst_7diffnormalized)[:, :, :3]  # Extract RGB channels (ignore alpha channel)
sst_7diffrgb = (sst_7diffrgb * 255).astype(np.uint8)  # Scale to 0-255 for image representation

# Convert to xarray for export
sst_7rgb_xarray = xr.DataArray(sst_7rgb, dims=("y", "x", "band"), 
                               coords={"y": sst_anom7_mercator.y, "x": sst_anom7_mercator.x, "band": [1, 2, 3]})
sst_7rgb_xarray = sst_7rgb_xarray.transpose("band", "y", "x")
sst_7rgb_xarray = sst_7rgb_xarray.rio.write_crs(sst_anom7_mercator.rio.crs)

sst_7diffrgb_xarray = xr.DataArray(sst_7diffrgb, dims=("y", "x", "band"),
                                   coords={"y": sst_anomdiff_mercator.y, "x": sst_anomdiff_mercator.x, "band": [1, 2, 3]})
sst_7diffrgb_xarray = sst_7diffrgb_xarray.transpose("band", "y", "x")
sst_7diffrgb_xarray = sst_7diffrgb_xarray.rio.write_crs(sst_anomdiff_mercator.rio.crs)

#check crs is right
sst_7rgb_xarray = sst_7rgb_xarray.rio.write_crs("EPSG:3857")
sst_7diffrgb_xarray = sst_7diffrgb_xarray.rio.write_crs("EPSG:3857")
# Export to GeoTIFF
sst_7rgb_xarray.rio.to_raster(os.path.join(figure_dir, 'sst_mercator7.png'), nodata=np.nan)
sst_7diffrgb_xarray.rio.to_raster(os.path.join(figure_dir, 'sst_mercator7diff.png'), nodata=np.nan)