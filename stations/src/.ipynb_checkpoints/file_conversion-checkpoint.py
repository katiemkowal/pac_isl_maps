import os
import numpy as np
import xarray as xr

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