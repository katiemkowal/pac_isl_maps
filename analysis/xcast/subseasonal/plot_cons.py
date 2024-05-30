import xcast as xc
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import pacisl_config as cfg
import os

fmodel1 = xr.open_dataset('./gefs_week34_cons_tercile.nc')
fmodel = fmodel1.precip
fmodel = fmodel.sel(lon=slice(0.01,179), lat=slice(-20,-16))#.rename({'time':'T', 'lon': 'LON', 'lat': 'LAT'})

os.path.join(cfg.output_path,'gefs_week_34_cons.png')

print (fmodel.isel(time=0))
from mprob import mview_probabilistic
mview_probabilistic(fmodel.isel(time=0), title='GEFS, Week-34, S. Weighted Cons., Valid: 14Jun2024 - 27Jun2024', savefig=os.path.join(cfg.output_path,'gefs_week_34_cons.png'))
