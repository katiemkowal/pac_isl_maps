import xcast as xc
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import pacisl_config as cfg
import os

fmodel1 = xr.open_dataset('./gefs_week34_cons_tercile.nc')
fmodel = fmodel1.precip
fmodel = fmodel.sel(lon=slice(cfg.final_predictand_zone['west'],
cfg.final_predictand_zone['east']), lat=slice(cfg.final_predictand_zone['south'],cfg.final_predictand_zone['north']))

print (fmodel.shape)
from mprob import mview_probabilistic
mview_probabilistic(fmodel.isel(time=0), title='GEFS, Week-34, S. Weighted Cons., Valid: 05Jun2024 - 18Jun2024', savefig=os.path.join(cfg.output_path, './gefs_week_34_cons.png'))