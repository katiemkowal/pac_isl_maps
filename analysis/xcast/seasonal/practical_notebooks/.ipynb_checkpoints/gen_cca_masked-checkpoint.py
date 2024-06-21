#!/usr/bin/env python
# coding: utf-8

import xcast as xc 
import xarray as xr 
import cartopy.crs as ccrs 
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping

model1 = xr.open_dataset('/cpc/africawrf/ebekele/projects/PREPARE_pacific/prep_data/data/gefs_week34_hind.nc')
obs1 = xr.open_dataset('/cpc/africawrf/ebekele/projects/PREPARE_pacific/prep_data/data/cmorph_week34_hind.nc')
fmodel1 = xr.open_dataset('/cpc/africawrf/ebekele/projects/PREPARE_pacific/prep_data/data/gefs_week34_fcst.nc')
msk = xr.open_dataset('/cpc/africawrf/ebekele/projects/PREPARE_pacific/notebooks/masked/libs/pacific_mask.nc')

model = model1.precip.expand_dims({'M':[0]})
fmodel = fmodel1.precip.expand_dims({'M':[0]})
obs = obs1.precip.expand_dims({'M':[0]})
mskk = msk.amask.expand_dims({'M':[0]})

obs = obs.assign_coords({'lon': [i + 360 if i <= 0 else i for i in obs.coords['lon'].values]}).sortby('lon').drop_duplicates('lon')
model = model.assign_coords({'lon': [i + 360 if i <= 0 else i for i in model.coords['lon'].values]}).sortby('lon').drop_duplicates('lon')
fmodel = fmodel.assign_coords({'lon': [i + 360 if i <= 0 else i for i in fmodel.coords['lon'].values]}).sortby('lon').drop_duplicates('lon')
mskk = mskk.assign_coords({'lon': [i + 360 if i <= 0 else i for i in mskk.coords['lon'].values]}).sortby('lon').drop_duplicates('lon')

drymask = xc.drymask(obs, dry_threshold=0.5, quantile_threshold=0.3)

# obs = xc.regrid(obs, mskk.lon, mskk.lat)
mskk = xc.regrid(mskk, obs.lon, obs.lat)
mask_missing = mskk.mean('time', skipna=False).mean('M', skipna=False)
mask_missing = xr.ones_like(mask_missing).where(~np.isnan(mask_missing), other=np.nan )
obs = obs * mask_missing

model = model.sel(lon=slice(122, 215), lat=slice(-32,19))
fmodel = fmodel.sel(lon=slice(122, 215), lat=slice(-32,19))
obs = obs.sel(lon=slice(132, 205), lat=slice(-22,9))

mask_missing = model.mean('time', skipna=False).mean('M', skipna=False)
mask_missing = xr.ones_like(mask_missing).where(~np.isnan(mask_missing), other=np.nan )
model = model * mask_missing

model = xc.regrid(model, obs.lon, obs.lat)
fmodel = xc.regrid(fmodel, obs.lon, obs.lat)

hindcasts_prob = []
i=1
for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model, obs, window=5):
    print("window {}".format(i))
    i += 1
    reg = xc.CCA(search_override=(5,5,3))
    reg.fit(xtrain, ytrain)
    probs =  reg.predict_proba(xtest)
    hindcasts_prob.append(probs.isel(time=2))
hindcasts_prob = xr.concat(hindcasts_prob, 'time')

hindcasts_prob = xc.gaussian_smooth(hindcasts_prob, kernel=3)
obs = xc.gaussian_smooth(obs, kernel=3)

ohc = xc.OneHotEncoder() 
ohc.fit(obs)
T = ohc.transform(obs)
clim = xr.ones_like(T) * 0.333

fprobs =  reg.predict_proba(fmodel)

fprobs20 =  reg.predict_proba(fmodel,quantile=0.2)
fprobs80 =  1-(reg.predict_proba(fmodel,quantile=0.8))
fprobs10 =  reg.predict_proba(fmodel,quantile=0.1)
fprobs90 =  1-(reg.predict_proba(fmodel,quantile=0.9))

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import sys
sys.path.append('./libs')

from pupper import puview
wet = ['80', '90']
for p in wet:
    fr = eval("fprobs"+p).transpose("lat", "lon", "time", "M")
    frr = fr.rename("Exceedance_Probability")
    frrr = frr.mean(dim=['time', 'M'])*100
    # view probabilistic and view, with cross-dateline=True will force back to (0,360) for plotting , won't change the original data tho
    ax = puview(frrr, x_feature_dim=None, coastlines=True, ocean=False, cross_dateline=True, title = f"GEFS, Week-34, CCA > {p}th, Valid: 05Jul2024 - 18Jul2024")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='blue', alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator([ 130, 140, 150, 160, 170, 180, -170, -160, -150, -140])
    gl.ylocator = mticker.FixedLocator([-25, -20, -15, -10, -5, 0, 5, 10])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}
    gl.top_labels = False
    gl.left_labels = True
    gl.right_labels=False
    gl.bottom_labels=True
    plt.savefig(f'fig_dir/gefs_week_34_cca_{p}.png', dpi=300)

from plower import puview
dry = ['10', '20']
for p in dry:
    fr = eval("fprobs"+p).transpose("lat", "lon", "time", "M")
    frr = fr.rename("Exceedance_Probability")
    frrr = frr.mean(dim=['time', 'M'])*100
    # view probabilistic and view, with cross-dateline=True will force back to (0,360) for plotting , won't change the original data tho
    ax = puview(frrr, x_feature_dim=None, ocean=False, cross_dateline=True, title = f"GEFS, Week-34, CCA < {p}th, Valid: 05Jul2024 - 18Jul2024")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='blue', alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator([ 130, 140, 150, 160, 170, 180, -170, -160, -150, -140])
    gl.ylocator = mticker.FixedLocator([-25, -20, -15, -10, -5, 0, 5, 10])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}
    gl.top_labels = False
    gl.left_labels = True
    gl.right_labels=False
    gl.bottom_labels=True
    #plt.savefig(f'fig_dir/gefs_week_34_cca_{p}.png', dpi=300)

from prob import pview_probabilistic
# view probabilistic and view, with cross-dateline=True will force back to (0,360) for plotting , won't change the original data tho
ax = pview_probabilistic(fprobs.isel(time=0), ocean=False, cross_dateline=True, title='GEFS, Week-34, CCA, Valid: 05Jul2024 - 18Jul2024')
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='blue', alpha=0.5, linestyle='--', draw_labels=True)
gl.top_labels = False
gl.left_labels = False
gl.right_labels=True
gl.xlines = True
gl.xlocator = mticker.FixedLocator([ 130, 140, 150, 160, 170, 180, -170, -160, -150, -140])
gl.ylocator = mticker.FixedLocator([-25, -20, -15, -10, -5, 0, 5, 10])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black'}
gl.top_labels = False
gl.left_labels = True
gl.right_labels=False
gl.bottom_labels=True
#plt.savefig('fig_dir/gefs_week_34_cca.png', dpi=300)

from prpssview import rview
clim_rps = xc.RankProbabilityScore(clim, T)
pred_rps = xc.RankProbabilityScore(hindcasts_prob, T)
rpss = 1 - pred_rps / clim_rps
ax = rview(rpss, cross_dateline=True, ocean=False, title='GEFS, Week-34, CCA-RPSS, Valid: 05Jul - 18Jul')
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl.xlocator = mticker.FixedLocator([ 130, 140, 150, 160, 170, 180, -170, -160, -150, -140])
gl.ylocator = mticker.FixedLocator([-25, -20, -15, -10, -5, 0, 5, 10])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black'} 
gl.top_labels = False
gl.left_labels = True
gl.right_labels=False
gl.bottom_labels=True
gl.xlines = True
#plt.savefig('fig_dir/gefs_week_34_cca_rpss.png')

from pgrocsview import rview
groc = xc.GROCS(hindcasts_prob, T)
ax = rview(groc, cross_dateline=True, ocean=False, title='GEFS, Week-34, CCA-GROCS, Valid: 05Jul - 18Jul')
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl.xlocator = mticker.FixedLocator([ 130, 140, 150, 160, 170, 180, -170, -160, -150, -140])
gl.ylocator = mticker.FixedLocator([-25, -20, -15, -10, -5, 0, 5, 10])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black'}
gl.top_labels = False
gl.left_labels = True
gl.right_labels=False
gl.bottom_labels=True
gl.xlines = True
#plt.savefig('fig_dir/gefs_week_34_cca_groc.png')


from rocc import view_mroc
ds = hindcasts_prob.assign_coords(M=(['BN', 'NN', 'AN']))
view_mroc(ds, T, savefig='fig_dir/gefs_week_34_cca_roc.png')

xc.view_reliability(hindcasts_prob, T)
plt.savefig('fig_dir/gefs_week_34_cca_relib.png', dpi=100)

import os
fr = fprobs.assign_coords(M=([0, 1, 2]))
frr = fr.transpose("lat", "lon", "time", "M")
ttt = frr.rename("prob")
#ttt.to_netcdf('./gefs_week34_cca.nc')

ttt = rpss.rename("skill")
#ttt.to_netcdf('./gefs_week34_cca_rpss.nc')

ttt = groc.rename("skill")
#ttt.to_netcdf('./gefs_week34_cca_groc.nc')
