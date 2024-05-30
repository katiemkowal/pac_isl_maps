import xcast as xc
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import os
import pacisl_config as cfg
from datetime import datetime

start=datetime.now()

model1 = xr.open_dataset(os.path.join(cfg.hcst_dir, 'gefs_week34_hind.nc'))
obs1 = xr.open_dataset(os.path.join(cfg.obs_dir, 'chirps_week34_hind.nc'))
fmodel1 = xr.open_dataset(os.path.join(cfg.fcst_dir, 'gefs_week34_fcst.nc'))
modelt = model1.precip.expand_dims({'M':[0]})
fmodelt = fmodel1.precip.expand_dims({'M':[0]})
obs = obs1.precip.expand_dims({'M':[0]})
drymask = xc.drymask(obs, dry_threshold= cfg.dry_threshold, quantile_threshold=cfg.quantile_threshold)
obs = obs*drymask
drymask = xc.drymask(modelt, cfg.dry_threshold, cfg.quantile_threshold)
modelt = modelt*drymask

#obs, model = xc.match(obs, model)

mask_missing = modelt.mean('time', skipna=False).mean('M', skipna=False)
mask_missing = xr.ones_like(mask_missing).where(~np.isnan(mask_missing), other=np.nan )
modelt = modelt * mask_missing

model = modelt.sel(lon=slice(170, 
                             179),
                   lat=slice(-30,
                             -13))
fmodel = fmodelt.sel(lon=slice(170, 
                             179),
                   lat=slice(-30,
                             -13))
obs = obs.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))

model = xc.regrid(model, obs.lon, obs.lat)
fmodel = xc.regrid(fmodel, obs.lon, obs.lat)

print(model)
print(obs)

i=1
for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model, obs, window=cfg.lov_window):
    print("window {}".format(i))
    i += 1
    reg = xc.CCA(search_override=(cfg.x_modes,
                                  cfg.y_modes,
                                 cfg.cca_modes))
    reg.fit(xtrain, ytrain)

fprobs =  reg.predict_proba(fmodel)

os.remove(os.path.join(cfg.tmp_path,'temp.nc'))
os.remove(os.path.join(cfg.tmp_path, 'temp_cca.nc'))
fr = fprobs.assign_coords(M=([1, 2, 3]))
fr = fr.rename({'M' : 'e'})
#tt = fr.to_dataset(name = 'prob')
#print(tt)
#tt.to_netcdf(os.path.join(cfg.tmp_path, 'temp_cca.nc'))

fr.to_netcdf(os.path.join(cfg.tmp_path, 'temp.nc'))
tt = xr.open_dataset(os.path.join(cfg.tmp_path, 'temp.nc'))
ttt = tt.rename({'predicted_values' : 'prob'})
ttt.to_netcdf(os.path.join(cfg.tmp_path, 'temp_cca.nc'))

model = modelt.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))
fmodel = fmodelt.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))
obs = obs.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))
print(obs)
                             
model = xc.regrid(model, obs.lon, obs.lat)
fmodel = xc.regrid(fmodel, obs.lon, obs.lat)

obs, model = xc.match(obs, model)

i=1
for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model, obs, window=cfg.lov_window):
    print("window {}".format(i))
    i += 1
    reg = xc.ELR()
    reg.fit(xtrain, ytrain)

fprobs =  reg.predict_proba(fmodel)

os.remove(os.path.join(cfg.tmp_path, 'temp.nc'))
os.remove(os.path.join(cfg.tmp_path, 'temp_elr.nc'))
fr = fprobs.assign_coords(M=([1, 2, 3]))
fr = fr.rename({'M' : 'e'})
#print(fr)
#tt = fr.to_dataset(name = 'prob')
#print(tt)
#tt.to_netcdf(os.path.join(cfg.tmp_path, 'temp_cca.nc'))

fr.to_netcdf(os.path.join(cfg.tmp_path,'temp.nc'))
tt = xr.open_dataset(os.path.join(cfg.tmp_path,'temp.nc'))
ttt = tt.rename({'predicted_probability' : 'prob'})
#print(ttt)
ttt.to_netcdf(os.path.join(cfg.tmp_path,'temp_elr.nc'))

i=1
for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model, obs, window=cfg.lov_window):
    print("window {}".format(i))
    i += 1
    reg = xc.EPOELM()
    reg.fit(xtrain, ytrain)

fprobs =  reg.predict_proba(fmodel)

os.remove(os.path.join(cfg.tmp_path, 'temp.nc'))
os.remove(os.path.join(cfg.tmp_path, 'temp_epoelm.nc'))
fr = fprobs.assign_coords(M=([1, 2, 3]))
fr = fr.rename({'M' : 'e'})
#print(fr)
#tt = fr.to_dataset(name = 'prob')
#print(tt)
#tt.to_netcdf(os.path.join(cfg.tmp_path,'temp_epoelm.nc'))

fr.to_netcdf(os.path.join(cfg.tmp_path,'temp.nc'))
tt = xr.open_dataset(os.path.join(cfg.tmp_path,'temp.nc'))
ttt = tt.rename({'predicted_probability' : 'prob'})
ttt.to_netcdf(os.path.join(cfg.tmp_path,'temp_epoelm.nc'))

os.remove(os.path.join(cfg.tmp_path, 'temp_rpss_cca.nc'))
model = modelt.sel(lon=slice(170, 
                             179),
                   lat=slice(-30,
                             -13))
fmodel = fmodelt.sel(lon=slice(170, 
                             179),
                   lat=slice(-30,
                             -13))
obs = obs.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))
model = xc.regrid(model, obs.lon, obs.lat)
fmodel = xc.regrid(fmodel, obs.lon, obs.lat)

obs = xc.gaussian_smooth(obs)
ohc = xc.OneHotEncoder()
ohc.fit(obs)
T = ohc.transform(obs)
clim = xr.ones_like(T) * 0.333

reg = xc.CCA(search_override=(cfg.x_modes,
                              cfg.y_modes,
                             cfg.cca_modes))
reg.fit(model, obs)
probs =  reg.predict_proba(model)
probs = xc.gaussian_smooth(probs)

clim_rps = xc.RankProbabilityScore(clim, T)
pred_rps = xc.RankProbabilityScore(probs, T)
rpss = 1 - pred_rps / clim_rps
rpss.to_netcdf(os.path.join(cfg.tmp_path,'temp_rpss_cca.nc'))


obs, model = xc.match(obs, modelt)

mask_missing = modelt.mean('time', skipna=False).mean('M', skipna=False)
mask_missing = xr.ones_like(mask_missing).where(~np.isnan(mask_missing), other=np.nan )
modelt = modelt * mask_missing
model = modelt.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))
fmodel = fmodelt.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))
obs = obs.sel(lon=slice(175, 
                             179),
                   lat=slice(-20,
                             -16))
model = xc.regrid(model, obs.lon, obs.lat)
fmodel = xc.regrid(fmodel, obs.lon, obs.lat)

obs = xc.gaussian_smooth(obs)
ohc = xc.OneHotEncoder()
ohc.fit(obs)
T = ohc.transform(obs)
clim = xr.ones_like(T) * 0.333

os.remove(os.path.join(cfg.tmp_path,'temp_rpss_elr.nc'))
reg = xc.ELR()
reg.fit(model, obs)
probs =  reg.predict_proba(model)
probs = xc.gaussian_smooth(probs)

clim_rps = xc.RankProbabilityScore(clim, T)
pred_rps = xc.RankProbabilityScore(probs, T)
rpss = 1 - pred_rps / clim_rps
rpss.to_netcdf(os.path.join(cfg.tmp_path,'temp_rpss_elr.nc'))

os.remove(os.path.join(cfg.tmp_path, 'temp_rpss_epoelm.nc'))
reg = xc.EPOELM()
reg.fit(model, obs)
probs =  reg.predict_proba(model)
probs = xc.gaussian_smooth(probs)

clim_rps = xc.RankProbabilityScore(clim, T)
pred_rps = xc.RankProbabilityScore(probs, T)
rpss = 1 - pred_rps / clim_rps
rpss.to_netcdf(os.path.join(cfg.tmp_path,'temp_rpss_epoelm.nc'))

print(datetime.now()-start)
