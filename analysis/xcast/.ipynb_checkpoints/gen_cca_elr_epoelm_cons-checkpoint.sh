cd /cpc/int_desk/pac_isl/analysis/xcast/

#######large prediction zone for models
#central america / carib coordinates
# large_w=-95
# large_e=-48
# large_s=0
# large_n=38

#pacific island coordinates
large_w=170
large_e=179
large_s=-30
large_n=-13

######final prediction zone
#central america / carib coordinates
# pred_w=-85
# pred_e=-58
# pred_s=10
# pred_n=28

#pacific island coordinates
pred_w=175
pred_e=179
pred_s=-20
pred_n=-16

#######to setup nc files for .ctl docs

#fraction to divide by to get correct predictor resolution (2 for 0.5 res)
multiplier_x=2
multiplier_y=2

predictor_res_x=$(echo "scale=1;1/$multiplier_x" | bc)
predictor_res_y=$(echo "scale=1;1/$multiplier_y" | bc)

nc_lon=$(((${pred_w}-${pred_e})*${multiplier_x}))
nc_lat=$(((${pred_n}-${pred_s})*${multiplier_y}))
nc_lat=$((${nc_lat/-/}+1))
nc_lon=$((${nc_lon/-/}+1))
echo $nc_lat
echo $nc_lon

dtt=`date --date "0 day ago" "+%Y%m%d"`
dt=`date --date "0 day ago" "+%d%b%Y"`
iwk1=`date --date "-1 day ago" "+%d%b%Y"`
iwk2=`date --date "-8 day ago" "+%d%b%Y"`
iwk3=`date --date "-15 day ago" "+%d%b%Y"`
iwk34=`date --date "-15 day ago" "+%d%b%Y"`
iwk1234=`date --date "-1 day ago" "+%d%b%Y"`

fwk1=`date --date "-7 day ago" "+%d%b%Y"`
fwk2=`date --date "-14 day ago" "+%d%b%Y"`
fwk3=`date --date "-21 day ago" "+%d%b%Y"`
fwk34=`date --date "-28 day ago" "+%d%b%Y"`
fwk1234=`date --date "-28 day ago" "+%d%b%Y"`

for wk in 1 2 3 34; do

if [ $wk = 1 ]; then iwk=$iwk1; fwk=$fwk1; fi
if [ $wk = 2 ]; then iwk=$iwk2; fwk=$fwk2; fi
if [ $wk = 3 ]; then iwk=$iwk3; fwk=$fwk3; fi
if [ $wk = 34 ]; then iwk=$iwk34; fwk=$fwk34; fi
if [ $wk = 1234 ]; then iwk=$iwk1234; fwk=$fwk1234; fi

mmndy=`date +"%m"-"%d"`
mondy=`date +'%-m, %-d'`
mndy=`date -d "1 day ago" "+%m%d"`
yr=`date -d "1 day ago" "+%Y"`
mn=`date -d "1 day ago" "+%m"`
dy=`date -d "1 day ago" "+%d"`

wkn=`date '+%W'`

cp /cpc/africawrf/ebekele/cca/xcast/subseason/notebooks_caribb/cons/tmp/*temp*.nc .

cat>gen_cons.py<<eofPY
import xcast as xc
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import os
import pacisl_config as cfg
from datetime import datetime

start=datetime.now()

model1 = xr.open_dataset(os.path.join(cfg.hcst_dir, 'gefs_week${wk}_hind.nc'))
obs1 = xr.open_dataset(os.path.join(cfg.obs_dir, 'chirps_week${wk}_hind.nc'))
fmodel1 = xr.open_dataset(os.path.join(cfg.fcst_dir, 'gefs_week${wk}_fcst.nc'))
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

model = modelt.sel(lon=slice(${large_w}, 
                             ${large_e}),
                   lat=slice(${large_s},
                             ${large_n}))
fmodel = fmodelt.sel(lon=slice(${large_w}, 
                             ${large_e}),
                   lat=slice(${large_s},
                             ${large_n}))
obs = obs.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))

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

model = modelt.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))
fmodel = fmodelt.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))
obs = obs.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))
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
model = modelt.sel(lon=slice(${large_w}, 
                             ${large_e}),
                   lat=slice(${large_s},
                             ${large_n}))
fmodel = fmodelt.sel(lon=slice(${large_w}, 
                             ${large_e}),
                   lat=slice(${large_s},
                             ${large_n}))
obs = obs.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))
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
model = modelt.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))
fmodel = fmodelt.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))
obs = obs.sel(lon=slice(${pred_w}, 
                             ${pred_e}),
                   lat=slice(${pred_s},
                             ${pred_n}))
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
eofPY

#/cpc/home/kkowal/.conda/envs/xcast_env/bin/python gen_cons.py
/cpc/home/ebekele/.conda/envs/xcast_env/bin/python gen_cons.py

cat>temp_cca.ctl<<eofCTL
dset temp_cca.nc
title model
undef 9.96921e+36
dtype netcdf
xdef ${nc_lon} linear ${pred_w} ${predictor_res_x}
ydef ${nc_lat} linear ${pred_s} ${predictor_res_y}
zdef 1 linear 0 1
tdef 1 linear 00Z10Oct2023 1mn
edef 3
1 1 00Z10Oct2023
2 1 00Z10Oct2023
3 1 00Z10Oct2023
endedef
vars 1
prob=>prob  0  e,t,y,x  probability
endvars
eofCTL

for mdl in 'elr' 'epoelm'; do

cat>temp_${mdl}.ctl<<eofCTL
dset temp_${mdl}.nc
title model
undef 9.96921e+36
dtype netcdf
xdef ${nc_lon} linear ${pred_w} ${predictor_res_x}
ydef ${nc_lat} linear ${pred_s} ${predictor_res_y}
zdef 1 linear 0 1
tdef 1 linear 00Z10Oct2023 1mn
edef 3
1 1 00Z10Oct2023
2 1 00Z10Oct2023
3 1 00Z10Oct2023
endedef
vars 1
prob=>prob  0  y,x,t,e  probability
endvars
eofCTL
done

for mdl in 'cca' 'elr' 'epoelm'; do

cat>temp_rpss_${mdl}.ctl<<eofCTL
dset temp_rpss_${mdl}.nc
title model
undef 9.96921e+36
dtype netcdf
xdef ${nc_lon} linear ${pred_w} ${predictor_res_x}
ydef ${nc_lat} linear ${pred_s} ${predictor_res_y}
zdef 1 linear 0 1
tdef 1 linear 00Z10Oct2023 1mn
vars 1
RankProbabilityScore=>prob  0  y,x  probability
endvars
eofCTL
done

cat>gen_cons.gs<<eofGS
'reinit'
'open temp_cca.ctl'
'set lat ${pred_s} ${pred_n}'
'set lon ${pred_w} ${pred_e}'
'define bncca = prob(e=1)'
'define nncca = prob(e=2)'
'define ancca = prob(e=3)'
'close 1'
'open temp_elr.ctl'
'set lat ${pred_s} ${pred_n}'
'set lon ${pred_w} ${pred_e}'
'define bnelr = prob(e=1)'
'define nnelr = prob(e=2)'
'define anelr = prob(e=3)'
'close 1'
'open temp_epoelm.ctl'
'set lat ${pred_s} ${pred_n}'
'set lon ${pred_w} ${pred_e}'
'define bnepoelm = prob(e=1)'
'define nnepoelm = prob(e=2)'
'define anepoelm = prob(e=3)'
'close 1'
'open temp_rpss_cca.ctl'
'set lat ${pred_s} ${pred_n}'
'set lon ${pred_w} ${pred_e}'
'define rcca = prob'
'define scca = prob*prob'
'close 1'
'open temp_rpss_elr.ctl'
'set lat ${pred_s} ${pred_n}'
'set lon ${pred_w} ${pred_e}'
'define relr = prob'
'define selr = prob*prob'
'close 1'
'open temp_rpss_epoelm.ctl'
'set lat ${pred_s} ${pred_n}'
'set lon ${pred_w} ${pred_e}'
'define repoelm = prob'
'define sepoelm = prob*prob'

'define rr1 = const(const(maskout(rcca,rcca),1),-1,-u)'
'define rr2 = const(const(maskout(relr,relr),1),-1,-u)'
'define rr3 = const(const(maskout(repoelm,repoelm),1),-1,-u)'

'define rr = rr1 + rr2 + rr3'

'define consan = (scca*ancca) + (selr*anelr) + (sepoelm*anepoelm) /(scca + selr + sepoelm)'
'define consnn = (scca*nncca) + (selr*nnelr) + (sepoelm*nnepoelm) /(scca + selr + sepoelm)'
'define consbn = (scca*bncca) + (selr*bnelr) + (sepoelm*bnepoelm) /(scca + selr + sepoelm)'

'define pconsan = maskout((consan /(consan+consnn+consbn)),rr-0.05)'
'define pconsnn = maskout((consnn /(consan+consnn+consbn)),rr-0.05)'
'define pconsbn = maskout((consbn /(consan+consnn+consbn)),rr-0.05)'

'set gxout fwrite'
'set fwrite cons_tercile.dat'

'd re(pconsbn,${nc_lon},linear,${pred_w},${predictor_res_x},${nc_lat},linear,${pred_s},${predictor_res_y},ba)'
'd re(pconsnn,${nc_lon},linear,${pred_w},${predictor_res_x},${nc_lat},linear,${pred_s},${predictor_res_y},ba)'
'd re(pconsan,${nc_lon},linear,${pred_w},${predictor_res_x},${nc_lat},linear,${pred_s},${predictor_res_y},ba)'
'quit'
eofGS

pperl="/cpc/africawrf/ebekele/perl/bin/perl"

$pperl /cpc/home/ebekele/opengrads-2.2.1.oga.1/Contents/grads -blc gen_cons.gs

cat>cons_tercile.py<<eofPY
import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta
import pacisl_config as cfg

f1 = "cons_tercile.dat"

#res1 = ${predictor_res_x} # Predictor horizontal resolution
#res2 = ${predictor_res_y} # Predictor vertical resolution 

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(${pred_s},${pred_n}+0.5,${predictor_res_y}); ny = len(nlat);
nlon = np.arange(${pred_w}, ${pred_e}+0.5,${predictor_res_x}); nx = len(nlon);

nn = 3
nt = 1
ntime = nt
nlat = ny
nlon = nx
nnum = nn

fid = open(f1, 'rb');
print(fid)
precipt = np.zeros( (nn, nt, ny, nx) );
t = 0
for ts in range(nn):
    precipt[t,:,:] = np.reshape(np.fromfile(fid,dtype='<f',count=ny*nx),(ny,nx));
    t += 1;
fid.close();

precipt[precipt <= -999] = np.nan


ncfile = netCDF4.Dataset('gefs_week${wk}_cons_tercile.nc',mode='w',format='NETCDF4_CLASSIC')
lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
M_dim = ncfile.createDimension('M', nnum) # Ensemble axis

lat = ncfile.createVariable('lat', 'f8', ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', 'f8', ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
M = ncfile.createVariable('M', 'f8', ('M',))
M.units = 'e'
M.axis = 'e'
M.long_name = 'Ens_Number'


units = 'days since ${yr}-$mmndy'
calendar = 'proleptic_gregorian'
time = ncfile.createVariable('time', np.float64, ('time',))
time.long_name = 'time'
time.units = 'days since ${yr}-${mmndy} 00:00:00'
time.calendar = 'proleptic_gregorian'
time.axis = 'T'
times = [datetime.datetime(${yr}, ${mondy}) + relativedelta(years=x) for x in range(0,nt)]
time[:] = netCDF4.date2num(times, units=units, calendar=calendar)
precip = ncfile.createVariable('precip',np.float64,('M', 'time', 'lat','lon')) # note: unlimited dimension is leftmost
precip.units = 'mm' #
precip.standard_name = 'Sea_surface_temperature' # this is a CF standard name
nlats = len(lat_dim); nlons = len(lon_dim); ntimes = nt; nlons=nn
time[:] = netCDF4.date2num(times, units=time.units, calendar=time.calendar)
lat[:] = ${pred_s} + ${predictor_res_y}*np.arange(nlat)
lon[:] = ${pred_w} + ${predictor_res_x}*np.arange(nlon)
M[:] = 1 + np.arange(nnum)
precip[:,:,:,:] = precipt # Appends data along unlimited dimension
eofPY

/cpc/prod/cpcwebapps/software/anaconda/bin/python cons_tercile.py

cat>plot_cons.py<<eofPY
import xcast as xc
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import pacisl_config as cfg
import os

fmodel1 = xr.open_dataset('./gefs_week${wk}_cons_tercile.nc')
fmodel = fmodel1.precip
fmodel = fmodel.sel(lon=slice(${pred_w+0.01},${pred_e-0.01}), lat=slice(${pred_s},${pred_n}))#.rename({'time':'T', 'lon': 'LON', 'lat': 'LAT'})

os.path.join(cfg.output_path,'gefs_week_${wk}_cons.png')

print (fmodel.isel(time=0))
from mprob import mview_probabilistic
mview_probabilistic(fmodel.isel(time=0), title='GEFS, Week-${wk}, S. Weighted Cons., Valid: ${iwk} - ${fwk}', savefig=os.path.join(cfg.output_path,'gefs_week_${wk}_cons.png'))
eofPY

#/cpc/home/kkowal/.conda/envs/xcast_env/bin/python plot_cons.py
/cpc/home/ebekele/.conda/envs/xcast_env/bin/python plot_cons.py

convert -trim ../../output/ens_cons/gefs_week_${wk}_cons.png ../../output/ens_cons/gefs_week_${wk}_cons.png
srcdir=/cpc/int_desk/pac_isl/output/ens_cons

#ssh -x ebekele@rzdm chmod -R 755 ../ftp/International/wk34/caribb_xcast/fig_dir
#scp $srcdir/gefs_week_${wk}_cons.png ebekele@cpcrzdm:"../ftp/International/wk34/caribb_xcast/fig_dir"
#ssh -x ebekele@rzdm chmod -R 755 ../ftp/International/wk34/caribb_xcast/fig_dir

done
