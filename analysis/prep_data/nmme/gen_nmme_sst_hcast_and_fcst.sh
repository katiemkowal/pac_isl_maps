#!/bin/sh
wdir=/cpc/int_desk/pac_isl/analysis/prep_data/nmme
datdir=/cpc/int_desk/pac_isl/data/processed/nmme/dat_files
ncdir=/cpc/int_desk/pac_isl/data/processed/nmme/nc_files

cd $wdir
grads=/cpc/home/ebekele/grads-2.1.0.oga.1//Contents/grads
py=/cpc/home/ebekele/.conda/envs/xcast_env/bin/python
pperl=/cpc/africawrf/ebekele/perl/bin/perl


mn=`date +"%b"`

yrmndy=`date +"%Y"-"%m"-"%d"`
yrmondy=`date +'%Y, %-m, %-d'`
mon=$(date +%b | tr A-Z a-z)

for ld in {1..3}; do

if [ $ld = 1 ]; then

if [ $mn == "Jan" ]; then prd='FMA'; fi
if [ $mn == "Feb" ]; then prd='MAM'; fi
if [ $mn == "Mar" ]; then prd='AMJ'; fi
if [ $mn == "Apr" ]; then prd='MJJ'; fi
if [ $mn == "May" ]; then prd='JJA'; fi
if [ $mn == "Jun" ]; then prd='JAS'; fi
if [ $mn == "Jul" ]; then prd='ASO'; fi
if [ $mn == "Aug" ]; then prd='SON'; fi
if [ $mn == "Sep" ]; then prd='OND'; fi
if [ $mn == "Oct" ]; then prd='NDJ'; fi
if [ $mn == "Nov" ]; then prd='DJF'; fi
if [ $mn == "Dec" ]; then prd='JFM'; fi
fi

if [ $ld = 2 ]; then

if [ $mn == "Jan" ]; then prd='MAM'; fi
if [ $mn == "Feb" ]; then prd='AMJ'; fi
if [ $mn == "Mar" ]; then prd='MJJ'; fi
if [ $mn == "Apr" ]; then prd='JJA'; fi
if [ $mn == "May" ]; then prd='JAS'; fi
if [ $mn == "Jun" ]; then prd='ASO'; fi
if [ $mn == "Jul" ]; then prd='SON'; fi
if [ $mn == "Aug" ]; then prd='OND'; fi
if [ $mn == "Sep" ]; then prd='NDJ'; fi
if [ $mn == "Oct" ]; then prd='DJF'; fi
if [ $mn == "Nov" ]; then prd='JFM'; fi
if [ $mn == "Dec" ]; then prd='FMA'; fi
fi

if [ $ld = 3 ]; then

if [ $mn == "Jan" ]; then prd='AMJ'; fi
if [ $mn == "Feb" ]; then prd='MJJ'; fi
if [ $mn == "Mar" ]; then prd='JJA'; fi
if [ $mn == "Apr" ]; then prd='JAS'; fi
if [ $mn == "May" ]; then prd='ASO'; fi
if [ $mn == "Jun" ]; then prd='SON'; fi
if [ $mn == "Jul" ]; then prd='OND'; fi
if [ $mn == "Aug" ]; then prd='NDJ'; fi
if [ $mn == "Sep" ]; then prd='DJF'; fi
if [ $mn == "Oct" ]; then prd='JFM'; fi
if [ $mn == "Nov" ]; then prd='FMA'; fi
if [ $mn == "Dec" ]; then prd='MAM'; fi
fi

if test -f ${datdir}/nmme_hind_sst_ld_${ld}.dat; then
    rm ${datdir}/nmme_hind_sst_ld_${ld}.dat
fi

if test -f ${ncdir}/nmme_hind_sst_ld_${ld}.nc; then
    rm ${ncdir}/nmme_hind_sst_ld_${ld}.nc
fi

if test -f ${ncdir}/nmme_fcst_sst_ld_${ld}.nc; then
    rm ${ncdir}/nmme_fcst_sst_ld_${ld}.nc
fi

cat>${mon}ic_ENSM_MEAN_1991-2022.ctl<<eofCTL
dset /cpc/int_desk/NMME/hindcast/raw_sst_precip_tmp2m/tmpsfc_monthly/${mon}ic_ENSM_MEAN_1991-2022.dat
undef 9.999E+20
title tmpsfc.bin
options little_endian
xdef 360 linear 0 1.0
ydef 181 linear -90.0 1.0
tdef 32 linear 15may1991 1yr
zdef 9 linear 1 1
vars 1
fcst 9,1,0   0,1,7,0 ** sst DegC
ENDVARS
eofCTL

# Generate NMME hindcast data
cat>nmme_hind.gs<<eofGS
'reinit'
'open ${mon}ic_ENSM_MEAN_1991-2022.ctl'
'set lat -90 90'
'set lon -180 180'
zz = ${ld} + 1 
'set gxout fwrite'
'set fwrite ${datdir}/nmme_hind_sst_ld_${ld}.dat'
i=1
while(i<=32)
'set t 'i
'define tt = ave(fcst,z='zz+0',z='zz+2')-273.14'
'd re(tt,361,linear,-180,1.0,181,linear,-90,1.0,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
eofGS

$pperl $grads -blc nmme_hind.gs

# Generate NMME current forecast data
cat>nmme_fcst.gs<<eofGS
'reinit'
'open nmme_tmpsfc_ensmean_fcst.ctl'
'set lat -90 90'
'set lon -180 180'
zz = ${ld} + 1
'set z 'zz
'set gxout fwrite'
'set fwrite ${datdir}/nmme_fcst_sst_ld_${ld}.dat'

'define tt = ave(fcst,z='zz+0',z='zz+2')'
'd re(tt,361,linear,-180,1.0,181,linear,-90,1.0,ba)'
'disable fwrite'
'quit'
eofGS

$pperl $grads -blc nmme_fcst.gs

cat>gen_nmme_hind_sst.py<<eofPY
import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta
import nmme_config as cfg

f1 = "${datdir}/nmme_hind_sst_ld_${ld}.dat"

# Predictor spatial dimension (Global tropics)
#lats = -90; latn = 90; lonw = -180; lone = 180

#res1 = 1.0 # Predictor horizontal resolution

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(cfg.lats,cfg.latn+cfg.h_res1,cfg.h_res1); ny = len(nlat);
nlon = np.arange(cfg.lonw,cfg.lone+cfg.h_res1,cfg.h_res1); nx = len(nlon);

nt = cfg.years
ntime = nt
nlat = ny
nlon = nx

fid = open(f1, 'rb');
sstt = np.zeros( (nt, ny, nx) );
t = 0
for ts in range(nt):
    sstt[t,:,:] = np.reshape(np.fromfile(fid,dtype='<f',count=ny*nx),(ny,nx));
    t += 1;
fid.close();

sstt[sstt <= -999] = np.nan

ncfile = netCDF4.Dataset('nmme_hind_sst_ld${ld}.nc',mode='w',format='NETCDF4_CLASSIC')
lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'

units = 'days since $yrmndy'
calendar = 'proleptic_gregorian'
time = ncfile.createVariable('time', np.float64, ('time',))
time.long_name = 'time'
time.units = 'days since ${yrmndy} 00:00:00'
time.calendar = 'proleptic_gregorian'
time.axis = 'T'
times = [datetime.datetime(${yrmondy}) + relativedelta(years=x) for x in range(0,nt)]
time[:] = netCDF4.date2num(times, units=units, calendar=calendar)
sst = ncfile.createVariable('sst',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
sst.units = 'K' # degrees Kelvin
sst.standard_name = 'Sea_surface_temperature' # this is a CF standard name
nlats = len(lat_dim); nlons = len(lon_dim); ntimes = nt
time[:] = netCDF4.date2num(times, units=time.units, calendar=time.calendar)
lat[:] = cfg.lats + 1.0*np.arange(nlat)
lon[:] = cfg.lonw + 1.0*np.arange(nlon)
sst[:,:,:] = sstt # Appends data along unlimited dimension
eofPY

$py gen_nmme_hind_sst.py


cat>gen_nmme_fcst_sst.py<<eofPY
import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta
import nmme_config as cfg

f2 = "${datdir}/nmme_fcst_sst_ld_${ld}.dat"

# Predictor spatial dimension (Global tropics)
#lats = -90; latn = 90; lonw = -180; lone = 180

#res1 = 1.0 # Predictor horizontal resolution

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(cfg.lats,cfg.latn+cfg.h_res1,cfg.h_res1); ny = len(nlat);
nlon = np.arange(cfg.lonw,cfg.lone+cfg.h_res1,cfg.h_res1); nx = len(nlon);

nt = 1
ntime = nt
nlat = ny
nlon = nx

fid = open(f2, 'rb');
sstt = np.zeros( (nt, ny, nx) );
t = 0
for ts in range(nt):
    sstt[t,:,:] = np.reshape(np.fromfile(fid,dtype='<f',count=ny*nx),(ny,nx));
    t += 1;
fid.close();

sstt[sstt <= -999] = np.nan

ncfile = netCDF4.Dataset('nmme_fcst_sst_ld${ld}.nc',mode='w',format='NETCDF4_CLASSIC')
lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'

units = 'days since $yrmndy'
calendar = 'proleptic_gregorian'
time = ncfile.createVariable('time', np.float64, ('time',))
time.long_name = 'time'
time.units = 'days since ${yrmndy} 00:00:00'
time.calendar = 'proleptic_gregorian'
time.axis = 'T'
times = [datetime.datetime(${yrmondy}) + relativedelta(years=x) for x in range(0,nt)]
time[:] = netCDF4.date2num(times, units=units, calendar=calendar)
sst = ncfile.createVariable('sst',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
sst.units = 'K' # degrees Kelvin
sst.standard_name = 'Sea_surface_temperature' # this is a CF standard name
nlats = len(lat_dim); nlons = len(lon_dim); ntimes = nt
time[:] = netCDF4.date2num(times, units=time.units, calendar=time.calendar)
lat[:] = cfg.lats + 1.0*np.arange(nlat)
lon[:] = cfg.lonw + 1.0*np.arange(nlon)
sst[:,:,:] = sstt # Appends data along unlimited dimension
eofPY

$py gen_nmme_fcst_sst.py

done

mv *.nc ${ncdir}