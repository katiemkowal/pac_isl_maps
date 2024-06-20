#!/bin/sh

wdir=/cpc/int_desk/endalk/PREPARE_pacific/seasonal/prep_data
cd $wdir
grads=/cpc/home/ebekele/grads2.1/grads-2.1.0/bin/grads
py=/cpc/home/ebekele/.conda/envs/xcast_env/bin/python

mn=`date +"%b"`

yrmndy=`date +"%Y"-"%m"-"%d"`
yrmondy=`date +'%Y, %-m, %-d'`
mon=$(date +%b | tr A-Z a-z)

if [ $mon == "jan" ]; then mn1='dec'; mn2='jan'; mn3='feb'; fi
if [ $mon == "feb" ]; then mn1='jan'; mn2='feb'; mn3='mar'; fi
if [ $mon == "mar" ]; then mn1='feb'; mn2='mar'; mn3='apr'; fi
if [ $mon == "apr" ]; then mn1='mar'; mn2='apr'; mn3='may'; fi
if [ $mon == "mAy" ]; then mn1='apr'; mn2='may'; mn3='jun'; fi
if [ $mon == "jun" ]; then mn1='may'; mn2='jun'; mn3='jul'; fi
if [ $mon == "jul" ]; then mn1='jun'; mn2='jul'; mn3='aug'; fi
if [ $mon == "aug" ]; then mn1='jul'; mn2='aug'; mn3='sep'; fi
if [ $mon == "sep" ]; then mn1='aug'; mn2='sep'; mn3='oct'; fi
if [ $mon == "oct" ]; then mn1='sep'; mn2='oct'; mn3='nov'; fi
if [ $mon == "nov" ]; then mn1='oct'; mn2='nov'; mn3='dec'; fi
if [ $mon == "dec" ]; then mn1='nov'; mn2='dec'; mn3='jan'; fi

for ld in {1..3}; do

rm nmmec_hind_sst_ld_${ld}.dat
# Generate NMME hindcast data
cat>nmmec_hind.gs<<eofGS
'reinit'
'open /cpc/int_desk/NMME/hindcast/raw_sst_precip_tmp2m/tmpsfc_monthly/${mn1}ic_ENSM_MEAN_1991-2022.ctl'
'open /cpc/int_desk/NMME/hindcast/raw_sst_precip_tmp2m/tmpsfc_monthly/${mn2}ic_ENSM_MEAN_1991-2022.ctl'
'open /cpc/int_desk/NMME/hindcast/raw_sst_precip_tmp2m/tmpsfc_monthly/${mn3}ic_ENSM_MEAN_1991-2022.ctl'
'set lat -90 90'
'set lon -180 180'
zz = ${ld} + 1
'set gxout fwrite'
'set fwrite nmmec_hind_sst_ld_${ld}.dat'
i=8
while(i<=32)
'set t 'i
'set dfile 1'
'define tt = ave(fcst.1,z='zz+0',z='zz+2')-273.14'
'd tt'
'set dfile 2'
'define tt = ave(fcst.2,z='zz+0',z='zz+2')-273.14'
'd tt'
'set dfile 3'
'define tt = ave(fcst.3,z='zz+0',z='zz+2')-273.14'
'd tt'
i = i + 1
endwhile
'disable fwrite'
'quit'
eofGS

$grads -blc nmmec_hind.gs

rm nmmec_fcst_sst_ld_${ld}.dat
# Generate NMME current forecast data
cat>nmme_fcst.gs<<eofGS
'reinit'
'open /cpc/fews/production/NMME/inputs/filtered/nmme_tmpsfc_ensmean_fcst.ctl'
'set lat -90 90'
'set lon -180 180'
zz = ${ld} + 1
'set z 'zz
'set gxout fwrite'
'set fwrite nmme_fcst_sst_ld_${ld}.dat'

'define tt = ave(fcst,z='zz+0',z='zz+2')'
'd tt'
'disable fwrite'
'quit'
eofGS

$grads -blc nmme_fcst.gs

cat>gen_nmmec_hind_sst.py<<eofPY
import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta

f1 = "nmmec_hind_sst_ld_${ld}.dat"

# Predictor spatial dimension (Global tropics)
lats = -90; latn = 90; lonw = -180; lone = 180

res1 = 1.0 # Predictor horizontal resolution

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(lats,latn+res1,res1); ny = len(nlat);
nlon = np.arange(lonw,lone+res1,res1); nx = len(nlon);

nt = 75
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

ncfile = netCDF4.Dataset('nmmec_hind_sst_ld${ld}.nc',mode='w',format='NETCDF4_CLASSIC')
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
lat[:] = lats + 1.0*np.arange(nlat)
lon[:] = lonw + 1.0*np.arange(nlon)
sst[:,:,:] = sstt # Appends data along unlimited dimension
eofPY

$py gen_nmmec_hind_sst.py

rm data/nmme_fcst_sst_ld${ld}.nc
cat>gen_nmme_fcst_sst.py<<eofPY
import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta

f2 = "nmme_fcst_sst_ld_${ld}.dat"

# Predictor spatial dimension (Global tropics)
lats = -90; latn = 90; lonw = -180; lone = 180

res1 = 1.0 # Predictor horizontal resolution

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(lats,latn+res1,res1); ny = len(nlat);
nlon = np.arange(lonw,lone+res1,res1); nx = len(nlon);

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
lat[:] = lats + 1.0*np.arange(nlat)
lon[:] = lonw + 1.0*np.arange(nlon)
sst[:,:,:] = sstt # Appends data along unlimited dimension
eofPY

$py gen_nmme_fcst_sst.py

done

mv *.nc data/
