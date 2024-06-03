#!/bin/sh

wdir=/cpc/int_desk/pac_isl/analysis/prep_data/chirps
datdir=/cpc/int_desk/pac_isl/data/processed/chirps/dat_files
ncdir=/cpc/int_desk/pac_isl/data/processed/chirps/nc_files

cd $wdir
grads=/cpc/home/ebekele/grads-2.1.0.oga.1//Contents/grads
py=/cpc/home/ebekele/.conda/envs/xcast_env/bin/python
pperl=/cpc/africawrf/ebekele/perl/bin/perl

if test -f ${datdir}/chirps_ld1.dat; then
    rm ${datdir}/chirps*.dat
fi

if test -f ${ncdir}/chirps_ld1.nc; then
    rm ${ncdir}/chirps*.nc
fi


mn=`date +"%b"`
yrmndy=`date +"%Y"-"%m"-"%d"`
yrmondy=`date +'%Y, %-m, %-d'`
mon=$(date +%b | tr A-Z a-z)

cat>chirps.ctl<<eofCTL
DSET /cpc/fews/production/rshukla/CMORPH1/CMORPH1_BLD_EOD/CMORPH_V1.0BETA_BLD_0.25deg-DLY_EOD_%y4%m2%d2
OPTIONS template little_endian
UNDEF  -999.0
TITLE  gauge - CMORPH_Adj Blended Analysis
XDEF 1440 LINEAR   0.125  0.25
YDEF  720 LINEAR -89.875  0.25
ZDEF   01 LEVELS 1
TDEF 60000 LINEAR  01jan1998 1dy
VARS 1
r    1  99  blended daily precip (mm) ending at EOD
ENDVARS
eofCTL

echo $mon

for ld in {1..3}; do

if [ $ld = 1 ]; then

if [ $mn == "Jan" ]; then prd='FMA'; moni='Feb'; monf='Apr';ln=89;emon=30; fi
if [ $mn == "Feb" ]; then prd='MAM'; moni='Mar'; monf='May';ln=92;emon=31; fi
if [ $mn == "Mar" ]; then prd='AMJ'; moni='Apr'; monf='Jun';ln=91;emon=30; fi
if [ $mn == "Apr" ]; then prd='MJJ'; moni='May'; monf='Jul';ln=92;emon=31; fi
if [ $mn == "May" ]; then prd='JJA'; moni='Jun'; monf='Aug';ln=92;emon=31; fi
if [ $mn == "Jun" ]; then prd='JAS'; moni='Jul'; monf='Sep';ln=92;emon=30; fi
if [ $mn == "Jul" ]; then prd='ASO'; moni='Aug'; monf='Oct';ln=92;emon=31; fi
if [ $mn == "Aug" ]; then prd='SON'; moni='Sep'; monf='Nov';ln=91;emon=30; fi
if [ $mn == "Sep" ]; then prd='OND'; moni='Oct'; monf='Dec';ln=92;emon=31; fi
if [ $mn == "Oct" ]; then prd='NDJ'; moni='Nov'; monf='Jan';ln=92;emon=31; fi
if [ $mn == "Nov" ]; then prd='DJF'; moni='Dec'; monf='Feb';ln=90;emon=28; fi
if [ $mn == "Dec" ]; then prd='JFM'; moni='Jan'; monf='Mar';ln=90;emon=31; fi
fi

if [ $ld = 2 ]; then

if [ $mn == "Jan" ]; then prd='MAM'; moni='Mar'; monf='May';ln=92;emon=31; fi
if [ $mn == "Feb" ]; then prd='AMJ'; moni='Apr'; monf='Jun';ln=91;emon=30; fi
if [ $mn == "Mar" ]; then prd='MJJ'; moni='May'; monf='Jul';ln=92;emon=31; fi
if [ $mn == "Apr" ]; then prd='JJA'; moni='Jun'; monf='Aug';ln=92;emon=31; fi
if [ $mn == "May" ]; then prd='JAS'; moni='Jul'; monf='Sep';ln=92;emon=30; fi
if [ $mn == "Jun" ]; then prd='ASO'; moni='Aug'; monf='Oct';ln=92;emon=31; fi
if [ $mn == "Jul" ]; then prd='SON'; moni='Sep'; monf='Nov';ln=91;emon=30; fi
if [ $mn == "Aug" ]; then prd='OND'; moni='Oct'; monf='Dec';ln=92;emon=31; fi
if [ $mn == "Sep" ]; then prd='NDJ'; moni='Nov'; monf='Jan';ln=92;emon=31; fi
if [ $mn == "Oct" ]; then prd='DJF'; moni='Dec'; monf='Feb';ln=90;emon=28; fi
if [ $mn == "Nov" ]; then prd='JFM'; moni='Jan'; monf='Mar';ln=90;emon=31; fi
if [ $mn == "Dec" ]; then prd='FMA'; moni='Feb'; monf='Apr';ln=89;emon=30; fi
fi

if [ $ld = 3 ]; then

if [ $mn == "Jan" ]; then prd='AMJ'; moni='Apr'; monf='Jun';ln=91;emon=30; fi
if [ $mn == "Feb" ]; then prd='MJJ'; moni='May'; monf='Jul';ln=92;emon=31; fi
if [ $mn == "Mar" ]; then prd='JJA'; moni='Jun'; monf='Aug';ln=92;emon=31; fi
if [ $mn == "Apr" ]; then prd='JAS'; moni='Jul'; monf='Sep';ln=92;emon=30; fi
if [ $mn == "May" ]; then prd='ASO'; moni='Aug'; monf='Oct';ln=92;emon=31; fi
if [ $mn == "Jun" ]; then prd='SON'; moni='Sep'; monf='Nov';ln=91;emon=30; fi
if [ $mn == "Jul" ]; then prd='OND'; moni='Oct'; monf='Dec';ln=92;emon=31; fi
if [ $mn == "Aug" ]; then prd='NDJ'; moni='Nov'; monf='Jan';ln=92;emon=31; fi
if [ $mn == "Sep" ]; then prd='DJF'; moni='Dec'; monf='Feb';ln=90;emon=28; fi
if [ $mn == "Oct" ]; then prd='JFM'; moni='Jan'; monf='Mar';ln=90;emon=31; fi
if [ $mn == "Nov" ]; then prd='FMA'; moni='Feb'; monf='Apr';ln=89;emon=30; fi
if [ $mn == "Dec" ]; then prd='MAM'; moni='Mar'; monf='May';ln=92;emon=31; fi
fi


# Generate chirps hindcast data
cat>chirps.gs<<eofGS
'reinit'
'open ../grid.ctl'
'set y 1 201'
'set x 1 721'
'define grd=prc'
'close 1' 
'open ../globe_mask0p1.ctl'
'set lat -50 50'
'set lon -180 180'
'define mm = mask'
'close 1'
'set gxout fwrite'
'set fwrite ${datdir}/chirps_ld${ld}.dat'
i=1991
while(i<=2022)
'open chirps_daily.ctl'
'set lat -50 50'
'set lon -180 180'
yr = i
yrr = i
say $moni
say $monf
say $ln
if(${moni}=Nov & ${monf}=Jan);yr=i;yrr=i+1;endif
if(${moni}=Dec & ${monf}=Feb);yr=i;yrr=i+1;endif
if(${moni}=Jan & ${monf}=Mar);yr=i+1;yrr=i+1;endif
if(${moni}=Feb & ${monf}=Apr);yr=i+1;yrr=i+1;endif
if(${moni}=Mar & ${monf}=May);yr=i+1;yrr=i+1;endif

say yr ' ' yrr

'define tt = ave(precip,time=1${moni}'yr',time=${emon}${monf}'yrr')'
'define ttt = maskout(lterp(tt,mm),mm)'
'd re(ttt,721,linear,-180,0.5,201,linear,-50,0.5,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
eofGS

$pperl $grads -blc chirps.gs

cat>chirps_ld${ld}.ctl<<eofCTL
dset ${datdir}/chirps_ld${ld}.dat
undef -999000000.000000
xdef 721 linear -180 0.5
ydef 201 linear -50 0.5
zdef 1 levels 0
tdef 32 linear 00Z01JAN0001 1yr
vars 1
precip 0 99 precip
endvars
eofCTL

cat>chirps_hind.gs<<eofGS
'reinit'
'open chirps_ld${ld}.ctl'
'set y 1 201'
'set x 1 721'
'set gxout fwrite'
'set fwrite ${datdir}/chirps_hind_precip_ld${ld}.dat'
'define clm = ave(precip,t=1,t=30)'
i=1
while(i<=32)
'set t 'i
'define pp = maskout(precip,(clm-0.5))' 
'd re(pp,721,linear,-180,0.5,201,linear,-50,0.5,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
eofGS

$pperl $grads -blc chirps_hind.gs

cat>gen_chirps_hind_precip.py<<eofPY
import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta
import chirps_config as cfg

f1 = "${datdir}/chirps_hind_precip_ld${ld}.dat"

# Predictor spatial dimension (Global tropics)
#lats = -50; latn = 50; lonw = -180; lone = 180

#h_res1 = 0.5 # Predictor horizontal resolution

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(cfg.lats,cfg.latn+cfg.h_res1,cfg.h_res1); ny = len(nlat);
nlon = np.arange(cfg.lonw,cfg.lone+cfg.h_res1,cfg.h_res1); nx = len(nlon);

nt = cfg.years
ntime = nt
nlat = ny
nlon = nx

fid = open(f1, 'rb');
precipt = np.zeros( (nt, ny, nx) );
t = 0
for ts in range(nt):
    precipt[t,:,:] = np.reshape(np.fromfile(fid,dtype='<f',count=ny*nx),(ny,nx));
    t += 1;
fid.close();

precipt[precipt <= -999] = np.nan

ncfile = netCDF4.Dataset('chirps_hind_precip_ld${ld}.nc',mode='w',format='NETCDF4_CLASSIC')
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
precip = ncfile.createVariable('precip',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
precip.units = 'K' # degrees Kelvin
precip.standard_name = 'Sea_surface_temperature' # this is a CF standard name
nlats = len(lat_dim); nlons = len(lon_dim); ntimes = nt
time[:] = netCDF4.date2num(times, units=time.units, calendar=time.calendar)
lat[:] = cfg.lats + cfg.h_res1*np.arange(nlat)
lon[:] = cfg.lonw + cfg.h_res1*np.arange(nlon)
precip[:,:,:] = precipt # Appends data along unlimited dimension
eofPY

$py gen_chirps_hind_precip.py

done

mv *.nc $ncdir
