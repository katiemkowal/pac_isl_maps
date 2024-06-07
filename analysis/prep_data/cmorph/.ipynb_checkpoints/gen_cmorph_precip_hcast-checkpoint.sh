#!/bin/sh

wdir=/cpc/int_desk/pac_isl/analysis/prep_data/cmorph
datdir=/cpc/int_desk/pac_isl/data/processed/cmorph/dat_files
ncdir=/cpc/int_desk/pac_isl/data/processed/cmorph/nc_files

cd $wdir
grads=/cpc/home/ebekele/grads-2.1.0.oga.1//Contents/grads
py=/cpc/home/ebekele/.conda/envs/xcast_env/bin/python
pperl=/cpc/africawrf/ebekele/perl/bin/perl

if test -f ${datdir}/cmorph*.dat; then
    rm ${datdir}/cmorph*.dat
fi

mn=`date +"%b"`
yrmndy=`date +"%Y"-"%m"-"%d"`
yrmondy=`date +'%Y, %-m, %-d'`
mon=$(date +%b | tr A-Z a-z)

cat>cmorph.ctl<<eofCTL
DSET /cpc/fews/production/rshukla/CMORPH1/CMORPH1_ADJ_EOD/CMORPH_V1.0_ADJ_0.25deg-DLY_EOD_%y4%m2%d2
OPTIONS template little_endian
UNDEF -999.0
TITLE Bias-Corrected CMORPH V1.0 Daily (End-of-Day) Precip Accumulation
XDEF  1440 LINEAR   0.125 0.25
YDEF   480 LINEAR -59.875 0.25
ZDEF     1 LEVELS   1
TDEF 99999 LINEAR 00z01jan1998 1dy
VARS 1
r 1 99 Daily (End-of-Day) Precip Accum [mm/day]
ENDVARS
eofCTL

echo $mon

for ld in {1..3}; do

if [ $ld = 1 ]; then

if [ $mn == "Jan" ]; then prd1='JFM'; prd2='FMA'; prd3='MAM'; mon1i='Jan'; mon1f='Mar';ln1=90;emon=31; mon2i='Feb'; mon2f='Apr';ln2=89;emon=30; mon3i='Mar'; mon3f='May';ln3=92;emon=31;fi
if [ $mn == "Feb" ]; then prd1='FMA'; prd2='MAM'; prd3='AMJ'; mon1i='Feb'; mon1f='Apr';ln1=89;emon=30; mon2i='Mar'; mon2f='May';ln2=92;emon=31; mon3i='Apr'; mon3f='Jun';ln3=91;emon=30;fi
if [ $mn == "Mar" ]; then prd1='MAM'; prd2='AMJ'; prd3='MJJ'; mon1i='Mar'; mon1f='May';ln1=92;emon=31; mon2i='Apr'; mon2f='Jun';ln2=91;emon=30; mon3i='May'; mon3f='Jul';ln3=92;emon=31; fi
if [ $mn == "Apr" ]; then prd1='AMJ'; prd2='MJJ'; prd3='JJA'; mon1i='Apr'; mon1f='Jun';ln1=91;emon=30; mon2i='May'; mon2f='Jul';ln2=92;emon=31; mon3i='Jun'; mon3f='Aug';ln3=92;emon=31; fi
if [ $mn == "May" ]; then prd1='MJJ'; prd2='JJA'; prd3='JAS'; mon1i='May'; mon1f='Jul';ln1=92;emon=31; mon2i='Jun'; mon2f='Aug';ln2=92;emon=31; mon3i='Jul'; mon3f='Sep';ln3=92;emon=30; fi
if [ $mn == "Jun" ]; then prd1='JJA'; prd2='JAS'; prd3='ASO'; mon1i='Jun'; mon1f='Aug';ln1=92;emon=31; mon2i='Jul'; mon2f='Sep';ln2=92;emon=30; mon3i='Aug'; mon3f='Oct';ln3=92;emon=31; fi
if [ $mn == "Jul" ]; then prd1='JAS'; prd2='ASO'; prd3='SON'; mon1i='Jul'; mon1f='Sep';ln1=92;emon=30; mon2i='Aug'; mon2f='Oct';ln2=92;emon=31; mon3i='Sep'; mon3f='Nov';ln3=91;emon=30; fi
if [ $mn == "Aug" ]; then prd1='ASO'; prd2='SON'; prd3='OND'; mon1i='Aug'; mon1f='Oct';ln1=92;emon=31; mon2i='Sep'; mon2f='Nov';ln2=91;emon=30; mon3i='Oct'; mon3f='Dec';ln3=92;emon=31; fi
if [ $mn == "Sep" ]; then prd1='SON'; prd2='OND'; prd3='NDJ'; mon1i='Sep'; mon1f='Nov';ln1=91;emon=30; mon2i='Oct'; mon2f='Dec';ln2=92;emon=31; mon3i='Nov'; mon3f='Jan';ln3=92;emon=31; fi
if [ $mn == "Oct" ]; then prd1='OND'; prd2='NDJ'; prd3='DJF'; mon1i='Oct'; mon1f='Dec';ln1=92;emon=31; mon2i='Nov'; mon2f='Jan';ln2=92;emon=31; mon3i='Dec'; mon3f='Feb';ln3=90;emon=28; fi
if [ $mn == "Nov" ]; then prd1='NDJ'; prd2='DJF'; prd3='JFM'; mon1i='Nov'; mon1f='Jan';ln1=92;emon=31; mon2i='Dec'; mon2f='Feb';ln2=90;emon=28; mon3i='Jan'; mon3f='Mar';ln3=90;emon=31; fi
if [ $mn == "Dec" ]; then prd1='DJF'; prd2='JFM'; prd3='FMA'; mon1i='Dec'; mon1f='Feb';ln1=90;emon=28; mon2i='Jan'; mon2f='Mar';ln2=90;emon=31; mon3i='Feb'; mon3f='Apr';ln3=89;emon=30; fi
fi

if [ $ld = 2 ]; then

if [ $mn == "Jan" ]; then prd1='FMA'; prd2='MAM'; prd3='AMJ'; mon1i='Feb'; mon1f='Apr';ln1=89;emon=30; mon2i='Mar'; mon2f='May';ln2=92;emon=31; mon3i='Apr'; mon3f='Jun';ln3=91;emon=30;fi   
if [ $mn == "Feb" ]; then prd1='MAM'; prd2='AMJ'; prd3='MJJ'; mon1i='Mar'; mon1f='May';ln1=92;emon=31; mon2i='Apr'; mon2f='Jun';ln2=91;emon=30; mon3i='May'; mon3f='Jul';ln3=92;emon=31; fi   
if [ $mn == "Mar" ]; then prd1='AMJ'; prd2='MJJ'; prd3='JJA'; mon1i='Apr'; mon1f='Jun';ln1=91;emon=30; mon2i='May'; mon2f='Jul';ln2=92;emon=31; mon3i='Jun'; mon3f='Aug';ln3=92;emon=31; fi   
if [ $mn == "Apr" ]; then prd1='MJJ'; prd2='JJA'; prd3='JAS'; mon1i='May'; mon1f='Jul';ln1=92;emon=31; mon2i='Jun'; mon2f='Aug';ln2=92;emon=31; mon3i='Jul'; mon3f='Sep';ln3=92;emon=30; fi   
if [ $mn == "May" ]; then prd1='JJA'; prd2='JAS'; prd3='ASO'; mon1i='Jun'; mon1f='Aug';ln1=92;emon=31; mon2i='Jul'; mon2f='Sep';ln2=92;emon=30; mon3i='Aug'; mon3f='Oct';ln3=92;emon=31; fi   
if [ $mn == "Jun" ]; then prd1='JAS'; prd2='ASO'; prd3='SON'; mon1i='Jul'; mon1f='Sep';ln1=92;emon=30; mon2i='Aug'; mon2f='Oct';ln2=92;emon=31; mon3i='Sep'; mon3f='Nov';ln3=91;emon=30; fi   
if [ $mn == "Jul" ]; then prd1='ASO'; prd2='SON'; prd3='OND'; mon1i='Aug'; mon1f='Oct';ln1=92;emon=31; mon2i='Sep'; mon2f='Nov';ln2=91;emon=30; mon3i='Oct'; mon3f='Dec';ln3=92;emon=31; fi   
if [ $mn == "Aug" ]; then prd1='SON'; prd2='OND'; prd3='NDJ'; mon1i='Sep'; mon1f='Nov';ln1=91;emon=30; mon2i='Oct'; mon2f='Dec';ln2=92;emon=31; mon3i='Nov'; mon3f='Jan';ln3=92;emon=31; fi   
if [ $mn == "Sep" ]; then prd1='OND'; prd2='NDJ'; prd3='DJF'; mon1i='Oct'; mon1f='Dec';ln1=92;emon=31; mon2i='Nov'; mon2f='Jan';ln2=92;emon=31; mon3i='Dec'; mon3f='Feb';ln3=90;emon=28; fi   
if [ $mn == "Oct" ]; then prd1='NDJ'; prd2='DJF'; prd3='JFM'; mon1i='Nov'; mon1f='Jan';ln1=92;emon=31; mon2i='Dec'; mon2f='Feb';ln2=90;emon=28; mon3i='Jan'; mon3f='Mar';ln3=90;emon=31; fi   
if [ $mn == "Nov" ]; then prd1='DJF'; prd2='JFM'; prd3='FMA'; mon1i='Dec'; mon1f='Feb';ln1=90;emon=28; mon2i='Jan'; mon2f='Mar';ln2=90;emon=31; mon3i='Feb'; mon3f='Apr';ln3=89;emon=30; fi   
if [ $mn == "Dec" ]; then prd1='JFM'; prd2='FMA'; prd3='MAM'; mon1i='Jan'; mon1f='Mar';ln1=90;emon=31; mon2i='Feb'; mon2f='Apr';ln2=89;emon=30; mon3i='Mar'; mon3f='May';ln3=92;emon=31;fi   
fi

if [ $ld = 3 ]; then

if [ $mn == "Jan" ]; then prd1='MAM'; prd2='AMJ'; prd3='MJJ'; mon1i='Mar'; mon1f='May';ln1=92;emon1=31; mon2i='Apr'; mon2f='Jun';ln2=91;emon2=30; mon3i='May'; mon3f='Jul';ln3=92;emon3=31; fi   
if [ $mn == "Feb" ]; then prd1='AMJ'; prd2='MJJ'; prd3='JJA'; mon1i='Apr'; mon1f='Jun';ln1=91;emon1=30; mon2i='May'; mon2f='Jul';ln2=92;emon2=31; mon3i='Jun'; mon3f='Aug';ln3=92;emon3=31; fi   
if [ $mn == "Mar" ]; then prd1='MJJ'; prd2='JJA'; prd3='JAS'; mon1i='May'; mon1f='Jul';ln1=92;emon1=31; mon2i='Jun'; mon2f='Aug';ln2=92;emon2=31; mon3i='Jul'; mon3f='Sep';ln3=92;emon3=30; fi   
if [ $mn == "Apr" ]; then prd1='JJA'; prd2='JAS'; prd3='ASO'; mon1i='Jun'; mon1f='Aug';ln1=92;emon1=31; mon2i='Jul'; mon2f='Sep';ln2=92;emon2=30; mon3i='Aug'; mon3f='Oct';ln3=92;emon3=31; fi   
if [ $mn == "May" ]; then prd1='JAS'; prd2='ASO'; prd3='SON'; mon1i='Jul'; mon1f='Sep';ln1=92;emon1=30; mon2i='Aug'; mon2f='Oct';ln2=92;emon2=31; mon3i='Sep'; mon3f='Nov';ln3=91;emon3=30; fi   
if [ $mn == "Jun" ]; then prd1='ASO'; prd2='SON'; prd3='OND'; mon1i='Aug'; mon1f='Oct';ln1=92;emon1=31; mon2i='Sep'; mon2f='Nov';ln2=91;emon2=30; mon3i='Oct'; mon3f='Dec';ln3=92;emon3=31; fi   
if [ $mn == "Jul" ]; then prd1='SON'; prd2='OND'; prd3='NDJ'; mon1i='Sep'; mon1f='Nov';ln1=91;emon1=30; mon2i='Oct'; mon2f='Dec';ln2=92;emon2=31; mon3i='Nov'; mon3f='Jan';ln3=92;emon3=31; fi   
if [ $mn == "Aug" ]; then prd1='OND'; prd2='NDJ'; prd3='DJF'; mon1i='Oct'; mon1f='Dec';ln1=92;emon1=31; mon2i='Nov'; mon2f='Jan';ln2=92;emon2=31; mon3i='Dec'; mon3f='Feb';ln3=90;emon3=28; fi   
if [ $mn == "Sep" ]; then prd1='NDJ'; prd2='DJF'; prd3='JFM'; mon1i='Nov'; mon1f='Jan';ln1=92;emon1=31; mon2i='Dec'; mon2f='Feb';ln2=90;emon2=28; mon3i='Jan'; mon3f='Mar';ln3=90;emon3=31; fi   
if [ $mn == "Oct" ]; then prd1='DJF'; prd2='JFM'; prd3='FMA'; mon1i='Dec'; mon1f='Feb';ln1=90;emon1=28; mon2i='Jan'; mon2f='Mar';ln2=90;emon2=31; mon3i='Feb'; mon3f='Apr';ln3=89;emon3=30; fi   
if [ $mn == "Nov" ]; then prd1='JFM'; prd2='FMA'; prd3='MAM'; mon1i='Jan'; mon1f='Mar';ln1=90;emon1=31; mon2i='Feb'; mon2f='Apr';ln2=89;emon2=30; mon3i='Mar'; mon3f='May';ln3=92;emon3=31;fi   
if [ $mn == "Dec" ]; then prd1='FMA'; prd2='MAM'; prd3='AMJ'; mon1i='Feb'; mon1f='Apr';ln1=89;emon1=30; mon2i='Mar'; mon2f='May';ln2=92;emon2=31; mon3i='Apr'; mon3f='Jun';ln3=91;emon3=30;fi   
fi

# Generate NMME hindcast data
cat>cmorph.gs<<eofGS
'reinit'
'set gxout fwrite'
'set fwrite ${datdir}/cmorph_ld${ld}.dat'
i=1998
while(i<=2022)
'open cmorph.ctl'
'set lat -50 50'
'set lon -180 180'
yr = i
yrr = i
say $mon1i ' ' $mon2i ' ' $mon3i
say $mon1f ' ' $mon2f ' ' $mon3f
say $ln1 ' ' $ln2 ' ' $ln3
if(${mon1i}=Nov & ${mon1f}=Jan);yr=i;yrr=i+1;endif
if(${mon1i}=Dec & ${mon1f}=Feb);yr=i;yrr=i+1;endif
if(${mon1i}=Jan & ${mon1f}=Mar);yr=i+1;yrr=i+1;endif
if(${mon1i}=Feb & ${mon1f}=Apr);yr=i+1;yrr=i+1;endif
if(${mon1i}=Mar & ${mon1f}=May);yr=i+1;yrr=i+1;endif

if(${mon2i}=Nov & ${mon2f}=Jan);yr=i;yrr=i+1;endif
if(${mon2i}=Dec & ${mon2f}=Feb);yr=i;yrr=i+1;endif
if(${mon2i}=Jan & ${mon2f}=Mar);yr=i+1;yrr=i+1;endif
if(${mon2i}=Feb & ${mon2f}=Apr);yr=i+1;yrr=i+1;endif
if(${mon2i}=Mar & ${mon2f}=May);yr=i+1;yrr=i+1;endif

if(${mon3i}=Nov & ${mon3f}=Jan);yr=i;yrr=i+1;endif
if(${mon3i}=Dec & ${mon3f}=Feb);yr=i;yrr=i+1;endif
if(${mon3i}=Jan & ${mon3f}=Mar);yr=i+1;yrr=i+1;endif
if(${mon3i}=Feb & ${mon3f}=Apr);yr=i+1;yrr=i+1;endif
if(${mon3i}=Mar & ${mon3f}=May);yr=i+1;yrr=i+1;endif

say yr ' ' yrr

'define tt = ave(r,time=1${mon1i}'yr',time=${emon1}${mon1f}'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
'define tt = ave(r,time=1${mon2i}'yr',time=${emon2}${mon2f}'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
'define tt = ave(r,time=1${mon3i}'yr',time=${emon3}${mon3f}'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
eofGS

$pperl $grads -blc cmorph.gs

cat>cmorph_ld${ld}.ctl<<eofCTL
dset ${datdir}/cmorph_ld${ld}.dat
undef -999000000.000000
xdef 1441 linear -180 0.25
ydef 401 linear -50 0.25
zdef 1 levels 0
tdef 75 linear 00Z01JAN0001 1yr
vars 1
precip 0 99 precip
endvars
eofCTL

cat>cmorph_hind.gs<<eofGS
'reinit'
'open cmorph_ld${ld}.ctl'
'set y 1 401'
'set x 1 1441'
'set gxout fwrite'
'set fwrite ${datdir}/cmorph_hind_precip_ld${ld}.dat'
'define clm = ave(precip,t=26,t=50)'
i=1
while(i<=75)
'set t 'i
'define pp = maskout(precip,(clm-0.5))' 
'd re(pp,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
eofGS

$pperl $grads -blc cmorph_hind.gs

cat>gen_cmorph_hind_precip.py<<eofPY
import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta

f1 = "${datdir}/cmorph_hind_precip_ld${ld}.dat"

# Predictor spatial dimension (Global tropics)
lats = -50; latn = 50; lonw = -180; lone = 180

res1 = 0.25 # Predictor horizontal resolution

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(lats,latn+res1,res1); ny = len(nlat);
nlon = np.arange(lonw,lone+res1,res1); nx = len(nlon);

nt = 75
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

ncfile = netCDF4.Dataset('cmorph_hind_precip_ld${ld}.nc',mode='w',format='NETCDF4_CLASSIC')
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
lat[:] = lats + 0.25*np.arange(nlat)
lon[:] = lonw + 0.25*np.arange(nlon)
precip[:,:,:] = precipt # Appends data along unlimited dimension
eofPY

$py gen_cmorph_hind_precip.py

done

mv *.nc ${ncdir}
