import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta

f1 = "/cpc/int_desk/pac_isl/data/processed/cmorph/dat_files/cmorph_hind_precip_ld3.dat"

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

ncfile = netCDF4.Dataset('May_ld3_CMORPH_hind_precip.nc',mode='w',format='NETCDF4_CLASSIC')
lat_dim = ncfile.createDimension('lat', nlat) # latitude axis
lon_dim = ncfile.createDimension('lon', nlon) # longitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'

units = 'days since 2024-05-01'
calendar = 'proleptic_gregorian'
time = ncfile.createVariable('time', np.float64, ('time',))
time.long_name = 'time'
time.units = 'days since 2024-05-01 00:00:00'
time.calendar = 'proleptic_gregorian'
time.axis = 'T'
times = [datetime.datetime(2024,5,1) + relativedelta(years=x) for x in range(0,nt)]
time[:] = netCDF4.date2num(times, units=units, calendar=calendar)
precip = ncfile.createVariable('precip',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
precip.units = 'mm' 
precip.standard_name = 'Precip' # this is a CF standard name
nlats = len(lat_dim); nlons = len(lon_dim); ntimes = nt
time[:] = netCDF4.date2num(times, units=time.units, calendar=time.calendar)
lat[:] = lats + 0.25*np.arange(nlat)
lon[:] = lonw + 0.25*np.arange(nlon)
precip[:,:,:] = precipt # Appends data along unlimited dimension
