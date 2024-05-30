import netCDF4
from netCDF4 import Dataset
from numpy import dtype
import numpy as np
import datetime
from netCDF4 import date2num,num2date
from dateutil.relativedelta import relativedelta
import pacisl_config as cfg

f1 = "cons_tercile.dat"

#res1 = .5 # Predictor horizontal resolution
#res2 = .5 # Predictor vertical resolution 

# Calculate zonal and meridional grid size (for predictor and predictand)
nlat = np.arange(-20,-16+0.5,.5); ny = len(nlat);
nlon = np.arange(175, 179+0.5,.5); nx = len(nlon);

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


ncfile = netCDF4.Dataset('gefs_week34_cons_tercile.nc',mode='w',format='NETCDF4_CLASSIC')
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


units = 'days since 2024-05-30'
calendar = 'proleptic_gregorian'
time = ncfile.createVariable('time', np.float64, ('time',))
time.long_name = 'time'
time.units = 'days since 2024-05-30 00:00:00'
time.calendar = 'proleptic_gregorian'
time.axis = 'T'
times = [datetime.datetime(2024, 5, 30) + relativedelta(years=x) for x in range(0,nt)]
time[:] = netCDF4.date2num(times, units=units, calendar=calendar)
precip = ncfile.createVariable('precip',np.float64,('M', 'time', 'lat','lon')) # note: unlimited dimension is leftmost
precip.units = 'mm' #
precip.standard_name = 'Sea_surface_temperature' # this is a CF standard name
nlats = len(lat_dim); nlons = len(lon_dim); ntimes = nt; nlons=nn
time[:] = netCDF4.date2num(times, units=time.units, calendar=time.calendar)
lat[:] = -20 + .5*np.arange(nlat)
lon[:] = 175 + .5*np.arange(nlon)
M[:] = 1 + np.arange(nnum)
precip[:,:,:,:] = precipt # Appends data along unlimited dimension
