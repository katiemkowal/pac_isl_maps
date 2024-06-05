#import libraries
import xcast as xc
import xarray as xr
import cartopy.crs as ccrs
import cartopy as cf
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path 
from datetime import datetime

#import function and config files written to process code
import chirps_pacisl_config as cfg

#to import functions from another folder
import importlib.util
# passing the file name and path as argument
spec1 = importlib.util.spec_from_file_location(
  "helper_functions", "/cpc/int_desk/pac_isl/analysis/helper_functions.py")    

# importing the module as helper
helper = importlib.util.module_from_spec(spec1) 
spec1.loader.exec_module(helper)

spec2 = importlib.util.spec_from_file_location(
  "helper_dictionaries", "/cpc/int_desk/pac_isl/analysis/helper_dictionaries.py")    
hdict = importlib.util.module_from_spec(spec2) 
spec2.loader.exec_module(hdict)

#setup years, months
years = helper.getYears(cfg.first_year,cfg.last_year)
months = [1,2,3,4,5,6,7,8,9,10,11,12]

#main function
def convert_monthly_to_3monthmean(ds):
    seasons = []
    for m in months:
        print(m)
        if m in [11,12]:
            if m == 11:
                #setup exception for crossing years if starting in november
                first_year_months = [11,12]
                second_year_months = [1]
                season_name = ['NDJ']
                #this code will save the first year as the season's year if crossing years***
            elif m == 12:
                first_year_months = [12]
                second_year_months = [1,2]
                season_name = ['DJF']
                
            season = []
            for y in np.unique(ds.time.dt.year):
                seas_1 = ds.sel(time = ds.time.dt.month.isin(first_year_months))
                seas_1 = seas_1.sel(time = seas_1.time.dt.year.isin(y))
                seas_2 = monthly_chirps.sel(time = monthly_chirps.time.dt.month.isin(second_year_months))
                seas_2 = seas_2.sel(time = seas_2.time.dt.year.isin(y+1))
                season_year =  xr.concat([seas_1, seas_2], dim = 'time').mean(dim = 'time')
                season_year = season_year.expand_dims({'year':int(y)})
                season.append(season_year)
            season = xr.concat(season, dim = 'year')
            season = season.expand_dims({'season':season_name})
    
        else:
            seas_months = ds.sel(time = ds.time.dt.month.isin([m, m+1, m+2]))
            season = seas_months.groupby('time.year').mean()
            season = season.expand_dims({'season':[hdict.month_number_dict[m] + hdict.month_number_dict[m+1] + hdict.month_number_dict[m+2]]})
        season = season.sel(year = season.year.isin(years))
        seasons.append(season)
    seasons = xr.concat(seasons, dim = 'season')
    return seasons

######### main script
#open all files
raw_chirps_daily = xr.open_mfdataset(os.path.join(cfg.raw_chirps_dir,
                                            'chirps-v2.0.*days_p05.nc'), 
                               combine = "by_coords", 
                               engine = "netcdf4", 
                               chunks={'time': 121, 'latitude': 2000, 'longitude': 720})#tested by opening one file first, divided longitude into a few chunks to open at a time (7200 total longitude coordinates for chirps high res global)


monthly_chirps = raw_chirps_daily.resample(time='1M').mean()#monthly chirps
seasonal_chirps = convert_monthly_to_3monthmean(monthly_chirps)

##### slice seasonal chirps data to region of interest to limit file size
#convert chirps to 0-360 if needed
if (cfg.slice_w | cfg.slice_e) >= 180:
    chirps_360 = helper.adjust_longitude_to_360(seasonal_chirps, 'longitude').sortby('latitude', ascending = False).sortby('longitude', ascending = True)
else:
    chirps_360 = seasonal_chirps.sortby('longitude', ascending=True).sortby('latitude', ascending = False)

regional_chirps = chirps_360.sel(longitude = slice(cfg.slice_w, cfg.slice_e),
                              latitude = slice(cfg.slice_n, cfg.slice_s))
print(regional_chirps)

file_name_to_save = 'chirps05' + cfg.region_name
regional_chirps.to_netcdf(os.path.join(cfg.nc_folder, file_name_to_save))

## save files by season if desired
# for s in np.unique(regional_chirps.season):
#     file_name_to_save = 'chirps05' + s
#     if Path(os.path.join(cfg.nc_folder, file_name_to_save)).is_file():
#         os.remove(os.path.join(cfg.nc_folder, file_name_to_save))
#     regional_chirps.sel(season=s).to_netcdf(os.path.join(cfg.nc_folder, file_name_to_save))






