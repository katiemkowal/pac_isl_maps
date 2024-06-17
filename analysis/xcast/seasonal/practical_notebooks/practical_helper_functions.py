import numpy as np
import xarray as xr

#this cell is setup to calculate your target forecast months based on your initialization date
#the forecast months are currently seto to be 1-3, 2-4 and 3-5 months ahead
number_to_month_name_dictionary = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec',
    0: 'Dec'
}

coordinate_conversion = {
    'latitude': 'Y',
    'lat': 'Y',
    'Y': 'Y',
    'longitude': 'X',
    'lon': 'X',
    'X': 'X',
    'time': 'T',
    'year': 'T',
    'season': 'season',
    'T': 'T'
}

#make sure all coordinate names follow the naming conventions set up in coordinate dictionary
def prep_names(ds, coordinate_conversion):
    og_coords = list(ds.coords)
    for o in og_coords:
        ds = ds.rename({o:coordinate_conversion[o]})
    return ds

#check if all leads have the same number of years
def check_leads_years(ds_list):
    nyears = []
    for ds in ds_list:
        nyears.append(len(np.unique(ds.S.values)))
    return  all(i==nyears[0] for i in nyears)

#check all years are available for all lead times in hindcast period, only keep intersecting years
def keep_intersecting_years(ds_list):
    if check_leads_years(ds_list) == True:
        ds_update = xr.concat(ds_list, dim = 'L')
    else:
        unique_years = []
        for ds in ds_list:
            base = ds.swap_dims({'S':'T'}).to_dataset(name = 'prec')
            unique_years.append(np.unique(base.T.dt.year.values))
        intersecting_years = [x for x in unique_years[0] if x in unique_years[1] and x in unique_years[2]]

        ds_update = []
        for ds in ds_list:
            ds_check = ds.swap_dims({'S':'T'}).to_dataset(name = 'prec')
            ds_update.append(ds_check.sel(T=ds_check.T.dt.year.isin(intersecting_years)).swap_dims({'T':'S'}).prec)
        ds_update = xr.concat(ds_update, dim = 'L')
    return ds_update

#adjust longitude from 0-360 to -180 to 180
#inputs: xarray dataset, name of longitude variable (string)
#outputs: xarray data with longitude 0-360 grid
def adjust_longitude_to_360(ds, name_of_longitude_var):
    ds_to_change = ds.copy()
    ds_to_change.coords[name_of_longitude_var] = (ds_to_change.coords[name_of_longitude_var] + 360) % 360
    ds_to_change = ds_to_change.sortby(ds_to_change[name_of_longitude_var])
    return ds_to_change

#get a list of year integers from a start and an end date
#inputs: start_year (integer), end_year (integer)
#outputs: list of all years beginning at start and ending at end year as integers
def getYears(start, end):
    #start year, to end year inclusive
    # >get Years (1981, 1982... 2016)
    #return [expression for var is iterable if condition]
    return [year for year in range(start, end+1)]