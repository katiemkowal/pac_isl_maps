#Functions designed to help with simple tasks like adjusting coordinates, dates, getting years

#to import these functions as a library named 'helper' into your other python files add the following to your other python files:

#     import importlib.util

#     spec = importlib.util.spec_from_file_location(
#     "helper_functions", #"/cpc/int_desk/pac_isl/analysis/helper_functions.py")    

#      helper = importlib.util.module_from_spec(spec) 
#      spec.loader.exec_module(helper)

import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location(
    "helper_dictionaries", "/cpc/int_desk/pac_isl/analysis/helper_dictionaries.py")    
hdict = importlib.util.module_from_spec(spec) 
spec.loader.exec_module(hdict)

#adjust longitude from 0-360 to -180 to 180
#inputs: xarray dataset, name of the longitude variable (string)
#outputs: xarray dataset with longitude -180 to 180 grid
def adjust_longitude_to_180180(ds, name_of_longitude_var):
    ds.coords[name_of_longitude_var] = (ds.coords[name_of_longitude_var] + 180) % 360 - 180
    ds = ds.sortby(ds[name_of_longitude_var])
    return ds

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

#change all day values to first of the month
#input: xarray dataset, name of the time variable (string), e.g. 'T' or 'time'
#output: xarray dataset with all time variable entries listed as first of the month
def change_dates_to_first(ds, time_var):
    ds.coords[time_var] = ds.coords[time_var].dt.strftime('%Y-%m-01')
    ds.coords[time_var] = pd.to_datetime(ds.coords[time_var])
    return ds

#calculate a future 3-month season name based on a current month and lead time
#designed to create season names while crossing year
#inputs: current_month (integer), lead (integer)
#outputs: str, e.g. JJA, or DJF
def calc_season_name_from_month_lead(current_month, lead):
    #calculate future month of interest, math helps cross years
    month1 = hdict.month_number_dict[(current_month +lead)%12]
    month2 = hdict.month_number_dict[(current_month +lead+1)%12]
    month3 = hdict.month_number_dict[(current_month+lead+2)%12]
    seas = month1 + month2 + month3
    return seas

#calculate whether all items in a list are identical, 
#useful to checking if all years or coordinates exist in two datasets
#inputs: list of integers
#outputs: True or False statement
def check_list_items_identical(list):
    return all(i==list[0] for i in list)
    