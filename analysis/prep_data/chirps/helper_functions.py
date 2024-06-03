import pandas as pd

#change all day values to first of the month
def change_dates_to_first(ds, time_var):
    ds.coords[time_var] = ds.coords[time_var].dt.strftime('%Y-%m-01')
    ds.coords[time_var] = pd.to_datetime(ds.coords[time_var])
    return ds

#get a list of year integers from a start and an end date
def getYears(start, end):
    #start year, to end year inclusive
    # >get YearStrings ("1981", "1982"... "2016")
    #return [expression for var is iterable if condition]
    return [year for year in range(start, end+1)]


#adjust longitude from 0-360 to -180 to 180
def adjust_longitude_to_360(ds, name_of_longitude_var):
    ds_to_change = ds.copy()
    ds_to_change.coords[name_of_longitude_var] = (ds_to_change.coords[name_of_longitude_var] + 360) % 360
    ds_to_change = ds_to_change.sortby(ds_to_change[name_of_longitude_var])
    return ds_to_change