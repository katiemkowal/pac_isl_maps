import pandas as pd

#dictionary to identify month numbers as a letter, designed to name seasons (e.g. JAS)
month_number_dict = {
    1: 'J',
    2: 'F',
    3: 'M',
    4: 'A',
    5: 'M',
    6: 'J',
    7: 'J',
    8: 'A',
    9: 'S',
    10:'O',
    11:'N',
    12:'D'}

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