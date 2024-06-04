
#adjust longitude from 0-360 to -180 to 180
def adjust_longitude_to_180180(ds, name_of_longitude_var):
    ds.coords[name_of_longitude_var] = (ds.coords[name_of_longitude_var] + 180) % 360 - 180
    ds = ds.sortby(ds[name_of_longitude_var])
    return ds

#adjust longitude from 0-360 to -180 to 180
def adjust_longitude_to_360(ds, name_of_longitude_var):
    ds_to_change = ds.copy()
    ds_to_change.coords[name_of_longitude_var] = (ds_to_change.coords[name_of_longitude_var] + 360) % 360
    ds_to_change = ds_to_change.sortby(ds_to_change[name_of_longitude_var])
    return ds_to_change

#get a list of year integers from a start and an end date
def getYears(start, end):
    #start year, to end year inclusive
    # >get YearStrings ("1981", "1982"... "2016")
    #return [expression for var is iterable if condition]
    return [year for year in range(start, end+1)]