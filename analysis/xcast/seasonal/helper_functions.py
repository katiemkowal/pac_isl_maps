
#adjust longitude from 0-360 to -180 to 180
def adjust_longitude_to_180180(ds, name_of_longitude_var):
    ds.coords[name_of_longitude_var] = (ds.coords[name_of_longitude_var] + 180) % 360 - 180
    ds = ds.sortby(ds[name_of_longitude_var])
    return ds