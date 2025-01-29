

#get a string of dates that will match the cmorph binary file naming conventions
def get_date_str(date):
    if date.month <10:
        cmonth_str = '0' + str(date.month)
    else:
        cmonth_str = str(date.month)
    if date.day <10:
        cday_str = '0' + str(date.day)
    else: cday_str = str(date.day)
    current_str = str(date.year) + cmonth_str + cday_str
    return current_str, cmonth_str+cday_str


## average and summation functions
def calc_totalavg_pastdays(ds, ds_var, days):
    days_ds = ds.isel(time=slice(-days, None)).mean(dim='time')
    return days_ds

def calc_total_pastdays(ds, ds_var, days):
    days_ds = ds.isel(time=slice(-days, None)).sum(dim='time')
    return days_ds

def calc_percent_pastdays(ds, ds_var, ds_clim, ds_clim_var, days, type):
    if type == 'avg':
        days_ds = ds.isel(time=slice(-days, None)).mean(dim='time')
        days_clim = ds_clim.isel(time=slice(-days, None)).mean(dim = 'time')
    elif type == 'total':
        days_ds = ds.isel(time=slice(-days, None)).sum(dim='time')
        days_clim = ds_clim.isel(time=slice(-days, None)).sum(dim = 'time')
    days_percent = ((days_ds[ds_var]/days_clim[ds_clim_var])*100).to_dataset(name = 'percent')
    return days_percent

#calculate the average anomaly over past days, e.g. last 90, 30, 7, etc...
#must give function a xarray dataset with last days and a dataset with climatology values
def calc_anom_pastdays(ds, ds_var, ds_clim, ds_clim_var, days, type):
    if type == 'avg':
        days_ds = ds.isel(time=slice(-days, None)).mean(dim='time')
        days_clim = ds_clim.isel(time=slice(-days, None)).mean(dim = 'time')
    elif type == 'total':
        days_ds = ds.isel(time=slice(-days, None)).sum(dim='time')
        days_clim = ds_clim.isel(time=slice(-days, None)).sum(dim = 'time')
    days_anom = (days_ds[ds_var] - days_clim[ds_clim_var]).to_dataset(name = 'anom')
    return days_anom