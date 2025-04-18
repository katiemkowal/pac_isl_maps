import datetime

def getYears(start, end):
    #start year, to end year inclusive
    # >get YearStrings ("1981", "1982"... "2016")
    #return [expression for var is iterable if condition]
    return [year for year in range(start, end+1)]

def getDates(start_date, end_date):
    # Return list of datetime.date objects (inclusive) between start_date and end_date (inclusive).
    dates_list = []
    curr_date = start_date
    while curr_date <= end_date:
        dates_list.append(curr_date)
        curr_date += datetime.timedelta(days=1)
    return dates_list

def get_months_days_from_dates(date_list):
    days_list = []
    months_list = []
    for date in date_list:
        days_needed = date.strftime('%d')
        months_needed = date.strftime('%m')
        days_list.append(days_needed)
        months_list.append(months_needed)
    day_need = sorted(set(days_list))
    month_need = sorted(set(months_list))
    days_months = {
        'day' : list(day_need),
        'month' : list(month_need)}
    return days_months

#calculates an anomaly for a given array - remember to groupby T.month first before applying for monthly anomalies
def calc_anomaly(x):
    return x - x.mean(dim = 'T')

# update tprate values to mm/month values based on days in month (feb is approximated) -- this is done here manually for precip to compare with chirps
def convert_ms_to_mmmonth(ds):
    fcst_month = ds.Ti.values[0]
    fcst_month = fcst_month.astype('datetime64[M]').astype(int) % 12 + 1
    if fcst_month in [1,3,5,7,8,10,12]:
        days_in_month = 31
    elif fcst_month in [4,6,9,11]:
        days_in_month = 30
    elif fcst_month in [2]:
        days_in_month = 28
    ds_month = ds*60*60*24*days_in_month*1000 #update to mm/month
    return(ds_month)

#convert an object to a list if not yet a list
def convert_to_list(item):
    if isinstance(item, list):
        # If it's already a list, return it as is
        return item
    else:
        # If it's not a list, convert it to a list
        return [item]
