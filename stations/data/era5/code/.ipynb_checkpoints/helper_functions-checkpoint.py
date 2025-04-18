import datetime
from datetime import datetime as dt

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

def get_years_months_days_from_dates(date_list):
    days_list = []
    months_list = []
    years_list = []
    for date in date_list:
        days_needed = date.strftime('%d')
        months_needed = date.strftime('%m')
        years_needed = date.strftime('%Y')
        days_list.append(days_needed)
        months_list.append(months_needed)
        years_list.append(years_needed)
    day_need = sorted(set(days_list))
    month_need = sorted(set(months_list))
    years_need = sorted(set(years_list))
    days_months_years = {
        'day' : list(day_need),
        'month' : list(month_need),
        'year': list(years_need)}
    return days_months_years

def get_monthsdays_from_dates(date_list):
    
    months_list = []
    for date in date_list:
        months_needed = date.strftime('%m')
        months_list.append(months_needed)
    month_need = sorted(set(months_list))
    
    month_day_yr_list = []
    for month_of_interest in month_need:
        date_month = [date for date in date_list if date.strftime("%m") == month_of_interest]
        days_list = []
        years_list = []
        for date in date_month:
            days_needed = date.strftime('%d')
            years_needed = date.strftime('%Y')
            years_list.append(years_needed)
            days_list.append(days_needed)
        day_need = sorted(set(days_list))
        years_need = sorted(set(years_list))
        month_day_yr_list.append({
            'day' : list(day_need),
            'month' : [str(month_of_interest)],
            'year': list(years_need)})
    return month_day_yr_list

#calculates an anomaly for a given array - remember to groupby T.month first before applying for monthly anomalies
def calc_anomaly(x):
    return x - x.mean(dim = 'T')