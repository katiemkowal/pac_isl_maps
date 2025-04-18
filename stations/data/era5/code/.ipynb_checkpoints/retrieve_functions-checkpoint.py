from pathlib import Path 
import os
import cdsapi
from os import path
import helper_functions as helper


cds_var_dict = {
    't2m': '2m_temperature',
    'prcp': 'total_precipitation'}

  
def download_ERA5_daily_single(first_clim_year, last_clim_year, current_years, var_list, months, days, hours, area_dict, filename, download_dir):
    import cdsapi
    c = cdsapi.Client()
    
    area = [area_dict['north'],
            area_dict['west'],
            area_dict['south'],
            area_dict['east']]

    years = helper.getYears(first_clim_year,last_clim_year)
    years = years + current_years

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': ['reanalysis'],
            'variable': var_list,
            'year': years,
            'month': months,
            'day': days,
            'area': area,
            'time': hours,
            'data_format': 'grib',
            'download_format': 'unarchived'
        },
        os.path.join(download_dir,'{}.grib'.format(filename))
    )