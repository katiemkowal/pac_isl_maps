import helper_functions as helper
import retrieve_functions as retrieve
import era5_config as cfg
import datetime
from datetime import date
import os
from pathlib import Path 

#create download_file_folders
os.makedirs(os.path.join(cfg.download_raw_data_dir, 'nc_files'), exist_ok=True)
os.makedirs(os.path.join(cfg.download_raw_data_dir, 'grib_files'), exist_ok=True) 
nc_files = os.path.join(cfg.download_raw_data_dir, 'nc_files')
grib_files = os.path.join(cfg.download_raw_data_dir, 'grib_files')
#calculate dates to pull (e.g. last two weeks)
today = date.today()
d = datetime.timedelta(days = cfg.days_back)
first_date = today - d

dates = helper.getDates(first_date, today)
# to_download = helper.get_years_months_days_from_dates(dates)
to_download = helper.get_monthsdays_from_dates(dates)

print(to_download)
#download ERA5 daily data from cds
for e, extent in enumerate(cfg.download_extents):
    for month_download in to_download:
        months_to_download = month_download['month']
        days_to_download = month_download['day']
        current_years = month_download['year']

        download_file_name = '_'.join(['dailysingle', cfg.extent_names[e], cfg.var, months_to_download[0], str(today), str('ERA5')])
        print(download_file_name)
        if not Path(os.path.join(grib_files,'{}.grib'.format(download_file_name))).is_file():
            print('STARTING DOWNLOAD OF {}'.format(download_file_name))
            retrieve.download_ERA5_daily_single(cfg.first_clim_year, 
                                            cfg.last_clim_year,
                                            current_years,
                                            [retrieve.cds_var_dict[cfg.var]], 
                                            months_to_download,
                                            days_to_download,
                                            cfg.hours_to_download,
                                            extent,
                                            download_file_name,
                                            grib_files)