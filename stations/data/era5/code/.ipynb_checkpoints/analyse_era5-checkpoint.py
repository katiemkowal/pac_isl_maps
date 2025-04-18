import xarray as xr
import os
from datetime import date
import era5_config as cfg

today = date.today()
download_file_name = '_'.join(['dailysingle',cfg.var, str(today), str('ERA5.nc')])
data = xr.open_dataset(os.path.join(cfg.download_raw_data_dir, 'nc_files', download_file_name))
print(data)