import helper_functions as helper
import retrieve_functions as retrieve
import c3s_config as cfg
import datetime
from datetime import date
import os
from pathlib import Path 
   
#download c3s monthly data for variables in config file
years = helper.getYears(cfg.first_year, cfg.last_year)

for var in cfg.vars:
    
    if var in ['t2m', 'prcp']:
        cds_var = retrieve.cds_var_dict[var]
    else:
        cds_var = var
        
    if var in ['t2m', 'prcp', 'sst', 'msl']:
        download_type = 'single'
        pressure_level = 'na'
    elif var in ['u', 'v', 'gph']:
        download_type = 'pressure'
        pressure_level = getattr(cfg,  var + '_pressure_level')
        
    area = [getattr(cfg,  var + '_download_extent')['north'],
            getattr(cfg,  var + '_download_extent')['west'],
            getattr(cfg,  var + '_download_extent')['south'],
            getattr(cfg,  var + '_download_extent')['east']]
    area_name = getattr(cfg,  var + '_download_name')
    
    retrieve.download_cds_initial(download_type,
                                          pressure_level,
                                          cfg.gcms,
                                          var,
                                          cds_var,
                                          area,
                                          area_name,
                                          cfg.initialized_months,
                                          years,
                                          cfg.leadtimes,
                                          cfg.download_raw_data_dir)
    