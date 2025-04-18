import xarray as xr
import os
import c3s_config_daily as cfg
import retrieve_functions as retrieve
import cfgrib

for var in cfg.vars:
    if var in ['t2m', 'prcp', 'sst']:
        cds_var = retrieve.cds_var_dict[var]
    else:
        cds_var = var
        
    area = [getattr(cfg,  var + '_download_extent')['north'],
            getattr(cfg,  var + '_download_extent')['west'],
            getattr(cfg,  var + '_download_extent')['south'],
            getattr(cfg,  var + '_download_extent')['east']]
    area_name = getattr(cfg,  var + '_download_name')
    
    ds_prepped = retrieve.prep_cds_grib_files_members(cfg.gcms, 
                                     var, 
                                     cds_var,
                                     area,
                                     area_name,
                                     cfg.initialized_months,
                                     cfg.download_type,
                                     cfg.download_raw_data_dir,
                                     cfg.download_processed_data_dir)
                                     
                                     
                          
                            

