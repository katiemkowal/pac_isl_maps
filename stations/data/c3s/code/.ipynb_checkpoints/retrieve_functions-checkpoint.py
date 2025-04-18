from pathlib import Path 
import os
import cdsapi
from os import path
import helper_functions as helper
import xarray as xr
import pandas as pd

#variable conversion dictionary, check cds store to see how vars are named
cds_var_dict = {
    't2m': '2m_temperature',
    'prcp': 'total_precipitation',
    #total_precipitation': 'tprate',#for monthly
    'total_precipitation':'tp', #for daily
    'sst': 'sea_surface_temperature'}

#naming conventions can continue adding models in here as needed
c3s ={
    # this is:   ( 'originating_centre', 'system' )
    'ECMWF-SEAS5': ('ecmwf', '5'),
    'ECMWF-SEAS5.1': ('ecmwf', '51'),  
    'JMA-CPS3': ('jma', '3'),
    'JMA-CPS2': ('jma', '2'),
    'NCEP-CFSV2': ('ncep', '2'), 
    'ECCC-CANSIPS-IC3': ('eccc', '3'), 
    'ECCC-CANSIPS-IC4': ('eccc', '4'),
    'DWD-GCFSV2.1': ('dwd', '21'), 
    'DWD-GCFSV2.0': ('dwd', '2'),
    'CMCC-SPSV3.5': ('cmcc', '35'), 
    'CMCC-SPSV3.0': ('cmcc', '3'),
    'METEOFRANCE-SYSTEM5': ('meteo_france', '5'),
    'METEOFRANCE-SYSTEM6': ('meteo_france', '6'),
    'METEOFRANCE-SYSTEM7': ('meteo_france', '7'),
    'METEOFRANCE-SYSTEM8': ('meteo_france', '8'),
    'UKMO-GLOSEA603': ('ukmo', '603'),
    'UKMO-GLOSEA602': ('ukmo', '602'),
    'UKMO-GLOSEA601': ('ukmo', '601'),
    'UKMO-GLOSEA6': ('ukmo', '6') 
}

#convert names in C3S to names to make them more aligned with names used in NMME
c3s_to_nmme ={
 #naming convention
    'ECMWF-SEAS5': 'SEAS5',
    'ECMWF-SEAS5.1': 'SEAS51',
    'NCEP-CFSV2': 'CFSv2', 
    'ECCC-CANSIPS-IC3': 'CanSIPSIC3', 
    'ECCC-CANSIPS-IC4': 'CanSIPSIC4',
    'DWD-GCFSV2.1': 'GCFS2p1', 
    'CMCC-SPSV3.5': 'SPSv3p5', 
    'UKMO-GLOSEA601': 'GLOSEA6',
    'UKMO-GLOSEA603': 'GLOSEA6',
    'METEOFRANCE-SYSTEM8': 'METEOFRANCE8'
}


#dictionary to help convert month names to numbers for c3s download and back again
months_str_dict = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12',
    '01': 'Jan',
    '02': 'Feb',
    '03': 'Mar',
    '04': 'Apr',
    '05': 'May',
    '06': 'Jun',
    '07': 'Jul',
    '08': 'Aug',
    '09': 'Sep',
    '10': 'Oct',
    '11': 'Nov',
    '12': 'Dec'
}


# #downloads c3s forecasts from cds at the monthly scale using single or pressure level downloads
# #inputs: 
# #        download_type: string that specifies whether downloading single or pressure level data, options: 'single', 'pressure', or 'single_daily'
# #        pressure_level: string of pressure_level to download, can sa 'na' if using single levels
# #        gcms: string of gcms list to include in download - see c3s dictionary above for options
# #        variable: string of variable to download - 'prcp' is precipitation
# #        cds_variable: how cds labels the variable
# #        area: [N,W,S,E] integer list extent to download data - use larger zone if calibrating later
# #        initialized_months: list of month numbers for forecast initialization lists, e..g ['01', '02'...]
# #        years: years to download data
# #        leadtimes: list of string numbers ['01', '02'... up to '06'] if months
# #                  OR light of string numbers for hours ["168","336","504","672"] #for 1wk, 2wk, 3wk, 4wk
# #        staging_dir: raw download folder for grib files from cds
# returns all downloaded raw grib files into a staging directory ready for further manipulation
def download_cds_initial(download_type, pressure_level, gcms, variable, cds_variable, area, area_name, initialized_months, years, leadtimes, staging_dir):
    
    c = cdsapi.Client()

    initialized_dates = []
    for i in initialized_months:
        print(i)
        models = []
        #for each gcm you wish to download
        for g in gcms:
            file_name = g + '_' + i + variable + '_' + area_name + download_type
            staging_file = os.path.join(staging_dir, file_name + '.grib')
            #if the grib file has not been downloading from cds yet
            if not Path(staging_file).is_file():
                print('downloading ' + staging_file)
                #to download single level variables
                if 'GLOSEA6' in g:
                    h_years = years
                    g_system = c3s['UKM0-GLOSEA603'][0]
                    
                if download_type == 'single':
                    c.retrieve(
                        'seasonal-monthly-single-levels',
                        {
                            'originating_centre': c3s[g][0],
                            'system': c3s[g][1],
                            'product_type': ['hincast_climate_mean', 'monthly_mean'],
                            'variable': cds_variable,
                            'year': years,
                            'month': months_str_dict[i],
                            'area': area,
                            'leadtime_month': leadtimes,
                            'data_format': 'grib',
                        },
                        os.path.join(staging_dir,'{}.grib'.format(file_name))
                    )
                elif download_type == 'daily_single':
                    c.retrieve(
                        'seasonal-original-single-levels',
                        {
                            'originating_centre': c3s[g][0],
                            'system': c3s[g][1],
                            'variable': cds_variable,
                            'year': years,
                            'month': months_str_dict[i],
                            'area': area,
                            'day': ['01'],
                            'leadtime_hour': leadtimes,
                            'data_format': 'grib',
                        },
                        os.path.join(staging_dir,'{}.grib'.format(file_name))
                    )
                #to download pressure level variables
                elif download_type == 'pressure':
                    c.retrieve(
                        'seasonal-monthly-pressure-levels',
                        {
                            'originating_centre': c3s[g][0],
                            'system': c3s[g][1],
                            'product_type': ['hindcast_climate_mean', 'monthly_mean'],
                            'variable': cds_variable,
                            'year': years,
                            'month': months_str_dict[i],
                            'area': area,
                            'pressure_level': pressure_level,
                            'leadtime_month': leadtimes,
                            'data_format': 'grib',
                        },
                        os.path.join(staging_dir,'{}.grib'.format(file_name))
                    )
            else: print(file_name + ' already downloaded')
            
            
#prepare raw grib files from cds as xarray datasets - 
# this function calculates leadtimes as months and regrids gcms to observations of choice
# #inputs: 
# #        forecast_type: str of type of forecast 'predictand' or 'predictor' used to save space (same predictor sets can be used for multipe predictand locations
# #        gcms: string of gcms list to include in download - see c3s dictionary above for options
# #        variable: string of variable to download - e.g. 'prcp' is precipitation
# #        cds_variable: how cds labels the variable
# #        area: list of strings of extents to crop region
# #        area_name: string of name of predictand region that is being assessed
# #        initialized_months: list of month numbers for forecast initialization lists, e..g ['01', '02'...]
# #        years: years to download data
# #        obs_to_regrid: observations to use as baseline to regrid the models to a common grid
# #        staging_dir: download folder for grib files from cds
# returns an xarray dataset with all lead times and models concatenated
def prep_cds_grib_files(gcms, variable, cds_variable, area, area_name, initialized_months, download_type, staging_dir, final_dir):
    initialized_dates = []
    for i in initialized_months:
        models = []
        #for each gcm you wish to use
        for g in gcms:
            print(i + ' ' + g + ' ' + variable)
            file_name = g + '_' + i + variable + '_' + area_name + download_type
            final_file = g + '_' + i + variable + '_' + area_name + download_type + '.nc'
            staging_file = os.path.join(staging_dir, file_name + '.grib')
    #once the grib file has been downloaded, some formatting adjustment steps to get the initialization and steps right and convert to a netcdf file
            if not Path(os.path.join(final_dir, final_file)).is_file():
                if Path(staging_file).is_file(): 
                    if download_type == 'daily_single':
                        ds = xr.open_dataset(staging_file, engine='cfgrib')
                        if variable in ['prcp']:
                            ds = getattr(ds, 'tp')
                    else:
                        ds = xr.open_dataset(staging_file, engine='cfgrib', filter_by_keys={'dataType': 'fcmean'})
                        if variable in ['prcp']:
                            ds = getattr(ds, cds_var_dict[cds_variable]) #adjusting name because ECMWF stores its total_precipitation call as 'tprate'
                        else: ds = getattr(ds, variable)
                    ds = ds.mean(dim = 'number') # this will convert individual member downloads into the ensemble mean to process
                    
                    leads = []
                    unique_steps = helper.convert_to_list(ds.step.values)
                    if isinstance(ds.step.values, list) or (len(ds.step.values) > 1):
                        step_number = unique_steps[0]
                    else: step_number = [1]
                    
                    for s, step in enumerate(step_number):
                        if len(step_number) > 1:
                            one_lead = ds.isel(step = s)
                        else: one_lead = ds
                        one_lead = one_lead.to_dataset(name = variable)
                        if download_type == 'daily_single':
                            days = one_lead.step.values/1000000000/60/60/24
                            if any(substring in str(days) for substring in ['7']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (7))
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':1}).expand_dims('L'))
                            elif any(substring in str(days) for substring in ['14']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (14))
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':2}).expand_dims('L'))
                            elif any(substring in str(days) for substring in ['21']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (21))
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':3}).expand_dims('L'))
                            elif any(substring in str(days) for substring in ['28']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (28))
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':4}).expand_dims('L'))
                        else:
                            days = one_lead.step.values/1000000000/60/60/24
                            if any(substring in str(days) for substring in ['30', '31']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values)
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':1}).expand_dims('L'))
                            elif any(substring in str(days) for substring in ['59', '60', '61','62']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(months = (1))
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':2}).expand_dims('L'))
                            elif any(substring in str(days) for substring in ['89', '90', '91', '92']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(months = (2))
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':3}).expand_dims('L'))
                            elif any(substring in str(days) for substring in ['120', '121', '122', '123']):
                                one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(months = (3))
                                one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                leads.append(one_lead.assign_coords({'L':4}).expand_dims('L'))
                    leads = xr.concat(leads, dim = 'L')
                    leads = leads.rename({'time': 'Ti', 'latitude': 'Y', 'longitude':'X'})
                    leads = leads.assign_coords({'M':c3s_to_nmme[g]})

                    if variable == 'prcp':
                        #convert m/s units to mm/month units to align with other models
                        leads = helper.convert_ms_to_mmmonth(leads)
                    leads.to_netcdf(os.path.join(final_dir,final_file))
                else: 
                    print(staging_file + ' not downloaded yet')
            if Path(os.path.join(final_dir,final_file)).is_file():
                print('file found')
                leads = xr.open_dataset(os.path.join(final_dir, final_file))
                models.append(leads)
        print(models)
        models = xr.concat(models, dim = 'M').sortby('Ti.month')
        initialized_dates.append(models)
    initialized_dates = xr.concat(initialized_dates, dim = 'Ti')
    return initialized_dates


#############
# code to prep C3S data individually by members not ensemble mean
# can regrid data if desired
def prep_cds_grib_files_members(gcms, variable, cds_variable, area, area_name, initialized_months, download_type, staging_dir, final_dir):
    initialized_dates = []
    for i in initialized_months:
        print(i)
        models = []
        #for each gcm you wish to use
        for g in gcms:
            print(i + ' ' + g + ' ' + variable)
            file_name = g + '_' + i + variable + '_' + area_name + download_type
            final_file = g + '_' + i + variable + '_' + area_name + download_type + '.nc'
            staging_file = os.path.join(staging_dir, file_name + '.grib')
    #once the grib file has been downloaded, some formatting adjustment steps to get the initialization and steps right and convert to a netcdf file
            if not Path(final_file).is_file():
                if Path(staging_file).is_file(): 
                    ds = xr.open_dataset(staging_file, engine='cfgrib')
                    if download_type == 'daily_single':
                        if variable in ['prcp']:
                            ds = getattr(ds, 'tp')
                        else: ds = getattr(ds, variable)#cds_var_dict[cds_variable.lower()])
                    elif variable == 't2m':
                        ds = getattr(ds, variable.lower())
                    else: ds = getattr(ds, cds_var_dict[cds_variable.lower()])
                    #for NCEP-CFSv2 only keep the members with actual values for hindcasts
                    if g == 'NCEP-CFSV2':
                        ds = ds.sel(number = ds.number.isin([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18, 19, 20, 21, 22, 23]))
                        
                    #for each member individually:
                    mems=[]
                    for n in range(ds.shape[list(ds.dims).index('number')]):
                        one_member = ds.isel(number=n)
                        leads = []
                        unique_steps = helper.convert_to_list(one_member.step.values)
                        if isinstance(one_member.step.values, list) or (len(one_member.step.values) > 1):
                            step_number = unique_steps[0]
                        else: step_number = [1]
                        #if #len(one_member.step.values) == 1:
                            #step_number = [1]
                        #else: step_number = unique_steps[0]
                        for s, step in enumerate(step_number):
                            if len(step_number) > 1:
                                one_lead = one_member.isel(step = s)
                            else: one_lead = one_member
                            one_lead = one_lead.to_dataset(name = variable)
                            
                            if download_type == 'daily_single':
                                days = one_lead.step.values/1000000000/60/60/24
                                if any(substring in str(days) for substring in ['7']):
                                    one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (7))
                                    one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                    leads.append(one_lead.assign_coords({'L':1}).expand_dims('L'))
                                elif any(substring in str(days) for substring in ['14']):
                                    one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (14))
                                    one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                    leads.append(one_lead.assign_coords({'L':2}).expand_dims('L'))
                                elif any(substring in str(days) for substring in ['21']):
                                    one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (21))
                                    one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                    leads.append(one_lead.assign_coords({'L':3}).expand_dims('L'))
                                elif any(substring in str(days) for substring in ['28']):
                                    one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(days = (28))
                                    one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                    leads.append(one_lead.assign_coords({'L':4}).expand_dims('L'))
                            else:
                                days = one_lead.step.values/1000000000/60/60/24
                                if any(substring in str(days) for substring in ['30', '31']):
                                    one_lead['T'] = pd.to_datetime(one_lead['time'].values)
                                    one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                    leads.append(one_lead.assign_coords({'L':1}).expand_dims('L'))
                                elif any(substring in str(days) for substring in ['59', '60', '61','62']):

                                    one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(months = (1))
                                    one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                    leads.append(one_lead.assign_coords({'L':2}).expand_dims('L'))
                                elif any(substring in str(days) for substring in ['89', '90', '91', '92']):
                                    one_lead['T'] = pd.to_datetime(one_lead['time'].values) + pd.DateOffset(months = (2))
                                    one_lead['T'] = one_lead['T'].swap_dims({'T':'time'})
                                    leads.append(one_lead.assign_coords({'L':3}).expand_dims('L'))
                        leads = xr.concat(leads, dim = 'L')

                        leads = leads.assign_coords({'M':c3s_to_nmme[g] + '_' + str(n+1)}).expand_dims('M')
                        leads = leads.rename({'time': 'Ti', 'latitude': 'Y', 'longitude':'X'})
                        mems.append(leads)
                    mems = xr.concat(mems, 'M')
                    mems = mems.assign_coords({'GCM':g})

                    if variable == 'prcp':
                        #convert m/s units to mm/month units to align with other models
                        mems= helper.convert_ms_to_mmmonth(mems)
                    mems.to_netcdf(os.path.join(final_dir,final_file))
                
                else: print(staging_file + ' not downloaded yet')
            if Path(os.path.join(final_dir,final_file)).is_file():
                print('file found')
                mems = xr.open_dataset(os.path.join(final_dir, final_file))
                models.append(mems)
        models = xr.concat(models, dim = 'M').sortby('Ti.month')
        initialized_dates.append(models)
    initialized_dates = xr.concat(initialized_dates, dim = 'Ti')
    return initialized_dates