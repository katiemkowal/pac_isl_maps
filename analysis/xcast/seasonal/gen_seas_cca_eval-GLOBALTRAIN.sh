#!/bin/sh
# CHOOSE YOUR CONDA ENVIRONMENT
#py=/cpc/home/kkowal/.conda/envs/xcast_new/bin/python
#py=/cpc/home/kkowal/.conda/envs/intdesk_train/bin/python
py=/cpc/home/ebekele/.conda/envs/xcast_env/bin/python

cat>gen_seas_cca_and_eval.py<<eofPY
################## LIBRARIES
import xcast as xc 
import xarray as xr 
import cartopy.crs as ccrs 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from pathlib import Path
import os
import time
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.feature import NaturalEarthFeature
import matplotlib.pyplot as plt
import glob

#yes or no depending on xcast_env version
#using older xcast_env for elr/epoelem, newer version for probabilistic forecasts
use_one_hot_update = 'no'

## add the workaround onehot-update and drymask-update script as workaround for new xcast envioront
#grab the necessary functions
import importlib.util

#onehot-update.py file and where it is located
function_folder1 = "/cpc/int_desk/pac_isl/analysis/xcast/seasonal/onehotupdate.py"

spec1 = importlib.util.spec_from_file_location(
"onehotupdate", function_folder1)    

onehot = importlib.util.module_from_spec(spec1) 
spec1.loader.exec_module(onehot)

#drymask-update.py file and where it is located
function_folder2 = "/cpc/int_desk/pac_isl/analysis/xcast/seasonal/drymaskupdate.py"

spec2 = importlib.util.spec_from_file_location(
"drymaskupdate", function_folder2)    

dry = importlib.util.module_from_spec(spec2) 
spec2.loader.exec_module(dry)

##################### CONSTANTS
#data directory
ddir='/cpc/int_desk/pac_isl/analysis/xcast/seasonal/practical_notebooks/practical_data'
ddir_processed = os.path.join(ddir, 'processed')

# figure directory
fdir='/Users/katie/Desktop/practical_data'
#'/cpc/int_desk/pac_isl/analysis/xcast/seasonal/presentation_figures'

#obs_name CMORPH or CHIRPS
obs_name='CMORPH'

#initialized dates
initial_dates = [(2024, 6, 1),(2023, 8, 1), (2023, 9, 1), (2023, 10, 1), (2023, 11,1), (2023, 12,1),(2024, 1, 1), (2024, 2, 1), (2024, 3, 1), (2023, 7,1),(2024, 4, 1), (2024, 5, 1)] 
#SET UP SOME REGIONS TO PLOT OR TRAIN

pacislands_coordinates = {
    'west': 130,
    'east': 205,
    'north': 8,
    'south': -20
    }

chuuk_coordinates = {
    'west': 151,
    'east': 153,
    'north': 8,
    'south': 6
    }

fiji_coordinates = {
    'west':  177,
    'east': 182,  
    'north': -15,  
    'south': -20}

kiribati_coordinates = {
        'west':  202,
        'east': 203,  
        'north': 2.5,  
        'south': 1}

solomon_coordinates = {
        'west':  155,
        'east': 167,  
        'north': -6,  
        'south': -13}

png_coordinates = {
        'west':  130,
        'east': 156,  
        'north': 1,  
        'south': -12}
        
palau_coordinates = {
        'west':  133,
        'east': 135,  
        'north': 8,  
        'south': 6
}
        
vanuatu_coordinates = {
        'west':  165,
        'east': 170,  
        'north': -12.5,  
        'south': -20
}

samoa_coordinates = {
        'west':  187,
        'east': 191,  
        'north': -13,  
        'south': -15
}

tuvalu_coordinates = {
        'west':  179,
        'east': 180,  
        'north': -8,  
        'south': -9
}

#Pacific region, encompassing all islands
pacific_extent = {
    'west': 120,
    'east': 210,
    'north': 10,
    'south': -30
}

#Global extent
global_extent = {
    'west': 0,
    'east': 360,
    'north': 30,
    'south': -30
}


predictor_train_extent = global_extent
predictor_train_extent_name = 'global'

regions = [pacislands_coordinates, fiji_coordinates, kiribati_coordinates, solomon_coordinates, png_coordinates, palau_coordinates, samoa_coordinates, tuvalu_coordinates, vanuatu_coordinates, chuuk_coordinates]
region_names = ['Pacific Islands', 'Fiji', 'Kiribati', 'Solomon Islands', 'Papua New Guinea', 'Palau', 'Samoa', 'Tuvalu', 'Vanuatu', 'Chuuk']
#'Chuuk', chuuk_coordinates,


#month dictionary
number_to_month_name_dictionary = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec',
    0: 'Dec'
}


############ MAIN CODE ###########################

#################### CALCULATE
##### generate seasons of interest for initialization dates
initial_months, initial_month_names, target_seasons, target_months = [], [], [], []
 
for i in initial_dates:
    leads = [['1', '3'],['2', '4'], ['3','5']]
    initial_month = dt.datetime(*i).month
    initial_months.append(initial_month)
    initial_month_names.append(number_to_month_name_dictionary[initial_month])
    target_month = []
    target_seas = []
    for l in leads:
        target_low = number_to_month_name_dictionary[(initial_month + float(l[0]))%12]
        target_mid = number_to_month_name_dictionary[(initial_month + float(l[0])+1)%12]
        target_high = number_to_month_name_dictionary[(initial_month + float(l[1]))%12]
        target_seas.append('-'.join([target_low, target_high]))
        target_month.append(target_low[0] + target_mid[0] + target_high[0])
    target_seasons.append(target_seas)
    target_months.append(target_month)

### for every initialized month, do the following
for t, initial_month_name in enumerate(initial_month_names):
    target_seas = target_seasons[t]
    ###### read in observations
    if obs_name == 'CMORPH':
        training_length = 'threeseas'
    elif obs_name == 'CHIRPS':
        training_length = 'oneseas'
    obs_leads = xr.open_dataset(os.path.join(ddir, '_'.join([initial_month_name, training_length, obs_name, 'precip.nc'])))

    hindcast_data_precip = xr.open_dataset(os.path.join(ddir, '_'.join([initial_month_name, training_length, 'NMME_hcst_precip.nc'])))
    forecast_data_precip = xr.open_dataset(os.path.join(ddir, '_'.join([initial_month_name, training_length, 'NMME_fcst_precip.nc'])))  
    
    hindcast_data_sst = xr.open_dataset(os.path.join(ddir, '_'.join([initial_month_name, training_length, 'NMME_hcst_sst.nc'])))
    forecast_data_sst = xr.open_dataset(os.path.join(ddir, '_'.join([initial_month_name, training_length, 'NMME_fcst_sst.nc'])))  

    msk = xr.open_dataset('/cpc/africawrf/ebekele/projects/PREPARE_pacific/notebooks/masked/libs/pacific_mask.nc')
    mskk = msk.amask.expand_dims({'M':[0]})
    mskk = mskk.assign_coords({'lon': [i + 360 if i <= 0 else i for i in mskk.coords['lon'].values]}).sortby('lon').drop_duplicates('lon')
    mskk = mskk.rename({'lon':'X', 'lat':'Y', 'time':'T'})
    mskk = xc.regrid(mskk, obs_leads.X, obs_leads.Y)
    mask_missing = mskk.mean('T', skipna=False).mean('M', skipna=False)
    mask_missing = xr.ones_like(mask_missing).where(~np.isnan(mask_missing), other=np.nan )
    
    hindcast_data_precip = hindcast_data_precip.sel(X=slice(predictor_train_extent['west'], predictor_train_extent['east']), Y=slice(predictor_train_extent['south'], predictor_train_extent['north']))
    forecast_data_precip = forecast_data_precip.sel(X=slice(predictor_train_extent['west'], predictor_train_extent['east']), Y=slice(predictor_train_extent['south'], predictor_train_extent['north']))
    
    #NMME sst data
    hindcast_data_sst = hindcast_data_sst.sel(X=slice(predictor_train_extent['west'], predictor_train_extent['east']), Y=slice(predictor_train_extent['south'], predictor_train_extent['north']))
    forecast_data_sst = forecast_data_sst.sel(X=slice(predictor_train_extent['west'], predictor_train_extent['east']), Y=slice(predictor_train_extent['south'], predictor_train_extent['north']))
    
    ########### for every predictand region you want to train on
    for r, region in enumerate(regions):
        #expand the training area to help create more to train for smaller islands
        predictand_train_extent = {
            'west':  region['west']-5,
                'east': region['east']+5,  
                'north': region['north']+3,  
                'south': region['south']-2
        }
        
        #chuuk fell out of ocean mask, workaround below
        if region == chuuk_coordinates:
            obs_leads = obs_leads.drop_duplicates(dim = 'X')
        else:
            obs_leads = obs_leads.drop_duplicates(dim = 'X') * mask_missing.drop_duplicates(dim = 'X')
        
        #crop the observations to your training region of choice
        obs_leads = obs_leads.sel(X=slice(predictand_train_extent['west'], predictand_train_extent['east']), Y=slice(predictand_train_extent['south'], predictand_train_extent['north']))

        ##### run cca on NMME precip forecasts
        start_time = time.time()
        cca_fcsts_prob_precip, cca_fcsts_det_precip, cca_hcasts_det_precip, cca_hcasts_prob_precip, obs_to_test_precip = [],[],[],[],[]

        for l in np.unique(hindcast_data_precip.L):
            model = hindcast_data_precip.sel(L=l).precip
            obs = obs_leads.sel(L=l).precip
            fmodel = forecast_data_precip.sel(L=l).precip

            #create a dry mask to avoid training over zero values
            drymask = xc.drymask(obs, dry_threshold=0.1, quantile_threshold=0.3)
            obs = obs * mask_missing.drop_duplicates(dim='X')
            obs = obs * drymask.drop_duplicates(dim = 'X')
            #run CCA
            hindcasts_det, hindcasts_prob, obs_test = [], [], []
            i=1
            for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model, obs, window=5):
                print("window {}".format(i))
                i += 1
                reg = xc.CCA(search_override=(5,
                                              5,
                                             3))
                reg.fit(xtrain, ytrain)
                preds = reg.predict(xtest)
                probs =  reg.predict_proba(xtest)
                obs_test.append(ytest)
                hindcasts_det.append(preds)
                hindcasts_prob.append(probs)
            hindcasts_det = xr.concat(hindcasts_det, 'T')
            hindcasts_prob = xr.concat(hindcasts_prob, 'T')
            obs_test = xr.concat(obs_test, 'T')

            fprobs =  reg.predict_proba(fmodel)

            cca_fcsts_prob_precip.append(fprobs)
            cca_hcasts_det_precip.append(hindcasts_det)
            cca_hcasts_prob_precip.append(hindcasts_prob)
            obs_to_test_precip.append(obs_test)
        cca_fcsts_prob_precip = xr.concat(cca_fcsts_prob_precip, dim = 'L')
        cca_hcasts_det_precip = xr.concat(cca_hcasts_det_precip, dim = 'L')
        cca_hcasts_prob_precip = xr.concat(cca_hcasts_prob_precip, dim = 'L')
        obs_to_test_precip = xr.concat(obs_to_test_precip, dim = 'L')
        print('cca precip processing time is ' + str(time.time() - start_time))
        
        ## run cca on NMME SST forecasts
        start_time = time.time()
        cca_fcsts_prob_sst, cca_fcsts_det_sst, cca_hcasts_det_sst, cca_hcasts_prob_sst, obs_to_test_sst = [],[],[],[],[]

        for l in np.unique(hindcast_data_sst.L):
            model = hindcast_data_sst.sel(L=l).sst
            obs = obs_leads.sel(L=l).precip
            fmodel = forecast_data_sst.sel(L=l).sst

            #create a dry mask to avoid training over zero values
            drymask = xc.drymask(obs, dry_threshold=0.1, quantile_threshold=0.3)
            obs = obs * mask_missing.drop_duplicates(dim = 'X')
            obs = obs * drymask.drop_duplicates(dim = 'X')
            #run CCA
            hindcasts_det, hindcasts_prob, obs_test = [], [], []
            i=1
            for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model, obs, window=5):
                print("window {}".format(i))
                i += 1
                reg = xc.CCA(search_override=(5,
                                              5,
                                             3))
                reg.fit(xtrain, ytrain)
                preds = reg.predict(xtest)
                probs =  reg.predict_proba(xtest)
                obs_test.append(ytest)
                hindcasts_det.append(preds)
                hindcasts_prob.append(probs)
            hindcasts_det = xr.concat(hindcasts_det, 'T')
            hindcasts_prob = xr.concat(hindcasts_prob, 'T')
            obs_test = xr.concat(obs_test, 'T')

            fprobs =  reg.predict_proba(fmodel)

            cca_fcsts_prob_sst.append(fprobs)
            cca_hcasts_det_sst.append(hindcasts_det)
            cca_hcasts_prob_sst.append(hindcasts_prob)
            obs_to_test_sst.append(obs_test)
        cca_fcsts_prob_sst = xr.concat(cca_fcsts_prob_sst, dim = 'L')
        cca_hcasts_det_sst = xr.concat(cca_hcasts_det_sst, dim = 'L')
        cca_hcasts_prob_sst = xr.concat(cca_hcasts_prob_sst, dim = 'L')
        obs_to_test_sst = xr.concat(obs_to_test_sst, dim = 'L')
        print('cca sst processing time is ' + str(time.time() - start_time))
        
        
        ##create ELR and EPOELM forecasts
        start_time = time.time()
        elr_fcsts_prob, elr_fcsts_det, elr_hcasts_det, elr_hcasts_prob = [], [], [], []
        epoelm_fcsts_prob, epoelm_fcsts_det, epoelm_hcasts_det, epoelm_hcasts_prob = [], [], [], []
        obs_to_test_grid, raw_to_test_grid = [],[]

        for l in np.unique(hindcast_data_precip.L):
            obs = obs_leads.sel(L=l).precip
            model = hindcast_data_precip.sel(L=l).precip
            fmodel = forecast_data_precip.sel(L=l).precip

            model_regrid = xc.regrid(model, obs.X, obs.Y)
            fmodel_regrid = xc.regrid(fmodel, obs.X, obs.Y)

            obs, model_regrid = xc.match(obs, model_regrid)

            #run ELR, EPOELM
            hindcasts_prob_ELR, hindcasts_prob_EPOELM, obs_test_grid, raw_test_grid = [], [], [], []
            i=1
            for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model_regrid, obs, window=5):
                print("window {}".format(i))
                i += 1
                reg_ELR = xc.ELR()
                reg_ELR.fit(xtrain, ytrain)

                reg_EPOELM = xc.EPOELM()
                reg_EPOELM.fit(xtrain, ytrain)

                probs_ELR =  reg_ELR.predict_proba(xtest)
                probs_EPOELM =  reg_EPOELM.predict_proba(xtest)

                obs_test_grid.append(ytest)
                raw_test_grid.append(xtest)
                hindcasts_prob_ELR.append(probs_ELR)
                hindcasts_prob_EPOELM.append(probs_EPOELM)

            hindcasts_prob_ELR = xr.concat(hindcasts_prob_ELR, 'T')
            hindcasts_prob_EPOELM = xr.concat(hindcasts_prob_EPOELM, 'T')
            obs_test_grid = xr.concat(obs_test_grid, 'T')
            raw_test_grid = xr.concat(raw_test_grid, 'T')

            fprobs_ELR =  reg_ELR.predict_proba(fmodel_regrid)
            fprobs_EPOELM =  reg_EPOELM.predict_proba(fmodel_regrid)

            elr_fcsts_prob.append(fprobs_ELR)
            elr_hcasts_prob.append(hindcasts_prob_ELR)
            epoelm_fcsts_prob.append(fprobs_EPOELM)
            epoelm_hcasts_prob.append(hindcasts_prob_EPOELM)
            obs_to_test_grid.append(obs_test_grid)
            raw_to_test_grid.append(raw_test_grid)

        elr_fcsts_prob = xr.concat(elr_fcsts_prob, dim = 'L')
        elr_hcasts_prob = xr.concat(elr_hcasts_prob, dim = 'L')
        epoelm_fcsts_prob = xr.concat(epoelm_fcsts_prob, dim = 'L')
        epoelm_hcasts_prob = xr.concat(epoelm_hcasts_prob, dim = 'L')
        obs_to_test_grid = xr.concat(obs_to_test_grid, dim = 'L')
        raw_to_test_grid = xr.concat(raw_to_test_grid, dim = 'L')
        print('elr/epoelm processing time is ' + str(time.time() - start_time))

    ################# EVALUATE the skill of the raw and bias corrected forecasts

        #PEARSON CALCULATION
        #run this to compare cca-sst cca-precip and raw nmme hindcasts
        start_time = time.time()
        #calculate pearson correlation score for hindcasts
        pearson_cca_precip, pearson_cca_sst, pearson_raw = [], [], []
        for l, lead in enumerate(np.unique(hindcast_data_precip.L.values)):

            obs_raw = obs_to_test_grid.isel(L=l)
            raw_regrid = raw_to_test_grid.isel(L=l)

            cca_pearson_calc_precip = xc.Pearson(cca_hcasts_det_precip.isel(L=l),obs_to_test_precip.isel(L=l))
            cca_pearson_calc_precip = cca_pearson_calc_precip.expand_dims({'M':['NMME CCA (Precip)']})

            cca_pearson_calc_sst = xc.Pearson(cca_hcasts_det_sst.isel(L=l),obs_to_test_sst.isel(L=l))
            cca_pearson_calc_sst = cca_pearson_calc_sst.expand_dims({'M':['NMME CCA (SST)']})

            #calc pearson correlation
            pearson_raw_calc = []
            for m, model in enumerate(np.unique(raw_regrid.M.values)):
                pearson_raw_c = xc.Pearson(raw_regrid.sel(M=model).expand_dims({'M':[model + ' Raw']}), 
                                                   obs_raw)
                pearson_raw_c = pearson_raw_c.expand_dims({'M':[model + ' Raw']})
                pearson_raw_calc.append(pearson_raw_c)
            pearson_raw_calc = xr.concat(pearson_raw_calc, dim = 'M')
            pearson_cca_precip.append(cca_pearson_calc_precip)
            pearson_cca_sst.append(cca_pearson_calc_sst)
            pearson_raw.append(pearson_raw_calc)
        pearson_cca_precip = xr.concat(pearson_cca_precip, dim = 'L')
        pearson_cca_sst = xr.concat(pearson_cca_sst, dim = 'L')
        pearson_raw = xr.concat(pearson_raw, dim = 'L')
        pearsons = xr.concat([pearson_cca_precip, pearson_cca_sst, pearson_raw], dim = 'M')
        print('Pearson processing time is ' + str(time.time() - start_time))
        print(pearsons)

        #GROCS CALCULATION
        start_time = time.time()
        grocs_cca_precip, grocs_cca_sst, grocs_elr, grocs_epoelm = [], [], [], []
        for l, lead in enumerate(np.unique(hindcast_data_precip.L.values)):

            hind_prob_cca_precip = xc.gaussian_smooth(cca_hcasts_prob_precip.isel(L=l), kernel=3)
            obs_cca_precip = xc.gaussian_smooth(obs_to_test_precip.isel(L=l), kernel=3)

            hind_prob_cca_sst = xc.gaussian_smooth(cca_hcasts_prob_sst.isel(L=l), kernel=3)
            obs_cca_sst = xc.gaussian_smooth(obs_to_test_sst.isel(L=l), kernel=3)

            hind_prob_elr = xc.gaussian_smooth(elr_hcasts_prob.isel(L=l), kernel=3)
            hind_prob_epoelm = xc.gaussian_smooth(epoelm_hcasts_prob.isel(L=l), kernel=3)
            obs_grid = xc.gaussian_smooth(obs_to_test_grid.isel(L=l), kernel=3)


            #transform obs into tercile based categories
            ohc_precip = xc.OneHotEncoder() 
            #swap these if using new xcast environment
            #ohc = onehot.OneHotEncoder()
            ohc_precip.fit(obs_cca_precip)
            T_precip = ohc_precip.transform(obs_cca_precip)
            clim = xr.ones_like(T_precip) * 0.333

            #transform obs into tercile based categories
            ohc_sst = xc.OneHotEncoder() 
            #swap these if using new xcast environment
            #ohc = onehot.OneHotEncoder()
            ohc_sst.fit(obs_cca_sst)
            T_sst = ohc_sst.transform(obs_cca_sst)
            clim = xr.ones_like(T_sst) * 0.333

            #transform obs into tercile based categories
            ohc_grid = xc.OneHotEncoder() 
            #swap these if using new xcast environment
            #ohc = onehot.OneHotEncoder()
            ohc_grid.fit(obs_grid)
            T_grid = ohc_sst.transform(obs_grid)
            clim = xr.ones_like(T_grid) * 0.333

            grocs_cca_l_precip = xc.GROCS(hind_prob_cca_precip, T_precip)
            grocs_cca_l_sst = xc.GROCS(hind_prob_cca_sst, T_sst)
            grocs_elr_l = xc.GROCS(hind_prob_elr, T_grid)
            grocs_epoelm_l = xc.GROCS(hind_prob_epoelm, T_grid)
            grocs_cca_l_precip = grocs_cca_l_precip.expand_dims({'M':['CCA (Precip)']})
            grocs_cca_l_sst = grocs_cca_l_sst.expand_dims({'M':['CCA (SST)']})
            grocs_elr_l = grocs_elr_l.expand_dims({'M':['ELR']})
            grocs_epoelm_l = grocs_epoelm_l.expand_dims({'M':['EPOELM']})

            grocs_cca_precip.append(grocs_cca_l_precip)
            grocs_cca_sst.append(grocs_cca_l_sst)
            grocs_elr.append(grocs_elr_l)
            grocs_epoelm.append(grocs_epoelm_l)

        grocs_cca_precip = xr.concat(grocs_cca_precip, dim = 'L')
        grocs_cca_sst = xr.concat(grocs_cca_sst, dim = 'L')
        grocs_elr = xr.concat(grocs_elr, dim = 'L')
        grocs_epoelm = xr.concat(grocs_epoelm, dim = 'L')
        grocs = xr.concat([grocs_cca_precip, grocs_cca_sst, grocs_elr, grocs_epoelm], dim = 'M')
        print('GROCS processing time is ' + str(time.time() - start_time))
        print(grocs)

        ########## CREATE A CONSOLIDATED WEIGHTED ENSEMBLE BASED ON THE GROCS SCORES
   
        sr_grocs = []
        for model in grocs.M.values:
            sr_grocs_l = []
            for l in grocs.L.values:
                grocs_test = grocs.sel(M=model, L=l)
                mask = grocs_test < 0.5
                grocs_test = grocs_test.where(~mask,0)
                sr_grocs_test = grocs_test * grocs_test
                sr_grocs_test = sr_grocs_test.to_dataset(name = 'GROCS_WEIGHT')
                sr_grocs_test = sr_grocs_test.expand_dims({'M': [model]})
                sr_grocs_l.append(sr_grocs_test)
            sr_grocs_l = xr.concat(sr_grocs_l, dim = 'L')
            sr_grocs.append(sr_grocs_l)
        sr_grocs = xr.concat(sr_grocs, dim = 'M')

        utt = cca_fcsts_prob_precip * sr_grocs.sel(M = sr_grocs.M.isin('CCA (Precip)'))  + cca_fcsts_prob_sst * sr_grocs.sel(M = sr_grocs.M.isin('CCA (SST)')) + elr_fcsts_prob * sr_grocs.sel(M = sr_grocs.M.isin('ELR')) + epoelm_fcsts_prob * sr_grocs.sel(M = sr_grocs.M.isin('EPOELM'))
        btt = sr_grocs.sel(M = sr_grocs.M.isin('CCA (Precip)')) + sr_grocs.sel(M = sr_grocs.M.isin('CCA (SST)')) + sr_grocs.sel(M = sr_grocs.M.isin('ELR')) + sr_grocs.sel(M = sr_grocs.M.isin('EPOELM'))
       
        pcons = []
        for l, lead in enumerate(grocs.L.values):
            pcons_m = []
            for m, tercile in enumerate(cca_fcsts_prob_precip.M.values):
                cca_fcsts_prob_precip_m = cca_fcsts_prob_precip.isel(L=l, M=m, T = 0)
                cca_fcsts_prob_sst_m = cca_fcsts_prob_sst.isel(L=l, M=m, T = 0)
                elr_fcsts_prob_m = elr_fcsts_prob.isel(L=l, M=m, T = 0).transpose('Y', 'X')
                epoelm_fcsts_prob_m = epoelm_fcsts_prob.isel(L=l, M=m, T = 0).transpose('Y', 'X')

                weighted_cca_precip = cca_fcsts_prob_precip_m* sr_grocs.sel(M = sr_grocs.M.isin('CCA (Precip)')).isel(L=l, M=0).GROCS_WEIGHT.values
                weighted_cca_sst = cca_fcsts_prob_sst_m * sr_grocs.sel(M = sr_grocs.M.isin('CCA (SST)')).isel(L=l, M=0).GROCS_WEIGHT.values
                weighted_elr = elr_fcsts_prob_m * sr_grocs.sel(M = sr_grocs.M.isin('ELR')).isel(L=l, M=0).GROCS_WEIGHT.values
                weighted_epoelm = epoelm_fcsts_prob_m * sr_grocs.sel(M = sr_grocs.M.isin('EPOELM')).isel(L=l, M=0).GROCS_WEIGHT.values 

                squared_cca_precip_grocs = sr_grocs.sel(M = sr_grocs.M.isin('CCA (Precip)')).isel(L=l, M=0).GROCS_WEIGHT.values
                squared_cca_sst_grocs = sr_grocs.sel(M = sr_grocs.M.isin('CCA (SST)')).isel(L=l, M= 0).GROCS_WEIGHT.values
                squared_elr_grocs = sr_grocs.sel(M = sr_grocs.M.isin('ELR')).isel(L=l, M= 0).GROCS_WEIGHT.values
                squared_epoelm_grocs = sr_grocs.sel(M = sr_grocs.M.isin('EPOELM')).isel(L=l, M=0).GROCS_WEIGHT.values

                utt = weighted_cca_precip + weighted_cca_sst + weighted_elr + weighted_epoelm
                btt = squared_cca_precip_grocs + squared_cca_sst_grocs + squared_elr_grocs + squared_epoelm_grocs

                p = (utt)/btt
                p = p.to_dataset(name = 'pcons').expand_dims({'M': [tercile]})
                pcons_m.append(p)
            pcons_m = xr.concat(pcons_m, dim = 'M')
            pcons_m = pcons_m.expand_dims({'L':[l]})
            pcons.append(pcons_m)
        pcons = xr.concat(pcons, dim = 'L')

        pcons.to_netcdf(os.path.join(ddir_processed, '_'.join([initial_month_name, region_names[r], predictor_train_extent_name, obs_name.split('.')[0],'pcons.nc'])))
        

    ################ PLOT THE RESULTS
        ###### PEARSONS PLOTS

        models = np.unique(pearsons.M.values)
        models = np.flip(models, axis = 0)

        if region_names[r] == 'Pacific Islands':
            for r1, region_of_interest in enumerate(region_names):

                fig, axes = plt.subplots(nrows=len(models), ncols=len(target_seas), figsize=(10, (len(models))*2 + 2), 
                                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

                # Set the extent to cover the entire world
                for ax in axes.flat:
                    ax.set_global()

                for j, model in enumerate(models):
                    for i, season in enumerate(target_seas):
                        ax = axes[j, i]
                        # Your plotting code here using the specific model and season
                        xplot = pearsons.isel(L=i, M=j).plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=21, vmin=-1, vmax=1, add_colorbar=False)
                        ax.coastlines()
                        c = ax.coastlines()
                        c = ax.gridlines(draw_labels=True, linewidth=0.3)
                        c.right_labels = False
                        c.top_labels = False 
                        # Add country borders
                        ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                                            scale='50m', edgecolor='black', facecolor='none'))
                        # Set the extent to cover the specific area
                        ax.set_extent([regions[r1]['west'], regions[r1]['east'], regions[r1]['south'], regions[r1]['north']], crs=ccrs.PlateCarree())
                        ax.set_title(f'{model} - {season}')

                # Add a single horizontal colorbar below the panel plot
                cbar_ax = fig.add_axes([0.15, 0.002, 0.6, 0.02])  # [left, bottom, width, height]
                cbar = fig.colorbar(xplot, cax=cbar_ax, orientation='horizontal', shrink =1, pad = 0.3)
                cbar.set_label(region_of_interest + ' Pearson Correlation', fontsize=13)
                cbar.ax.tick_params(labelsize=14)
                # Adjust layout
                plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)

                # Show plot
                plt.savefig(os.path.join(fdir, '_'.join([initial_month_name, 'PacIsltrain', region_of_interest, 'pearson_CCA', obs_name.split('.')[0]])), bbox_inches='tight', dpi=100)
                plt.close()
        else:
            fig, axes = plt.subplots(nrows=len(models), ncols=len(target_seas), figsize=(10, (len(models))*2 + 2), 
                                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

            # Set the extent to cover the entire world
            for ax in axes.flat:
                ax.set_global()

            for j, model in enumerate(models):
                for i, season in enumerate(target_seas):
                    ax = axes[j, i]
                    # Your plotting code here using the specific model and season
                    xplot = pearsons.isel(L=i, M=j).plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=21, vmin=-1, vmax=1, add_colorbar=False)
                    ax.coastlines()
                    c = ax.coastlines()
                    c = ax.gridlines(draw_labels=True, linewidth=0.3)
                    c.right_labels = False
                    c.top_labels = False 
                    # Add country borders
                    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                                        scale='50m', edgecolor='black', facecolor='none'))
                    # Set the extent to cover the specific area
                    ax.set_extent([regions[r]['west'], regions[r]['east'], regions[r]['south'], regions[r]['north']], crs=ccrs.PlateCarree())
                    ax.set_title(f'{model} - {season}')

            # Add a single horizontal colorbar below the panel plot
            cbar_ax = fig.add_axes([0.15, 0.002, 0.6, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(xplot, cax=cbar_ax, orientation='horizontal', shrink =1, pad = 0.3)
            cbar.set_label(region_names[r] + ' Pearson Correlation', fontsize=13)
            cbar.ax.tick_params(labelsize=14)
            # Adjust layout
            plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)

            # Show plot
            plt.savefig(os.path.join(fdir, '_'.join([initial_month_name, region_names[r], 'pearson_CCA', predictor_train_extent_name + '-train', obs_name.split('.')[0]])), bbox_inches='tight', dpi=100)
            plt.close()

        ### GROCS WEIGHTS PLOTS
        models = np.unique(sr_grocs.M.values)
        models = np.flip(models, axis = 0)

        if region_names[r] == 'Pacific Islands':
            for r1, region_of_interest in enumerate(region_names):
                fig, axes = plt.subplots(nrows=len(models), ncols=len(target_seas), figsize=(6, (len(models)+1)*2 + 2), 
                                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

                # Set the extent to cover the entire world
                for ax in axes.flat:
                    ax.set_global()

                for j, model in enumerate(sr_grocs.M.values):
                    for i, season in enumerate(target_seas):
                        ax = axes[j,i]
                        # Your plotting code here using the specific model and season
                        xplot = sr_grocs.GROCS_WEIGHT.isel(L=i, M=j).plot(ax=ax,transform=ccrs.PlateCarree(), cmap='coolwarm', levels=21, vmin=0, vmax=.7, add_colorbar=False)
                        ax.coastlines()
                        c = ax.coastlines()
                        c = ax.gridlines(draw_labels=True, linewidth=0.3)
                        c.right_labels = False
                        c.top_labels = False 
                        # Add country borders
                        ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                                            scale='50m', edgecolor='black', facecolor='none'))
                        # Set the extent to cover the specific area
                        ax.set_extent([regions[r1]['west'], regions[r1]['east'], regions[r1]['south'], regions[r1]['north']], crs=ccrs.PlateCarree())
                        ax.set_title(f'{model} - {season}')
                # Add a single horizontal colorbar below the panel plot
                cbar_ax = fig.add_axes([0.15, 0.002, 0.6, 0.02])  # [left, bottom, width, height]
                cbar = fig.colorbar(xplot, cax=cbar_ax, orientation='horizontal', shrink =1, pad = 0.3)
                cbar.set_label(region_of_interest + ' GROCS', fontsize=13)
                cbar.ax.tick_params(labelsize=14)
                # Adjust layout
                plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)

                # Show plot
                plt.savefig(os.path.join(fdir, '_'.join([initial_month_name, 'PacIsltrain' + region_of_interest, 'GROCS', obs_name.split('.')[0]])), bbox_inches='tight', dpi=100)
                plt.close()
        else:
            fig, axes = plt.subplots(nrows=len(models), ncols=len(target_seas), figsize=(5, (len(models))*2 + 2), 
                                         subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

            # Set the extent to cover the entire world
            #for ax in axes.flat:
            #    ax.set_global()

            for j, model in enumerate(sr_grocs.M.values):
                for i, season in enumerate(target_seas):
                    ax = axes[j,i]

                    # Your plotting code here using the specific model and season
                    xplot = sr_grocs.GROCS_WEIGHT.isel(L=i, M=j).plot(ax=ax, transform=ccrs.PlateCarree(), 
                                                                      cmap='coolwarm', levels=21, vmin=0, vmax=.7,
                                                                      add_colorbar=False)

                    ax.coastlines()
                    c = ax.coastlines()
                    c = ax.gridlines(draw_labels=True, linewidth=0.3)
                    c.right_labels = False
                    c.top_labels = False 
                    c.left_labels = False
                    c.bottom_labels = False


                    if i == 0:
                        ax.text(-0.07, 0.55, f'{model}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',transform=ax.transAxes, weight = 'bold')
                    if j == 0:
                        ax.set_title(f'{season}', fontweight = 'bold')
                    else:
                        ax.set_title(None)
                    # Add country borders
                    ax.add_feature(NaturalEarthFeature(category='cultural', name='admin_0_countries', 
                                                       scale='50m', edgecolor='black', facecolor='none'))
                    # Set the extent to cover the specific area
                    ax.set_extent([regions[r]['west'], regions[r]['east'], regions[r]['south'], regions[r]['north']], crs=ccrs.PlateCarree())



            # Add a single horizontal colorbar below the panel plot
            cbar_ax = fig.add_axes([0.15, 0.002, 0.6, 0.02])  # [left, bottom, width, height]
            cbar = fig.colorbar(xplot, cax=cbar_ax, orientation='horizontal', shrink =1, pad = 0.1)
            cbar.set_label(region_names[r] + ' GROCS Consolidated Ensemble Weights', fontsize=13)
            cbar.ax.tick_params(labelsize=10)
            # Adjust layout
            plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)

            # Show plot
            plt.savefig(os.path.join(fdir, '_'.join([initial_month_name, region_names[r], 'GROCS', predictor_train_extent_name + '-train', obs_name.split('.')[0]])), bbox_inches='tight', dpi=100)
            #plt.close()

    #only works in new XCast environment
    #        ##### FORECAST PLOTS, plot the probabalistic forecasts
    #        for r, region_of_interest in enumerate(region_names):
    #            for l, lead in enumerate(np.unique(cca_fcsts_prob.L)):
    #                im = xc.view_probabilistic(cca_fcsts_prob.isel(T=0, L=l).sel(X=slice(regions[r]['west'], regions[r]['east']),Y=slice(regions[r]['south'], regions[r]['north'])), cross_dateline=True, title= region_of_interest + ' CCA MME Probabalistic Forecast for ' + target_seas[l], savefig=os.path.join(fdir, '_'.join(['im' + initial_month_name, target_seas[l],region_of_interest,'CCA_forecast',obs_name + '.png'])))
    #                plt.close()
eofPY

$py gen_seas_cca_and_eval.py

