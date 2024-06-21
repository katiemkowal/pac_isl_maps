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

##################### CONSTANTS
#data directory
ddir='/cpc/int_desk/pac_isl/analysis/xcast/seasonal/practical_notebooks/practical_data'

# figure directory
fdir='/cpc/int_desk/pac_isl/analysis/xcast/seasonal/presentation_figures'

#obs_name CMORPH or CHIRPS
obs_name='CMORPH'

#initialized dates
initial_dates = [(2023, 7,1), (2024, 1, 1), (2024, 2, 1), (2024, 3, 1), (2024, 4, 1), (2024, 5, 1), (2024, 6, 1)]
#(2023, 8, 1), (2023, 9, 1), (2023, 10, 1), (2023, 11,1), (2023, 12,1),

#SET UP SOME REGIONS TO PLOT

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
        'north': 3,  
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
        'north': -12,  
        'south': -20
}

samoa_coordinates = {
        'west':  187,
        'east': 191,  
        'north': -13,  
        'south': -15
}

tuvalu_coordinates = {
        'west':  178,
        'east': 180,  
        'north': -8,  
        'south': -9
}

regions = [pacislands_coordinates, fiji_coordinates, kiribati_coordinates, solomon_coordinates, png_coordinates, palau_coordinates, samoa_coordinates, tuvalu_coordinates, vanuatu_coordinates]
region_names = ['Pacific Islands', 'Fiji', 'Kiribati', 'Solomon Islands', 'Papua New Guinea', 'Palau', 'Samoa', 'Tuvalu', 'Vanuatu']
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
    ###### read in observations
    if obs_name == 'CMORPH':
        training_length = 'threeseas'
    elif obs_name == 'CHIRPS':
        training_length = 'oneseas'
    obs_leads = xr.open_dataset(os.path.join(ddir, '_'.join([initial_month_name, training_length, obs_name, 'precip.nc'])))

    #if obs_name == 'CHIRPS':
        #r1_coordinates = {
        #'west':  130,
        #'east': 170,  
        #'north': 1,  
        #'south': -12}
        
        #r2_coordinates = {
        #'west':  170,
        #'east': 203,  
        #'north': 3,  
        #'south': -20}
        
    #masked_obs1 = obs_leads.sel(X=slice(r1_coords['west'], r1_coords['east']),
    #                  Y=slice(r1_coords['south'], r1_coords['north']))
    #masked_obs2 = obs_leads.sel(X=slice(r2_coords['west'], r2_coords['east']),
    #                  Y=slice(r2_coords['south'], r2_coords['north']))
    
    ###### read in hindcast and forecast data
    hindcast_data = xr.open_dataset(os.path.join(ddir,
    '_'.join([initial_month_name, training_length, 'NMME_hcst_precip.nc'])))
    forecast_data = xr.open_dataset(os.path.join(ddir, '_'.join([initial_month_name, training_length, 'NMME_fcst_precip.nc'])))  
    
    ###### read in the ocean mask for the data
    msk = xr.open_dataset('/cpc/africawrf/ebekele/projects/PREPARE_pacific/notebooks/masked/libs/pacific_mask.nc')
    mskk = msk.amask.expand_dims({'M':[0]})
    mskk = mskk.assign_coords({'lon': [i + 360 if i <= 0 else i for i in mskk.coords['lon'].values]}).sortby('lon').drop_duplicates('lon')
    mskk = mskk.rename({'lon':'X', 'lat':'Y', 'time':'T'})
    mskk = xc.regrid(mskk, obs_leads.X, obs_leads.Y)
    mask_missing = mskk.mean('T', skipna=False).mean('M', skipna=False)
    mask_missing = xr.ones_like(mask_missing).where(~np.isnan(mask_missing), other=np.nan )
    
    ##### run cca
    start_time = time.time()
    cca_fcsts_prob, cca_fcsts_det, cca_hcasts_det, cca_hcasts_prob, obs_to_test = [],[],[],[],[]

    for l in np.unique(hindcast_data.L):
        model = hindcast_data.sel(L=l).precip
        obs = obs_leads.sel(L=l).precip
        fmodel = forecast_data.sel(L=l).precip
        
        #create a dry mask to avoid training over zero values
        drymask = xc.drymask(obs, dry_threshold=0.5, quantile_threshold=0.3)
        obs = obs * mask_missing
        obs = obs * drymask
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

        cca_fcsts_prob.append(fprobs)
        cca_hcasts_det.append(hindcasts_det)
        cca_hcasts_prob.append(hindcasts_prob)
        obs_to_test.append(obs_test)
    cca_fcsts_prob = xr.concat(cca_fcsts_prob, dim = 'L')
    cca_hcasts_det = xr.concat(cca_hcasts_det, dim = 'L')
    cca_hcasts_prob = xr.concat(cca_hcasts_prob, dim = 'L')
    obs_to_test = xr.concat(obs_to_test, dim = 'L')
    print('cca processing time is ' + str(time.time() - start_time))
    
################# EVALUATE the skill of the raw and bias corrected forecasts
    
    #PEARSON CALCULATION
    start_time = time.time()
    #calculate pearson correlation score for hindcasts
    pearson_cca, pearson_raw = [], []
    for l, lead in enumerate(np.unique(hindcast_data.L.values)):
        #regrid raw data for pearson calculation on one to one grid
        raw_regrid = xc.regrid(hindcast_data.isel(L=l).precip, obs_leads.X, obs_leads.Y)
        raw_regrid = raw_regrid * mask_missing
        
        obs_raw = obs_leads.isel(L=l).precip* mask_missing

        cca_pearson_calc = xc.Pearson(cca_hcasts_det.isel(L=l),obs_to_test.isel(L=l))
        cca_pearson_calc = cca_pearson_calc.expand_dims({'M':['CCA on NMME']})

        #calc pearson correlation
        pearson_raw_calc = []
        for m, model in enumerate(np.unique(raw_regrid.M.values)):
            pearson_raw_c = xc.Pearson(raw_regrid.sel(M=model).expand_dims({'M':[model]}), 
                                               obs_raw)
            pearson_raw_c = pearson_raw_c.expand_dims({'M':[model]})
            pearson_raw_calc.append(pearson_raw_c)
        pearson_raw_calc = xr.concat(pearson_raw_calc, dim = 'M')
        pearson_cca.append(cca_pearson_calc)
        pearson_raw.append(pearson_raw_calc)
    pearson_cca = xr.concat(pearson_cca, dim = 'L')
    pearson_raw = xr.concat(pearson_raw, dim = 'L')
    pearsons = xr.concat([pearson_cca, pearson_raw], dim = 'M')
    print('pearson processing time is ' + str(time.time() - start_time))
    
    #RMSE-SS CALCULATION
    #anomalize your data
    def std_anomalize(x):
        clim = x.mean(dim='T')
        standard_dev = x.std(dim='T')
        anom = x - clim
        std_anom = anom/standard_dev
        return std_anom

    #calculate the root mean squared error for each point across time
    def RMSE(x, y): 
      # x is Nx1, y is Nx1
      squared_error = (x - y)**2 
      mean_squared_error = squared_error.mean(dim = 'T') 
      return np.sqrt(mean_squared_error) 

    #calculate the skill score using RMSE by comparing performance of the climatology against your data
    def RMSESS(x,y):
        clim = y.mean(dim='T')
        rmse_model = RMSE(x,y)
        rmse_clim = RMSE(clim,y)
        rmsess = 1 - rmse_model/rmse_clim
        return rmsess
        
    start_time = time.time()
    #anomalize the cca and obs_test data
    obs_test_anom = std_anomalize(obs_to_test)
    cca_hcasts_anom = std_anomalize(cca_hcasts_det)

    #calculate RMSESS score for hindcasts
    rmse_cca, rmse_raw = [], []
    for l, lead in enumerate(np.unique(hindcast_data.L.values)):
        cca_rmse_calc = RMSESS(cca_hcasts_anom.isel(L=l),obs_test_anom.isel(L=l))
        cca_rmse_calc = cca_rmse_calc.isel(M=0).expand_dims({'M':['CCA on NMME']})

        raw_regrid = xc.regrid(hindcast_data.isel(L=l).precip, obs_leads.X, obs_leads.Y)
        raw_regrid = raw_regrid * mask_missing
        
        obs_raw = obs_leads.isel(L=l).precip * mask_missing

        #anomalize the raw model data and obs data over the original time frame
        raw_anom = std_anomalize(raw_regrid)
        obs_anom = std_anomalize(obs_raw)

        #calc rmse for raw values
        rmse_raw_calc = []
        for m, model in enumerate(np.unique(raw_regrid.M.values)):
            rmse_raw_c = RMSESS(raw_anom.sel(M=model), 
                                               obs_anom)
            rmse_raw_c = rmse_raw_c.isel(M=m).expand_dims({'M':[model]})
            rmse_raw_calc.append(rmse_raw_c)
        rmse_raw_calc = xr.concat(rmse_raw_calc, dim = 'M')
        rmse_cca.append(cca_rmse_calc)
        rmse_raw.append(rmse_raw_calc)
    rmse_cca = xr.concat(rmse_cca, dim = 'L')
    rmse_raw = xr.concat(rmse_raw, dim = 'L')
    rmsess = xr.concat([rmse_cca, rmse_raw], dim = 'M')
    print('RMSE-SS processing time is ' + str(time.time() - start_time))
    
    #GROCS CALCULATION
    start_time = time.time()
    grocs_cca, grocs_raw = [], []
    for l, lead in enumerate(np.unique(hindcast_data.L.values)):
    
        hind_prob = xc.gaussian_smooth(cca_hcasts_prob.isel(L=l), kernel=3)
        obs = xc.gaussian_smooth(obs_to_test.isel(L=l), kernel=3)
        
        #transform obs into tercile based categories
        ohc = xc.OneHotEncoder() 
        ohc.fit(obs)
        T = ohc.transform(obs)
        clim = xr.ones_like(T) * 0.333
        
        grocs_cca_l = xc.GROCS(hind_prob, T)
        grocs_cca_l = grocs_cca_l.expand_dims({'M':['CCA on NMME']})
        grocs_cca.append(grocs_cca_l)
        ## CALCULATE THE RAW GROCS SCORE HERE AFTER PREPARING INDIVIDUAL MODELS
    grocs_cca = xr.concat(grocs_cca, dim = 'L')
    print('GROCS processing time is ' + str(time.time() - start_time))
    
################ PLOT THE RESULTS
    ###### PEARSONS PLOTS

    models = np.unique(pearsons.M.values)
    for r, region_of_interest in enumerate(region_names):

        fig, axes = plt.subplots(nrows=len(models), ncols=len(target_seas), figsize=(10, (2)*2 + 2), 
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
        cbar.set_label(region_of_interest + ' Pearson Correlation', fontsize=13)
        cbar.ax.tick_params(labelsize=14)
        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)

        # Show plot
        plt.savefig(os.path.join(fdir, '_'.join([initial_month_name, region_of_interest, 'pearson_CCA', obs_name.split('.')[0]])), bbox_inches='tight', dpi=100)
        
    ###### RMSE-SS PLOTS

    for r, region_of_interest in enumerate(region_names):
        fig, axes = plt.subplots(nrows=len(models), ncols=len(target_seas), figsize=(10, (2)*2 + 2), 
                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        # Set the extent to cover the entire world
        for ax in axes.flat:
            ax.set_global()

        for j, model in enumerate(models):
            for i, season in enumerate(target_seas):
                ax = axes[j, i]
                # Your plotting code here using the specific model and season
                xplot = rmsess.isel(L=i, M=j).plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=21, vmin=-1, vmax=1, add_colorbar=False)
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
        cbar.set_label(region_of_interest + ' RMSE-SS', fontsize=13)
        cbar.ax.tick_params(labelsize=14)
        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)

        # Show plot
        plt.savefig(os.path.join(fdir, '_'.join([initial_month_name, region_of_interest, 'RMSE-SS_CCA', obs_name.split('.')[0]])), bbox_inches='tight', dpi=100)
        
        
    ### GROCS PLOTS
    for r, region_of_interest in enumerate(region_names):
        fig, axes = plt.subplots(nrows=1, ncols=len(target_seas), figsize=(10, (2)*2 + 2), 
                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        # Set the extent to cover the entire world
        for ax in axes.flat:
            ax.set_global()

        for j, model in enumerate(grocs_cca.M.values):
            for i, season in enumerate(target_seas):
                ax = axes[i]##, i]
                # Your plotting code here using the specific model and season
                xplot = grocs_cca.isel(L=i, M=j).plot(ax=ax,transform=ccrs.PlateCarree(), cmap='coolwarm', levels=21, vmin=-1, vmax=1, add_colorbar=False)
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
        cbar.set_label(region_of_interest + ' GROCS', fontsize=13)
        cbar.ax.tick_params(labelsize=14)
        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.01, hspace=0.2)

        # Show plot
        plt.savefig(os.path.join(fdir, '_'.join([initial_month_name, region_of_interest, 'GROCS_CCA', obs_name.split('.')[0]])), bbox_inches='tight', dpi=100)
        
#only works in new XCast environment
        ##### FORECAST PLOTS, plot the probabalistic forecasts
#        for r, region_of_interest in enumerate(region_names):
#            for l, lead in enumerate(np.unique(cca_fcsts_prob.L)):
#                im = xc.view_probabilistic(cca_fcsts_prob.isel(T=0, L=l).sel(X=slice(regions[r]['west'], regions[r]['east']),Y=slice(regions[r]['south'], regions[r]['north'])), cross_dateline=True, title= region_of_interest + ' CCA MME Probabalistic Forecast for ' + target_seas[l], savefig=os.path.join(fdir, '_'.join(['im' + initial_month_name, target_seas[l],region_of_interest,'CCA_forecast',obs_name + '.png'])))
