import os
from datetime import datetime,timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib as mpl
import rioxarray as rio
from rasterio.warp import calculate_default_transform

#local functions
import src.colors as colors
import src.helper as helper
import src.file_conversion as fc

print('prepping cmorph files')
cmorph_prd = '/cpc/fews/production/cmorph_cpcwork5/cmorph_RH6_BC_ADJ_EOD/output/bin/CMORPH_V1.0_ADJ_0.25deg-DLY_EOD_'
cmorph_clim = '/cpc/fews/production/cmorph_cpcwork5/cmorph_RH6_BC_ADJ_EOD/output/bin/clim_dly/clim_dly.'
figure_dir = '/cpc/int_desk/pac_isl/stations/images'

#from Ravi's ctl file for cmorph
xdim = 1440
ydim = 480
xmin = 0.125
xmax = 360
ymin = -59.875
ymax = 60

#anomaly max/mins on the cmorph anomaly figures for color bar
minanom = -600
maxanom = 600
minanomavg= -25
maxanomavg = 25
minpercent = 0
maxpercent = 800
mintotal = 0
maxtotal = 3500
mintotalavg = 0
maxtotalavg = 50

# Define custom intervals for the colormap
#anomaly color bar breaks
anomavg_intervals = [-25, -20, -15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15, 20, 25]
percentavg_intervals = [0, 5, 10, 25, 50, 80, 120, 150, 200, 400, 600, 800]
totalavg_intervals = [0, 1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 50]
anom_intervals = [-600,-500, -300, -200, -100, -50, -25, -10, 10, 25, 50, 100, 200, 300, 500,600]
percent_intervals = [0,1,5,25,50,80,120,150,200,400,600,800]
total_intervals = [0,2,5,10,25,50,75,100,150,200,300,500,750,1000,1500,2500]

# Define colors for the colormap (corresponding to your value intervals)
#these were found usuing find_dominant_colors function in prep_data.ipynb in this folder
anom_colors = [
    (204/255, 51/255, 51/255), #dark red,
    (254/255,  4/255, 0/255), #bright red
    (255/255,  158/255, 0/255), #orange
    (255/255, 232/255, 123/255), #yellow
    (119/255,  79/255,  70/255), #dark brown
    (180/255, 140/255, 128/255), #med brown
    (237/255, 221/255, 212/255), #light brown
    (254/255, 254/255, 254/255), # off white
    (201/255, 254/255, 192/255), #light green
    (119/255, 244/255, 114/255), #bright green
    (30/255, 180/255,  30/255), #dark green
    (152/255, 210/255, 250/255), #lightmed blue
    (39/255, 129/255, 240/255), #darker blue
    (220/255, 220/255, 254/255), #light purple
    (127/255, 111/255, 234/255) #med purple
]

percent_colors = [
    (221/255, 193/255, 185/255), #beige
    (192/255,   0/255,   0/255), #dark red
    (254/255,  49/255,   0/255), #bright red
    (255/255, 158/255,   0/255), #orange
    (255/255, 232/255, 123/255), #yellow
    (254/255, 254/255, 254/255), #off white
    (201/255, 254/255, 192/255), #light green
    (124/255, 245/255, 119/255), #bright green
    (30/255, 180/255,  29/255), #dark green
    (153/255, 211/255, 250/255), #light blue
    (40/255, 130/255, 240/255) #med blue
]

totalavg_colors = [
    (254/255, 254/255, 254/255), #off white
    (201/255, 254/255, 192/255), #light green
    (124/255, 245/255, 119/255), #bright green
    (30/255, 180/255,  29/255), #dark green
    (153/255, 211/255, 250/255), #light blue
    (40/255, 130/255, 240/255), #med blue
    (39/255, 129/255, 240/255), #darker blue
    (254/255, 249/255, 170/255), #light yellow
    (254/255, 160/255,   1/255), #orange
    (225/255,  20/255,   0/255), #red
    (165/255,   0/255,   0/255), #dark red
    (229/255, 139/255, 139/255) #rose
]

total_colors = [
    (254/255, 254/255, 254/255), #off white
    (201/255, 254/255, 192/255), #light green
    (124/255, 245/255, 119/255), #bright green
    (30/255, 180/255,  29/255), #dark green
    (153/255, 211/255, 250/255), #light blue
    (40/255, 130/255, 240/255), #med blue
    (39/255, 129/255, 240/255), #darker blue
    (236/255, 228/255, 238/255), #light purple
    (158/255, 139/255, 253/255), #bright purple
    (110/255,  94/255, 216/255), #dark purple
    (254/255, 249/255, 170/255), #light yellow
    (254/255, 160/255,   1/255), #orange
    (225/255,  20/255,   0/255), #red
    (165/255,   0/255,   0/255), #dark red
    (227/255, 138/255, 138/255), #rose
    (244/255, 232/255, 232/255) #light pink
]


########## MAIN SCRIPT
# Get the current date
current_date = datetime.now().date()

# Get the date up to 185 days prior (because of lags in data availability, grab a few extra days to be safe
last180_days_ago = current_date - timedelta(days=185)

# Generate a list of dates from seven_days_ago to current_date
last180 = [last180_days_ago + timedelta(days=i) for i in range(186)]  # 8 to include both endpoints

date_strs = []
for date in last180:
    date_strs.append((helper.get_date_str(date)[0], helper.get_date_str(date)[1]))
    
#read in files 
panom_da = []
panomclim_da = []
for d, date in enumerate(date_strs):
    da = fc.read_in_binary(cmorph_prd + date[0], xdim, ydim, xmin, xmax, ymin, ymax)
    da_clim = fc.read_in_binary(cmorph_clim + date[1], xdim, ydim, xmin, xmax, ymin, ymax)
    if da is not None:
        da = da.expand_dims({'time': [last180[d]]})
        da_clim = da_clim.expand_dims({'time':[last180[d]]})
        panom_da.append(da)
        panomclim_da.append(da_clim)
allptotal = xr.concat(panom_da, dim = 'time').to_dataset(name = 'ptotal') #.sum(dim = 'date')
allptotalclim = xr.concat(panomclim_da, dim = 'time').to_dataset(name = 'ptotalclim')
allptotal = allptotal.rename({'lon':'x', 'lat':'y'})
allptotalclim = allptotalclim.rename({'lon':'x', 'lat':'y'})

#compute anomalies and convert to mercator for list of dates specified
last90_anom = helper.calc_anom_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 90, 'total') 
last90_anom_crs = last90_anom.rio.write_crs('EPSG:4326', inplace = True)
last90_anom_clipped = last90_anom.sel(x=slice(0,359.999), y = slice(-59,59))
last90_anom_mc = fc.convert_to_mercator(last90_anom_clipped, 'anom')
last90_anomnorm = np.clip(last90_anom_mc.anom, minanom, maxanom)

last90_percent = helper.calc_percent_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 90, 'total') 
last90_percent_crs = last90_percent.rio.write_crs('EPSG:4326', inplace = True)
last90_percent_clipped = last90_percent.sel(x=slice(0,359.999), y = slice(-59,59))
last90_percent_mc = fc.convert_to_mercator(last90_percent_clipped, 'percent')
last90_percentnorm = np.clip(last90_percent_mc.percent, minpercent, maxpercent)

last30_anom = helper.calc_anom_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 30, 'total') 
last30_anom_crs = last30_anom.rio.write_crs('EPSG:4326', inplace = True)
last30_anom_clipped = last30_anom.sel(x=slice(0,359.999), y = slice(-59,59))
last30_anom_mc = fc.convert_to_mercator(last30_anom_clipped, 'anom')
last30_anomnorm = np.clip(last30_anom_mc.anom, minanom, maxanom)

last30_percent = helper.calc_percent_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 30, 'total') 
last30_percent_crs = last30_percent.rio.write_crs('EPSG:4326', inplace = True)
last30_percent_clipped = last30_percent.sel(x=slice(0,359.999), y = slice(-59,59))
last30_percent_mc = fc.convert_to_mercator(last30_percent_clipped, 'percent')
last30_percentnorm = np.clip(last30_percent_mc.percent, minpercent, maxpercent)

last7_anom = helper.calc_anom_pastdays(allptotal, 'ptotal', allptotalclim, 'ptotalclim', 7, 'total') 
last7_anom_crs = last7_anom.rio.write_crs('EPSG:4326', inplace = True)
last7_anom_clipped = last7_anom.sel(x=slice(0,359.999), y = slice(-59,59))
last7_anom_mc = fc.convert_to_mercator(last7_anom_clipped, 'anom')
last7_anomnorm = np.clip(last7_anom_mc.anom, minanom, maxanom)


last7_total = helper.calc_total_pastdays(allptotal, 'ptotal', 7) 
last7_total_crs = last7_total.rio.write_crs('EPSG:4326', inplace = True)
last7_total_clipped = last7_total.sel(x=slice(0,359.999), y = slice(-59,59))
last7_total_mc = fc.convert_to_mercator(last7_total_clipped, 'ptotal')
last7_totalnorm = np.clip(last7_total_mc.ptotal, mintotal, maxtotal)

# Create a ListedColormap using your defined colors
anom_cmap = ListedColormap(anom_colors)
percent_cmap = ListedColormap(percent_colors)
total_cmap = ListedColormap(total_colors)

# Create a BoundaryNorm to map the data to the value intervals
anom_norm = BoundaryNorm(boundaries=anom_intervals, ncolors=len(anom_colors))
percent_norm = BoundaryNorm(boundaries=percent_intervals, ncolors=len(percent_colors))
total_norm = BoundaryNorm(boundaries=total_intervals, ncolors=len(total_colors))

last90_rgb = colors.apply_colormap(last90_anomnorm, anom_cmap, anom_norm, anom_intervals)
last90p_rgb =  colors.apply_colormap(last90_percentnorm, percent_cmap, percent_norm, percent_intervals)
last30_rgb = colors.apply_colormap(last30_anomnorm, anom_cmap, anom_norm, anom_intervals)
last30p_rgb =  colors.apply_colormap(last30_percentnorm, percent_cmap, percent_norm, percent_intervals)
last7_rgb = colors.apply_colormap(last7_anomnorm, anom_cmap, anom_norm, anom_intervals)
last7t_rgb = colors.apply_colormap(last7_totalnorm, total_cmap, total_norm, total_intervals) 

#write to raster
last90_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph90anom.tif'), dtype="uint8")
last90p_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph90percent.tif'), dtype="uint8")
last30_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph30anom.tif'), dtype="uint8")
last30p_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph30percent.tif'), dtype="uint8")
last7_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph7anom.tif'), dtype="uint8")
last7t_rgb.rio.to_raster(os.path.join(figure_dir, 'cmorph7total.tif'), dtype="uint8")