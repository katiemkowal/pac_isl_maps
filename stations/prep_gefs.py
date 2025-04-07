import os
from pathlib import Path
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
import src.file_conversion as fc
import src.colors as colors
import src.station_locations as sl

print('prepping gefs files')
gefs_procdir = '/cpc/africawrf/ebekele/projects/PREPARE_pacific/notebooks/unmasked'
gefs_rawdir = '/cpc/africawrf/ebekele/projects/PREPARE_pacific/subseason_unmasked'
figure_dir = '/cpc/int_desk/pac_isl/stations/images/'

#get current date for raw gefs
c_date = datetime.now().date()
c_month = c_date.month
c_day = c_date.day
if c_month < 10:
    c_month = '0' + str(c_month)
else: c_month = str(c_month)
if c_day < 10:
    c_day = '0' + str(c_day)
else: c_day = str(c_day)
date_str = str(c_date.year) + c_month + c_day

## gefs raw ctl file from endalk's script
xdimgef = 720
ydimgef = 361
xmingef= 0
xmaxgef=360
ymingef=-90
ymaxgef=89.5
zdimgef = 15
zdimgeft = 5

minptotal = 0
maxptotal = 3500
minpanom= -75
maxpanom = 75
minp50 = 0
maxp50 = 100
minp100 = 0
maxp100 = 100
minttotal = 24
maxttotal = 35
mintanom = -4
maxtanom = 4

ptotal_intervals = [0, 2, 5, 10, 25, 50, 75, 100,
                    150, 200, 300, 500, 750,1000,
                    1500, 2500, 3500]
panom_intervals = [-75, -50, -40,-30,-20,-10,-5,
                    5,10,20,30,40,50,75]
poe_intervals = [0,5,10,20,30,40,50,60,70,80,90,95,100]
ttotal_intervals = [24,25,26,27,28,29,30,31,32,33,34,35]
tanom_intervals = [-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4]

ptotal_colors = [
    (254/255, 254/255, 254/255), #off white
    (198/255, 252/255, 188/255), #light green
    (118/255, 241/255, 113/255), #bright green
    (29/255, 178/255,  29/255), #dark green
    (178/255, 238/255, 248/255), #light blue
    (79/255, 163/255, 243/255), #med blue
    (29/255, 108/255, 231/255), #darker blue
    (236/255, 228/255, 238/255), #light purple
    (158/255, 139/255, 253/255), #bright purple
    (110/255,  94/255, 216/255), #dark purple
    (253/255, 248/255, 168/255), #light yellow
    (250/255, 156/255,   0/255), #orange
    (223/255,  19/255,   0/255), #bright red
    (163/255,   0/255,   0/255), #dark red
    (227/255, 138/255, 138/255), #rose
    (244/255, 232/255, 232/255) #light pink
]

panom_colors = [
    (96/255,  59/255,  49/255), #darkest brown
    (113/255,  81/255,  73/255), #second darkest brown
    (139/255,  99/255,  89/255), #med brown
    (171/255, 142/255, 135/255), #med light brown
    (216/255, 187/255, 178/255), #light brown
    (243/255, 238/255, 232/255), #tan
    (254/255, 254/255, 254/255), #white
    (238/255, 253/255, 223/255), #off white green
    (178/255, 247/255, 168/255), #light green
    (119/255, 243/255, 114/255), #bright green
    (54/255, 207/255,  59/255), #med green
    (22/255, 167/255,  22/255), #dark green
    (14/255,  83/255,  15/255) #darkest green)
]

poe_colors = [
    (254/255, 254/255, 254/255), #off white
    (230/255, 252/255, 226/255), #lightest green
    (177/255, 247/255, 168/255), #light green
    (118/255, 241/255, 113/255), #bright green
    (54/255, 207/255,  59/255), #med green
    (118/255, 241/255, 113/255), #dark med green
    (148/255, 208/255, 247/255), #sky blue
    (190/255, 178/255, 252/255), #light purple
    (126/255, 110/255, 232/255), #med purple
    (71/255,  60/255, 197/255), #violet
    (44/255,  30/255, 162/255), #dark purple
    (29/255, 108/255, 231/255) #royal blue 
]

ttotal_colors = [
    (148/255, 239/255, 138/255), #lightmed green
    (180/255, 247/255, 170/255), #light green
    (200/255, 254/255, 190/255), #light light green
    (252/255, 247/255, 168/255), #light yellow
    (251/255, 189/255,  59/255), #light orange
    (252/255,  94/255,   0/255), #med orange
    (222/255,  19/255,   0/255), #bright red
    (162/255,   0/255,   0/255), #dark red
    (226/255, 110/255, 110/255), #darker rose
    (226/255, 138/255, 138/255), #med rose
    (248/255, 160/255, 160/255) #light rose
]

tanom_colors = [
    (19/255,  98/255, 206/255), #dark blue
    (39/255, 128/255, 237/255), #med blue
    (79/255, 163/255, 24/255), #lightmed blue
    (148/255, 207/255, 247/255), #skyblue
    (224/255, 254/255, 255/255), #lightest blue
    (254/255, 254/255, 254/255), #white
    (252/255, 247/255, 168/255), #light yellow
    (251/255, 189/255,  59/255), #light orange
    (251/255,  94/255,   0/255), #bright orange
    (222/255,  19/255,   0/255), #bright red
    (162/255,   0/255,   0/255) #dark red
]

# Define intervals and colors
bn_intervals = [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,100]
bn_colors = [
    (245/255, 245/255, 245/255),#0-35
    (245/255, 230/255, 193/255),#40-45
    (233/255, 212/255, 159/255),
    (222/255, 192/255, 123/255),
    (206/255, 160/255, 83/255),
    (190/255, 128/255, 44/255),
    (164/255, 104/255, 26/255),
    (139/255, 81/255, 10/255),
    (111/255, 63/255, 6/255),
    (100/255, 55/255, 6/255),
    (82/255, 48/255, 6/255)
]

nn_intervals = [0, 35, 40, 45, 50, 55,100]
nn_colors = [
    (245/255, 245/255, 245/255),#0-35
    (238/255, 238/255, 233/255),#35-40
    (194/255, 194/255, 194/255),#40-45
    (176/255, 176/255, 176/255),
    (144/255, 144/255, 144/255),
    (144/255, 144/255, 144/255)
]

an_intervals = [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,100]
an_colors = [
    (245/255, 245/255, 245/255),#0-35
    (198/255, 233/255, 227/255),#35-40
    (162/255, 218/255, 210/255),#40-45
    (144/255, 211/255, 201/255),#45-50
    (127/255, 203/255, 191/255),
    (89/255, 176/255, 167/255),
    (52/255, 150/255, 142/255),
    (26/255, 125/255, 117/255),
    (0/255, 101/255, 93/255),
    (0/255, 80/255, 71/255),
    (3/255, 56/255, 47/255)
]

## read in raw gefs datacon
#variables based on endalk's documentation
print(date_str)
if Path(os.path.join(gefs_rawdir,'gefs_week1_precip_' + date_str + 'IC.dat')).is_file():
    gefs_raw1 = fc.read_in_binary_gefs(os.path.join(gefs_rawdir,'gefs_week1_precip_' + date_str + 'IC.dat'), xdimgef, ydimgef, zdimgef, xmingef, xmaxgef, ymingef, ymaxgef)
    gefs1_totalp = gefs_raw1.isel(var=1).drop('var')
    gefs1_totalclim = gefs_raw1.isel(var=0).drop('var')
    gefs1_panom = gefs1_totalp - gefs1_totalclim
    gefs1_probs50 = gefs_raw1.isel(var=6)
    gefs1_probs100 = gefs_raw1.isel(var=7)
    
    #gefs wk 1 raw total precip
    gefs1_totalp = gefs1_totalp.to_dataset(name = 'tp')
    gefs1_totalp = gefs1_totalp.rename({'lon':'x', 'lat':'y'})
    gefs1_total_slice = gefs1_totalp.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs1_total_crs = gefs1_total_slice.rio.write_crs('EPSG:4326', inplace = True)
    gefs1_tp_mc = fc.convert_to_mercator(gefs1_total_crs, 'tp')
    gefs1_tpnorm = np.clip(gefs1_tp_mc.tp, minptotal, maxptotal)
    ptotal_cmap = ListedColormap(ptotal_colors, N=len(ptotal_colors))
    ptotal_norm = BoundaryNorm(boundaries=ptotal_intervals, ncolors=len(ptotal_colors))
    ptotal1_rgb = colors.apply_colormap(gefs1_tpnorm, ptotal_cmap, ptotal_norm, ptotal_intervals)
    ptotal1_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk1ptotal.tif'), dtype="uint8")
    
    #gefs wk 1 raw precip anomaly
    gefs1_panom = gefs1_panom.to_dataset(name='anom')
    gefs1_panom = gefs1_panom.rename({'lon':'x', 'lat':'y'})
    gefs1_panomslice = gefs1_panom.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs1_panom_crs = gefs1_panomslice.rio.write_crs('EPSG:4326', inplace = True)
    gefs1_panom_mc = fc.convert_to_mercator(gefs1_panom_crs, 'anom')
    gefs1_panom_norm = np.clip(gefs1_panom_mc.anom, minpanom, maxpanom)
    panom_cmap = ListedColormap(panom_colors, N=len(panom_colors))
    panom_norm = BoundaryNorm(boundaries=panom_intervals, ncolors=len(panom_colors))
    panom1_rgb = colors.apply_colormap(gefs1_panom_norm, panom_cmap, panom_norm, panom_intervals)
    panom1_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk1panom.tif'), dtype="uint8")
    
    #gefs wk1 raw poe50
    gefs1_probs50 = gefs1_probs50.to_dataset(name='p50')
    gefs1_probs50 = gefs1_probs50.rename({'lon':'x', 'lat':'y'})
    gefs1_probs50slice = gefs1_probs50.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs1_probs50_crs = gefs1_probs50slice.rio.write_crs('EPSG:4326', inplace = True)
    gefs1_probs50_mc = fc.convert_to_mercator(gefs1_probs50_crs, 'p50')
    gefs1_probs50_mc['p50'] = gefs1_probs50_mc.p50 * 100
    gefs1_probs50_norm = np.clip(gefs1_probs50_mc.p50, minp50, maxp50)
    poe_cmap = ListedColormap(poe_colors, N=len(poe_colors))
    poe_norm = BoundaryNorm(boundaries=poe_intervals, ncolors=len(poe_colors))
    poe50_rgb1 = colors.apply_colormap(gefs1_probs50_norm, poe_cmap, poe_norm, poe_intervals)
    poe50_rgb1.rio.to_raster(os.path.join(figure_dir, 'gefswk1poe50.tif'), dtype="uint8")
    
    #gefs wk1 raw poe100
    gefs1_probs100 = gefs1_probs100.to_dataset(name='p100')
    gefs1_probs100 = gefs1_probs100.rename({'lon':'x', 'lat':'y'})
    gefs1_probs100slice = gefs1_probs100.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs1_probs100_crs = gefs1_probs100slice.rio.write_crs('EPSG:4326', inplace = True)
    gefs1_probs100_mc = fc.convert_to_mercator(gefs1_probs100_crs, 'p100')
    gefs1_probs100_mc['p100'] = gefs1_probs100_mc.p100 * 100
    gefs1_probs100_norm = np.clip(gefs1_probs100_mc.p100, minp100, maxp100)
    poe100_rgb1 = colors.apply_colormap(gefs1_probs100_norm, poe_cmap, poe_norm, poe_intervals)
    poe100_rgb1.rio.to_raster(os.path.join(figure_dir, 'gefswk1poe100.tif'), dtype="uint8")
else: print('no raw dat file available for gefs week 1 precip')

if Path(os.path.join(gefs_rawdir,'gefs_week2_precip_' + date_str + 'IC.dat')).is_file():
    gefs_raw2 = fc.read_in_binary_gefs(os.path.join(gefs_rawdir,'gefs_week2_precip_' + date_str + 'IC.dat'), xdimgef, ydimgef, zdimgef, xmingef, xmaxgef, ymingef, ymaxgef)
    gefs2_totalp = gefs_raw2.isel(var=1).drop('var')
    gefs2_totalclim = gefs_raw2.isel(var=0).drop('var')
    gefs2_panom = gefs2_totalp - gefs2_totalclim
    gefs2_probs50 = gefs_raw2.isel(var=6)
    gefs2_probs100 = gefs_raw2.isel(var=7)
    
    #gefs wk2 raw total p
    gefs2_totalp = gefs2_totalp.to_dataset(name = 'tp')
    gefs2_totalp = gefs2_totalp.rename({'lon':'x', 'lat':'y'})
    gefs2_total_slice = gefs2_totalp.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs2_total_crs = gefs2_total_slice.rio.write_crs('EPSG:4326', inplace = True)
    gefs2_tp_mc = fc.convert_to_mercator(gefs2_total_crs, 'tp')
    gefs2_tpnorm = np.clip(gefs2_tp_mc.tp, minptotal, maxptotal)
    ptotal2_rgb = colors.apply_colormap(gefs2_tpnorm, ptotal_cmap, ptotal_norm, ptotal_intervals)
    ptotal2_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk2ptotal.tif'), dtype="uint8")
    
    #gefs wk2 raw anomaly precip
    gefs2_panom = gefs2_panom.to_dataset(name='anom')
    gefs2_panom = gefs2_panom.rename({'lon':'x', 'lat':'y'})
    gefs2_panomslice = gefs2_panom.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs2_panom_crs = gefs2_panomslice.rio.write_crs('EPSG:4326', inplace = True)
    gefs2_panom_mc = fc.convert_to_mercator(gefs2_panom_crs, 'anom')
    gefs2_panom_norm = np.clip(gefs2_panom_mc.anom, minpanom, maxpanom)
    panom_cmap = ListedColormap(panom_colors, N=len(panom_colors))
    panom_norm = BoundaryNorm(boundaries=panom_intervals, ncolors=len(panom_colors))
    panom2_rgb = colors.apply_colormap(gefs2_panom_norm, panom_cmap, panom_norm, panom_intervals)
    panom2_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk2panom.tif'), dtype="uint8")
    
    #gefs wk2 raw poe50
    gefs2_probs50 = gefs2_probs50.to_dataset(name='p50')
    gefs2_probs50 = gefs2_probs50.rename({'lon':'x', 'lat':'y'})
    gefs2_probs50slice = gefs2_probs50.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs2_probs50_crs = gefs2_probs50slice.rio.write_crs('EPSG:4326', inplace = True)
    gefs2_probs50_mc = fc.convert_to_mercator(gefs2_probs50_crs, 'p50')
    gefs2_probs50_mc['p50'] = gefs2_probs50_mc.p50 * 100
    gefs2_probs50_norm = np.clip(gefs2_probs50_mc.p50, minp50, maxp50)
    poe_cmap = ListedColormap(poe_colors, N=len(poe_colors))
    poe_norm = BoundaryNorm(boundaries=poe_intervals, ncolors=len(poe_colors))
    poe50_rgb2 = colors.apply_colormap(gefs2_probs50_norm, poe_cmap, poe_norm, poe_intervals)
    poe50_rgb2.rio.to_raster(os.path.join(figure_dir, 'gefswk2poe50.tif'), dtype="uint8")
    
    #gefs wk2 raw poe100
    gefs2_probs100 = gefs2_probs100.to_dataset(name='p100')
    gefs2_probs100 = gefs2_probs100.rename({'lon':'x', 'lat':'y'})
    gefs2_probs100slice = gefs2_probs100.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs2_probs100_crs = gefs2_probs100slice.rio.write_crs('EPSG:4326', inplace = True)
    gefs2_probs100_mc = fc.convert_to_mercator(gefs2_probs100_crs, 'p100')
    gefs2_probs100_mc['p100'] = gefs2_probs100_mc.p100 * 100
    gefs2_probs100_norm = np.clip(gefs2_probs100_mc.p100, minp100, maxp100)
    poe100_rgb2 = colors.apply_colormap(gefs2_probs100_norm, poe_cmap, poe_norm, poe_intervals)
    poe100_rgb2.rio.to_raster(os.path.join(figure_dir, 'gefswk2poe100.tif'), dtype="uint8")

else: print('no raw dat file available for gefs week 2 precip')

if Path(os.path.join(gefs_rawdir,'gefs_week1_t2m_' + date_str + 'IC.dat')).is_file():
    gefs_t2mraw1 = fc.read_in_binary_gefs(os.path.join(gefs_rawdir,'gefs_week1_t2m_' + date_str + 'IC.dat'), xdimgef, ydimgef, zdimgeft, xmingef, xmaxgef, ymingef, ymaxgef)
    gefs1_totalt = gefs_t2mraw1.isel(var=1).drop('var')
    gefs1_totaltclim = gefs_t2mraw1.isel(var=0).drop('var')
    gefs1_tanom = gefs1_totalt - gefs1_totaltclim
    
    #gefs wk1 raw total t2m
    gefs1_totalt = gefs1_totalt.to_dataset(name = 't2m')
    gefs1_totalt = gefs1_totalt.rename({'lon':'x', 'lat':'y'})
    gefs1_total_tslice = gefs1_totalt.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs1_totalt_crs = gefs1_total_tslice.rio.write_crs('EPSG:4326', inplace = True)
    gefs1_t2m_mc = fc.convert_to_mercator(gefs1_totalt_crs, 't2m')
    gefs1_t2mnorm = np.clip(gefs1_t2m_mc.t2m, minttotal, maxttotal)
    ttotal_cmap = ListedColormap(ttotal_colors, N=len(ttotal_colors))
    ttotal_norm = BoundaryNorm(boundaries=ttotal_intervals, ncolors=len(ttotal_colors))
    ttotal1_rgb = colors.apply_colormap(gefs1_t2mnorm, ttotal_cmap, ttotal_norm, ttotal_intervals)
    ttotal1_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk1ttotal.tif'), dtype="uint8")
    
    #gefs wk1 raw t2m anomaly
    gefs1_tanom = gefs1_tanom.to_dataset(name='anom')
    gefs1_tanom = gefs1_tanom.rename({'lon':'x', 'lat':'y'})
    gefs1_tanomslice = gefs1_tanom.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs1_tanom_crs = gefs1_tanomslice.rio.write_crs('EPSG:4326', inplace = True)
    gefs1_tanom_mc = fc.convert_to_mercator(gefs1_tanom_crs, 'anom')
    gefs1_tanom_norm = np.clip(gefs1_tanom_mc.anom, mintanom, maxtanom)
    tanom_cmap = ListedColormap(tanom_colors, N=len(tanom_colors))
    tanom_norm = BoundaryNorm(boundaries=tanom_intervals, ncolors=len(tanom_colors))
    tanom1_rgb = colors.apply_colormap(gefs1_tanom_norm, tanom_cmap, tanom_norm, tanom_intervals)
    tanom1_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk1tanom.tif'), dtype="uint8")
else: print('no raw dat file available for gefs week 1 t2m')

if Path(os.path.join(gefs_rawdir,'gefs_week2_t2m_' + date_str + 'IC.dat')).is_file():
    gefs_t2mraw2 = fc.read_in_binary_gefs(os.path.join(gefs_rawdir,'gefs_week2_t2m_' + date_str + 'IC.dat'), xdimgef, ydimgef, zdimgeft, xmingef, xmaxgef, ymingef, ymaxgef)
    gefs2_totalt = gefs_t2mraw2.isel(var=1).drop('var')
    gefs2_totaltclim = gefs_t2mraw2.isel(var=0).drop('var')
    gefs2_tanom = gefs2_totalt - gefs2_totaltclim
    
    #gefs wk2 raw total t2m
    gefs2_totalt = gefs2_totalt.to_dataset(name = 't2m')
    gefs2_totalt = gefs2_totalt.rename({'lon':'x', 'lat':'y'})
    gefs2_total_tslice = gefs2_totalt.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs2_totalt_crs = gefs2_total_tslice.rio.write_crs('EPSG:4326', inplace = True)
    gefs2_t2m_mc = fc.convert_to_mercator(gefs2_totalt_crs, 't2m')
    gefs2_t2mnorm = np.clip(gefs2_t2m_mc.t2m, minttotal, maxttotal)
    ttotal2_rgb = colors.apply_colormap(gefs2_t2mnorm, ttotal_cmap, ttotal_norm, ttotal_intervals)
    ttotal2_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk2ttotal.tif'), dtype="uint8")
    
    #gefs wk2 raw total t2m
    gefs2_tanom = gefs2_tanom.to_dataset(name='anom')
    gefs2_tanom = gefs2_tanom.rename({'lon':'x', 'lat':'y'})
    gefs2_tanomslice = gefs2_tanom.sel(x=slice(0,359.999), y=slice(-80,80))
    gefs2_tanom_crs = gefs2_tanomslice.rio.write_crs('EPSG:4326', inplace = True)
    gefs2_tanom_mc = fc.convert_to_mercator(gefs2_tanom_crs, 'anom')
    gefs2_tanom_norm = np.clip(gefs2_tanom_mc.anom, mintanom, maxtanom)
    tanom2_rgb = colors.apply_colormap(gefs2_tanom_norm, tanom_cmap, tanom_norm, tanom_intervals)
    tanom2_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk2tanom.tif'), dtype="uint8")
else: print('no raw dat file available for gefs week 1 t2m')

############### CONSOLIDATED FORECASTS ##########################################
## prep tercile forecasts
categories = ["Below-Normal", "Near-Normal", "Above-Normal"]
# Create colormaps
bn_cmap = ListedColormap(bn_colors, N=len(bn_colors))
nn_cmap = ListedColormap(nn_colors, N=len(nn_colors))
an_cmap = ListedColormap(an_colors, N=len(an_colors))
colormaps = {"Below-Normal": bn_cmap, "Near-Normal": nn_cmap, "Above-Normal": an_cmap}
intervals = {"Below-Normal": bn_intervals, "Near-Normal": nn_intervals, "Above-Normal": an_intervals}

gefs_wk1cons = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week_1_cons.nc'))
gefs_wk1cca =  xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week1_cca.nc'))
gefs_wk1elr = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week1_elr.nc'))
if Path(os.path.join(gefs_procdir, 'gefs_week_2_cons.nc')).is_file():
    gefs_wk2cons = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week_2_cons.nc'))
else: gefs_wk2cons = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week_1_cons.nc'))
if Path(os.path.join(gefs_procdir, 'gefs_week2_cca.nc')).is_file():
    gefs_wk2cca = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week2_cca.nc'))
else: gefs_wk2cca = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week1_cca.nc'))
if Path(os.path.join(gefs_procdir, 'gefs_week2_elr.nc')).is_file():
    gefs_wk2elr = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week2_elr.nc'))
else: gefs_wk2elr = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week1_elr.nc'))

gefs_wk1cons = gefs_wk1cons.rename({'lon':'x', 'lat':'y'})
gefs_wk1cca = gefs_wk1cca.rename({'lon':'x', 'lat':'y', 'M':'e'})
gefs_wk1elr = gefs_wk1elr.rename({'lon':'x', 'lat':'y', 'M':'e'})
gefs_wk2cons = gefs_wk2cons.rename({'lon':'x', 'lat':'y'})
gefs_wk2cca = gefs_wk2cca.rename({'lon':'x', 'lat':'y', 'M':'e'})
gefs_wk2elr = gefs_wk2elr.rename({'lon':'x', 'lat':'y', 'M':'e'})

gefswk1_pcons_rgba = colors.process_gefs_probabilities(gefs_wk1cons, categories, colormaps, intervals,
                                               crs="EPSG:4326", time_index=0)

gefswk1pcons_prep = gefswk1_pcons_rgba.to_dataset(name = 'color')
gefswk1_pcons_mc = fc.convert_to_mercator(gefswk1pcons_prep, 'color')
gefswk1_pcons_mc['color'].rio.to_raster(os.path.join(figure_dir,'gefswk1pcons.tif'), dtype = 'uint8')

gefswk2_pcons_rgba = colors.process_gefs_probabilities(gefs_wk2cons, categories, colormaps, intervals,
                                               crs="EPSG:4326", time_index=0)
gefswk2pcons_prep = gefswk2_pcons_rgba.to_dataset(name = 'color')
gefswk2_pcons_mc = fc.convert_to_mercator(gefswk2pcons_prep, 'color')
if Path(os.path.join(gefs_procdir, 'gefs_week_2_cons.nc')).is_file():
    gefswk2_pcons_mc['color'].rio.to_raster(os.path.join(figure_dir,'gefswk2pcons.tif'), dtype = 'uint8')


cons1_stations = []
cca1_stations = []
elr1_stations = []
cons2_stations = []
cca2_stations = []
elr2_stations = []
for station in sl.stations:
    cons1_station = gefs_wk1cons.sel(x=station['lon'], y = station['lat'], method = 'nearest')
    cons1_station['station'] = station['name']
    cons1_stations.append(cons1_station)
    
    cca1_station = gefs_wk1cca.sel(x=station['lon'], y = station['lat'], method = 'nearest')
    cca1_station['station'] = station['name']
    cca1_stations.append(cca1_station)
    
    elr1_station = gefs_wk1elr.sel(x=station['lon'], y = station['lat'], method = 'nearest')
    elr1_station['station'] = station['name']
    elr1_stations.append(elr1_station)
    
    cca2_station = gefs_wk2cca.sel(x=station['lon'], y = station['lat'], method = 'nearest')
    cca2_station['station'] = station['name']
    cca2_stations.append(cca2_station)
    
    elr2_station = gefs_wk2elr.sel(x=station['lon'], y = station['lat'], method = 'nearest')
    elr2_station['station'] = station['name']
    elr2_stations.append(elr2_station)
    
    cons2_station = gefs_wk2cons.sel(x=station['lon'], y = station['lat'], method = 'nearest')
    cons2_station['station'] = station['name']
    cons2_stations.append(cons2_station)
cons1_stations = xr.concat(cons1_stations, dim = 'station')
cca1_stations = xr.concat(cca1_stations, dim = 'station')
elr1_stations = xr.concat(elr1_stations, dim = 'station')
cca2_stations = xr.concat(cca2_stations, dim = 'station')
elr2_stations = xr.concat(elr2_stations, dim = 'station')
cons2_stations = xr.concat(cons2_stations, dim = 'station')

cons1_stations['prob_colors'] = cons1_stations['prob']
cons1_stations['prob'] = cons1_stations['prob']*100
cca1_stations['prob_colors'] = cca1_stations['prob']
cca1_stations['prob'] = cca1_stations['prob']*100
elr1_stations['prob_colors'] = elr1_stations['prob']
elr1_stations['prob'] = elr1_stations['prob']*100
cca2_stations['prob_colors'] = cca2_stations['prob']
cca2_stations['prob'] = cca2_stations['prob']*100
elr2_stations['prob_colors'] = elr2_stations['prob']
elr2_stations['prob'] = elr2_stations['prob']*100
cons2_stations['prob_colors'] = cons2_stations['prob']
cons2_stations['prob'] = cons2_stations['prob']*100

for s, station in enumerate(sl.stations):
    bar1_consdata = []
    bar1_consdata_colors = []
    bar1_ccadata = []
    bar1_ccadata_colors = []
    bar1_elrdata = []
    bar1_elrdata_colors = []
    bar2_data = []
    bar2_data_colors = []
    bar2_ccadata = []
    bar2_ccadata_colors = []
    bar2_elrdata = []
    bar2_elrdata_colors = []

    # Prepare the data for each category (bn, nn, an)
    for c, cat in enumerate(categories):
        bar1_consdata.append(cons1_stations.isel(time=0,station=s,e=c).prob.values)
        bar1_consdata_colors.append(cons1_stations.isel(time=0,station=s,e=c).prob_colors.values)
        bar1_ccadata.append(cca1_stations.isel(time=0,station=s,e=c).prob.values)
        bar1_ccadata_colors.append(cca1_stations.isel(time=0,station=s,e=c).prob_colors.values)
        bar1_elrdata.append(elr1_stations.isel(time=0,station=s,e=c).prob.values)
        bar1_elrdata_colors.append(elr1_stations.isel(time=0,station=s,e=c).prob_colors.values)
        bar2_data.append(cons2_stations.isel(time=0,station=s,e=c).prob.values)
        bar2_data_colors.append(cons2_stations.isel(time=0,station=s,e=c).prob_colors.values)
        bar2_ccadata.append(cca2_stations.isel(time=0,station=s,e=c).prob.values)
        bar2_ccadata_colors.append(cca2_stations.isel(time=0,station=s,e=c).prob_colors.values)
        bar2_elrdata.append(elr2_stations.isel(time=0,station=s,e=c).prob.values)
        bar2_elrdata_colors.append(elr2_stations.isel(time=0,station=s,e=c).prob_colors.values)
        
        #normalize the intervals given colors/intervals defined above
        bnnorm = BoundaryNorm(boundaries = bn_intervals, ncolors=bn_cmap.N+1)
        nnnorm = BoundaryNorm(boundaries = nn_intervals, ncolors=nn_cmap.N+1)
        annorm = BoundaryNorm(boundaries = an_intervals, ncolors=an_cmap.N+1)
    
    colors1 = [
        bn_cmap(bnnorm(bar1_consdata_colors[0]*(bn_intervals[-1]-bn_intervals[0])+bn_intervals[0])),
        nn_cmap(nnnorm(bar1_consdata_colors[1]*(nn_intervals[-1]-nn_intervals[0])+nn_intervals[0])),
        an_cmap(annorm(bar1_consdata_colors[2]*(an_intervals[-1]-an_intervals[0])+an_intervals[0]))
    ]
    
    colors1cca = [
        bn_cmap(bnnorm(bar1_ccadata_colors[0]*(bn_intervals[-1]-bn_intervals[0])+bn_intervals[0])),
        nn_cmap(nnnorm(bar1_ccadata_colors[1]*(nn_intervals[-1]-nn_intervals[0])+nn_intervals[0])),
        an_cmap(annorm(bar1_ccadata_colors[2]*(an_intervals[-1]-an_intervals[0])+an_intervals[0]))
    ]
    
    colors1elr = [
        bn_cmap(bnnorm(bar1_elrdata_colors[0]*(bn_intervals[-1]-bn_intervals[0])+bn_intervals[0])),
        nn_cmap(nnnorm(bar1_elrdata_colors[1]*(nn_intervals[-1]-nn_intervals[0])+nn_intervals[0])),
        an_cmap(annorm(bar1_elrdata_colors[2]*(an_intervals[-1]-an_intervals[0])+an_intervals[0]))
    ]

    colors2 = [
        
        bn_cmap(bnnorm(bar2_data_colors[0]*(bn_intervals[-1]-bn_intervals[0])+bn_intervals[0])),
        nn_cmap(nnnorm(bar2_data_colors[1]*(nn_intervals[-1]-nn_intervals[0])+nn_intervals[0])),
        an_cmap(annorm(bar2_data_colors[2]*(an_intervals[-1]-an_intervals[0])+an_intervals[0]))
    ]

    colors2cca = [
        bn_cmap(bnnorm(bar2_ccadata_colors[0]*(bn_intervals[-1]-bn_intervals[0])+bn_intervals[0])),
        nn_cmap(nnnorm(bar2_ccadata_colors[1]*(nn_intervals[-1]-nn_intervals[0])+nn_intervals[0])),
        an_cmap(annorm(bar2_ccadata_colors[2]*(an_intervals[-1]-an_intervals[0])+an_intervals[0]))
    ]
    
    colors2elr = [
        bn_cmap(bnnorm(bar2_elrdata_colors[0]*(bn_intervals[-1]-bn_intervals[0])+bn_intervals[0])),
        nn_cmap(nnnorm(bar2_elrdata_colors[1]*(nn_intervals[-1]-nn_intervals[0])+nn_intervals[0])),
        an_cmap(annorm(bar2_elrdata_colors[2]*(an_intervals[-1]-an_intervals[0])+an_intervals[0]))
    ]

############### BAR PLOTS

    # Consolidated Week 1 Bar plot
    plt.bar(categories, bar1_consdata, color=colors1)
    # Add labels and title
    plt.ylabel("Probability (%)")
    plt.ylim(0,85)
    plt.title(station['name'] + ' GEFS Week 1 Consolidated Precip Probabilities')
    # Save the box plot as a PNG file with the station name
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'station_data', f"{station['name']}_pconswk1bar.png"))  # Save as PNG file
    plt.close()  # Close the plot to avoid memory issues
    
    # CCA tercile bar plot
    plt.bar(categories, bar1_ccadata, color=colors1cca)
    plt.ylabel("Probability (%)")
    plt.ylim(0,85)
    plt.title(station['name'] + ' GEFS Week 1 CCA Precip Probabilities')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'station_data', f"{station['name']}_pccaswk1bar.png"))  # Save as PNG file
    plt.close()  # Close the plot to avoid memory issues
    
    # ELR tercile bar plot
    plt.bar(categories, bar1_elrdata, color=colors1elr)
    plt.ylabel("Probability (%)")
    plt.ylim(0,85)
    plt.title(station['name'] + ' GEFS Week 1 ELR Precip Probabilities')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'station_data', f"{station['name']}_pelrwk1bar.png"))  # Save as PNG file
    plt.close()  # Close the plot to avoid memory issues
    
    # Week 2 consolidated plot
    if Path(os.path.join(gefs_procdir, 'gefs_week2_cons.nc')).is_file():
        plt.bar(categories, bar2_data, color=colors2)
        # Add labels and title
        plt.ylabel("Probability (%)")
        plt.ylim(0,85)
        plt.title(station['name'] + ' GEFS Week 2 Consolidated Precip Probabilities')
        # Save the box plot as a PNG file with the station name
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, 'station_data', f"{station['name']}_pconswk2bar.png"))  # Save as PNG file
        plt.close()  # Close the plot to avoid memory issues
    
    # CCA tercile bar plot WEEK 2
    if Path(os.path.join(gefs_procdir, 'gefs_week2_cca.nc')).is_file():
        plt.bar(categories, bar2_ccadata, color=colors2cca)
        plt.ylabel("Probability (%)")
        plt.ylim(0,85)
        plt.title(station['name'] + ' GEFS Week 2 CCA Precip Probabilitish states')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, 'station_data', f"{station['name']}_pccaswk2bar.png"))  # Save as PNG file
        plt.close()  # Close the plot to avoid memory issues

    # ELR tercile bar plot WEEK 2
    if Path(os.path.join(gefs_procdir, 'gefs_week2_elr.nc')).is_file():
        plt.bar(categories, bar2_elrdata, color=colors2elr)
        plt.ylabel("Probability (%)")
        plt.ylim(0,85)
        plt.title(station['name'] + ' GEFS Week 2 ELR Precip Probabilities')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, 'station_data', f"{station['name']}_pelrwk2bar.png"))  # Save as PNG file
        plt.close()  # Close the plot to avoid memory issues
