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
ymaxgef=90
zdimgef = 15

minptotal = 0
maxptotal = 3500

ptotal_intervals = [0, 2, 5, 10, 25, 50, 75, 100,
                    150, 200, 300, 500, 750,1000,
                    1500, 2500, 3500]

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

## read in raw gefs data
gefs_raw = fc.read_in_binary_gefs(os.path.join(gefs_rawdir,'gefs_week1_precip_' + date_str + 'IC.dat'), xdimgef, ydimgef, zdimgef, xmingef, xmaxgef, ymingef, ymaxgef)

gefs_totalp = gefs_raw.isel(var=1)
gefs_totalp = gefs_totalp.to_dataset(name = 'tp')
gefs_totalp = gefs_totalp.rename({'lon':'x', 'lat':'y'})
gefs_total_crs = gefs_totalp.rio.write_crs('EPSG:4326', inplace = True)
gefs_totalp = gefs_total_crs.sel(x=slice(0,359.999), y = slice(-90,90))
gefs_tp_mc = fc.convert_to_mercator(gefs_totalp, 'tp')
gefs_tpnorm = np.clip(gefs_tp_mc.tp, minptotal, maxptotal)
ptotal_cmap = ListedColormap(ptotal_colors, N=len(ptotal_colors))
ptotal_norm = BoundaryNorm(boundaries=ptotal_intervals, ncolors=len(ptotal_colors))
ptotal_rgb = colors.apply_colormap(gefs_tpnorm, ptotal_cmap, ptotal_norm, ptotal_intervals)
ptotal_rgb.rio.to_raster(os.path.join(figure_dir, 'gefswk1ptotal.png'), dtype="uint8")

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
