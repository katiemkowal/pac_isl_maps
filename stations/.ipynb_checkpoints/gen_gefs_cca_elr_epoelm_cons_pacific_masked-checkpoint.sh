cd /cpc/africawrf/ebekele/projects/PREPARE_pacific/notebooks/unmasked

dtt=`date --date "0 day ago" "+%Y%m%d"`
dt=`date --date "0 day ago" "+%d%b%Y"`
iwk1=`date --date "-1 day ago" "+%d%b%Y"`
iwk2=`date --date "-8 day ago" "+%d%b%Y"`
iwk3=`date --date "-15 day ago" "+%d%b%Y"`
iwk34=`date --date "-15 day ago" "+%d%b%Y"`
iwk1234=`date --date "-1 day ago" "+%d%b%Y"`

fwk1=`date --date "-7 day ago" "+%d%b%Y"`
fwk2=`date --date "-14 day ago" "+%d%b%Y"`
fwk3=`date --date "-21 day ago" "+%d%b%Y"`
fwk34=`date --date "-28 day ago" "+%d%b%Y"`
fwk1234=`date --date "-28 day ago" "+%d%b%Y"`

for wk in $1; do

if [ $wk = 1 ]; then iwk=$iwk1; fwk=$fwk1; fi
if [ $wk = 2 ]; then iwk=$iwk2; fwk=$fwk2; fi
if [ $wk = 3 ]; then iwk=$iwk3; fwk=$fwk3; fi
if [ $wk = 34 ]; then iwk=$iwk34; fwk=$fwk34; fi
if [ $wk = 1234 ]; then iwk=$iwk1234; fwk=$fwk1234; fi

iwkk=$(date -d"$iwk + 0 day" +"%d%b")
fwkk=$(date -d"$fwk + 0 day" +"%d%b")

cat>gen_cons_masked.py<<eofPY
import xcast as xc
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
#import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
#import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
import matplotlib
matplotlib.use('Agg')

pcca = xr.open_dataset('./gefs_week${wk}_cca.nc').prob
pelr = xr.open_dataset('./gefs_week${wk}_elr.nc').prob
pepoelm = xr.open_dataset('./gefs_week${wk}_epoelm.nc').prob

rcca = xr.open_dataset('./gefs_week${wk}_cca_groc.nc').skill
relr = xr.open_dataset('./gefs_week${wk}_elr_groc.nc').skill
repoelm = xr.open_dataset('./gefs_week${wk}_epoelm_groc.nc').skill

mask1 = rcca < 0.5
rcca = rcca.where(~mask1,0)
mask2 = relr < 0.5
relr = relr.where(~mask2,0)
mask3 = repoelm < 0.5
repoelm = repoelm.where(~mask3,0)

srcca = rcca * rcca
srelr = relr * relr
srepoelm = repoelm * repoelm

utt = (pcca*srcca) + (pelr*srelr) + (pepoelm*srepoelm)
btt = srcca + srelr + srepoelm
pcons = (utt) / btt

import sys
sys.path.append('./libs')

filename = 'ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'
world = gpd.read_file(filename)
extent = [132, 205, -22, 9]  # Adjusted to match the domain you mentioned
exclude_countries = ['Marshall Is.', 'Australia', 'New Zealand']

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from prob import pview_probabilistic
# view probabilistic and view, with cross-dateline=True will force back to (0,360) for plotting , won't change the original data tho
ax = pview_probabilistic(pcons.isel(time=0), ocean=False, cross_dateline=True, title='GEFS, Week-${wk}, CCA/ELR/EPOELM Cons. Valid: ${iwk} - ${fwk}')

for country, geom in zip(world['NAME'], world['geometry']):
    if country not in exclude_countries and \
       geom.bounds[0] < extent[1] and geom.bounds[2] > extent[0] and \
       geom.bounds[1] < extent[3] and geom.bounds[3] > extent[2]:
        centroid = geom.representative_point().coords[:][0]
        ax.text(centroid[0], centroid[1], country, fontsize=6, color='red', ha='center',
                transform=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='lightsteelblue', alpha=0.5, linestyle='--', draw_labels=True)
gl.top_labels = False
gl.left_labels = False
gl.right_labels=True
gl.xlines = True
gl.xlocator = mticker.FixedLocator([ 130, 140, 150, 160, 170, 180, -170, -160, -150, -140])
gl.ylocator = mticker.FixedLocator([-25, -20, -15, -10, -5, 0, 5, 10])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'black'}
gl.top_labels = False
gl.left_labels = True
gl.right_labels=False
gl.bottom_labels=True
plt.savefig('fig_dir/gefs_week_${wk}_cons.png', dpi=300)
ttt = pcons.rename("prob")
ttt = ttt.rename({'M' : 'e'})
ttt.to_netcdf(f'./gefs_week_${wk}_cons.nc')

eofPY

/cpc/home/ebekele/.conda/envs/xcast_env/bin/python gen_cons_masked.py

convert -trim fig_dir/gefs_week_${wk}_cons.png fig_dir/gefs_week_${wk}_cons.png
convert -bordercolor white -border 10  fig_dir/gefs_week_${wk}_cons.png fig_dir/gefs_week_${wk}_cons.png

cat>gefs_week_${wk}_cons.ctl<<eofCTL
dset gefs_week_${wk}_cons.nc
title model
undef 9.96921e+36
dtype netcdf
xdef 293 linear 132 0.25
ydef 125 linear -22 0.25
zdef 1 linear 0 1
tdef 1 linear 00Z10Jul2024 1mn
edef 3
1 1 00Z10jul2024
2 1 00Z10jul2024
3 1 00Z10jul2024
endedef
vars 1
prob=>prob  0  y,x,t,e  probability
endvars
eofCTL

cat>gen_geotiff.gs<<eofGS
'reinit'
'open gefs_week_${wk}_cons.ctl'
'set x 1 293'
'set y 1 125'
'set geotiff gefs_week${wk}_precip_below.tif'
'set gxout geotiff'
'set e 1'
'd prob'
'c'
'set geotiff gefs_week${wk}_precip_above.tif'
'set gxout geotiff'
'set e 3'
'd prob'
'quit'
eofGS
/cpc/home/ebekele/grads2.1/grads-2.1.0/bin/grads -blc gen_geotiff.gs

mv gefs_week${wk}_precip_*.tif fig_dir/
done
