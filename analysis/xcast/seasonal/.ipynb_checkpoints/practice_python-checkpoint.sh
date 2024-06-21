#!/bin/sh

wdir=/cpc/int_desk/pac_isl/analysis/prep_data/nmme
py=/cpc/home/ebekele/.conda/envs/xcast_env/bin/python


cat>practice_python.py<<eofPY
import xcast as xc 
import xarray as xr 
import cartopy.crs as ccrs 
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping
print('works')

eofPY

$py practice_python.py

