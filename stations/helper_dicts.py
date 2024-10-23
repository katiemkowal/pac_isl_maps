# import pptx
# from pptx import Presentation
# from pptx.util import Inches,Pt
# from pptx.dml.color import RGBColor
# from pptx.enum.text import PP_ALIGN
# import os
# from io import BytesIO
# import requests
# from datetime import datetime,timedelta
# import warnings
# import aspose.slides as slides
# import rasterio
# import xarray as xr
# import io
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import geopandas as gpd
# from shapely.geometry import Point, LineString, Polygon
# import pandas as pd
# import matplotlib.patheffects as path_effects
# from matplotlib.colors import ListedColormap
# from matplotlib.colorbar import ColorbarBase
# import matplotlib.colors as mcolors
# import random
# import helper_functions as helper
# import plotting_functions as plotf
# import image_functions as imagef
# import text_functions as textf
# import importlib.util
# import sys

color_bar_crop = (150, 1430, 2350, 1450)#left, top, right, bottom

color_categories = {
    'black' : [0,0,0],
    'dark_red': [170,0,0],
    'bright_red': [255, 30, 0],
    'salmon':[230,140,140],
    'light_pink':[255,230,230],
    'orange': [255, 160, 0],
    'yellow': [255,240,140],
    'dark_brown': [120,80,70],
    'med_brown':[180,140,130],
    'light_brown':[240,220,210],
    'white':[255,255,255],
    'light_green':[200,255,190],
    'bright_green':[120,245,115],
    'dark_green':[30,180,30],
    'light_blue':[180,240,250],
    'lightmed_blue':[150,210,250],
    'med_blue':[80,165,245],
    'dark_blue':[35,120,235],
    'light_purple':[220,220,255],
    'med_purple':[160,140,255],
    'dark_purple':[120,105,230],
}

total_colorbar = ['white', 'light_green', 'bright_green', 'dark_green',
                  'light_blue', 'med_blue', 'dark_blue',
                  'light_purple', 'med_purple', 'dark_purple',
                  'yellow', 'orange', 'bright_red',
                  'dark_red', 'salmon', 'light_pink']

anom_colorbar = ['dark_red', 'bright_red', 'orange', 'yellow',
                 'dark_brown', 'med_brown', 'light_brown', 'white',
                 'light_green', 'bright_green', 'dark_green',
                 'lightmed_blue', 'dark_blue', 'light_purple', 'dark_purple']


#unique colors
black = (0,0,0)
white = (255,255,255)
dark_blue_abs =(30,110,235)
dark_blue_anom = (40,130,240)
dark_green =(30,180,30)
med_blue = (80,165,245)
dark_purple_abs = (112,96, 220)
dark_purple_anom = (128, 112, 235)
bright_green = (120,245,115)
med_purple = (160,140,255)
dark_red_abs = (165,0,0)
dark_red_anom = (192,0,0)
gray = (170,170,170)
light_blue = (180,240, 250)
lightmed_blue = (150,210,250)
light_green = (200, 255,190)
light_purple = (220,220,255)
red_abs = (225,20,0)
red_anom = (255,50,0)
salmon = (230,140,140)
orange = (255,160, 0)
light_pink = (255,230, 230)

light_yellow = (255, 250,170)
med_yellow = (255,232,120)
dark_brown = (120,80,70)
med_brown = (180,140,130)
light_brown = (240,220,210)

dark_green_anom = (30,180,30)
bright_green_anom = (120,245,115)
light_green_anom = (200,255,190)

#text colors
blue_text = (21,73,125)


med_rainfall = ['dark_blue', 'med_blue', 'dark_purple', 'med_purple', 'light_purple', 'light_blue', 'yellow', 'orange']
med_rainfall_under30 = ['dark_blue', 'med_blue', 'light_purple']
med_anom = ['dark_brown', 'med_brown', 'light_brown', 'white', 'light_green', 'bright_green', 'dark_green']
med_anom_under30 = ['white', 'light_green']
high_rainfall = ['bright_red', 'dark_red', 'salmon', 'light_pink']
high_rainfall_under30 = ['med_purple', 'dark_purple', 'yellow', 'orange', 'bright_red', 'dark_red', 'salmon', 'light_pink']
low_rainfall_under30 = ['white', 'light_green', 'bright_green']
low_rainfall = ['white', 'light_green', 'bright_green', 'dark_green']
high_anom_cat =  ['lightmed_blue', 'dark_blue', 'light_purple', 'dark_purple']
high_anom_under30 = ['bright_green', 'dark_green', 'lightmed_blue', 'dark_blue', 'light_purple', 'dark_purple']
low_anom_cat =  ['dark_red', 'bright_red', 'orange', 'yellow']
low_anom_under30 = ['dark_red', 'bright_red', 'orange', 'yellow', 'dark_brown', 'light_brown']

convert_threshold_color = {
    'white': 0,
    'light_green': 2,
    'bright_green': 5,
    'dark_green': 10,
    'light_blue': 25,
    'med_blue': 50,
    'dark_blue': 75,
    'light_purple': 100,
    'med_purple': 150, 
    'dark_purple': 200,
   'yellow': 300,
    'orange': 500,
    'bright_red': 750,
    'dark_red': 1000,
    'salmon': 1500,
   'light_pink': 2500}
convert_color_threshold = {v: k for k, v in convert_threshold_color.items()}

convert_anom_color = {
    'dark_red': -500,
    'bright_red': -300,
    'orange': -200,
    'yellow': -100,
    'dark_brown': -50,
    'med_brown': -25,
    'light_brown': -10,
    'white': 0,
    'light_green': 10,
    'bright_green': 25,
    'dark_green': 50,
    'lightmed_blue': 100,
    'dark_blue': 200,
    'light_purple': 300,
    'dark_purple': 500}

convert_color_anom = {v: k for k, v in convert_anom_color.items()}

subsequent_threshold = {
    0:2,
    2:5,
    5:10,
    10:25,
    25:50,
    50:75,
    75:100,
    100:150,
    150:200,
    200:300,
    300:500,
    500:750,
    750:1000,
    1000:1500,
    1500:2500}

subsequent_anom = {
    -500: -300,
    -300: -200,
    -200: -100,
    -100: -50,
    -50: -25,
    -25: -10,
    -10: 0,
    0: 10,
    10:25,
    25:50,
    50:100,
    100:200,
    200:300,
    300:500}

convert_threshold_number = {
    0: '0-2mm',
    2: '2-5mm',
    5: '5-10mm',
    10: '10-25mm',
    25: '25-50mm',
    50: '50-75mm',
    75: '75-100mm',
    100: '100-150mm',
    150: '150-200mm', 
    200: '200-300mm',
   300: '300-500mm',
    500: '500-750mm',
    750: '750-1000mm',
    1000: '1000-1500mm',
    1500: '1500-2500mm',
   2500: '>2500mm'
}

convert_anom_number = {
    -500: '>500mm',
    -300: '300-500mm',
    -200: '200-300mm',
    -100: '100-200mm',
    -50: '50-100mm',
    -25: '25-50mm',
    -10: '10-25mm',
    0: '0-10mm',
    10: '10-25mm',
    25: '25-50mm',
    50: '50-100mm',
    100: '100-200mm',
    200: '200-300mm',
    300: '300-500mm',
    500: '>500mm'}

#for calculating issue dates and valid period text
month_number_dict = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December',
    'January': 'Jan',
    'February': 'Feb',
    'March': 'Mar',
    'April': 'Apr',
    'May': 'May',
    'June': 'Jun',
    'July': 'Jul',
    'August': 'Aug',
    'September': 'Sep',
    'October': 'Oct',
    'November': 'Nov',
    'December': 'Dec'}

resolution_dict = {
    'low_res': {'res_name': 'low_res',
                'Mexico':[5,5,15],
                 'Central America': [5,4,50],
                 'Caribbean':[6,5,15]},
    'med_res':{'res_name': 'med_res',
               'Mexico':[3,3,13],
                 'Central America': [3,3,35],
                 'Caribbean':[4,4,20]},
    'high_res':{'res_name': 'high_res',
                'Mexico':[2,2,13],
                 'Central America': [2,2,15],
                 'Caribbean':[3,3,15]}
}
previous_week_dict = {
    'Monday': 7,
    'Tuesday': 8,
    'Wednesday': 9,
    'Thursday': 10,
    'Friday': 11
}

#picture setups
mex_coords = (-120,-85,35,10)
mex_box = (145, 145,1120,1028)
ca_coords = (-95, -75, 20, 5)
ca_box = (835, 675, 1388, 1203)
carib_coords = (-85,-55,30,10) 
carib_box = (1112,325,1940, 1028)