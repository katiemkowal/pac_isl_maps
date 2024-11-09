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
# import helper_dicts as dicts
# import plotting_functions as plotf
# import image_functions as imagef
# import text_functions as textf
# import importlib.util
# from docx import Document
# import sys

def can_be_integer(s):
    try:
        int(s)  # Attempt to convert the string to an integer
        return True
    except ValueError:
        return False

def invert_dict(dict):
    # Invert the dictionary
    inverted_map = {v: k for k, v in dict.items()}
    return inverted_map

def get_variable_name(variable, scope=globals()):
    # Search for the variable name in the given scope (globals or locals)
    names = [name for name, value in scope.items() if value is variable]
    if names:
        return names[0]
    else: 
        scope2 = vars(globals().get('dicts')).items()
        names = [name for name, value in scope2 if value is variable]
        if names != None:
            return names[0]
        else: return None

# Initialize an empty dictionary to hold the maximum value for each state
def collapse_list(states, maxmin):
    state_max_values = {}
    
    # Iterate through the list of tuples
    for state, value in states:
        # Update the dictionary with the maximum value for each state
        if maxmin == 'max':
            if state not in state_max_values or value > state_max_values[state]:
                state_max_values[state] = value
        elif maxmin == 'min':
             if state not in state_max_values or value < state_max_values[state]:
                state_max_values[state] = value
    
    # Convert the dictionary back into a list of tuples
    collapsed_list = list(state_max_values.items())
    return collapsed_list

def get_extremes_list(state_list, extreme_type):
    if extreme_type == 'max':
        extreme_value = max(num for state, num in state_list)
    elif extreme_type == 'min':
        extreme_value = min(num for state, num in state_list)
    return [(state, num) for state, num in state_list if num == extreme_value]