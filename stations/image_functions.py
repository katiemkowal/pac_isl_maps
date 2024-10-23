# import pptx
# from pptx import Presentation
# from pptx.util import Inches,Pt
# from pptx.dml.color import RGBColor
# from pptx.enum.text import PP_ALIGN
# import os
from io import BytesIO
# import requests
# from datetime import datetime,timedelta
# import warnings
# import aspose.slides as slides
# import rasterio
# import xarray as xr
import io
from PIL import Image
import numpy as np
# import matplotlib as matplotlib
# import matplotlib.pyplot as plt
import geopandas as gpd
# from shapely.geometry import Point, LineString, Polygon
import pandas as pd
# import matplotlib.patheffects as path_effects
# from matplotlib.colors import ListedColormap
# from matplotlib.colorbar import ColorbarBase
# import matplotlib.colors as mcolors
# import random
# import helper_functions as helper
# import helper_dicts as dict
# import plotting_functions as plotf


def convert_image_to_array(image,crop_box):
    image_cropped = image.crop(crop_box)
    if image_cropped.mode == 'P':
        # Convert indexed image to RGB
        image_cropped = image_cropped.convert('RGB')
    image_np = np.array(image_cropped)
    return image_np

def mask_array_by_color(array, colors):
    masks = []
    for color in colors:
        mask = np.all(array == color, axis = -1)
        masks.append(mask)
    combined_mask = np.logical_or.reduce(masks)
    masked_image = np.zeros_like(array)
    masked_image[combined_mask] = array[combined_mask]
    return(masked_image)

def calc_unique_colors(image_array):
    # Reshape the array to a list of pixels
    pixels = image_array.reshape(-1, 3)
    
    # Find unique colors
    unique_colors = np.unique(pixels, axis=0)
    
    # Limit the number of colors to plot
    max_colors = 16
    unique_colors = unique_colors[:max_colors]
    return unique_colors

# Function to calculate the Euclidean distance
def euclidean_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

# Function to find the closest category
def closest_color_category(color, color_categories):
    distances = {category: euclidean_distance(color, rgb) for category, rgb in color_categories.items()}
    return min(distances, key=distances.get)

def convert_image_to_coordinates(image_to_process, minx, maxx, miny, maxy):
    if isinstance(image_to_process, np.ndarray):
        image = Image.fromarray(image_to_process)
    else:
        image_stream = io.BytesIO(image_to_process)
        image = Image.open(image_stream)
        if image.mode == 'P':
            # Convert indexed image to RGB
            image = image.convert('RGB')
    
    width, height = image.size

    coordinates = []
    values = []
    
    # Loop through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the RGB value of the pixel
            pixel_value = image.getpixel((x, y))
            
            # Store the coordinate and pixel value
            coordinates.append((x, y))
            values.append(pixel_value)
    
    # Create a DataFrame with coordinates and RGB values
    df = pd.DataFrame(coordinates, columns=['x', 'y'])
    df[['R', 'G', 'B']] = pd.DataFrame(values, index=df.index)
    
    # Scale x-coordinates from pixel indices to the target range
    df['x_scaled'] = minx + (df['x'] / (width - 1)) * (maxx - minx)
    df['y_scaled'] = miny + (df['y'] / (height - 1)) * (maxy - miny) 

    return df


def convert_df_to_reduced_gdf(df, xsample, ysample):
    gpd_df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.x_scaled, df.y_scaled))
    gpd_df.set_crs(epsg=4326, inplace = True)
    gpd_df = gpd_df[['R','G', 'B', 'geometry']]
    gpd_df = gpd_df.drop_duplicates()
    # Drop every other entry by selecting rows with even indices
    gpd_df_reduced = gpd_df.iloc[::xsample].reset_index(drop=True)
    gpd_df_reduced['x'] = gpd_df_reduced.geometry.x
    gpd_df_reduced['y'] = gpd_df_reduced.geometry.y
    gpd_df_sorted = gpd_df_reduced.sort_values(by = ['x','y'])
    gpd_df_reduced = gpd_df_sorted.iloc[::ysample].reset_index(drop=True)
    return gpd_df_reduced

def convert_color(df):
    df['color'] = df.apply(lambda row: (
        row['R'] / 255.0,  # Normalize Red
        row['G'] / 255.0,  # Normalize Green
        row['B'] / 255.0   # Normalize Blue
    ), axis=1)
    return df


def calc_states_in_extreme_threshold(df, states, metric, number_in_threshold, plot_resolution, colors_in_image, plotting):
    #set number of pixels required to to appear for a match to be considered
    if plot_resolution['res_name'] == 'low_res':
        pixels_required = 1
    elif plot_resolution['res_name'] =='med_res':
        pixels_required = 3
    elif plot_resolution['res_name'] == 'high_res':
        pixels_required = 5
    #how many colors to iterate through before location isn't considered to include that threshold, e.g. no dry locations
    number_of_colors_to_check_for_extreme_inclusion = 11
    
    #create some empty lists
    states_above_threshold = []
    states_below_threshold = []
    total_high = []
    total_low = []
    highest_value = False
    lowest_value = False
    
    #for each state with a shapefile geometry
    for state in range(0,len(states)):
        #crop df to state
        if isinstance(df, list):
            df_check = df[state]
        else: df_check = df
        df_state = df_check[df_check.geometry.within(states.loc[state, 'geometry'])]
        
        #print('testing ' + states.loc[state, 'ADM1_ES']) 
        if plotting == 'on':
            df_state.plot(color = df_state['color'])
            plt.title(states.loc[state,'ADM1_ES'])
            plt.show()
            matplotlib.pyplot.close()

        # #choose how the color will be interpreted as a value based on dictionaries
        if metric == 'absolute':
            color_bar = dict.total_colorbar
            conversion_dict = dict.convert_threshold_color
        elif metric == 'anom':
            color_bar = dict.anom_colorbar
            conversion_dict = dict.convert_anom_color
        
        #check each threshold to see top three thresholds of a state
        any_high = 0
        high_color_count = 0
        highest_state_value = False
        for hr in color_bar[::-1]:
            if (any_high < number_in_threshold):
                #check whether any grid points in a state meet the color of interest
                condition = (df_state['R'] == colors_in_image[hr][0]) & (df_state['G'] == colors_in_image[hr][1]) & (df_state['B'] == colors_in_image[hr][2])
                matching_high = df_state[condition]

                #if any match the color that is being checked
                if len(matching_high) >= pixels_required and (high_color_count < number_of_colors_to_check_for_extreme_inclusion):#not matching_high.empty:
                    #print('match found')
                    #add the state to to the list of matches if the color condition is matched
                    if len(total_high) < number_in_threshold:
                        total_high.append(conversion_dict[hr])
                    else:
                        min_to_check = min(total_high)
                        min_value_index = total_high.index(min(total_high))
                        if (conversion_dict[hr] > min_to_check) and (conversion_dict[hr] not in total_high):
                            total_high[min_value_index] = conversion_dict[hr]
                    if not highest_state_value or (conversion_dict[hr] > highest_state_value):
                        highest_state_value = conversion_dict[hr]
                    states_above_threshold.append((states.loc[state, 'ADM1_ES'], conversion_dict[hr]))
                    any_high = any_high + 1
                elif len(matching_high) >= pixels_required and (high_color_count >= number_of_colors_to_check_for_extreme_inclusion) and not highest_state_value:
                        highest_state_val = conversion_dict[hr]
            high_color_count = high_color_count + 1
            if not highest_value or (highest_state_value > highest_value):
                highest_value = highest_state_value
                    
        any_low = 0
        low_color_count = 0
        lowest_state_value = False
        for lr in color_bar:
            if (any_low < number_in_threshold):
                condition = (df_state['R'] == colors_in_image[lr][0]) & (df_state['G'] == colors_in_image[lr][1]) & (df_state['B'] == colors_in_image[lr][2])
                matching_low = df_state[condition]
                if len(matching_low) >= pixels_required  and (low_color_count < number_of_colors_to_check_for_extreme_inclusion):#not matching_low.empty:
                    if len(total_low) < number_in_threshold:
                        total_low.append(conversion_dict[lr])
                    else:
                        max_to_check = max(total_low)
                        max_value_index = total_low.index(max(total_low))
                        if (conversion_dict[lr] < max_to_check) and (conversion_dict[lr] not in total_low):
                            total_low[max_value_index] = conversion_dict[lr]
                    if not lowest_state_value or (conversion_dict[lr] < lowest_state_value):
                        lowest_state_value = conversion_dict[lr]
                    states_below_threshold.append((states.loc[state, 'ADM1_ES'], conversion_dict[lr]))
                    any_low = any_low + 1
                elif len(matching_low) >= pixels_required and (low_color_count >= number_of_colors_to_check_for_extreme_inclusion) and (not lowest_state_value):
                        lowest_state_value = conversion_dict[lr]
            low_color_count = low_color_count + 1
        if not lowest_value or (lowest_state_value < lowest_value):
            lowest_value = lowest_state_value
    #filter to only include top in threshold
    states_above_threshold = [(state, num) for state, num in states_above_threshold if num in total_high]
    states_below_threshold = [(state, num) for state, num in states_below_threshold if num in total_low]
     
    return states_above_threshold, states_below_threshold, highest_value, lowest_value