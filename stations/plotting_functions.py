# import os
# from io import BytesIO
# import warnings
# import io
from PIL import Image
# import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.table import Table
# import geopandas as gpd
# from shapely.geometry import Point, LineString, Polygon
# import pandas as pd
# import matplotlib.patheffects as path_effects
# from matplotlib.colors import ListedColormap
# from matplotlib.colorbar import ColorbarBase
# import matplotlib.colors as mcolors
# import random
# import helper_functions as helper
# import helper_dicts as dict
# import image_functions as imagef

def plot_original_image(image):
    # #get original 180day picture from slide 4
    # p_180_raw = shape
    # image_stream = io.BytesIO(p_180_raw.image.blob)
    print(image)
    if image.mode == 'P':
        # Convert indexed image to RGB
        image = image.convert('RGB')
    
    # Plot the image using Matplotlib
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')  # Hide axis labels
    plt.show()

# Define jitter function
def jitter_coords(x, y, x_scale, y_scale):
    """Add random jitter to coordinates."""
    return x + np.random.uniform(-x_scale, x_scale), y + np.random.uniform(-y_scale, y_scale)

def plot_threshold_states(df, state_shapes, extra_shapes, state_thresholds, number_in_threshold, 
                          unique_low, unique_high,
                          point_size,
                          threshold_type, color_mapping,
                         days_name, region_name, figure_dir, plot_show):
    fig, ax = plt.subplots(figsize = (12,8))

    states_projected = state_shapes.to_crs(epsg=4326)
    unique_colors = sorted(set(number for state, number in state_thresholds))

    if len(unique_high) > 0:
        max_states = helper.get_extremes_list(state_thresholds, 'max')
        max_state = [state for state,number in max_states]
    else: max_state = []
    if len(unique_low) > 0:
        min_states = helper.get_extremes_list(state_thresholds, 'min')
        min_state = [state for state,number in min_states]
    else:
        min_state = []
        
    if threshold_type == 'absolute':
        inverted_dict = helper.invert_dict(dict.convert_threshold_color)
        regular_dict = dict.convert_threshold_color
    elif threshold_type == 'anom':
        inverted_dict = helper.invert_dict(dict.convert_anom_color)
        regular_dict = dict.convert_anom_color

    
    annotated_list = []
    state_shapes.geometry.plot(ax = ax, color = 'lightgrey', edgecolor = 'black')
    for state in state_thresholds:
        if isinstance(df, list):
            for d in df:
                if d['country'].iloc[0] == state[0]:
                    print('found df')
                    df_check = d
        else: df_check = df
        df_state = df_check[df_check.geometry.within(state_shapes[state_shapes['ADM1_ES'] == state[0]].union_all())]
        hr = color_mapping[inverted_dict[state[1]]]

        condition = (df_state['R'] == hr[0]) & (df_state['G'] == hr[1]) & (df_state['B'] == hr[2])
        matching_high = df_state[condition]

        if not matching_high.empty:
            matching_high.plot(ax = ax, color = matching_high['color'], markersize=point_size)
            hr_color = (hr[0]/255, hr[1]/255, hr[2]/255)
            
            if 'Veracruz' in state[0]:
                text = 'Veracruz'
            elif 'Michoacán' in state[0]:
                text = 'Michoacán'
            elif 'Dominican' in state[0]:
                text = 'DR'
            else: text = state[0]

            if (state[0] in (max_state + min_state)) and (text not in annotated_list):
                centroid = states_projected[states_projected['ADM1_ES'] == state[0]]['geometry'].centroid
                # Plot the state name at the centroid of each geometry
                if state[0] in ['Bahamas', 'Lesser Antilles']:
                    x,y = float(centroid.x.iloc[0]), float(centroid.y.iloc[0]) + 3.3
                elif 'Pacific' in state[0]:
                    x,y = float(centroid.x.iloc[0]), float(centroid.y.iloc[0]) - 1
                elif 'Central' in state[0]:
                    x,y = float(centroid.x.iloc[0]), float(centroid.y.iloc[0]) - 0.5
                else: x,y = float(centroid.x.iloc[0]), float(centroid.y.iloc[0]) + 1

                jittered_x, jittered_y = jitter_coords(x, y, x_scale = 0.01, y_scale=0.1)
                ax.text(
                    jittered_x, 
                    jittered_y,  # x and y coordinates
                    text,        # Text to display
                    fontsize=10,             # Font size
                    ha='center',             # Horizontal alignment
                    va='center',             # Vertical alignment
                    color='white' , 
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='black')],# Text color
                    weight = 'bold'
                )
                annotated_list.append(text)
            if state[0] != 'Bahamas':
                state_shapes[state_shapes['ADM1_ES'] == state[0]].geometry.plot(ax = ax, color = 'none', edgecolor = 'black')
        else: print(state[0] + ' producing an empty list')
    if not extra_shapes.empty:
        extra_shapes.geometry.plot(ax = ax, color = 'none', edgecolor = 'black')
    
    # Define your custom color thresholds and corresponding colors
    color_map = {}
    #print(len(unique_colors))
    if len(unique_colors) > (number_in_threshold) or (len(unique_low) >= 1 and len(unique_high) >= 1):
        if threshold_type =='absolute':
            sub_number = dict.subsequent_threshold[unique_colors[len(unique_low)-1]]
        elif threshold_type == 'anom':
            sub_number = dict.subsequent_anom[unique_colors[len(unique_low)-1]]
        unique_colors.insert(len(unique_low),sub_number)

        for count, h in enumerate(unique_colors):
            if count != len(unique_low):
                color_map[h] = (color_mapping[inverted_dict[h]][0]/255, 
                            color_mapping[inverted_dict[h]][1]/255, 
                            color_mapping[inverted_dict[h]][2]/255)
            else: color_map[h] = (211/255,211/255,211/255)
    else:
        for count, h in enumerate(unique_colors):
            color_map[h] = (color_mapping[inverted_dict[h]][0]/255, 
                            color_mapping[inverted_dict[h]][1]/255, 
                            color_mapping[inverted_dict[h]][2]/255)
    #print(unique_colors)
    # Create a colormap and norm based on your data
    cmap = ListedColormap([mcolors.to_hex(c) for c in color_map.values()])
    norm = mcolors.BoundaryNorm(boundaries=list(range(len(color_map) + 1)), ncolors=len(color_map))
    
    # Add the colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])  #x,y,width,height
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='horizontal')

    # Set colorbar ticks and labels
    if threshold_type == 'anom':
        # Initialize lists to hold tick positions and labels
        ticks = []
        tick_labels = []
        for i in range(len(color_map)):
            if (i == number_in_threshold) and (len(unique_colors) == number_in_threshold*2):
                continue
            elif unique_colors[i] < 0:
                # Assuming you want negative values on the right side
                ticks.append(1 + i)  # Adjust this based on your color map scale
                tick_labels.append(unique_colors[i])
            elif unique_colors[i] > 0:
                # Assuming you want positive values on the left side
                ticks.append(i)  # Adjust this based on your color map scale
                tick_labels.append(unique_colors[i])
        
            # Set the ticks and labels on the color bar
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
    else:
        cbar.set_ticks([i for i in range(len(color_map))])  # Place ticks in the center of each color block
        cbar.set_ticklabels(list(color_map.keys()))  # Set the labels according to your thresholds
    
    # Set a label for the colorbar
    if threshold_type == 'absolute':
        threshold_label = days_name + ' Day Total Rainfall (mm), regions are labelled if they include lowest or highest total rainfall category'
    elif threshold_type == 'anom':
        threshold_label = days_name + ' Day Total Rainfall Anomaly (mm), regions are labelled if they include lowest or highest anomaly category'
    cbar.set_label(threshold_label)
    plt.savefig(os.path.join(figure_dir, '_'.join([days_name, threshold_type, region_name + '.png'])))
    if plot_show == 'on':
        plt.show()
    plt.close()


def create_summary_plots(df_total, df_anom, states, extra_shapes, state_lists,
                         number_in_threshold, point_size,
                         colormap_total, colormap_anom, days_name,
                         region_name, figure_dir, plot_show):

    threshold_states = sorted(state_lists[0], key=lambda x:(x[0], x[1])) + sorted(state_lists[1], key=lambda x:(x[0], -x[1]))
    anom_states = sorted(state_lists[2], key=lambda x:(x[0], x[1])) + sorted(state_lists[3], key=lambda x:(x[0], -x[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        unique_high =  sorted(set(number for state, number in state_lists[0]))
        unique_low = sorted(set(number for state, number in state_lists[1]))

        plot_threshold_states(df_total, 
                              states, 
                              extra_shapes,
                              threshold_states,
                              number_in_threshold,
                              unique_low,
                              unique_high,
                              point_size,
                              'absolute',
                              colormap_total,
                              days_name,
                              region_name,
                              figure_dir,
                             plot_show)

        unique_high =  sorted(set(number for state, number in state_lists[2]))
        unique_low = sorted(set(number for state, number in state_lists[3]))
        plot_threshold_states(df_anom, 
                              states, 
                              extra_shapes,
                              anom_states,
                              number_in_threshold,
                              unique_low,
                              unique_high,
                              point_size,
                              'anom',
                              colormap_anom,
                              days_name,
                              region_name,
                              figure_dir,
                             plot_show)

def rgb_to_hex(rgb):
    """Convert an RGB tuple to hex."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])