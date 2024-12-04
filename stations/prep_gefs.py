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
# from pyproj import CRS
# from affine import Affine

gefs_procdir = '/cpc/africawrf/ebekele/projects/PREPARE_pacific/notebooks/unmasked'
gefs_rawdir = '/cpc/africawrf/ebekele/projects/PREPARE_pacific/subseason_unmasked'
figure_dir = '/cpc/int_desk/pac_isl/stations/images/'

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
    (250/255, 250/255, 250/255),#0-35
    (250/255, 250/255, 250/255),#0-35
    (250/255, 250/255, 250/255),#0-35
    (245/255, 230/255, 193/255),#40-45
    (233/255, 212/255, 159/255),
    (222/255, 192/255, 123/255),
    (206/255, 160/255, 83/255),
    (190/255, 128/255, 44/255),
    (164/255, 104/255, 26/255),
    (139/255, 81/255, 10/255),
    (111/255, 63/255, 6/255),
    (100/255, 55/255, 6/255),
    (82/255, 48/255, 6/255),
    (82/255, 48/255, 6/255)
]

nn_intervals = [0, 35, 40, 45, 50, 55,100]
nn_colors = [
    (250/255, 250/255, 250/255),#0-35
    (250/255, 250/255, 250/255),#0-35
    (250/255, 250/255, 250/255),#0-35
    (238/255, 238/255, 233/255),#35-40
    (194/255, 194/255, 194/255),#40-45
    (176/255, 176/255, 176/255),
    (144/255, 144/255, 144/255),
    (144/255, 144/255, 144/255)
]

an_intervals = [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,100]
an_colors = [
    #(254/255, 254/255, 254/255),#0-35
    (254/255, 254/255, 254/255),#0-35
    (254/255, 254/255, 254/255),#0-35
    (254/255, 254/255, 254/255),#0-35
    (198/255, 233/255, 227/255),#35-40
    (162/255, 218/255, 210/255),#40-45
    (144/255, 211/255, 201/255),#45-50
    (127/255, 203/255, 191/255),
    (89/255, 176/255, 167/255),
    (52/255, 150/255, 142/255),
    (26/255, 125/255, 117/255),
    (0/255, 101/255, 93/255),
    (0/255, 80/255, 71/255),
    (3/255, 56/255, 47/255),
    (3/255, 56/255, 47/255)
]

categories = ["Below-Normal", "Near-Normal", "Above-Normal"]
# Create colormaps
bn_cmap = LinearSegmentedColormap.from_list("browns", bn_colors, N=len(bn_colors))
nn_cmap = LinearSegmentedColormap.from_list("grays", nn_colors, N=len(nn_colors))
an_cmap = LinearSegmentedColormap.from_list("greens", an_colors, N=len(an_colors))
colormaps = {"Below-Normal": bn_cmap, "Near-Normal": nn_cmap, "Above-Normal": an_cmap}
intervals = {"Below-Normal": bn_intervals, "Near-Normal": nn_intervals, "Above-Normal": an_intervals}

#convert lat/lon coords to mercator projection for tiles
def convert_to_mercator(ds, var):
    # crs_mercator = CRS.from_epsg(3857)
    # mercator_bbox = (
    # *crs_mercator.transform(-22, (132+180)%360-180),  # Transform bottom-left corner
    # *crs_mercator.transform(9, (205+180)%360-180),  # Transform top-right corner
# )
    ds_mercator = ds.rio.reproject("EPSG:3857")
                                   # ,  transform=Affine.translation(mercator_bbox[0], mercator_bbox[1]),  # Adjust transform
    # shape=(mercator_bbox[3] - mercator_bbox[1], mercator_bbox[2] - mercator_bbox[0]),
    # resampling="bilinear")#, resolution = 10000)
    # ds_clip = ds_mercator.where(ds_mercator.notnull(), drop=True)
    return ds_mercator
#convert the data array to RGB values for image export using defined colorschemes

# Helper function to map probability to color
# def get_color(prob, intervals, cmap):
#     norm_prob = (prob - intervals[0]) / (intervals[-1] - intervals[0])  # Normalize to 0-1
#     norm_prob = min(max(norm_prob, 0), 1)  # Clamp to [0, 1]
#     color = cmap(norm_prob)
#     print(f"Probability: {prob}, Normalized: {norm_prob}, Color: {color}")  # Debugging
#     return color

# Apply the colormap and norm to the data
def apply_colormap(da, colormap, norm, value_intervals):
    """
    Apply a custom colormap to the data array based on specified boundaries.

    Parameters:
        da: xarray.DataArray
            The data array to which the colormap will be applied.
        colormap: matplotlib.colors.Colormap
            The custom colormap to apply.
        norm: matplotlib.colors.BoundaryNorm
            The normalizer that defines the color intervals.

    Returns:
        xarray.DataArray
            DataArray with RGBA values.
    """
    # Clip the data to the specified range (you could also use np.clip if needed)
    da_clipped = np.clip(da, value_intervals[0], value_intervals[-1])

    # Map the data values to the colormap using BoundaryNorm
    colormap_values = colormap(norm(da_clipped))

    # Scale to 0-255 for RGB and add an alpha channel
    da_rgb = (colormap_values[:, :, :3] * 255).astype(np.uint8)
    da_alpha = (~np.isnan(da_clipped)) * 255  # Transparency: 0 for NaN, 255 otherwise
    da_rgba = np.dstack((da_rgb, da_alpha.astype(np.uint8)))  # Combine RGB + Alpha

    # Convert to xarray for exporting with spatial coordinates
    da_rgba_xarray = xr.DataArray(
        da_rgba,
        dims=("y", "x", "band"),
        coords={"y": da.y, "x": da.x, "band": [1, 2, 3, 4]},
    )
    da_rgba_xarray = da_rgba_xarray.transpose("band", "y", "x").rio.write_crs(da.rio.crs)

    return da_rgba_xarray

def process_gefs_probabilities(gefs_data, categories, colormaps, intervals, crs="EPSG:4326", time_index=0):
    """
    Process GEFS probability data to create a combined RGBA map.

    Parameters:
        gefs_data: xarray.Dataset
            The input dataset containing probabilities.
        categories: list of str
            List of category names (e.g., ["Below-Normal", "Near-Normal", "Above-Normal"]).
        colormaps: dict
            Dictionary mapping category names to their colormaps.
        intervals: dict
            Dictionary mapping category names to their interval boundaries.
        crs: str, optional
            Coordinate Reference System to assign to the dataset. Default is "EPSG:4326".
        time_index: int, optional
            The time index to select for processing. Default is 0.

    Returns:
        xarray.DataArray
            Combined RGBA map as an xarray DataArray.
    """
    # Write CRS to the dataset
    gefs_data_crs = gefs_data.rio.write_crs(crs, inplace=False)

    # Select data for the given time step and scale probabilities
    data = gefs_data_crs['prob'].isel(time=time_index) * 100

    # Identify NaN mask
    nan_mask = data.isnull().all(dim='e')
    filled_data = data.where(~nan_mask, -1)

    # Compute the maximum category index
    max_cat_index = filled_data.argmax(dim="e")

    # Extract max probabilities for each category
    max_probs = {}
    for i, category in enumerate(categories):
        cat_prob = data.isel(e=i)  # Select probabilities for this category
        max_prob = cat_prob.where(max_cat_index == i)  # Retain only max category
        max_prob = max_prob.where(~nan_mask)  # Mask out all-NaN locations
        max_probs[category] = max_prob

    # Apply colormaps and combine RGBA maps
    rgba_maps = []
    for category in categories:
        prob = max_probs[category]
        cmap = colormaps[category]
        interval = intervals[category]
        norm = BoundaryNorm(interval, cmap.N, extend="both")
        
        # Apply the colormap
        rgba_map = apply_colormap(prob, cmap, norm, interval)
        rgba_maps.append(rgba_map)

    # Combine individual RGBA maps into a single map
    combined_rgba = np.zeros_like(rgba_maps[0].values)  # Initialize combined map
    for rgba_map in rgba_maps:
        mask = rgba_map[3, :, :] > 0  # Use alpha channel to identify valid data
        combined_rgba[:, mask] = rgba_map.values[:, mask]

    #Convert combined RGBA to xarray
    combined_rgba_xarray = xr.DataArray(
        combined_rgba,
        dims=("band", "y", "x"),
        coords={
            "band": [1, 2, 3, 4],
            "y": rgba_maps[0].y,
            "x": rgba_maps[0].x,
        },
    ).rio.write_crs(gefs_data_crs.rio.crs)

    return combined_rgba_xarray


gefs_wk1cons = xr.open_dataset(os.path.join(gefs_procdir, 'gefs_week_1_cons.nc'))
gefs_wk1cons = gefs_wk1cons.rename({'lon':'x', 'lat':'y'})
gefswk1_pcons_rgba = process_gefs_probabilities(gefs_wk1cons, categories, colormaps, intervals,
                                               crs="EPSG:4326", time_index=0)
gefswk1pcons_prep = gefswk1_pcons_rgba.to_dataset(name = 'color')
# gefswk1pcons_prep = gefswk1pcons_prep.isel(x=slice(132,205))
# gefswk1pcons_prep = gefswk1pcons_prep.isel(y=slice(None,None,-1))
#gefswk1pcons_prep['x'] = (gefswk1pcons_prep.x + 180)%360 -180
#gefswk1pcons_prep = gefswk1pcons_prep.sortby('x', ascending = False)
gefswk1_pcons_mc = convert_to_mercator(gefswk1pcons_prep, 'color')
gefswk1_pcons_mc['color'].rio.to_raster(os.path.join(figure_dir,'gefswk1pcons.tif'), dtype = 'uint8')


## station time series prep
stations = [{'name': 'Chuuk-Weno_FSM','lat': 7.45, 'lon': 151.85 },
            { 'name': 'Colonia-Yap_FSM', 'lat': 9.52, 'lon': 138.13 },
            { 'name': 'Lukunor_FSM', 'lat': 5.502, 'lon': 153.817 },
            { 'name': 'Kolonia-Pohnpei_FSM', 'lat': 6.984, 'lon': 158.207 },
            { 'name': 'Kosrae_FSM', 'lat': 5.326, 'lon': 163.011 },
            { 'name': 'Palikir-Pohnpei_FSM', 'lat': 6.923, 'lon': 158.159 },
            { 'name': 'Abaiang_Kiribati', 'lat': 1.799, 'lon': 173.039 },
            { 'name': 'Beru_Kiribati', 'lat': -1.337, 'lon': 175.996 },
            { 'name': 'Butaritari_Kiribati', 'lat': 3.072, 'lon': 172.792 },
            { 'name': 'Tarawa_Kiribati', 'lat': 1.382, 'lon': 173.147 }, 
            { 'name': 'Fanning_Kiribati', 'lat': 3.898, 'lon': (-159.386 + 360) % 360 }, 
            { 'name': 'Kirimati_Kiribati', 'lat': 1.986, 'lon': (-157.35 + 360) % 360 }, 
            { 'name': 'Washington_Kiribati', 'lat': 4.692, 'lon': (-160.408 + 360) % 360 }, 
            { 'name': 'Canton_Kiribati', 'lat': -2.769, 'lon': (-171.718 + 360) % 360 }, 
            { 'name': 'Koror_Palau', 'lat': 7.342, 'lon': 134.476 },
            { 'name': 'Ngerulmud_Palau', 'lat': 7.5, 'lon': 134.624 },
            { 'name': 'Alotau_PNG', 'lat': -10.314, 'lon': 150.458 },
            { 'name': 'Kavieng_PNG', 'lat': -2.581, 'lon': 150.806 },
            { 'name': 'Lae_PNG', 'lat': -6.728, 'lon': 146.996 },
            { 'name': 'Madang_PNG', 'lat': -5.225, 'lon': 145.792 },
            { 'name': 'Mendi_PNG', 'lat': -6.145, 'lon': 143.657 },
            { 'name': 'Mount-Hagen_PNG', 'lat': -5.859, 'lon': 144.235 },
            { 'name': 'Popondetta_PNG', 'lat': -8.764, 'lon': 148.239 },
            { 'name': 'Port Moresby_PNG', 'lat': -9.439, 'lon': 147.211 },
            { 'name': 'Tari_PNG', 'lat': -5.851, 'lon': 142.95 },
            { 'name': 'Wewak_PNG', 'lat': -3.584, 'lon': 143.669 },
            { 'name': 'Goroka_PNG', 'lat': -6.075, 'lon': 145.392 },
            { 'name': 'Vanimo_PNG', 'lat': -2.688, 'lon': 141.299 },
            { 'name': 'Nadzab_PNG', 'lat': -6.566, 'lon': 146.73 },
            { 'name': 'Tokua_PNG', 'lat': -4.343, 'lon': 152.377 },
            { 'name': 'Apia_Samoa', 'lat': -13.846, 'lon': (-171.763 + 360) % 360 },
            { 'name': 'Afega_Samoa', 'lat': -13.8, 'lon': (-171.853 + 360) % 360 },
            { 'name': 'Safotu_Samoa', 'lat': -13.452, 'lon': (-172.408 + 360) % 360 },
            { 'name': 'Leulumoega_Samoa', 'lat': -13.827, 'lon': (-171.961 + 360) % 360 },
            { 'name': 'Asau_Samoa', 'lat': -13.528, 'lon': (-172.628 + 360) % 360 },
            { 'name': 'Lufilufi_Samoa', 'lat': -13.874, 'lon': (-171.6 + 360) % 360 },
            { 'name': 'Vailoa_Samoa', 'lat': -13.755, 'lon': (-172.307 + 360) % 360 },
            { 'name': 'Saleaula_Samoa', 'lat': -13.449, 'lon': (-172.337 + 360) % 360 },
            { 'name': 'Samamea_Samoa', 'lat': -13.934, 'lon': (-171.532 + 360) % 360 },
            { 'name': 'Auki_Solomons', 'lat': -8.768, 'lon': 160.697 },
            { 'name': 'Buala_Solomons', 'lat': -8.146, 'lon': 159.593 },
            { 'name': 'Gizo_Solomons', 'lat': -8.105, 'lon': 156.846 },
            { 'name': 'Honiara_Solomons', 'lat': -9.442, 'lon': 159.983 },
            { 'name': 'Kirakira_Solomons', 'lat': -10.457, 'lon': 161.92 },
            { 'name': 'Lata_Solomons', 'lat': -10.723, 'lon': 165.799 },
            { 'name': 'Tigoa_Solomons', 'lat': -11.551, 'lon': 160.063 },
            { 'name': 'Tulagi_Solomons', 'lat': -9.11, 'lon': 160.152 },
            { 'name': 'Taro_Solomons', 'lat': -6.711, 'lon': 156.396 },
            { 'name': 'Munda_Solomons', 'lat': -8.326, 'lon': 157.267 },
            { 'name': 'Funafuti_Tuvalu', 'lat': -8.524, 'lon': 179.197 },
            { 'name': 'Nui_Tuvalu', 'lat': -7.244, 'lon': 177.147 },
            { 'name': 'Niulakita_Tuvalu', 'lat': -10.79, 'lon': 179.471 },
            { 'name': 'Isangel_Vanuatu', 'lat': -19.541, 'lon': 169.281 },
            { 'name': 'Lakatoro_Vanuatu', 'lat': -16.107, 'lon': 167.419 },
            { 'name': 'Luganville_Vanuatu', 'lat': -15.505, 'lon': 167.22 },
            { 'name': 'Port-Vila_Vanuatu', 'lat': -17.742, 'lon': 168.321 },
            { 'name': 'Sola_Vanuatu', 'lat': -13.873, 'lon': 167.547 },
            { 'name': 'Seratmata_Vanuatu', 'lat': -15.288, 'lon': 167.987 },
            { 'name': 'Aneityum_Vanuatu', 'lat': -20.234, 'lon': 169.781 },
            { 'name': 'Ba_Fiji', 'lat': -17.541, 'lon': 177.671 },
            { 'name': 'Labasa_Fiji', 'lat': -16.432, 'lon': 179.38 },
            { 'name': 'Lami_Fiji', 'lat': -18.112, 'lon': 178.415 },
            { 'name': 'Lautoka_Fiji', 'lat': -17.615, 'lon': 177.454 },
            { 'name': 'Nadi_Fiji', 'lat': -17.753, 'lon': 177.451 },
            { 'name': 'Nakasi_Fiji', 'lat': -18.068, 'lon': 178.524 },
            { 'name': 'Nausori_Fiji', 'lat': -18.03, 'lon': 178.531 },
            { 'name': 'Sigatoka_Fiji', 'lat': -18.142, 'lon': 177.503 },
            { 'name': 'Suva_Fiji', 'lat': -18.135, 'lon': 178.435 },
            { 'name': 'Rotuma_Fiji', 'lat': -12.482, 'lon': 177.071 },
            { 'name': 'Nabouwalu_Fiji', 'lat': -16.998, 'lon': 178.695 },
            { 'name': 'Yasawa_Fiji', 'lat': -16.699, 'lon': 177.575 },
            { 'name': 'Viwa_Fiji', 'lat': -17.149, 'lon': 176.913 },
            { 'name': 'VanuaBalavu_Fiji', 'lat': -17.246, 'lon': (-178.956 + 360) % 360 },
            { 'name': 'Tubou-Lakeba_Fiji', 'lat': -18.236 , 'lon': (-178.812 + 360) % 360 },
            { 'name': 'Vunisea_Fiji', 'lat': -19.047, 'lon': 178.163 }
           ]


cons_stations = []
for station in stations:
    cons_station = gefs_wk1cons.sel(x=station['lon'], y = station['lat'], method = 'nearest')
    cons_station['station'] = station['name']
    cons_stations.append(cons_station)
cons_stations = xr.concat(cons_stations, dim = 'station')
cons_stations['prob_colors'] = cons_stations['prob']
cons_stations['prob'] = cons_stations['prob']*100

for s, station in enumerate(stations):
    bar_data = []
    bar_data_colors = []

    # Prepare the data for each category (bn, nn, an)
    for c, cat in enumerate(categories):
        bar_data.append(cons_stations.isel(time=0,station=s,e=c).prob.values)
        bar_data_colors.append(cons_stations.isel(time=0,station=s,e=c).prob_colors.values)

    colors = [
        bn_cmap(bar_data_colors[0]),
        nn_cmap(bar_data_colors[1]),
        an_cmap(bar_data_colors[2])
        #get_color(bar_data[0], bn_intervals, bn_cmap),
        #get_color(bar_data[1], nn_intervals, nn_cmap),
        #get_color(bar_data[2], an_intervals, an_cmap),
    ]
    
    # Create the bar plot
    plt.bar(categories, bar_data, color=colors)
    # Add labels and title
    plt.ylabel("Probability (%)")
    plt.ylim(0,85)
    plt.title(station['name'] + ' GEFS Week 1 Consolidated Precip Probabilities')
    # Save the box plot as a PNG file with the station name
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'station_data', f"{station['name']}_pconswk1bar.png"))  # Save as PNG file
    plt.close()  # Close the plot to avoid memory issues

