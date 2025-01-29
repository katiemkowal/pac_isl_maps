import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
#convert the data array to RGB values for image export using defined colorschemes
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
        norm = BoundaryNorm(boundaries = interval, ncolors=cmap.N+1)#, extend="both")
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