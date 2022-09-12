import math
from itertools import product

from skimage import io as skio
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
import seaborn as sns
from palettable.cartocolors.sequential import OrYel_4_r

from helper import *


def point_to_pixels(lon, lat, zoom, tile_size):
    """convert gps coordinates to web mercator - this function is provided by the open Street maps wiki"""
    r = math.pow(2, zoom) * tile_size
    lat = math.radians(lat)
    x = int((lon + 180.0) / 360.0 * r)
    y = int((1.0 - math.log(math.tan(lat) + (1.0 / math.cos(lat))) / math.pi) / 2.0 * r)
    return x, y


def pixels_to_points(x, y, zoom, tile_size):
    """convert web mercator to gps coordinates - this function is provided by the open Street maps wiki"""
    r = math.pow(2, zoom) * tile_size
    lon_deg = x / r * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / r)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def generate_OSM_image(left_bound, right_bound, upper_bound, lower_bound, zoom, tile_size=256):
    # Convert to web mercator projection
    x0, y0 = point_to_pixels(left_bound, upper_bound, zoom, tile_size)
    x1, y1 = point_to_pixels(right_bound, lower_bound, zoom, tile_size)

    # Add buffer around ROI
    buffer_x = int(0.075 * abs(x1 - x0))
    buffer_y = int(0.075 * abs(y1 - y0))
    x0, x1 = x0 - buffer_x, x1 + buffer_x
    y0, y1 = y0 - buffer_y, y1 + buffer_y

    # Get outer bounds of needed tiles
    x0_tile, y0_tile = int(x0 / tile_size), int(y0 / tile_size)
    x1_tile, y1_tile = math.ceil(x1 / tile_size), math.ceil(y1 / tile_size)

    # Make sure not to download to many tiles from the servers
    assert (x1_tile - x0_tile) * (y1_tile - y0_tile) < 50, "The zoom level is impractical for this area!"

    # Initialize full image
    whole_img = np.zeros(((y1_tile - y0_tile) * tile_size, (x1_tile - x0_tile) * tile_size, 3), dtype=np.uint8)

    # Iterate through raster of tiles
    for x_tile, y_tile in product(range(x0_tile, x1_tile), range(y0_tile, y1_tile)):
        # Assemble image url
        #imgurl = 'http://a.tile.openstreetmap.fr/hot/%d/%d/%d.png' % (zoom, x_tile, y_tile)
        imgurl = 'https://stamen-tiles.a.ssl.fastly.net/toner-lite/%d/%d/%d.png' % (zoom, x_tile, y_tile)
        print(imgurl)

        # Get image via sklearn
        tile_img = skio.imread(imgurl)

        # Calculate tile origin
        tile_origin = ((x_tile - x0_tile) * tile_size, (y_tile - y0_tile) * tile_size)

        # Add to complete image
        whole_img[tile_origin[1]:tile_origin[1] + tile_size, tile_origin[0]:tile_origin[0] + tile_size, :] = np.array(
            tile_img[:, :, :-1])

    # Cut out the area of interest from the complete image
    x, y = x0_tile * tile_size, y0_tile * tile_size

    lat_max, lon_min = pixels_to_points(x0, y0, zoom, tile_size)
    lat_min, lon_max = pixels_to_points(x1, y1, zoom, tile_size)

    # Create geopandas dataframe for bounding box and set coordinate system
    bounding_box = gpd.GeoDataFrame(geometry=gpd.points_from_xy((lon_min, lon_max), (lat_min, lat_max)))
    bounding_box.set_crs(epsg=4326, inplace=True)
    bounding_box.to_crs(epsg=3857, inplace=True)

    # Return cutout and boundary tuple
    return whole_img[y0 - y:y1 - y, x0 - x:x1 - x, :], bounding_box


def determine_zoom_level(left_bound, right_bound, destination_width_px, tile_size=256):
    # Generate list of zoom levels (width in degrees per tile) as lookup table
    zoom_lvls = np.asarray([360 / (2 ** i) for i in range(20)])

    # Calculate destination number of tiles in width (from pixel width
    destination_width_tiles = destination_width_px / tile_size

    # Calculate width of tile in degrees
    tile_width_deg = (right_bound - left_bound) / destination_width_tiles

    # Get zoom level from lookup table
    return (np.abs(zoom_lvls - tile_width_deg)).argmin()


def calc_map_size(trajectory, destination_width):
    # Calculate data boundaries
    left_bound, lower_bound, right_bound, upper_bound = trajectory.to_crs(epsg=4326).total_bounds

    # Calculate zoom level from predefined destination width
    zoom = determine_zoom_level(left_bound, right_bound, destination_width)

    return left_bound, lower_bound, right_bound, upper_bound, zoom


def format_ticks_deg(ax, bounding_box):
    # Set axes to be equal
    ax.set_ylim(bounding_box.geometry.y[0], bounding_box.geometry.y[1])
    ax.set_xlim(bounding_box.geometry.x[0], bounding_box.geometry.x[1])

    # Reformat ticks to epsg:4326
    ax.set_xticks(np.linspace(bounding_box.geometry.x[0], bounding_box.geometry.x[1], 3))
    xlabel_array = np.linspace(bounding_box.to_crs(epsg=4326).geometry.x[0],
                               bounding_box.to_crs(epsg=4326).geometry.x[1], 3)
    xlabel_list = []
    for i, xlabel in enumerate(xlabel_array):
        xlabel_list.append(dec_2_dms(xlabel))
    ax.set_xticklabels(xlabel_list)

    ax.set_yticks(np.linspace(bounding_box.geometry.y[0], bounding_box.geometry.y[1], 4))
    ylabel_array = np.linspace(bounding_box.to_crs(epsg=4326).geometry.y[0],
                               bounding_box.to_crs(epsg=4326).geometry.y[1], 4)
    ylabel_list = []
    for i, ylabel in enumerate(ylabel_array):
        ylabel_list.append(dec_2_dms(ylabel))
    ax.set_yticklabels(ylabel_list)

    return ax


def create_map_plot(trajectory_resampled_df, secondary_data_df, secondary_data_key, trajectory_df, destination_width=500):
    # If no tertiary data given, replace with constant
    plot_sizes = 50

    # Apply the default theme
    sns.set_theme()

    # Calculate bounaries of map as well as zoom size
    left_bound, lower_bound, right_bound, upper_bound, zoom = calc_map_size(trajectory_df, destination_width)

    # Download tiles, fuse and crop to single image
    map_img, bounding_box = generate_OSM_image(left_bound, right_bound, upper_bound, lower_bound, zoom)

    # Start plot
    fig, ax = plt.subplots()

    # Scatter GNSS data on top
    # Trajectory
    axp = ax.plot(trajectory_df.iloc[1:].to_crs(epsg=3857).geometry.x, trajectory_df.iloc[1:].to_crs(epsg=3857).geometry.y, zorder=0, linewidth=3)
    # Positions of critical encounters
    axs = ax.scatter(x=trajectory_resampled_df.iloc[1:].to_crs(epsg=3857).geometry.x, y=trajectory_resampled_df.iloc[1:].to_crs(epsg=3857).geometry.y, c=secondary_data_df[secondary_data_key].iloc[1:]-30, vmin=0, vmax=150, cmap=OrYel_4_r.mpl_colormap, s=plot_sizes, zorder=1)
    # Add colorbar for scatter
    cbar = plt.colorbar(axs)
    cbar.set_label('Abstand aus Ultraschallsensor [cm]')

    # Insert image into bounds
    ax.imshow(map_img, extent=(bounding_box.geometry.x[0], bounding_box.geometry.x[1], bounding_box.geometry.y[0], bounding_box.geometry.y[1]), zorder=-1)

    # Set plotting and tick format
    ax = format_ticks_deg(ax, bounding_box)



    # Show the plot
    plt.show()