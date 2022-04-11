import math
from itertools import product

import matplotlib.pyplot as plt
from skimage import io as skio
import numpy as np


def point_to_pixels(lon, lat, zoom, tile_size):
    """convert gps coordinates to web mercator - this function is provided by Open Street maps"""
    r = math.pow(2, zoom) * tile_size
    lat = math.radians(lat)

    x = int((lon + 180.0) / 360.0 * r)
    y = int((1.0 - math.log(math.tan(lat) + (1.0 / math.cos(lat))) / math.pi) / 2.0 * r)

    return x, y

def pixels_to_points(x, y, zoom, tile_size):
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
        imgurl = 'http://a.tile.openstreetmap.fr/hot/%d/%d/%d.png' % (zoom, x_tile, y_tile)
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

    # Return cutout and boundary tuple
    return whole_img[y0 - y:y1 - y, x0 - x:x1 - x, :], (lon_min, lon_max, lat_min, lat_max)

def determine_zoom_level(left_bound, right_bound, destination_width_px, tile_size=256):
    # Generate list of zoom levels (width in degrees per tile) as lookup table
    zoom_lvls = np.asarray([360 / (2 ** i) for i in range(20)])

    # Calculate destination number of tiles in width (from pixel width
    destination_width_tiles = destination_width_px / tile_size

    # Calculate width of tile in degrees
    tile_width_deg = (right_bound - left_bound) / destination_width_tiles

    # Get zoom level from lookup table
    return (np.abs(zoom_lvls - tile_width_deg)).argmin()

"""# Main
# Define map bounds
left_bound, right_bound = 9.778970033148356, 9.792782600356505
upper_bound, lower_bound = 52.377849536057006, 52.370752181339995

# Get zoom level from predefined destination width
zoom = determine_zoom_level(left_bound, right_bound, 1000)

map_img, pixel_bound_tuple = generate_OSM_image(left_bound, right_bound, upper_bound, lower_bound, zoom)

plt.imshow(map_img)
plt.show()


# Other test


upper_bound, lower_bound = 52.377849536057006, 9.778970033148356

x0, y0 = point_to_pixels(upper_bound, lower_bound, 10, 256)


print(pixels_to_points(x0, y0, 10, 256))"""

