import math
from itertools import product

import matplotlib.pyplot as plt
from skimage import io as skio
import numpy as np


TILE_SIZE = 256


def point_to_pixels(lon, lat, zoom):
    """convert gps coordinates to web mercator - this function is provided by Open Street maps"""
    r = math.pow(2, zoom) * TILE_SIZE
    lat = math.radians(lat)

    x = int((lon + 180.0) / 360.0 * r)
    y = int((1.0 - math.log(math.tan(lat) + (1.0 / math.cos(lat))) / math.pi) / 2.0 * r)

    return x, y

# Main
# Define map bounds
zoom = 16
left_bound, right_bound = 9.778970033148356, 9.792782600356505
upper_bound, lower_bound = 52.377849536057006, 52.370752181339995

# Convert to web mercator projection
x0, y0 = point_to_pixels(left_bound, upper_bound, zoom)
x1, y1 = point_to_pixels(right_bound, lower_bound, zoom)

# Get outer bounds of needed tiles
x0_tile, y0_tile = int(x0 / TILE_SIZE), int(y0 / TILE_SIZE)
x1_tile, y1_tile = math.ceil(x1 / TILE_SIZE), math.ceil(y1 / TILE_SIZE)

# Make sure not to download to many tiles from the servers
assert (x1_tile - x0_tile) * (y1_tile - y0_tile) < 50, "The zoom level is impractical for this area!"

# Initialize full image
whole_img = np.zeros(((y1_tile - y0_tile) * TILE_SIZE, (x1_tile - x0_tile) * TILE_SIZE, 3), dtype=np.uint8)

# Iterate through raster of tiles
for x_tile, y_tile in product(range(x0_tile, x1_tile), range(y0_tile, y1_tile)):
    # Assemble image url
    imgurl = 'http://a.tile.openstreetmap.fr/hot/%d/%d/%d.png' % (zoom, x_tile, y_tile)
    print(imgurl)

    # Get image via sklearn
    tile_img = skio.imread(imgurl)

    # Calculate tile origin
    tile_origin = ((x_tile - x0_tile) * TILE_SIZE, (y_tile - y0_tile) * TILE_SIZE)

    # Add to complete image
    whole_img[tile_origin[1]:tile_origin[1] + TILE_SIZE, tile_origin[0]:tile_origin[0] + TILE_SIZE, :] = np.array(tile_img[:, :, :-1])

# Cut out the area of interest from the complete image
x, y = x0_tile * TILE_SIZE, y0_tile * TILE_SIZE
whole_img = whole_img[y0-y:y1-y, x0-x:x1-x, :]

plt.imshow(whole_img)
plt.show()