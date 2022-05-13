"""import geopandas as gpd
import pandas as pd
import requests

from vt2geojson.tools import vt_bytes_to_geojson
#(15, 17275, 10768)

x = 17275
y = 10768
z = 15

url = f"https://basisvisualisierung.niedersachsen.de/services/basiskarte/v1/tiles/{z}/{x}/{y}.pbf"
r = requests.get(url)
assert r.status_code == 200, r.content
vt_content = r.content

features = vt_bytes_to_geojson(vt_content, x, y, z)
gdf = gpd.GeoDataFrame.from_features(features)
gdf = gdf.dropna(subset = ['klasse'])

# Load style catalog
url_style = "https://basisvisualisierung.niedersachsen.de/services/basiskarte/styles/vt-style-grayscale.json"
catalog_style = requests.get(url_style).json()

# Load order catalog
catalog_order = requests.get(catalog_style['sources']['basiskarte']['url']).json()
order_frame = pd.DataFrame.from_dict(catalog_order['vector_layers'])

for layer in catalog_style['layers']:
    print(layer['id'])
    if len(gdf[gdf['klasse'] == layer['id']]) != 0:
        print(gdf[gdf['klasse'] == layer['id']])
    print('-------------------------------------------------------------------------------------------------------')
print(1)"""

import os
from qgis.core import *
from PyQt5 import *
from PyQt5.QtSvg import *
from PyQt5.Qt import *

app = QgsApplication([], True)
app.setPrefixPath(r"/usr/bin/qgis", True)
app.initQgis()

project = QgsProject.instance()
project.read("map_base.qgs.qgz")
layer = project.mapLayersByName("myLayer")[0]

options = QgsMapSettings()
options.setLayers([layer])
options.setBackgroundColor(QColor(255, 255, 255))
options.setOutputSize(QSize(800, 600))
options.setExtent(layer.extent())
render = QgsMapRendererParallelJob(options)
image_location = os.path.join(os.getcwd(), "render.png")

def finished():
    img = render.renderedImage()
    img.save(image_location, "png")
    print("saved")

render.finished.connect(finished)
render.start()