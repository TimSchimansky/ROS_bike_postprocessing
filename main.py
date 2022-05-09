import rosbag
import cv2
import numpy as np
import os
from datetime import datetime
import warnings
import pandas as pd
import geopandas as gpd
from io import StringIO
import json
import csv
from shapely.geometry import Point

import matplotlib.pyplot as plt
import seaborn as sns

import map_plotting
import timeline_plotting
from hesai_pandar_64_packets import *
from helper import *
from ingest import *
import fix_bag

class dataframe_with_meta:
    def __init__(self, dataframe, meta, orig_file_name):
        # Put dataframe into struct
        self.dataframe = dataframe

        # Add meta info
        self.message_type = meta['message_type']
        self.orig_topic_name = meta['topic_name']
        self.orig_file_name = orig_file_name
        self.frequency = meta['frequency']
        self.message_count = meta['message_count']
        self.is_geo = meta['is_geo']
        self.is_in_folder = meta['is_in_folder']
        self.orig_file_type = '.feather'

class data_as_pandas:
    def __init__(self, directory):
        self.working_directory = directory

        # Load overview json
        with open(os.path.join(self.working_directory, 'overview.json'), 'r') as f:
            self.overview = json.loads(f.read())

        # remove working directory entry
        self.overview.pop('working_directory')

        # Create sensor list
        self.sensor_list = list(self.overview.keys())

        # Create empty dict for pandas dataframes
        self.dataframes = dict()

    def load_from_working_directory(self, exclude=None):
        # TODO: implement exclude option

        # Iterate over available files
        for key, value in self.overview.items():
            # Check if folder or 1d data
            if value['is_in_folder']:
                # Case of image data
                if value['message_type'] == 'sensor_msgs/CompressedImage':
                    # Assemble feather file name
                    import_file_path = os.path.join(self.working_directory, key, 'camera_0.feather')

                    # Read feather file
                    self.dataframes[key] = dataframe_with_meta(pd.read_feather(import_file_path), value, key)

                    self.dataframes[key].dataframe = filter_max_width(self.dataframes[key].dataframe)

                # TODO: Handle pointclouds
                pass
            else:
                # Data is available as pandas dataframe in feather file
                import_file_path = os.path.join(self.working_directory, key + '.feather')

                # Decide between pandas and geopandas dataframe
                if value['is_geo']:
                    # Import data and save in dictionary as geopandas
                    self.dataframes[key] = dataframe_with_meta(gpd.read_feather(import_file_path), value, key)
                else:
                    # Import data and save in dictionary as pandas
                    self.dataframes[key] = dataframe_with_meta(pd.read_feather(import_file_path), value, key)

                # Change unix timestamp to datetime
                # TODO: add timestamp for sensor and for bag
                self.dataframes[key].dataframe['timestamp_sensor'] = pd.to_datetime(self.dataframes[key].dataframe['timestamp_sensor'], unit='s')
                self.dataframes[key].dataframe['timestamp_bagfile'] = pd.to_datetime(self.dataframes[key].dataframe['timestamp_bagfile'], unit='s')
                self.dataframes[key].dataframe.set_index('timestamp_bagfile', inplace=True)

def calculate_overlap(xmin, xmax, ymin, ymax):
    # Define center
    bb1 = [320, 960, 180, 540]

    bb2 = [xmin, xmax, ymin, ymax]

    assert bb1[0] < bb1[1]
    assert bb1[2] < bb1[3]
    assert bb2[0] < bb2[1]
    assert bb2[2] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0

    return iou

def filter_max_width(input_dataframe):
    # TODO: redo less weird
    # Calc overlap
    input_dataframe['overlap'] = list(map(calculate_overlap, input_dataframe['xmin'], input_dataframe['xmax'], input_dataframe['ymin'], input_dataframe['ymax']))

    # Get index for max widths per timestamp
    #idx = input_dataframe.groupby(['timestamp'])['ymax'].transform(max) == input_dataframe['ymax']
    idx = input_dataframe.groupby(['timestamp'])['overlap'].transform(max) == input_dataframe['overlap']

    # Only keep max values and width and timestamp column
    dataframe = input_dataframe[idx]
    dataframe = dataframe[['overlap', 'timestamp']]

    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s')
    dataframe.set_index('timestamp', inplace=True)

    # TODO: less weird
    dataframe = dataframe.drop(dataframe[dataframe.overlap < 0.7].index)

    return dataframe

#fix_bag.fix_bagfile_header("../2022-03-24-11-40-06.bag", "../test3.bag")

# 2022-04-28-track3.bag
# lidar_test

# with rosbag_reader("../debug_test_camera_lidar.bag") as reader_object:
with rosbag_reader("../2022-04-28-track3.bag") as reader_object:
    print(reader_object.topics)

    # TODO: Do this based on config file
    #reader_object.export_pointclouds('/hesai/pandar_packets', sensor_name='lidar_0')
    #reader_object.export_images('/phone1/camera/image/compressed', sensor_name='camera_0')

    #reader_object.export_1d_data('/phone1/android/magnetic_field', sensor_name='magnetic_field_sensor_0')
    #reader_object.export_1d_data('/phone1/android/illuminance', sensor_name='illuminance_sensor_0')
    #reader_object.export_1d_data('/phone1/android/imu', sensor_name='inertial_measurement_unit_0')
    reader_object.export_1d_data('/note9/android/barometric_pressure', sensor_name='pressure_sensor_0')
    reader_object.export_1d_data('/side_distance', sensor_name='left_range_sensor_0')
    reader_object.export_1d_data('/note9/android/fix', sensor_name='gnss_0')
    reader_object.export_images('/side_view/image_raw/compressed', sensor_name='camera_0', sampling_step=10)

# TODO: Add pretty print to json

bag_pandas = data_as_pandas('2022-04-28-track3')
bag_pandas.load_from_working_directory()
print(1)

timeline_plotting.create_timeline_plot(bag_pandas, ['pressure_sensor_0', 'left_range_sensor_0', 'camera_0'], ['fluid_pressure', 'range_cm', 'overlap'])
#timeline_plotting.create_timeline_plot(bag_pandas, ['left_range_sensor_0'], ['range_cm'])
# ---- Testing below --------------------------------------------

# det = bag_pandas.dataframes['camera_0'].dataframe
# bag_pandas.dataframes['sensor_msgs/MagneticField'].dataframe['timestamp']
# bag_pandas.dataframes['sensor_msgs/NavSatFix'].dataframe['longitude']


"""
#ran = bag_pandas.dataframes['left_range_sensor_0'].dataframe
nav = bag_pandas.dataframes['gnss_0'].dataframe
#pre = bag_pandas.dataframes['pressure_sensor_0'].dataframe
dist = bag_pandas.dataframes['left_range_sensor_0'].dataframe

width = bag_pandas.dataframes['camera_0'].dataframe


# Interpolate data
mixed_index = dist.index.join(nav.index, how='outer')
nav_pre = pd.DataFrame(nav.iloc[:,1:-1]).reindex(index=mixed_index).interpolate().reindex(dist.index)
nav_pre = gpd.GeoDataFrame(nav_pre, geometry=gpd.points_from_xy(nav_pre.lon, nav_pre.lat))
nav_pre.set_crs(epsg=4326, inplace=True)

mixed_index = dist.index.join(width.index, how='outer')
width_pre = width.reindex(index=mixed_index).interpolate().reindex(dist.index)

print(0)

# TODO: TEMPORÄR FILTEr ------------------------------------------------------------------------
#pre.loc[(pre.range_cm >= 150), 'range_cm'] = np.nan

# Plot as map
map_plotting.create_map_plot(nav_pre, dist, 'range_cm', tertiary_data_df=width_pre, tertiary_data_key='ymax', destination_width=200)

"""