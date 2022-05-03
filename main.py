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

        # Create empty dict for pandas dataframes
        self.dataframes = dict()

    def load_from_working_directory(self, exclude=None):
        # TODO: implement exclude option

        # Create custom time parser
        unix_time_parser = lambda x: datetime.fromtimestamp(float(x))

        # Iterate over available files
        for key, value in self.overview.items():
            # Check if folder or 1d data
            if value['is_in_folder']:
                # TODO: Handle images and pointclouds
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

bag_pandas = data_as_pandas('2022-04-28-track3')
bag_pandas.load_from_working_directory()
print(1)


# ---- Testing below --------------------------------------------

# bag_pandas.dataframes['sensor_msgs/MagneticField'].dataframe['timestamp']
# bag_pandas.dataframes['sensor_msgs/NavSatFix'].dataframe['longitude']

# Apply the default theme
sns.set_theme()

#ran = bag_pandas.dataframes['left_range_sensor_0'].dataframe
nav = bag_pandas.dataframes['gnss_0'].dataframe
pre = bag_pandas.dataframes['pressure_sensor_0'].dataframe
#pre = bag_pandas.dataframes['left_range_sensor_0'].dataframe



# Interpolate data
mixed_index = pre.index.join(nav.index, how='outer')
nav_pre = pd.DataFrame(nav.iloc[:,1:-1]).reindex(index=mixed_index).interpolate().reindex(pre.index)
nav_pre = gpd.GeoDataFrame(nav_pre, geometry=gpd.points_from_xy(nav_pre.lon, nav_pre.lat))
nav_pre.set_crs(epsg=4326, inplace=True)

# TODO: TEMPORÄR FILTEr ------------------------------------------------------------------------
#pre.loc[(pre.range_cm >= 150), 'range_cm'] = np.nan

# Plot as map
map_plotting.create_map_plot(nav_pre, pre, 'fluid_pressure', destination_width=200)

