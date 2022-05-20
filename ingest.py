import numpy as np
import os
from datetime import datetime, timedelta
import warnings
import pandas as pd
import geopandas as gpd
import rosbag
import cv2
import json
from io import StringIO
from hesai_pandar_64_packets import *
import yaml
import urllib.request

from helper import *

class rosbag_reader:
    def __init__(self, bag_file_name):
        """This function is used for initialization"""
        self.source_bag = rosbag.Bag(bag_file_name, 'r')
        self.topics = self.source_bag.get_type_and_topic_info()[1].keys()

        # Prepare export folder if not existing
        self.bag_unpack_dir = os.path.splitext(os.path.basename(bag_file_name))[0]
        if not os.path.exists(self.bag_unpack_dir):
            os.makedirs(self.bag_unpack_dir)

        # Assemble name for overview json file
        self.overview_file_name = os.path.join(self.bag_unpack_dir, 'overview.json')

        # See if overview json is already there
        if os.path.exists(self.overview_file_name):
            # Open existing overview file
            with open(self.overview_file_name, 'r') as f:
                self.overview = json.loads(f.read())

        else:
            # Create new dict for overview
            self.overview = dict()
            self.overview['working_directory'] = self.bag_unpack_dir

            # Create dict for sensor meta data as entry in overview dict
            self.overview['sensor_streams'] = dict()

            # Add general meta from ros bag
            self.get_ros_bag_meta()

    def __enter__(self):
        """This function is used for the (with .. as ..) call"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This function is used to close all connections to files when this class is not needed anymore"""
        self.source_bag.close()

        # Write overview json for bagfile
        with open(self.overview_file_name, 'w') as f:
            json.dump(self.overview, f, indent=4)

    def get_ros_bag_meta(self):
        # Extract info from bag file
        info_dict = yaml.safe_load(self.source_bag._get_yaml_info())

        # Save data into overview
        self.overview['general_meta'] = dict()
        self.overview['general_meta']['total_message_count'] = info_dict['messages']
        self.overview['general_meta']['original_path'] = info_dict['path']
        self.overview['general_meta']['start_time_unix'] = info_dict['start']
        self.overview['general_meta']['end_time_unix'] = info_dict['end']
        self.overview['general_meta']['version'] = info_dict['version']
        self.overview['general_meta']['is_indexed'] = info_dict['indexed']

    def get_weather_for_trajectory(self, geo_dataframe):
        # Calculate center of dataframe
        center_lon_string = str(np.mean((geo_dataframe.total_bounds[0], geo_dataframe.total_bounds[2])))
        center_lat_string = str(np.mean((geo_dataframe.total_bounds[1], geo_dataframe.total_bounds[3])))

        # Get date
        trajectory_date_string = datetime.utcfromtimestamp(self.overview['general_meta']['start_time_unix']).strftime('%Y-%m-%d')

        # Assemble request url
        brightsky_url = f'https://api.brightsky.dev/weather?lat={center_lat_string}&lon={center_lon_string}&date={trajectory_date_string}'

        # Get data through brightsky api
        with urllib.request.urlopen(brightsky_url) as api_handler:
            brightsky_data = json.loads(api_handler.read().decode())

        # Keep weather info for time of interest (WORKS ONLY WITH SUMMERTIME LIKE THIS!)
        start_datetime = datetime.utcfromtimestamp(self.overview['general_meta']['start_time_unix']) + timedelta(hours=2)
        end_datetime = datetime.utcfromtimestamp(self.overview['general_meta']['end_time_unix']) + timedelta(hours=2)
        self.overview['weather'] = dict()
        self.overview['weather']['hourly'] = brightsky_data['weather'][start_datetime.hour:end_datetime.hour + 2]

        # Save station info
        self.overview['weather']['stations'] = brightsky_data['sources']

    def export_images(self, topic, sensor_name='camera_0', sampling_step=1, pretty_print=None):
        topic_meta = self.source_bag.get_type_and_topic_info(topic_filters=topic)

        # Break if already exported and in overview
        if sensor_name in self.overview.keys():
            print(sensor_name + ' was already exported')
            return

        # Prepare export folder if not existing
        camera_unpack_subdir = sensor_name
        export_directory = os.path.join(self.bag_unpack_dir, camera_unpack_subdir)
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        for frame_number, (topic, msg, t) in enumerate(self.source_bag.read_messages(topics=[topic])):
            # Skip iteration for sub sampling purposes
            if frame_number % sampling_step != 0:
                continue

            # Convert msg to numpy then to opencv image
            temp_image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)

            # Write image into predefined folder
            image_file_name = ("%s.%s.png" % (msg.header.stamp.secs, msg.header.stamp.nsecs))
            cv2.imwrite(os.path.join(export_directory, image_file_name), temp_image)

        # Add to list of exported data
        self.add_to_meta_data(camera_unpack_subdir, topic_meta.topics[topic].msg_type, topic, topic_meta.topics[topic].frequency, topic_meta.topics[topic].message_count, pretty_print, is_in_folder=True)

    def export_pointclouds(self, topic, sensor_name='lidar_0', pretty_print='Hesai Pandar64'):
        topic_meta = self.source_bag.get_type_and_topic_info(topic_filters=topic)

        # Prepare export folder if not existing
        lidar_unpack_subdir = sensor_name
        export_directory = os.path.join(self.bag_unpack_dir, lidar_unpack_subdir)
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        for msg_number, (topic, msg, t) in enumerate(self.source_bag.read_messages(topics=[topic])):
            # Debug timing
            start_time = datetime.now()

            # Create empty array for point coordinates and reflectances
            point_cloud_array = np.empty((0, 5))

            # Retrieve sensor calibration from last udp package
            calibration_array_raw = msg.packets[-1].data[8:].decode('utf-8')
            self.calibration_array = np.genfromtxt(StringIO(calibration_array_raw), delimiter=",", skip_header=1)

            # Retrieve starting second of frame
            sec_first_stamp = msg.packets[0].stamp.secs
            nsec_first_stamp = msg.packets[0].stamp.nsecs

            for i, packet in enumerate(msg.packets[:-1]):
                # Get cartesian points from UDP packet and append to numpy array
                tmp_point_array, tmp_reflectance_array = self.raw_hesai_to_cartesian(packet.data)

                # Append new data to existing array
                point_cloud_array = np.append(point_cloud_array, np.hstack((tmp_point_array, tmp_reflectance_array, np.ones_like(tmp_reflectance_array) * packet.stamp.nsecs)), axis=0)

            # Assemble pandas dataframe
            frame_df = pd.DataFrame(data=point_cloud_array, columns=['x', 'y', 'z', 'reflectance', 'timestamp'])
            frame_df = frame_df.astype(dtype={'x':'f8', 'y':'f8', 'z':'f8', 'reflectance':'u1', 'timestamp':'u4'})

            # Export data
            comment = 'timestamp_first_package_of_frame: %s' % sec_first_stamp
            point_cloud_file_name = ("%s.%s.ply" % (sec_first_stamp, nsec_first_stamp))
            ply_data_types = [(name, dtype.type) for name, dtype in frame_df.dtypes.items()]
            with open(os.path.join(export_directory, point_cloud_file_name), "wb") as f:
                wr = PLYWriter(f, ply_data_types, comments='Placeholder')
                wr.writeDataFrame(frame_df)
                wr.finish()

            # Debug print timing
            end_time = datetime.now()
            print('Duration: {}'.format(end_time - start_time))

        # Add to list of exported data
        self.add_to_meta_data(lidar_unpack_subdir, topic_meta.topics[topic].msg_type, topic, topic_meta.topics[topic].frequency, topic_meta.topics[topic].message_count, pretty_print, is_in_folder=True)

    def raw_hesai_to_cartesian(self, packet_data):
        # Create data transfer object from binary UDP package
        hesai_udp_dto = HesaiPandar64Packets.from_bytes(packet_data)

        # Pull sensor info from self.calibration array
        elevation_array = (np.pi / 2) - np.deg2rad(self.calibration_array[:, 1])
        azimuth_offset_array = np.deg2rad(self.calibration_array[:, 2])

        # Check for dual return mode
        if hesai_udp_dto.tail.return_mode == 57:
            block_iter_start = 1
            block_iter_step = 2
        else:
            block_iter_start = 0
            block_iter_step = 1

        # Iterate over all blocks
        for block in hesai_udp_dto.blocks[block_iter_start::block_iter_step]:
            # Retrieve azimuth value for current block
            azimuth_rad = np.deg2rad(block.azimuth_deg)
            azimuth_array = (azimuth_offset_array + azimuth_rad) * -1

            # Get list of all distances and reflectances
            distance_array = np.asarray([channel.distance_value for channel in block.channel])/1000
            reflectance_array = np.asarray([channel.reflectance_value for channel in block.channel])

            # Create mask for zero distance entries
            valid_value_mask = distance_array != 0

            # Conversion from polar to cartesian
            x_array = distance_array[valid_value_mask] * np.sin(elevation_array[valid_value_mask]) * np.cos(azimuth_array[valid_value_mask])
            y_array = distance_array[valid_value_mask] * np.sin(elevation_array[valid_value_mask]) * np.sin(azimuth_array[valid_value_mask])
            z_array = distance_array[valid_value_mask] * np.cos(elevation_array[valid_value_mask])

            return np.vstack((x_array,y_array,z_array)).T, np.expand_dims(reflectance_array[valid_value_mask], axis=1)

    def export_1d_data(self, topic_filter, sensor_name=None, pretty_print=None):
        """Function to export data from topics that deliver 1 dimensional data"""

        # Throw warning if topic does not exist ant skip
        if topic_filter not in self.topics:
            warnings.warn('The topic ' + topic_filter + ' is not available in this bag file!')
            return

        # Break if already exported and in overview
        if sensor_name in self.overview.keys():
            print(sensor_name + ' was already exported')
            return

        # Load message type from msg for correct csv translation
        topic_meta = self.source_bag.get_type_and_topic_info(topic_filters=topic_filter)
        message_type = topic_meta.topics[topic_filter].msg_type

        # Flag for overview json
        is_geo = False

        # Debug output
        print('DEBUG: message type of topic: ' + topic_filter + ' is: ' + message_type)

        # Handle file export for barometric pressure data
        if message_type == 'sensor_msgs/FluidPressure':
            # Assemble export filename
            if sensor_name == None:
                export_filename = '2022-04-28-track3/pressure_sensor_0.feather'
            else:
                export_filename = sensor_name + '.feather'
            export_filename = os.path.join(self.bag_unpack_dir, export_filename)

            # Create empty list for appension
            dataframe_as_list = []

            for topic, msg, t in self.source_bag.read_messages(topics=[topic_filter]):
                dataframe_as_list.append([msg.header.stamp.to_sec(), t.to_sec(), msg.fluid_pressure])

            # Save as pandas dataframe in feather file
            dataframe = pd.DataFrame(dataframe_as_list, columns=['timestamp_sensor', 'timestamp_bagfile', 'fluid_pressure'])
            dataframe.to_feather(export_filename)

        # Handle file export for illuminance data
        elif message_type == 'sensor_msgs/Illuminance':
            # Assemble export filename
            if sensor_name == None:
                export_filename = 'illuminance_sensor_0.feather'
            else:
                export_filename = sensor_name + '.feather'
            export_filename = os.path.join(self.bag_unpack_dir, export_filename)

            # Create empty list for appension
            dataframe_as_list = []

            for topic, msg, t in self.source_bag.read_messages(topics=[topic_filter]):
                dataframe_as_list.append([msg.header.stamp.to_sec(), t.to_sec(), msg.illuminance])

            # Save as pandas dataframe in feather file
            dataframe = pd.DataFrame(dataframe_as_list, columns=['timestamp_sensor', 'timestamp_bagfile', 'illuminance'])
            dataframe.to_feather(export_filename)

        # Handle file export for IMU data
        elif message_type == 'sensor_msgs/Imu':
            # Assemble export filename
            if sensor_name == None:
                export_filename = 'inertial_measurement_unit_0.feather'
            else:
                export_filename = sensor_name + '.feather'
            export_filename = os.path.join(self.bag_unpack_dir, export_filename)

            # Create empty list for appension
            dataframe_as_list = []

            for topic, msg, t in self.source_bag.read_messages(topics=[topic_filter]):
                dataframe_as_list.append([msg.header.stamp.to_sec(), t.to_sec()] + quaternion_to_list(msg.orientation) + vec3_to_list(
                        msg.linear_acceleration) + vec3_to_list(msg.angular_velocity))

            # Save as pandas dataframe in feather file
            dataframe = pd.DataFrame(dataframe_as_list, columns=['timestamp_sensor', 'timestamp_bagfile', 'or_x', 'or_y', 'or_z', 'or_w', 'li_ac_x', 'li_ac_y', 'li_ac_z', 'an_ve_x', 'an_ve_y', 'an_ve_z'])
            dataframe.to_feather(export_filename)

        # Handle file export for magnetic field data
        elif message_type == 'sensor_msgs/MagneticField':
            # Assemble export filename
            if sensor_name == None:
                export_filename = 'magnetic_field_sensor_0.feather'
            else:
                export_filename = sensor_name + '.feather'
            export_filename = os.path.join(self.bag_unpack_dir, export_filename)

            # Create empty list for appension
            dataframe_as_list = []

            for topic, msg, t in self.source_bag.read_messages(topics=[topic_filter]):
                dataframe_as_list.append([msg.header.stamp.to_sec(), t.to_sec()] + vec3_to_list(msg.magnetic_field))

            # Save as pandas dataframe in feather file
            dataframe = pd.DataFrame(dataframe_as_list, columns=['timestamp_sensor', 'timestamp_bagfile', 'x', 'y', 'z'])
            dataframe.to_feather(export_filename)

        # Handle file export for range data
        elif message_type == 'sensor_msgs/Range':
            # Assemble export filename
            if sensor_name == None:
                export_filename = 'range_sensor_0.feather'
            else:
                export_filename = sensor_name + '.feather'
            export_filename = os.path.join(self.bag_unpack_dir, export_filename)

            # Create empty list for appension
            dataframe_as_list = []
            last_range_valid = True

            for topic, msg, t in self.source_bag.read_messages(topics=[topic_filter]):
                # Assemble line output by conversion of message into list
                if msg.range <= msg.max_range:
                    dataframe_as_list.append([msg.header.stamp.to_sec(), t.to_sec(), msg.range])
                    last_range_valid = True
                elif last_range_valid == True:
                    dataframe_as_list.append([msg.header.stamp.to_sec(), t.to_sec(), np.nan])
                    last_range_valid = False

            # Save as pandas dataframe in feather file
            dataframe = pd.DataFrame(dataframe_as_list, columns=['timestamp_sensor', 'timestamp_bagfile', 'range_cm'])
            dataframe.to_feather(export_filename)

        # Handle file export for gnss data
        elif message_type == 'sensor_msgs/NavSatFix':
            # Set flag for geodata
            is_geo = True

            # Assemble export filename
            if sensor_name == None:
                export_filename = '2022-04-28-track3/gnss_0.feather'
            else:
                export_filename = sensor_name + '.feather'
            export_filename = os.path.join(self.bag_unpack_dir, export_filename)

            # Create empty list for appension
            dataframe_as_list = []

            for topic, msg, t in self.source_bag.read_messages(topics=[topic_filter]):
                dataframe_as_list.append([msg.header.stamp.to_sec(), t.to_sec(), msg.altitude, msg.longitude, msg.latitude, msg.status.service, msg.status.status])

            # Save as pandas dataframe in feather file
            dataframe = pd.DataFrame(dataframe_as_list, columns=['timestamp_sensor', 'timestamp_bagfile', 'alt', 'lon', 'lat', 'ser', 'fix'])

            # Convert to geopandas dataframe
            dataframe = gpd.GeoDataFrame(dataframe, geometry=gpd.points_from_xy(dataframe.lon, dataframe.lat))
            dataframe.set_crs(epsg=4326, inplace=True)

            dataframe.to_feather(export_filename)

            # Add weather info to overview
            self.get_weather_for_trajectory(dataframe)

        else:
            # TODO: throw exception
            warnings.warn('The topic ' + topic_filter + ' is not available in this bag file!')
            pass

        # Add to list of exported data
        self.add_to_meta_data(export_filename, message_type, topic_filter, topic_meta.topics[topic_filter].frequency, topic_meta.topics[topic_filter].message_count, pretty_print, is_geo=is_geo)

    def add_to_meta_data(self, export_filepath, message_type, topic_name, frequency, message_count, pretty_print, is_in_folder=False, is_geo=False):
        # Split path, filename and suffix
        export_filename = os.path.basename(export_filepath)
        export_filename_pure = export_filename.split('.')[0]

        # Add to list of exported data
        self.overview['sensor_streams'][export_filename_pure] = dict()
        self.overview['sensor_streams'][export_filename_pure]['message_type'] = message_type
        self.overview['sensor_streams'][export_filename_pure]['topic_name'] = topic_name
        self.overview['sensor_streams'][export_filename_pure]['frequency'] = frequency
        self.overview['sensor_streams'][export_filename_pure]['message_count'] = message_count
        self.overview['sensor_streams'][export_filename_pure]['is_in_folder'] = is_in_folder
        self.overview['sensor_streams'][export_filename_pure]['is_geo'] = is_geo
        self.overview['sensor_streams'][export_filename_pure]['pretty_print'] = pretty_print