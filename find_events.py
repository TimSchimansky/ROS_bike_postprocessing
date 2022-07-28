# Native
from itertools import compress

# Installed
import pandas as pd
from scipy import ndimage
import numpy as np
import matplotlib as plt

# Own
from ingest import *
import detect_cars

class DataframeWithMeta:
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

class VehicleEncounter:
    def __init__(self, bag_directory, time_span, time_multiplicator=3):
        # Set unpacked bag path
        self.bag_directory = bag_directory[:-4]

        # Set Flags
        self.is_confirmed = False

        # Set moving direction (-1:undetermined, 0:overtake, 1:opposing)
        self.movement_direction = -1

        # Write time span to class and enlarge by factor
        self.encounter_begin = (time_span[0].astype(int) / 10**9)[0]
        self.encounter_end = (time_span[1].astype(int) / 10**9)[0]
        encounter_duration = self.encounter_end - self.encounter_begin
        single_buffer_only = (time_multiplicator - 1) / 2 * encounter_duration
        self.encounter_begin -= single_buffer_only
        self.encounter_end += single_buffer_only

        # Generate side image list
        self.image_list = self.generate_image_list()

        # Elongate the duration of the event by this factor since ultrasonic sensor only senses the center of the frame
        # but vehicles can be seen for longer time in images
        self.time_multiplicator = time_multiplicator

    def generate_image_list(self, camera_sub_directory='camera_0'):
        # Get list of all files in bagfile camera direcory
        image_file_list = os.listdir(os.path.join(self.bag_directory, camera_sub_directory))
        image_file_list.sort()

        # Generate List of floats from filenames
        image_time_list = np.array([float('.'.join(file_name.split('.')[:-1])) for file_name in image_file_list])

        # Get period of interest
        # TODO: Fix error
        boolean_list = list(np.logical_and((image_time_list >= self.encounter_begin), (image_time_list <= self.encounter_end)))

        # Get image names in period of interest
        image_file_list_crop = list(compress(image_file_list, boolean_list))

        # Return full path for each frame
        return [os.path.join(self.bag_directory, camera_sub_directory, file_name) for file_name in image_file_list_crop]

    def verify_encounter(self):
        # Run YOLO5
        #detect_cars.CarDetector()

        # Run Raft

        # Set Flag
        pass

class DataAsPandas:
    def __init__(self, directory):
        self.working_directory = directory

        # Load overview json
        with open(os.path.join(self.working_directory, 'overview.json'), 'r') as f:
            self.overview = json.loads(f.read())

        # remove working directory entry
        self.working_directory_original = self.overview['working_directory']

        # Create sensor list
        self.sensor_list = self.overview['sensor_streams']

        # Create empty dict for pandas dataframes
        self.dataframes = dict()

    def load_from_working_directory(self, exclude=None):
        # TODO: implement exclude option

        # Iterate over available files
        for key, value in self.sensor_list.items():
            # Check if folder or 1d data
            if value['is_in_folder']:
                # Case of image data
                """if value['message_type'] == 'sensor_msgs/CompressedImage':
                    # Assemble feather file name
                    import_file_path = os.path.join(self.working_directory, key, 'camera_0.feather')

                    # Read feather file
                    self.dataframes[key] = DataframeWithMeta(pd.read_feather(import_file_path), value, key)"""

                # TODO: Handle pointclouds
                pass
            else:
                # Data is available as pandas dataframe in feather file
                import_file_path = os.path.join(self.working_directory, key + '.feather')

                # Decide between pandas and geopandas dataframe
                if value['is_geo']:
                    # Import data and save in dictionary as geopandas
                    self.dataframes[key] = DataframeWithMeta(gpd.read_feather(import_file_path), value, key)
                else:
                    # Import data and save in dictionary as pandas
                    self.dataframes[key] = DataframeWithMeta(pd.read_feather(import_file_path), value, key)

                # Change unix timestamp to datetime
                self.dataframes[key].dataframe['timestamp_sensor'] = pd.to_datetime(self.dataframes[key].dataframe['timestamp_sensor'], unit='s')
                self.dataframes[key].dataframe['timestamp_bagfile'] = pd.to_datetime(self.dataframes[key].dataframe['timestamp_bagfile'], unit='s')
                self.dataframes[key].dataframe.set_index('timestamp_bagfile', inplace=True)

def traverse_bag_vehicle_encounters(bagfile_directory, encounter_max_dist, bagfile_pandas):
    """To look for moments in the data where a vehicle encounter is propable, the side distance sensor is checked for being under a certain value"""
    # TODO: Detect blocks of close distance and return
    side_dist = bagfile_pandas.dataframes['left_range_sensor_0'].dataframe

    side_dist['range_cm'].values[side_dist['range_cm'].values > 1000] = np.NAN

    # TODO: Threshold as parameter
    side_dist['below_threshold'] = False
    side_dist['below_threshold'].values[side_dist['range_cm'].values < 250] = True

    # Morphological opening to remove noise
    side_dist['below_threshold_filtered'] = ndimage.binary_erosion(ndimage.binary_dilation(np.array(side_dist['below_threshold']), iterations=10), iterations=10)

    # Detect rising and falling edges
    diff_edges = np.diff(side_dist['below_threshold_filtered'].astype(np.int8))

    # Get indicees of edges and shift rising edge by one to get index of first true entry
    ind_rise_edge = np.where(diff_edges == 1)[0] + 1
    ind_fall_edge = np.where(diff_edges == -1)[0]

    # Check if indicees for rising and falling are equal:
    if len(ind_rise_edge) != len(ind_fall_edge):
        # TODO: Save the situation
        print("Warning: Amount of edges not equal!! Handling not yet implemented")

    # Check if index of first rising edge is smaller than first falling edge
    if ind_rise_edge[0] >= ind_fall_edge[0]:
        # TODO: Save the situation
        print("Warning: falling edge before rising edge!! Handling not yet implemented")

    # Create empty list of encounters
    vehicle_encounter_list = []

    # Iterate over pair of indicees
    for begin_index, end_index in zip(ind_rise_edge, ind_fall_edge):
        # Get timestamps for begin and end index
        begin_timestamp = side_dist.iloc[[begin_index]].index
        end_timestamp = side_dist.iloc[[end_index]].index

        # Append event to list
        vehicle_encounter_list.append(VehicleEncounter(bagfile_directory, (begin_timestamp, end_timestamp)))

    # Return list of vehicle encounters
    return vehicle_encounter_list

if __name__ == "__main__":
    # Set bagfile path
    bagfile_directory = "../"
    bagfiles = ["2022-04-28-track3.bag"]
    bagfile_paths = [os.path.join(bagfile_directory, bagfile_i) for bagfile_i in bagfiles]

    # Set export directory
    unpack_direcory = ".."

    # Read config file for sensors to import
    sensor_export = pd.read_csv('export_sensors_tmp.csv')

    # Set threshold for encounter indication in cm
    encounter_max_dist = 300

    # Ingest bag files via config file
    for bagfile_path in bagfile_paths:
        with rosbag_reader(bagfile_path, unpack_direcory) as reader_object:
            #print(reader_object.topics)

            # Go through list of sensors to export
            for _, sensor_line in sensor_export.iterrows():
                reader_object.export_flex(sensor_line['sens_type'], sensor_line['topic_name'], sensor_line['sensor_name'], sensor_line['pretty_print'], sensor_line['subsampling'])


    # TODO: Traverse Trajectories(s) for TOIs
    for bagfile_path in bagfile_paths:
        # Load into pandas for operations
        bagfile_pandas = DataAsPandas(bagfile_path[:-4])
        bagfile_pandas.load_from_working_directory()

        # Go though trajectory for events with potential vehicle encounters
        encounter_list = traverse_bag_vehicle_encounters(bagfile_path, encounter_max_dist, bagfile_pandas)

        # Verify TOIs as real encounters
        for encounter in encounter_list:
            encounter.verify_encounter()