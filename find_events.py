# Native
from itertools import compress
import copy

# Installed
import pandas as pd
from scipy import ndimage
import numpy as np
import matplotlib as plt

# Own
from ingest import *
from helper import *
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
    def __init__(self, bag_directory, time_span, cropped_bagfile_pandas, time_multiplicator=3):
        # Set unpacked bag path
        self.bag_directory = bag_directory

        # Load cropped bagfile as pandas into self
        self.cropped_bagfile_pandas = cropped_bagfile_pandas

        # Set Flags
        self.is_confirmed = False

        # Set moving direction (-1:undetermined, 0:overtake, 1:opposing)
        self.movement_direction = -1

        # Write time span to class and enlarge by factor (external helper function)
        self.encounter_begin, self.encounter_end = extend_timespan_with_factor(time_span, time_multiplicator)

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

    def calc_meta_info(self):
        # Filter invalid measurements
        self.cropped_bagfile_pandas.dataframes['left_range_sensor_0'].dataframe['range_cm'].values[
            self.cropped_bagfile_pandas.dataframes['left_range_sensor_0'].dataframe['range_cm'].values > 1000] = np.NAN

        # Calculate average range measurement
        self.mean_distance = self.cropped_bagfile_pandas.dataframes['left_range_sensor_0'].dataframe['range_cm'].mean()

        # Calculate average speed

        pass

    def verify_encounter(self):
        # Run YOLO5
        self.yolo5 = detect_cars.CarDetector(self.image_list)
        self.yolo5_results = self.yolo5.manage_detection()

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

    def get_trimmed_copy(self, time_span, time_multiplicator):
        # Calc time span
        trim_begin, trim_end = extend_timespan_with_factor(time_span, time_multiplicator)

        # Copy self for trimming
        copy_of_self = copy.deepcopy(self)

        # Set boundaries for trim in copy
        copy_of_self.overview['general_meta']['start_time_unix'] = trim_begin
        copy_of_self.overview['general_meta']['end_time_unix'] = trim_end

        # Do the trimming and return
        copy_of_self.trim_to_set_period()
        return copy_of_self

    def trim_to_set_period(self):
        # Iterate over timeseries for trimming
        for sensor in self.dataframes.keys():
            self.dataframes[sensor].dataframe = self.dataframes[sensor].dataframe.truncate(before=pd.to_datetime(self.overview['general_meta']['start_time_unix'],unit='s'), after=pd.to_datetime(self.overview['general_meta']['end_time_unix'],unit='s'))

def traverse_bag_vehicle_encounters(bagfile_directory, encounter_max_dist, bagfile_pandas):
    """To look for moments in the data where a vehicle encounter is propable, the side distance sensor is checked for being under a certain value"""
    # load side distance into temp var
    side_dist = bagfile_pandas.dataframes['left_range_sensor_0'].dataframe.copy()

    # Filter out ranges over 10m
    side_dist['range_cm'].values[side_dist['range_cm'].values > 1000] = np.NAN

    # Use parameter threshold to filter for unusual close encounters
    side_dist['below_threshold'] = False
    side_dist['below_threshold'].values[side_dist['range_cm'].values < encounter_max_dist] = True

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

        # Get trimmed version of data as pandas
        cropped_bagfile_pandas = bagfile_pandas.get_trimmed_copy((begin_timestamp, end_timestamp), 10)

        # Append event to list
        vehicle_encounter_list.append(VehicleEncounter(bagfile_directory, (begin_timestamp, end_timestamp), cropped_bagfile_pandas))

    # Return list of vehicle encounters
    return vehicle_encounter_list

if __name__ == "__main__":
    # Get df of ingested tracks
    bagfile_unpack_direcory = "../bagfiles_unpack"
    bagfile_db_file = "trajectory_db.feather"
    bagfile_db = pd.read_feather(os.path.join(bagfile_unpack_direcory, bagfile_db_file))

    # Filter out files, that are completely processed
    bagfile_db = bagfile_db.loc[bagfile_db['processed'] == False]

    # Set threshold for encounter indication in cm
    encounter_max_dist = 300

    # TODO: Traverse Trajectories(s) for TOIs
    for i, bagfile_db_entry in bagfile_db.iterrows():
        # Get bagfile path
        bagfile_path = os.path.join(bagfile_db_entry['directory'], bagfile_db_entry['name'][:-4])

        # Load into pandas for operations
        bagfile_pandas = DataAsPandas(bagfile_path)
        bagfile_pandas.load_from_working_directory()

        # Go though trajectory for events with potential vehicle encounters
        encounter_list = traverse_bag_vehicle_encounters(bagfile_path, encounter_max_dist, bagfile_pandas)

        # Verify TOIs as real encounters
        for encounter in encounter_list:
            encounter.verify_encounter()