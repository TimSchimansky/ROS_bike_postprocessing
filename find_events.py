# Native
from itertools import compress
import copy
import os

# Installed
import pandas as pd
from scipy import ndimage
import numpy as np

# Own
from ingest import *
from helper import *
import detect_cars
import detect_flow

# Debug
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Set base rules for overtaking car
MIN_BOUNDING_BOX_HEIGHT = 150
MIN_SELF_SPEED_M_S = 2
MIN_ENCOUNTER_DURATION = 0.5
MAX_ENCOUNTER_DURATION = 6
MIN_FRAMES_WITH_DETECTION = 0.25
MAX_SINGLE_ENCOUNTER_VARIATION_CM = 100


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
        self.is_tested = False
        self.is_confirmed = False

        # Set moving direction (-1:undetermined, 0:opposing, 1:overtake)
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
        self.mean_side_distance_cm = self.cropped_bagfile_pandas.dataframes['left_range_sensor_0'].dataframe['range_cm'].mean()

        # Calculate speed for every pair of consecutive values
        utm_pos = self.cropped_bagfile_pandas.dataframes['gnss_0'].dataframe.geometry.to_crs(epsg=32632)
        speed_array = np.empty((0))
        vector_array = np.empty((0, 2))
        for (pos_0, pos_1, time_0, time_1) in zip(utm_pos.values, utm_pos.values[1:], utm_pos.index, utm_pos.index[1:]):
            pos_delta = pos_0.distance(pos_1)
            time_delta = time_1 - time_0

            # Append speed to list of speeds
            speed_array = np.append(speed_array, [pos_delta / time_delta.total_seconds()])

            # Get vector of two points and normalize
            vector_array = np.append(vector_array, [[pos_1.x - pos_0.x, pos_1.y - pos_0.y]], axis=0)

        # Calculate mean speed
        self.mean_speed_m_s = np.mean(speed_array)

        # Calculate mean heading
        self.mean_heading_rad = np.arctan2(np.mean(vector_array, axis=0)[0], np.mean(vector_array, axis=0)[1])

    def verify_encounter(self):
        # Calculate mete data
        self.calc_meta_info()

        # Run tests
        self.run_tests()

        # Set flags
        self.is_tested = True
        self.is_confirmed = self.results[0]
        if self.is_confirmed == True:
            self.movement_direction = int(self.results[1])

        return self.is_confirmed, self.movement_direction, self.mean_side_distance_cm, self.mean_speed_m_s, self.mean_heading_rad, self.encounter_begin, self.encounter_end, self.reason

    def run_tests(self):
        # Break immediately if duration does not meet criteria
        duration_s = self.encounter_end - self.encounter_begin
        if duration_s < MIN_ENCOUNTER_DURATION or duration_s > MAX_ENCOUNTER_DURATION:
            # print((False, False), self.encounter_begin, self.encounter_end)
            self.results = (False, False)
            self.reason = "Unrealistically long/short"
            return

        # Ensure minimal speed criteria is met
        if self.mean_speed_m_s < MIN_SELF_SPEED_M_S:
            # print((False, False), self.encounter_begin, self.encounter_end)
            self.results = (False, False)
            self.reason = "Self not moving (enough)"
            return

        # Run YOLO5
        yolo5 = detect_cars.CarDetector(self.image_list)
        self.yolo5_results = yolo5.manage_detection()

        # Check that result is not None
        if self.yolo5_results is None:
            self.results = (False, False)
            self.reason = "No visual detections"
            return

        # Filter out bounding boxes, that do not meet minimal size criterium
        self.yolo5_results_filtered = self.yolo5_results[
            self.yolo5_results['ymax'] - self.yolo5_results['ymin'] >= MIN_BOUNDING_BOX_HEIGHT]

        # Check that at least in n percent of the frames, there is a valid detection
        if len(self.yolo5_results_filtered.image_name.unique()) / len(self.image_list) <= MIN_FRAMES_WITH_DETECTION:
            # print((False, False), self.encounter_begin, self.encounter_end)
            self.results = (False, False)
            self.reason = "Not enough visual detections"
            return

        # Run Raft
        raft = detect_flow.FlowDetector(self.image_list, self.mean_side_distance_cm, self.mean_speed_m_s,
                                             standalone_mode=False, yolo_bounding_boxes=self.yolo5_results_filtered)
        self.results = raft.get_flow_results()

        if self.results[0]:
            # Get vehicle type most often vehicle type in top ten highest y values (lowest to bottom of frame)
            self.reason = self.yolo5_results_filtered.sort_values('ymax')['name'].iloc[-10:].value_counts().index[0]
        else:
            self.reason = "Ran through tests"

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
        return copy_of_self, (trim_begin, trim_end)

    def trim_to_set_period(self):
        # Iterate over timeseries for trimming
        before_value = pd.to_datetime(self.overview['general_meta']['start_time_unix'],unit='s')
        after_value = pd.to_datetime(self.overview['general_meta']['end_time_unix'],unit='s')

        for sensor in self.dataframes.keys():
            if sensor == "gnss_0":
                before_value_gnss = pd.to_datetime(self.overview['general_meta']['start_time_unix'] - 2, unit='s')
                after_value_gnss = pd.to_datetime(self.overview['general_meta']['end_time_unix'] + 2, unit='s')
                self.dataframes[sensor].dataframe = self.dataframes[sensor].dataframe.truncate(before=before_value_gnss, after=after_value_gnss)
            else:
                self.dataframes[sensor].dataframe = self.dataframes[sensor].dataframe.truncate(before=before_value, after=after_value)

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

    # Skip if no potential Regions
    if len(ind_rise_edge) == 0 and len(ind_fall_edge) == 0:
        return []

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

        # Check if sudden jump in range (Background, (but triggered) --> car --> background)
        tmp_range_values = side_dist.iloc[begin_index:end_index].range_cm.values

        # Clean out nan values
        tmp_range_values_nonan = tmp_range_values[~np.isnan(tmp_range_values)]
        potential_jumps = (np.where(np.abs(np.diff(tmp_range_values_nonan)) >= MAX_SINGLE_ENCOUNTER_VARIATION_CM))[0] + 1
        non_nan_map = np.where(~np.isnan(tmp_range_values))[0]

        if len(potential_jumps) != 0:
            # Create list of arrays with value indices each
            non_nan_blocks = np.split(non_nan_map, potential_jumps)

            for non_nan_block in non_nan_blocks:
                # Get bounding indices from block
                sub_begin_index = non_nan_block[0] + begin_index
                sub_end_index = non_nan_block[-1] + begin_index

                # Get new timestamps for begin and end index
                begin_timestamp = side_dist.iloc[[sub_begin_index]].index
                end_timestamp = side_dist.iloc[[sub_end_index]].index

                # Get back index before nan removal
                #current_values = non_nan_map[sub_begin_index:sub_end_index+1]

                # Get trimmed version of data as pandas
                cropped_bagfile_pandas, extended_timestamps = bagfile_pandas.get_trimmed_copy((begin_timestamp, end_timestamp), 2)

                # Append event to list
                vehicle_encounter_list.append(VehicleEncounter(bagfile_directory, extended_timestamps, cropped_bagfile_pandas))

        else:
            # Get trimmed version of data as pandas
            cropped_bagfile_pandas, extended_timestamps = bagfile_pandas.get_trimmed_copy((begin_timestamp, end_timestamp), 2)

            # Append event to list
            vehicle_encounter_list.append(VehicleEncounter(bagfile_directory, extended_timestamps, cropped_bagfile_pandas))

    # Return list of vehicle encounters
    return vehicle_encounter_list

if __name__ == "__main__":
    # Get df of ingested tracks
    bagfile_unpack_direcory = "H:/bagfiles_unpack"
    bagfile_db_file = "trajectory_db.feather"
    bagfile_db = pd.read_feather(os.path.join(bagfile_unpack_direcory, bagfile_db_file))

    # Get df of processed encounters
    encounter_db_file = "encounter_db_v2.feather"
    if not os.path.isfile(os.path.join(bagfile_unpack_direcory, encounter_db_file)):
        encounter_db = pd.DataFrame(columns=['is_encounter', 'direction', 'distance', 'begin', 'end', 'description', 'working_dir', 'bag_file', 'manual_override'])
    else:
        encounter_db = pd.read_feather(os.path.join(bagfile_unpack_direcory, encounter_db_file))

    # Filter out files, that are completely processed
    #bagfile_db = bagfile_db.loc[bagfile_db['processed'] == False]

    # Set threshold for encounter indication in cm
    encounter_max_dist = 350

    # Create List to collect encounters
    encounter_results_collector_list = []

    # TODO: Traverse Trajectories(s) for TOIs
    for count_bagfile, bagfile_db_entry in bagfile_db.iterrows():
        if bagfile_db_entry['processed'] == True and bagfile_db_entry['name'] not in ['2022-08-08-05-51-45_0.bag']:
            continue

        print(f"DEBUG: Checking bagfile {count_bagfile}")
        # Get bagfile path
        bagfile_path = os.path.join(bagfile_db_entry['directory'], bagfile_db_entry['name'][:-4])

        # Load into pandas for operations
        bagfile_pandas = DataAsPandas(os.path.join(bagfile_unpack_direcory, os.path.split(bagfile_path)[1]))
        bagfile_pandas.load_from_working_directory()

        # Go though trajectory for events with potential vehicle encounters
        encounter_list = traverse_bag_vehicle_encounters(os.path.join(bagfile_unpack_direcory, os.path.split(bagfile_path)[1]), encounter_max_dist, bagfile_pandas)

        # Verify TOIs as real encounters
        for count_encounter, encounter in enumerate(encounter_list):
            print(f"DEBUG: Checking encounter {count_encounter} in bafile number {count_bagfile} ({os.path.split(bagfile_path)[-1]})")
            tmp_results = list(encounter.verify_encounter())
            tmp_results.extend(os.path.split(bagfile_path))
            tmp_results.append(bagfile_pandas.overview['weather']['hourly'])
            tmp_results.append(False)
            print(tmp_results[:8])
            encounter_results_collector_list.append(tmp_results)

        # Concat to existing db_file
        tmp_encounter_db = pd.DataFrame(encounter_results_collector_list, columns=['is_encounter', 'direction', 'distance', 'bike_speed', 'bike_heading_rad', 'begin', 'end', 'description', 'working_dir', 'bag_file', 'weather', 'manual_override'])
        encounter_db = pd.concat([encounter_db, tmp_encounter_db])

        # Set bagfile db to processed
        bagfile_db.at[count_bagfile, 'processed'] = True

        # Save changes on bagfile_db
        bagfile_db.to_feather(os.path.join(bagfile_unpack_direcory, bagfile_db_file))

        # Save changes to encounter_db
        encounter_db.reset_index(drop=True).to_feather(os.path.join(bagfile_unpack_direcory, encounter_db_file))