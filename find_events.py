

import pandas as pd

from ingest import *

if __name__ == "__main__":
    # Set bagfile path
    bagfile_directory = "../"
    bagfiles = ["2022-04-28-track3.bag"]
    bagfile_paths = [os.path.join(bagfile_directory, bagfile_i) for bagfile_i in bagfiles]

    # Set export directory
    unpack_direcory = ".."

    # Read config file for sensors to import
    sensor_export = pd.read_csv('export_sensors.csv')

    # Ingest bag files via config file
    for bagfile_path in bagfile_paths:
        with rosbag_reader(bagfile_path, unpack_direcory) as reader_object:
            #print(reader_object.topics)

            # Go through list of sensors to export
            for sensor_line in sensor_export:
                reader_object.export_flex(sens_type, topic_name, sensor_name, pretty_print, subsampling)


    # TODO: Traverse Trajectories(s) for TOIs

    # Repeat for all trajecories


    # TODO: Do Classification for TOI