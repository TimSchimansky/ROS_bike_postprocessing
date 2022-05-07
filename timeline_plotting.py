import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def add_from_to_timestamp(frame, frequency):
    # Invert frequency for duration
    half_length_sensor_event = (1 / frequency) * 0.4

    # Add begin and end timestamp to frame
    frame['from_timestamp'] = frame.index - datetime.timedelta(seconds=half_length_sensor_event)
    frame['duration'] = pd.Timedelta(half_length_sensor_event * 2, unit="d")

    return frame


def generalize_dataframe(frame, key, sensor):
    # Add sensor name as column
    frame['sensor_name'] = sensor

    # Remove everything but chosen key, sensor name and index
    frame = frame[[key, 'sensor_name']]

    # Remove rows of nan values in key column
    frame = frame.dropna(axis=0)

    # Normalize value column (purely for plotting and color range restrictions!)
    frame[key] = (frame[key] - frame[key].min()) / (frame[key].max() - frame[key].min())

    # Rename main column to value
    return frame.rename(columns={key: 'value'})


def create_timeline_plot(bag_pandas_object, sensor_filter, sensor_keys):
    # Create empty pandas dataframe for appension
    collector_list = []

    # Iterate over sensors to use
    for sensor, key in zip(sensor_filter, sensor_keys):
        # Load values into tmp variables
        tmp_frame = bag_pandas_object.dataframes[sensor].dataframe
        tmp_freq = bag_pandas_object.dataframes[sensor].frequency

        # Bring dataframe into common form
        tmp_frame = generalize_dataframe(tmp_frame, key, sensor)

        # Add two timestamps as time period per measurement
        tmp_frame = add_from_to_timestamp(tmp_frame, tmp_freq)

        # Add to collector for concatenation
        collector_list.append(tmp_frame)

    # Concat all preprocessed data frames
    collector_frame = pd.concat(collector_list, axis=0)

    fig, ax = plt.subplots(1, figsize=(8, 3))

    ax.barh(collector_frame.sensor_name, collector_frame.duration, left=collector_frame.from_timestamp)
    plt.show()
    pass