import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    # Create empty pandas dataframe list and duration list for appension
    frame_collector_list = []
    duration_collector_list = []

    # Iterate over sensors to use
    for sensor, key in zip(sensor_filter, sensor_keys):
        # Load values into tmp variables
        tmp_frame = bag_pandas_object.dataframes[sensor].dataframe
        tmp_freq = bag_pandas_object.dataframes[sensor].frequency

        # Bring dataframe into common form
        tmp_frame = generalize_dataframe(tmp_frame, key, sensor)

        # Add two timestamps as time period per measurement
        #tmp_frame = add_from_to_timestamp(tmp_frame, tmp_freq)

        # Add duration to list
        duration_collector_list.append((1 / tmp_freq) * 10)

        # Add to collector for concatenation
        frame_collector_list.append(tmp_frame)

    # Concat all preprocessed data frames
    collector_frame = pd.concat(frame_collector_list, axis=0)

    """fig, ax = plt.subplots(1, figsize=(8, 3))

    ax.barh(collector_frame.sensor_name, collector_frame.duration, left=collector_frame.from_timestamp)"""

    sns.set_theme()
    cmap = plt.cm.get_cmap('Spectral')

    for i, (frame, duration) in enumerate(zip(frame_collector_list, duration_collector_list)):
        tmp_frame_as_list = [[el] for el in list(frame.index)]


        plt.eventplot(tmp_frame_as_list, linelengths=0.75, linewidths=duration*10, colors=cmap(frame.value), lineoffsets=[i] * len(frame)) #, colors=cmap(frame.value)) #np.array(frame.value)
    plt.show()

    """plt.eventplot([[1.5], [2.4], [5.6]], colors=[(1,0,0), (1,1,0), (1,0,0)], lineoffsets=[1] * 3)
    plt.show()"""
    """for i, (frame, duration) in enumerate(zip(frame_collector_list, duration_collector_list)):
        sns.rugplot(y=i, x=frame.index, hue=frame['value'], height=0.15, legend = False) #jitter=False, marker="$\u2223$", size=5
    plt.show()"""
    # y=frame_collector_list[0]["sensor_name"]
    """ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))"""
    #

    #pass