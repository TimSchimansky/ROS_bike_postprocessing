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

    # Resample frame for displaying purposes
    frame = frame.resample('0.25S', label='right', closed='right').median()

    # Normalize value column (purely for plotting and color range restrictions!)
    if frame[key].max() > 1 or frame[key].max() < 0 or frame[key].min() < 0 or frame[key].min() > 1:
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

    # Setup for plotting
    sns.set_theme()
    fig, ax = plt.subplots()

    # Set colormap
    # TODO: find nicer one
    cmap = plt.cm.get_cmap('autumn')
    #cmap = sns.color_palette("coolwarm", as_cmap=True)

    for i, frame in enumerate(frame_collector_list):
        # Convert stripe positions to list of list
        tmp_frame_as_list = [[el] for el in list(frame.index)]

        # Plot Stripes
        plt.eventplot(tmp_frame_as_list, linelengths=0.8, linewidths=1, colors=cmap(frame.value), lineoffsets=[i] * len(frame)) #, colors=cmap(frame.value)) #np.array(frame.value)

    # Replace y ticks with sensor names
    # TODO: Add pretty print names
    ax.set_yticks(np.linspace(0, i, i+1))
    ax.set_yticklabels(sensor_keys)

    # Show plot
    plt.show()