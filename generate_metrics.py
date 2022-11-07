# Native
import os
from datetime import timedelta, time

# 3rd party
import pandas as pd
import geopandas as gpd
import numpy as np
import sklearn.metrics as skm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.use('Qt5Agg')
import seaborn as sns
from shapely.geometry import Point, LineString
from palettable.cartocolors.sequential import agGrnYl_7_r
from mpl_toolkits.axes_grid1 import make_axes_locatable


from map_plotting import *

sns.set_theme()


def general_measurement_stats(encounter_df):
    print(f"Eventproposals: {len(encounter_df)}")
    print(f"\tdavon Überholungen: {len(encounter_df[encounter_df.direction == 1])}")
    print(f"\tdavon Überholungen: {len(encounter_df[encounter_df.direction == 0])}")


def find_neighbours(value, df):
    exactmatch = df[df.index == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        lowerneighbour_ind = max(df[df.index < value].index)
        upperneighbour_ind = min(df[df.index > value].index)
        return [lowerneighbour_ind, upperneighbour_ind]


def project_points_left(gnss, dist):
    gnss = gnss.to_crs(epsg=32632)

    projected_point_list_x = []
    projected_point_list_y = []
    projected_point_list_dist = []

    for timestamp, point in dist.iterrows():
        if np.isnan(point.range_cm_2):
            projected_point_list_x.append(np.nan)
            projected_point_list_y.append(np.nan)
            projected_point_list_dist.append(np.nan)
            continue

        # Get neighbouring timestamps od gnss dataframe
        lo_neigh, up_neigh = find_neighbours(timestamp, gnss)

        # Get GNSS points
        lo_point = gnss.at_time(lo_neigh)
        up_point = gnss.at_time(up_neigh)

        # Progress along line
        delta_t_points = up_point.index - lo_point.index
        delta_t_progress = timestamp - lo_point.index
        progress_perc = (delta_t_progress/delta_t_points).values[0]

        # Points
        lo_point_xy = np.array([lo_point.geometry.x, lo_point.geometry.y])
        up_point_xy = np.array([up_point.geometry.x, up_point.geometry.y])
        de_point_xy = up_point_xy - lo_point_xy

        # Progress and projected point
        range_m = point.values[-2]/100 - 0.19
        pr_point_xy = lo_point_xy + (up_point_xy - lo_point_xy) * progress_perc
        pr_angle_arc = np.arctan2(de_point_xy[0], de_point_xy[1]) - (np.pi / 2)
        ne_point_xy = pr_point_xy + np.array([range_m * np.cos(pr_angle_arc), range_m * np.sin(pr_angle_arc)])

        projected_point_list_x.append(ne_point_xy[0,0])
        projected_point_list_y.append(ne_point_xy[1,0])
        projected_point_list_dist.append(range_m)

    # Assemble geodataframe
    proj_point_df = pd.DataFrame(np.array([projected_point_list_x, projected_point_list_y, projected_point_list_dist]).T, columns=['x', 'y', 'col'])
    proj_point_df = gpd.GeoDataFrame(proj_point_df, geometry=gpd.points_from_xy(proj_point_df['x'], proj_point_df['y']))
    proj_point_df = proj_point_df.set_crs(epsg=32632)

    # For colorization
    """proj_point_df['col'] = 1
    proj_point_df.iloc[150:230, proj_point_df.columns.get_loc('col')] = 2
    proj_point_df.iloc[290:350, proj_point_df.columns.get_loc('col')] = 3
    proj_point_df.iloc[414:467, proj_point_df.columns.get_loc('col')] = 4
    proj_point_df.iloc[550:, proj_point_df.columns.get_loc('col')] = 5"""

    # Test plot
    #proj_point_df.plot(column='col')

    return proj_point_df


def create_meet_map_plot(projected_df, trajectory, bagfile_name, destination_px_width=1000):
    # Set up font
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Calculate bounaries of map as well as zoom size
    left_bound, lower_bound, right_bound, upper_bound, zoom = calc_map_size(projected_df, destination_px_width)

    # Download tiles, fuse and crop to single image
    map_img, bounding_box = generate_OSM_image(left_bound, right_bound, upper_bound, lower_bound, zoom)

    # Start plot
    fig, ax = plt.subplots(figsize=(8, 5))
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("left", size="5%", pad=1.75)

    # Scatter GNSS data on top
    # Trajectory
    projected_df.to_crs(epsg=3857).plot(column='col', s=15, cmap=agGrnYl_7_r.mpl_colormap, ax=ax, legend=True, cax=cax, vmax=3.5, vmin=0)
    ax.autoscale(False)
    ax.plot(trajectory.to_crs(epsg=3857).geometry.x, trajectory.to_crs(epsg=3857).geometry.y, linewidth=2)

    # Insert image into bounds
    ax.imshow(map_img, extent=(bounding_box.geometry.x[0], bounding_box.geometry.x[1], bounding_box.geometry.y[0], bounding_box.geometry.y[1]), zorder=-1)

    # Set citation string
    citation_string = r"Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
    ax.text(1, 0, citation_string, ha='right', va='bottom', fontsize=7.5, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='none', pad=2.0), fontproperties=font_property)

    # Set plotting and tick format
    ax = format_ticks_deg(ax, bounding_box)

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)
    for label in cax.get_yticklabels():
        label.set_fontproperties(font_property)

    cax.set_ylabel('seitlicher Abstand zum Fahrrad [m]', fontproperties=font_property)


    # Set capstyle
    ax.get_children()[0].set_capstyle('round')

    ax.legend(['Projizierte Abstände', 'GNSS-Trajektorie'], prop=font_property)

    # Export as pdf
    fig.savefig(f'{bagfile_name}_map_dist_projected_count.eps', bbox_inches='tight')

    # Show the plot
    plt.show()


def create_split_plot(dist_df, from_time, to_time, bagfile_name):
    # Start plot
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

    font_property = fm.FontProperties(fname='cmunrm.ttf')

    line1, = ax2.plot(dist_df.between_time(from_time, to_time).range_cm_2 / 100, label='akzeptierte Abstandswerte')
    line2, = ax.plot(dist_df.between_time(from_time, to_time).range_cm_3 / 100, c=sns.color_palette()[1], label='abgelehnte Abstandswerte')

    ax.set_ylim(11, 13)  # outliers only
    ax2.set_ylim(2, 4)  # most of the data

    # Set Computer Moedern as tick font
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_property)

    # plt.legend(['akzeptierte Abstandswerte', 'abgelehnte Abstandswerte'], prop=font_property)
    ax.legend(handles=[line1, line2], prop=font_property)
    ax2.set_xlabel('Zeit', fontproperties=font_property)
    fig.supylabel('seitlicher Abstand zum Fahrrad [m]', fontproperties=font_property)

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.tick_bottom()
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

    # Remove space
    plt.subplots_adjust(hspace=0.025)

    # Export as pdf
    fig.savefig(f'{bagfile_name}_dist_filter.eps', bbox_inches='tight')

    plt.show()


def create_nonsplit_plot(dist_df):
    # timeframe
    from_time = '16:41:14' # '16:09:39'
    to_time = '16:41:16' # '16:09:43'

    # Start plot
    fig, ax = plt.subplots(figsize=(10, 4))

    font_property = fm.FontProperties(fname='cmunrm.ttf')

    print(dist_df.between_time(from_time, to_time).range_cm_2.mean())

    ax.scatter(dist_df.between_time(from_time, to_time).index, dist_df.between_time(from_time, to_time).range_cm / 100, label='abgelehnte Abstandswerte', s=1)

    ax.set_ylim(1, 12.5)  # outliers only

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)

    # plt.legend(['akzeptierte Abstandswerte', 'abgelehnte Abstandswerte'], prop=font_property)
    #ax.legend(handles=[line1, line2], prop=font_property)
    #fig.supylabel('seitlicher Abstand zum Fahrrad [m]', fontproperties=font_property)

    # Export as pdf
    # fig.savefig('dist_filter.eps', bbox_inches='tight')

    plt.show()


def create_morpho_plot(dist_df):
    # timeframe
    from_time = '16:37:35' # '16:09:39'
    to_time = '16:37:40' # '16:09:43'

    nan_times = dist_df.between_time(time(16, 9, 39, 800000), time(16, 9, 41, 409000))[dist_df.range_cm_2.between_time(from_time, to_time).isna()].index
    nan_beg_offset = - timedelta(microseconds=57000)

    # Start plot
    fig, ax = plt.subplots(figsize=(10, 4))

    font_property = fm.FontProperties(fname='cmunrm.ttf')

    ax.plot(dist_df.range_cm_2.between_time(from_time,to_time)/100)

    for timestamp in nan_times:
        ax.axvspan(timestamp + nan_beg_offset, timestamp, alpha=0.25, color=sns.color_palette()[3])

    ax.set_ylim(0, 3.5)
    ax.set_xlim(pd.Timestamp(dist_df.between_time(from_time, to_time).index[0]), pd.Timestamp(dist_df.between_time(from_time, to_time).index[-1]))

    # Set Computer Moedern as tick font
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_property)

    # plt.legend(['akzeptierte Abstandswerte', 'abgelehnte Abstandswerte'], prop=font_property)
    ax.legend(['gültige Werte', 'fehlende Werte'], prop=font_property)
    #fig.supylabel('seitlicher Abstand zum Fahrrad [m]', fontproperties=font_property)

    # Export as pdf
    fig.savefig('morphology.svg', bbox_inches='tight')

    plt.show()


def plot_side_dist():
    bagfile_name = '2022-08-11-16-40-41_0' #'2022-08-16-15-27-34_0'
    from_time = '16:09:15' #'15:28:49'
    to_time = '16:09:45' #'15:29:35'
    """bagfile_name = '2022-08-16-15-27-34_0'
    from_time = '15:28:49'
    to_time = '15:29:35'"""
    dist_df = pd.read_feather(f'H:/bagfiles_unpack/{bagfile_name}/left_range_sensor_0.feather')
    dist_df['range_cm_2'] = dist_df['range_cm']
    dist_df['range_cm_3'] = dist_df['range_cm']
    dist_df['range_cm_2'].values[dist_df['range_cm_2'].values > 350] = np.nan
    dist_df['range_cm_3'].values[dist_df['range_cm_3'].values < 350] = np.nan

    # Change unix timestamp to datetime
    dist_df['timestamp_bagfile'] = pd.to_datetime(dist_df['timestamp_bagfile'], unit='s')
    dist_df.set_index('timestamp_bagfile', inplace=True)

    fix_df = gpd.read_feather(f'H:/bagfiles_unpack/{bagfile_name}/gnss_0.feather')
    # Change unix timestamp to datetime
    fix_df['timestamp_bagfile'] = pd.to_datetime(fix_df['timestamp_bagfile'], unit='s')
    fix_df.set_index('timestamp_bagfile', inplace=True)

    # Project points to left
    proj_point_df = project_points_left(fix_df.between_time(from_time, to_time), dist_df.between_time(from_time, to_time))

    # Start plotting
    #create_meet_map_plot(proj_point_df, fix_df.between_time(from_time, to_time), bagfile_name, destination_px_width=1000)
    #create_meet_map_plot(proj_point_df, fix_df, destination_px_width=1000)
    #create_split_plot(dist_df, from_time, to_time, bagfile_name)
    #create_morpho_plot(dist_df)

    create_nonsplit_plot(dist_df)



def print_metrics_overview(skm):
    # Import unedited version
    encounter_db_auto = pd.read_feather('H:/bagfiles_unpack/encounter_db_v2_backup_pre_manual.feather')
    encounter_db_auto = encounter_db_auto.drop_duplicates(subset=['begin', 'end', 'description']).reset_index().drop('index', axis=1)

    # Import version after manual revision
    encounter_db_manual = pd.read_feather('H:/bagfiles_unpack/encounter_db_v2_backup_after_manual.feather')

    # Remove certain vehicle classes
    vehicle_classes_to_remove = ['bicycle', 'person', 'train', 'motorcycle']
    encounter_db_auto = encounter_db_auto[~encounter_db_auto.description.isin(vehicle_classes_to_remove)]
    encounter_db_manual = encounter_db_manual[~encounter_db_manual.description.isin(vehicle_classes_to_remove)]

    # Print stats
    print(skm.confusion_matrix(encounter_db_manual.direction.values, encounter_db_auto.direction.values))
    print(skm.classification_report(encounter_db_manual.direction.values, encounter_db_auto.direction.values))




plot_side_dist()




"""# Fuse auto and manual label as well as distance for histogram
comparison_df = pd.concat([encounter_db_auto.direction,encounter_db_manual.direction, encounter_db_auto.distance], axis=1, keys=['direction_auto', 'direction_manual', 'distance'])
comparison_df["combination"] = comparison_df.direction_auto.astype(str) + comparison_df.direction_manual.astype(str)

# Plot histogram
#sns.histplot([comparison_df[comparison_df.combination == '11'].distance.values, comparison_df[comparison_df.combination == '-11'].distance.values, comparison_df[comparison_df.combination == '1-1'].distance.values], bins=15)
#sns.histplot([comparison_df[comparison_df.combination == '00'].distance.values, comparison_df[comparison_df.combination == '-10'].distance.values, comparison_df[comparison_df.combination == '0-1'].distance.values], bins=15)
#plt.legend(['True Positives', 'False Negatives', 'False Positives'])

#sns.histplot([comparison_df[comparison_df.direction_manual == 1].distance.values, comparison_df[comparison_df.direction_manual == 0].distance.values], bins=15)"""




print(1)