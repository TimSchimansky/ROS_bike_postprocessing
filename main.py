"""This script will plot time vs data for recorded rosbag files"""

import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

import fix_bag

#fix_bag.fix_bagfile_header("debug_test_with_cam.bag", "debug_test_fixed_header.bag")
#b = bagreader('debug_test_fixed_header.bag')
b = bagreader('debug_test_camera_lidar.bag')

#data = b.message_by_topic("/phone1/android/fix")

csvfiles = []
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)

df_imu = pd.read_csv(data)
df_imu


df_imu = pd.read_csv(data)
print(1)






"""
# Main function
if __name__ == '__main__':
    print_hi('test')"""


