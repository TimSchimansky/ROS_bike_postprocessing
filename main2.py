import rosbag
import cv2
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
import numpy as np
import os

class rosbag_reader:
    def __init__(self, bag_file_name):
        """This function is used for initialization"""
        self.source_bag = rosbag.Bag(bag_file_name, 'r')
        self.topics = self.source_bag.get_type_and_topic_info()[1].keys()

        # TODO: make these generated from name or as parameter
        self.bag_unpack_dir = 'debug_test_unpack'

    def __enter__(self):
        """This function is used for the (with .. as ..) call"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This function is used to close all connections to files when this class is not needed anymore"""
        self.source_bag.close()

    def export_images(self):
        # TODO: make this parameters of the function
        camera_unpack_subdir = 'camera'

        # Prepare export folders if not existing
        export_directory = os.path.join(self.bag_unpack_dir, camera_unpack_subdir)
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

        for topic, msg, t in self.source_bag.read_messages(topics=['/phone1/camera/image/compressed']):
            # Convert msg to numpy then to opencv image
            temp_image = cv2.imdecode(np.fromstring(msg.data, np.uint8), cv2.IMREAD_COLOR)

            # Write image into predefined folder
            image_file_name = ("%s.%s.png" %(msg.header.stamp.secs, msg.header.stamp.nsecs))
            cv2.imwrite(os.path.join(export_directory, image_file_name), temp_image)

    def export_1d_data(self, topic):
        # TODO: make functional
        pass

with rosbag_reader("debug_test_camera_lidar.bag") as reader_object:
    #reader_object.export_images()
    print(reader_object.topics)


