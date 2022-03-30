import rosbag
import cv2
import numpy as np
import os
import warnings

def vec3_to_list(vector3_in):
    return [vector3_in.x, vector3_in.y, vector3_in.z]

def quaternion_to_list(quaternion_in):
    return [quaternion_in.x, quaternion_in.y, quaternion_in.z, quaternion_in.w]

class rosbag_reader:
    def __init__(self, bag_file_name):
        """This function is used for initialization"""
        self.source_bag = rosbag.Bag(bag_file_name, 'r')
        self.topics = self.source_bag.get_type_and_topic_info()[1].keys()

        # Prepare export folder if not existing
        # TODO: make these generated from name or as parameter
        self.bag_unpack_dir = 'debug_test_unpack'
        if not os.path.exists(self.bag_unpack_dir):
            os.makedirs(self.bag_unpack_dir)

    def __enter__(self):
        """This function is used for the (with .. as ..) call"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """This function is used to close all connections to files when this class is not needed anymore"""
        self.source_bag.close()

    def export_images(self):
        # TODO: make this parameters of the function
        camera_unpack_subdir = 'camera'

        # Prepare export folder if not existing
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
        # Load message type from msg for correct csv translation
        topic_meta = self.source_bag.get_type_and_topic_info(topic_filters=topic)
        message_type = topic_meta.topics[topic].msg_type

        # Handle file export for barometric pressure data
        if message_type == 'sensor_msgs/FluidPressure':
            # Assemble export filename
            export_filename = os.path.join(self.bag_unpack_dir, 'barometric_pressure.csv')

            # TODO: Think about changing this into a binary format (maybe from Pandas)
            # Open file with context handler
            with open(export_filename, 'w') as f:
                # Write header
                f.write('timestamp fluid_pressure variance\n')

                # Iterate over sensor messages
                for topic, msg, t in self.source_bag.read_messages(topics=[topic]):
                    f.write('%.12f %.12f %.12f\n' % (msg.header.stamp.to_sec(), msg.fluid_pressure, msg.variance))

        # Handle file export for illuminance data
        elif message_type == 'sensor_msgs/Illuminance':
            # Assemble export filename
            export_filename = os.path.join(self.bag_unpack_dir, 'illuminance.csv')

            # TODO: Think about changing this into a binary format (maybe from Pandas)
            # Open file with context handler
            with open(export_filename, 'w') as f:
                # Write header
                f.write('timestamp illuminance variance\n')

                # Iterate over sensor messages
                for topic, msg, t in self.source_bag.read_messages(topics=[topic]):
                    f.write('%.12f %.12f %.12f\n' % (msg.header.stamp.to_sec(), msg.illuminance, msg.variance))

        # Handle file export for illuminance data
        elif message_type == 'sensor_msgs/Imu':
            # Assemble export filename
            export_filename = os.path.join(self.bag_unpack_dir, 'imu.csv')

            # TODO: Think about changing this into a binary format (maybe from Pandas)
            # Open file with context handler
            with open(export_filename, 'w') as f:
                # Write header
                f.write('timestamp or_x or_y or_z or_w li_ac_x li_ac_y li_ac_z an_ve_x an_ve_y an_ve_z\n')

                # Iterate over sensor messages
                for topic, msg, t in self.source_bag.read_messages(topics=[topic]):

                    # Assemble line output by conversion of message into list
                    content_list = [msg.header.stamp.to_sec()] + quaternion_to_list(msg.orientation) + vec3_to_list(msg.linear_acceleration) + vec3_to_list(msg.angular_velocity)
                    f.write('%.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n' %tuple(content_list))

        # Handle file export for illuminance data
        elif message_type == 'sensor_msgs/MagneticField':
            # Assemble export filename
            export_filename = os.path.join(self.bag_unpack_dir, 'magnetic_field.csv')

            # TODO: Think about changing this into a binary format (maybe from Pandas)
            # Open file with context handler
            with open(export_filename, 'w') as f:
                # Write header
                f.write('timestamp x y z\n')

                # Iterate over sensor messages
                for topic, msg, t in self.source_bag.read_messages(topics=[topic]):
                    # Assemble line output by conversion of message into list
                    f.write('%.12f %.12f %.12f %.12f\n' % tuple(
                        [msg.header.stamp.to_sec()] + vec3_to_list(msg.magnetic_field)))

        else:
            # TODO: throw exception
            warnings.warn('The topic ' + topic + ' is not available in this bag file!')
            pass

with rosbag_reader("../debug_test_camera_lidar.bag") as reader_object:
    reader_object.export_images()
    reader_object.export_1d_data('/phone1/android/barometric_pressure')
    reader_object.export_1d_data('/phone1/android/illuminance')
    reader_object.export_1d_data('/phone1/android/imu')
    reader_object.export_1d_data('/phone1/android/magnetic_field')
    print(reader_object.topics)

