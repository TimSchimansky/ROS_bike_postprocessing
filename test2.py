import numpy as np
import cv2 as cv
import os
import pandas as pd
from tqdm import tqdm

import torch
#from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_small
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import flow_to_image
import torchvision.io

class FlowDetector:
    def __init__(self, image_sequence_path, image_type='.png'):
        # Set working directory
        self.working_directory = image_sequence_path

        # Scrape directory for files of predefined type
        directory_content = []
        for file in sorted(os.listdir(self.working_directory)):
            if file.endswith(image_type):
                directory_content.append(file)
        self.image_path_list = [os.path.join(self.working_directory, image_name) for image_name in directory_content]  # batch of images

        # If you can, run this example on a GPU, it will be a lot faster.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Setup model
        self.model = raft_small(pretrained=True, progress=False).to(self.device)
        self.model = self.model.eval()

    def preprocess(self, batch):
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # map [0, 1] into [-1, 1]
                T.Resize(size=(360, 640)),
            ]
        )
        batch = transforms(batch)
        return batch

    def classify_by_model(self, image_0, image_1):
        """Frames have to be permuted [C, H, W]"""
        # Stack images
        image_0_batch = torch.stack([image_0])
        image_1_batch = torch.stack([image_1])

        # Crop images
        image_0_batch = self.preprocess(image_0_batch).to(self.device)
        image_1_batch = self.preprocess(image_1_batch).to(self.device)

        # Apply model
        return self.model(image_0_batch.to(self.device), image_1_batch.to(self.device))

    def manage_flow_on_sequence(self):
        # Create empty list for tensors
        image_flow_list = []

        # Iterate over all frames
        for i, (image_path_0, image_path_1) in tqdm(enumerate(zip(self.image_path_list[9:], self.image_path_list[10:])), desc="sequence", total=len(self.image_path_list[1:])):
            # Load images as tensors
            image_0_raw = torchvision.io.read_image(image_path_0)
            image_1_raw = torchvision.io.read_image(image_path_1)

            # Feed images to classifier
            tmp_flow = self.classify_by_model(image_0_raw, image_1_raw)

            # Put result into list
            image_flow_list.append(tmp_flow)

            # Save as jpeg
            tmp_flow_img = flow_to_image(tmp_flow[-1][0, ...]).to("cpu")
            torchvision.io.write_jpeg(tmp_flow_img, os.path.join(self.working_directory, "dense_optical_flow", f"predicted_flow_{i+9}.jpg"))

        return image_flow_list


def quick_show(image):
    # The function cv2.imshow() is used to display an image in a window.
    cv.imshow('quickshow', image)

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv.waitKey(0)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
    cv.destroyAllWindows()

def draw_boxes(image, dataframe):
    for index, line in dataframe.iterrows():
        bounds = list(line[['xmin', 'ymin', 'xmax', 'ymax']])
        bounds = [int(corner) for corner in bounds]

        # Blue color in BGR
        color = (255, 255, 255)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv.rectangle(image, tuple(bounds[:2]), tuple(bounds[2:]), color, thickness)

    return image


def detect_most_prominent_values(input_array, angular_mode=True):
    # Switch between angle and distance mode
    if angular_mode:
        # Return index of fullest histogram bin
        return np.argmax(np.histogram(input_array, bins=360, range=(0,np.pi*2))[0])

    else:
        # Return index of fullest histogram bin
        max_value_as_int = int(np.ceil(np.max(input_array)))
        return np.argmax(np.histogram(input_array, bins=max_value_as_int, range=(0, max_value_as_int))[0])

def print_average_vector(flow_mag, flow_ang, dataframe, angle_buffer=20, dist_buffer=0):
    # Get dimensions from flow result
    width, height = flow_mag.shape

    print('\x1b[6;30;43m' + '-----------------------------------------------------------' + '\x1b[0m')
    for index, line in dataframe.iterrows():
        # Get bounds from current row
        bounds = list(line[['ymin', 'ymax', 'xmin', 'xmax']])
        bounds = [int(corner) for corner in bounds]

        print(bounds[0] - bounds[1], bounds[2] - bounds[3])

        # Create mask array
        box_mask = np.zeros((width, height), dtype=bool)
        box_mask[bounds[0]:bounds[1], bounds[2]:bounds[3]] = True

        if np.isnan(np.rad2deg(np.average(flow_ang[box_mask]))):
            print('Hi')

        # Get main angle of movement inside and outside of box
        in_box_angle = detect_most_prominent_values(flow_ang[box_mask], angular_mode=True)
        out_box_angle = detect_most_prominent_values(flow_ang[~box_mask], angular_mode=True)
        in_out_angle_diff = abs((in_box_angle - out_box_angle + 180) % 360 - 180)

        # Get main amount of movement inside and outside of box
        in_box_movement = detect_most_prominent_values(flow_mag[box_mask],angular_mode=False)
        out_box_movement = detect_most_prominent_values(flow_mag[~box_mask], angular_mode=False)
        in_out_movement_diff = abs(in_box_movement - out_box_movement)

        print('')
        print('Winkel:')
        print('Innerhalb: {0}° \t Außerhalb: {1}° \t Differenz: {2}°'.format(in_box_angle, out_box_angle, in_out_angle_diff))
        print('Distanz:')
        print('Innerhalb: {0}px \t Außerhalb: {1}px \t Differenz: {2}px'.format(in_box_movement, out_box_movement, in_out_movement_diff))

        # Make prediction if car is moving in or against driving direction
        if in_out_angle_diff <= 180 + angle_buffer and in_out_angle_diff >= 180 - angle_buffer:
            #print('Entgegenkommendes Auto')
            print('\x1b[6;30;42m' + 'Entgegenkommendes Auto' + '\x1b[0m')

working_directory = '2022-04-28-track2/camera_0_subset'
image_type = 'png'

raft = FlowDetector(working_directory, image_type)
flow_list = raft.manage_flow_on_sequence()

"""# Scrape directory for files of predefined type
directory_content = []
for file in sorted(os.listdir(working_directory)):
    if file.endswith(image_type):
        directory_content.append(file)
image_path_list = [os.path.join(working_directory, image_name) for image_name in directory_content]




detection_frame = pd.read_feather(os.path.join('2022-04-28-track2/camera_0_subset', 'camera_0.feather'))



# Ugly code from documentation

frame1 = cv.imread(image_path_list[0])
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
for image_name in directory_content[1:]:
    # Einschub
    timestamp_from_file = pd.to_datetime(float(image_name[:-4]), unit='s')

    detection_subframe = detection_frame[detection_frame['timestamp'] == timestamp_from_file]


    # Einschub ende

    frame2 = cv.imread(os.path.join(working_directory, image_name))

    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 30, 3, 7, 1.5, 0)
    #flow = cv.calcOpticalFlowSparseToDense(prvs, next, grid_step=5, sigma=0.5)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    hsv = draw_boxes(hsv, detection_subframe)

    # Print average motion inside vs outside of detection box
    print_average_vector(mag, ang, detection_subframe)

    # Display dense optical flow
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff

    # Set current secondary frame as next primary frame
    prvs = next

# Close all windows
cv.destroyAllWindows()"""