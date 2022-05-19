import numpy as np
import cv2 as cv
import os
import pandas as pd
import matplotlib.pyplot as plt

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


def detect_most_prominent_values(input_array):
    # Round values to int
    input_array = np.around(input_array)

    # Return index of fullest histogram bin
    max_value_as_int = int(np.ceil(np.max(input_array)))
    min_value_as_int = int(np.min(input_array))

    # Ignore zero movements
    input_array[input_array == 0] = 'nan'

    # Index of max value
    hist = np.histogram(input_array, bins=abs(max_value_as_int - min_value_as_int), range=(min_value_as_int, max_value_as_int))
    prom_index = np.argmax(hist[0]) + 1

    #print(hist[0])
    #print(hist[1])

    # Return integer value of most prominent bin
    return hist[1][prom_index]

def print_average_vector(flow, dataframe, angle_buffer=20, dist_buffer=0):
    # Get dimensions from flow result
    width, height, _ = flow.shape

    print('-----------------------------------------------------------')
    for index, line in dataframe.iterrows():
        # Get bounds from current row
        bounds = list(line[['ymin', 'ymax', 'xmin', 'xmax']])
        bounds = [int(corner) for corner in bounds]

        print(bounds[0] - bounds[1], bounds[2] - bounds[3])

        # Create mask array
        box_mask = np.zeros((width, height), dtype=bool)
        box_mask[bounds[0]:bounds[1], bounds[2]:bounds[3]] = True

        # Get main angle of movement inside and outside of box
        in_box_x = detect_most_prominent_values(flow[box_mask, 0])
        in_box_y = detect_most_prominent_values(flow[box_mask, 1])
        #in_out_angle_diff = abs((in_box_angle - out_box_angle + 180) % 360 - 180)

        # Get main amount of movement inside and outside of box
        out_box_x = detect_most_prominent_values(flow[~box_mask, 0])
        out_box_y = detect_most_prominent_values(flow[~box_mask, 1])
        #in_out_movement_diff = abs(in_box_movement - out_box_movement)

        print('Inside ROI:')
        print('x: {0}px \t y: {1}px'.format(in_box_x, in_box_y))
        print('Outside ROI:')
        print('x: {0}px \t y: {1}px'.format(out_box_x, out_box_y))

        diff_norm = np.sqrt((in_box_x - out_box_x)**2 + (in_box_y - out_box_y)**2)
        print('Diff vector:')
        print('x: {0}px \t y: {1}px \t norm: {2}px'.format(in_box_x - out_box_x, in_box_y - out_box_y, diff_norm))

        CRED = '\33[91m'
        CGREEN = '\33[32m'
        CEND = '\33[0m'

        if diff_norm >= 6:
            print(CGREEN + "Vermutlich Auto" + CEND)
        else:
            print(CRED + "Vermutlich kein Auto" + CEND)

working_directory = '2022-04-28-track3/camera_0_subset_1'
image_type = 'png'

# Scrape directory for files of predefined type
directory_content = []
for file in sorted(os.listdir(working_directory)):
    if file.endswith(image_type):
        directory_content.append(file)
image_path_list = [os.path.join(working_directory, image_name) for image_name in directory_content]




detection_frame = pd.read_feather(os.path.join('2022-04-28-track3/camera_0_subset_1', 'camera_0.feather'))



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
    print_average_vector(flow, detection_subframe)

    # Display dense optical flow
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff

    # Set current secondary frame as next primary frame
    prvs = next

# Close all windows
cv.destroyAllWindows()