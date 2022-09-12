import numpy as np
import cv2 as cv
import os
import pandas as pd
from tqdm import tqdm
import imageio
import datetime

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
matplotlib.use('Qt5Agg')
import seaborn as sns

import scipy as scp
from scipy.signal import find_peaks

import torch
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import flow_to_image
import torchvision.io

# Set constant values
FLOW_IMG_WIDTH = 640
FLOW_IMG_HEIGHT = 360
CAMERA_OPENING_ANGLE_DEG = 117

class FlowDetector:
    def __init__(self, image_sequence, measured_distance, measured_speed, image_type='.png', standalone_mode=True, yolo_bounding_boxes=None, save_img=False):
        self.save_img = save_img

        if standalone_mode:
            # Set working directory
            self.working_directory = image_sequence

            # Scrape directory for files of predefined type
            directory_content = []
            for file in sorted(os.listdir(self.working_directory)):
                if file.endswith(image_type):
                    directory_content.append(file)
            self.image_path_list = [os.path.join(self.working_directory, image_name) for image_name in directory_content]  # batch of images

            # Get feather file of detections from same folder
            self.yolo_bounding_boxes = pd.read_feather(os.path.join(image_sequence, 'camera_0.feather'))

            self.img_out_path = working_directory

        elif not standalone_mode and yolo_bounding_boxes is not None:
            # Save list of image paths for later use
            self.image_path_list = image_sequence

            # Save bounding boxes for later use
            self.yolo_bounding_boxes = yolo_bounding_boxes

            self.img_out_path = None

        else:
            pass
            # TODO: Throw exception

        # Get dimensions of images
        self.input_img_height, self.input_img_width = imageio.imread(self.image_path_list[0]).shape[:-1]
        self.flow_img_height, self.flow_img_width = FLOW_IMG_HEIGHT, FLOW_IMG_WIDTH
        self.camera_opening_angle = np.deg2rad(CAMERA_OPENING_ANGLE_DEG)

        # Calculate multiplication factor for bounding box scaling
        self.bounding_box_scale = self.flow_img_width / self.input_img_width

        # If you can, run this example on a GPU, it will be a lot faster.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Save measured data for paralax threshold calculation
        self.obj_distance = measured_distance / 100
        self.bike_speed = measured_speed

        # Setup model
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
        self.model = self.model.eval()

        # Setup empty lists to acumulate results
        self.result_list_is_moving_vehicle = []
        self.result_list_is_overtaking = []

    def preprocess(self, batch):
        transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # map [0, 1] into [-1, 1]
                T.Resize(size=(self.flow_img_height, self.flow_img_width)),
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
        image_0_batch = self.preprocess(image_0_batch)
        image_1_batch = self.preprocess(image_1_batch)

        # Apply model
        return self.model(image_0_batch.to(self.device), image_1_batch.to(self.device))

    def image_path_to_unix(self, path_string):
        return float(os.path.split(path_string)[-1][:-4])

    def get_flow_results(self):
        # Run detection
        self.manage_flow_on_sequence()

        # Release video memory
        torch.cuda.empty_cache()

        # Decide if threshold is reached
        ratio_positives = sum(self.result_list_is_moving_vehicle) / len(self.result_list_is_moving_vehicle)

        if ratio_positives < 1/4: #1/2:
            # Not enough positives to confirm car
            return False, False
        else:
            # Cut out values for actual detections before determining movement direction
            positives_indicees = [i for i, x in enumerate(self.result_list_is_moving_vehicle) if x]
            cut_result_list_is_overtaking = [self.result_list_is_overtaking[i] for i in positives_indicees]

            # Catch case of no True values for overtaking
            if sum(cut_result_list_is_overtaking) == 0:
                # Return Values directly
                return True, False

            # Case of not overtaking but more than 0 values indicate overtaking
            elif sum(cut_result_list_is_overtaking)/ len(cut_result_list_is_overtaking) <= 1/2:
                return True, False

            # Case of overtaking car
            else:
                return True, True

    def manage_flow_on_sequence(self):
        # Iterate over all frames
        for image_index, (image_path_0, image_path_1) in tqdm(enumerate(zip(self.image_path_list, self.image_path_list[1:])), desc="sequence", total=len(self.image_path_list[1:])):

            # Load images as tensors
            image_0_raw = torchvision.io.read_image(image_path_0)
            image_1_raw = torchvision.io.read_image(image_path_1)

            if os.path.split(image_path_0)[-1] == "1659937922.879283239.png":
                print('hi')

            # Feed images to classifier
            tmp_flow = self.classify_by_model(image_0_raw, image_1_raw)

            # Put result into list
            flow_array = tmp_flow[-1][0, ...].cpu().detach().numpy()

            if self.img_out_path is not None and self.save_img == True:
                flow_img = flow_to_image(flow_array)


            # Determine timestamp of images as unix
            unix_timestamp_0 = pd.to_datetime(self.image_path_to_unix(image_path_0), unit='s')
            unix_timestamp_1 = pd.to_datetime(self.image_path_to_unix(image_path_1), unit='s')

            # Extract bounding boxes for image 0 from yolo bounding boxes (+- 1/100 sec to accommodate rounding errors)
            #curr_yolo_bounding_boxes = self.yolo_bounding_boxes.loc[(self.yolo_bounding_boxes['timestamp'] >= unix_timestamp_0 - datetime.timedelta(seconds=0.01)) & (self.yolo_bounding_boxes['timestamp'] < unix_timestamp_0 + datetime.timedelta(seconds=0.01))]
            curr_yolo_bounding_boxes = self.yolo_bounding_boxes.loc[self.yolo_bounding_boxes['timestamp'] == unix_timestamp_0]

            # Flow array is present as cartesian-displacement, convert to polar
            flow_array_distance = np.sqrt(flow_array[0, :, :] ** 2 + flow_array[1, :, :] ** 2)
            flow_array_angle = np.arctan(flow_array[0, :, :], flow_array[1, :, :])

            # Set up masks
            mask_list = []
            background_box_mask = np.ones((self.flow_img_height, self.flow_img_width), dtype=bool)
            for _, yolo_bounding_box in curr_yolo_bounding_boxes.iterrows():
                # Get bounds from current row, scale and convert to int
                yolo_bounding_box_values = list(yolo_bounding_box[['ymin', 'ymax', 'xmin', 'xmax']])
                yolo_bounding_box_values_scaled = [int(corner * self.bounding_box_scale) for corner in yolo_bounding_box_values]
                #yolo_bounding_box_values = [int(corner) for corner in yolo_bounding_box_values]

                # Create mask array
                tmp_box_mask = np.zeros((self.flow_img_height, self.flow_img_width), dtype=bool)
                tmp_box_mask[yolo_bounding_box_values_scaled[0]:yolo_bounding_box_values_scaled[1], yolo_bounding_box_values_scaled[2]:yolo_bounding_box_values_scaled[3]] = True
                mask_list.append(tmp_box_mask)

                # Remove area from background mask
                background_box_mask[yolo_bounding_box_values_scaled[0]:yolo_bounding_box_values_scaled[1], yolo_bounding_box_values_scaled[2]:yolo_bounding_box_values_scaled[3]] = False

            # Decide if moving object is present
            threshold_pixelshift = self.estimate_paralax_threshold(self.image_path_to_unix(image_path_1), self.image_path_to_unix(image_path_0))
            is_moving_vehicle, is_overtaking = self.check_for_moving_objects(mask_list, background_box_mask, flow_array_distance, flow_array_angle, threshold_pixelshift, curr_yolo_bounding_boxes)
            self.result_list_is_moving_vehicle.append(is_moving_vehicle)
            self.result_list_is_overtaking.append(is_overtaking)

            # Save as jpeg
            #tmp_flow_img = flow_to_image(tmp_flow[-1][0, ...]).to("cpu")
            #torchvision.io.write_jpeg(tmp_flow_img, os.path.join(self.working_directory, "dense_optical_flow", f"predicted_flow_{i}.jpg"))


    def check_for_moving_objects(self, mask_list, background_box_mask, flow_array_distance, flow_array_angle, threshold_pixelshift, yolo_bounding_boxes, threshold_multiplier=1.1):
        """Returns tuple of boolean (is_moving_vehicle, is_overtaking, estimated_speed)
        speed estimation not yet implemented"""
        # Initialize booleans for result
        is_moving_vehicle = False
        is_overtaking = False

        # Iterate over all bounding boxes of current image
        for box_index, (box_mask, (_, yolo_bounding_box)) in enumerate(zip(mask_list, yolo_bounding_boxes.iterrows())):
            # Calculate KDE for distance and angle
            kde_distance = calculate_kde(flow_array_distance, box_mask, background_box_mask)
            kde_anglular = calculate_kde(flow_array_angle, box_mask, background_box_mask, angular_mode=True)

            # Create limits for accepted angles
            lower_acceptance_angle_value = np.pi / 2 - np.pi / 8
            upper_acceptance_angle_value = np.pi / 2 + np.pi / 8

            if yolo_bounding_box["name"] == 'person' or yolo_bounding_box["name"] == 'bycicle':
                tmp_is_moving_vehicle = False
                tmp_is_overtaking = False

            elif kde_anglular[0] <= upper_acceptance_angle_value and kde_anglular[0] >= lower_acceptance_angle_value:
                # Case 1: Overtaking Vehicle
                tmp_is_moving_vehicle = True
                tmp_is_overtaking = True

            elif kde_anglular[0] >= - upper_acceptance_angle_value and kde_anglular[0] <= - lower_acceptance_angle_value:
                # Case 2: Opposing vehicle
                # Since ambiguous if just parallax, additional check is made
                if kde_distance[0] >= threshold_pixelshift * threshold_multiplier:
                    print(kde_distance[0], threshold_pixelshift * threshold_multiplier)
                    tmp_is_moving_vehicle = True
                    tmp_is_overtaking = False

                else:
                    tmp_is_moving_vehicle = False
                    tmp_is_overtaking = False

            else:
                tmp_is_moving_vehicle = False
                tmp_is_overtaking = False

            # Fuse result over all bounding boxes
            is_overtaking = is_overtaking or tmp_is_overtaking
            is_moving_vehicle = is_moving_vehicle or tmp_is_moving_vehicle

        return is_moving_vehicle, is_overtaking


    def estimate_paralax_threshold(self, timestamp_0, timestamp_1):
        """Determines pixelshift value for still standing objects at given distance. Moving objects would be faster than that."""
        # Calculate time span
        timestamp_delta = abs(timestamp_1 - timestamp_0)

        # Calculate moved distance between images (since time span is short, a straight trajectory is assumed)
        dist_movement_bike = self.bike_speed * timestamp_delta

        # Calculate angular difference between centered object in first and second frame
        angular_delta = np.arctan2(dist_movement_bike, self.obj_distance)

        # Transfer angle to pixels
        relative_angle = angular_delta / self.camera_opening_angle
        return relative_angle * self.flow_img_width


def quick_show(image):
    # The function cv2.imshow() is used to display an image in a window.
    cv.imshow('quickshow', image)

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv.waitKey(0)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
    cv.destroyAllWindows()


def calculate_kde(flow_array, box_mask, background_box_mask, angular_mode=False, plotting_mode=False, kappa=10):
    """Calculate kernel density estimation, switching between standard approach by scipy and custom von mises approach,
    that allows circular data (angles)"""

    in_box = flow_array[box_mask][~np.isnan(flow_array[box_mask])]
    out_box = flow_array[background_box_mask][~np.isnan(flow_array[background_box_mask])]

    # Switch based on mode
    if angular_mode:
        # Calculate circular kde for angle inside and outside of bounding box
        bin_centers, inside_kde = vonmises_fft_kde(in_box, kappa)
        _, outside_kde = vonmises_fft_kde(out_box, kappa)

        # Fill dummy
        upper_lim = None

        # Determine max values
        inside_max = bin_centers[np.argmax(inside_kde)]
        outside_max = bin_centers[np.argmax(outside_kde)]

    else:
        # Upper limit for kde
        upper_lim = int(np.ceil(np.nanmax(flow_array)))

        # Create x-grid for kde
        bin_centers = np.linspace(0, upper_lim, upper_lim + 1, endpoint=True)

        # Calculate classic kde for distance inside and outside of bounding box

        inside_kde_manager = scp.stats.gaussian_kde(in_box, bw_method=kappa/in_box.std(ddof=1))
        inside_kde = inside_kde_manager.evaluate(bin_centers)

        outside_kde_manager = scp.stats.gaussian_kde(out_box, bw_method=kappa/out_box.std(ddof=1))
        outside_kde = outside_kde_manager.evaluate(bin_centers)

        # Determine max values
        peak_in = find_peaks(inside_kde, height=np.std(inside_kde))
        inside_max = peak_in[0][np.argmax(peak_in[1])]

        inside_max = np.argmax(inside_kde)

        peak_out = find_peaks(outside_kde, height=np.std(outside_kde))
        outside_max = peak_out[0][np.argmax(peak_out[1])]

        outside_max = np.argmax(outside_kde)

    # Return results
    if plotting_mode:
        return bin_centers, inside_kde, inside_max, outside_kde, outside_max, upper_lim
    else:
        return inside_max, outside_max


def vonmises_pdf(x, mu, kappa):
    """Helper function for vonmises_fft_kde"""
    return np.exp(kappa * np.cos(x - mu)) / (2. * np.pi * scp.special.i0(kappa))


def vonmises_fft_kde(data, kappa, n_bins=128):
    """Circular kde method, that is based on FFT by rudolfbyker on stackoverflow"""
    bins = np.linspace(-np.pi, np.pi, n_bins + 1, endpoint=True)
    hist_n, bin_edges = np.histogram(data, bins=bins)
    bin_centers = np.mean([bin_edges[1:], bin_edges[:-1]], axis=0)
    kernel = vonmises_pdf(
        x=bin_centers,
        mu=0,
        kappa=kappa
    )
    kde = np.fft.fftshift(np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(hist_n)))
    kde /= np.trapz(kde, x=bin_centers)
    return bin_centers, kde


def plot_kde(flow_array_distance, flow_array_angle, box_mask, background_box_mask, kappa=2):

    # Set seaborn as theme
    sns.set_theme()

    flow_array_distance[np.logical_and(flow_array_distance <= 15, box_mask)] = np.nan
    flow_array_angle[np.logical_and(flow_array_distance <= 15, box_mask)] = np.nan

    # Calculate kde for distance and angle
    kde_result_distance = calculate_kde(flow_array_distance, box_mask, background_box_mask, angular_mode=False, plotting_mode=True, kappa=kappa)
    kde_result_angle = calculate_kde(flow_array_angle, box_mask, background_box_mask, angular_mode=True, plotting_mode=True, kappa=kappa)

    print('dist:', kde_result_distance)
    print('angl:', kde_result_angle)

    # Set up plot
    fig = plt.figure(figsize=(10,5))
    font_property = fm.FontProperties(fname='cmunrm.ttf')

    # Cartesian subplot for displacement distance
    ax1 = fig.add_subplot(121)
    ax1.set_box_aspect(1)
    ax1.plot(kde_result_distance[0], kde_result_distance[1], label="Außerhalb Bounding-Box")
    ax1.fill_between(x = kde_result_distance[0], y1 = kde_result_distance[1], alpha= 0.25, label='_nolegend_')
    ax1.axvline(kde_result_distance[2], 0, 1, linestyle='--', c=sns.color_palette()[0], label="Dominanter Wert")
    ax1.plot(kde_result_distance[0], kde_result_distance[3], label="Innerhalb Bounding-Box")
    ax1.fill_between(x = kde_result_distance[0], y1 = kde_result_distance[3], alpha= 0.25, label='_nolegend_')
    ax1.axvline(kde_result_distance[4], 0, 1, linestyle='--', c=sns.color_palette()[1], label="Dominanter Wert")
    ax1.set_xlabel("Verschiebungsbetrag [px]", fontproperties=font_property)
    ax1.set_xlim([0, kde_result_distance[5]])
    ax1.set_ylim(bottom=0)

    # Set Computer Moedern as tick font
    for label in ax1.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_property)

    # Set legend
    fig.legend(loc='upper center', bbox_to_anchor=[0.35, 0.85], prop=font_property)

    # Polar subplot for displacement angle
    ax2 = fig.add_subplot(122, polar=True)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.plot(kde_result_angle[0], kde_result_angle[1])
    ax2.fill(kde_result_angle[0], kde_result_angle[1], alpha=0.25)
    ax2.axvline(kde_result_angle[2], 0, 1, linestyle='--', c=sns.color_palette()[0])
    ax2.plot(kde_result_angle[0], kde_result_angle[3])
    ax2.fill(kde_result_angle[0], kde_result_angle[3], alpha=0.25)
    ax2.axvline(kde_result_angle[4], 0, 1, linestyle='--', c=sns.color_palette()[1])
    ax2.set_xlabel("Bewegungsrichtung [°]", fontproperties=font_property)

    # Set Computer Moedern as tick font
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_property)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_property)

    # Export as pdf
    fig.savefig('../kde1.svg', bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    """# For standalone debugging
    working_directory = '../kde_demo_for_thesis'
    image_type = 'png'

    event_range = 0.85
    bike_speed = 4.276

    raft = FlowDetector(working_directory, event_range, bike_speed, image_type, save_img=False)
    print(raft.get_flow_results())"""

    working_dir = os.path.join('..', 'kde_demo_for_thesis', 'out2')

    flow_as_array = np.load(os.path.join(working_dir, 'uff.npy'))

    bounding_boxen = pd.read_feather(os.path.join(working_dir, 'bounds.feather'))

    yolo_bounding_box = bounding_boxen.iloc[0]

    # Get bounds from current row, scale and convert to int
    yolo_bounding_box_values = list(yolo_bounding_box[['ymin', 'ymax', 'xmin', 'xmax']])
    yolo_bounding_box_values_scaled = [int(corner / 2) for corner in yolo_bounding_box_values]
    # yolo_bounding_box_values = [int(corner) for corner in yolo_bounding_box_values]

    # Create mask array
    tmp_box_mask = np.zeros((FLOW_IMG_HEIGHT, FLOW_IMG_WIDTH), dtype=bool)
    tmp_box_mask[yolo_bounding_box_values_scaled[0]:yolo_bounding_box_values_scaled[1],yolo_bounding_box_values_scaled[2]:yolo_bounding_box_values_scaled[3]] = True

    box_mask = tmp_box_mask
    bg_mask = ~tmp_box_mask

    # Flow array is present as cartesian-displacement, convert to polar
    flow_array_distance = np.sqrt(flow_as_array[0, :, :] ** 2 + flow_as_array[1, :, :] ** 2)
    flow_array_angle = np.arctan(flow_as_array[0, :, :], flow_as_array[1, :, :])

    plot_kde(flow_array_distance, flow_array_angle, box_mask, bg_mask, kappa=10)

    print(1)