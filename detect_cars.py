import os
import torch
import pandas as pd

class CarDetector:
    def __init__(self, image_sequence_path, image_type='.png'):
        # Classes the detector should look for
        self.classes_of_interest_list = ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'person']

        # Set working directory
        self.working_directory = image_sequence_path

        # Scrape directory for files of predefined type
        directory_content = []
        for file in sorted(os.listdir(self.working_directory)):
            if file.endswith(image_type):
                directory_content.append(file)
        self.image_path_list = [os.path.join(self.working_directory, image_name) for image_name in directory_content]  # batch of images

        # Download pretrained yolo5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def manage_detection(self, batch_size=100):
        # Split list of images into batches to avoid memory issues
        self.split_image_path_list = [self.image_path_list[i:i + batch_size] for i in range(0, len(self.image_path_list), batch_size)]

        # Create empty list of pandas dataframes
        self.results_frame_list = []

        # Iterate over batches
        for sub_image_path_list in self.split_image_path_list:
            self.results_frame_list.extend(self.batch_detection(sub_image_path_list))

        # Concatente results
        self.concatenated_results = pd.concat(self.results_frame_list, ignore_index=True, axis=0)

        # Export as feather file
        self.concatenated_results.to_feather(os.path.join(self.working_directory, 'camera_0.feather'))

    def batch_detection(self, sub_image_path_list):
        # Start detection on batch
        detector_results = self.model(sub_image_path_list)

        return self.clean_up_results(detector_results)

    def clean_up_results(self, detector_results):
        # Get list of pandas dataframes
        raw_results_frame_list = detector_results.pandas().xywh

        # List of indexes marked for removal
        empty_frames_list = []

        # Iterate over entries for postprocessing
        for i, frame_name in enumerate(detector_results.files):
            # Add empty frames to flag list and skip post processing
            if raw_results_frame_list[i].empty:
                empty_frames_list.append(i)
                continue

            # Remove unwanted entries (e.g. benches) and remove index
            raw_results_frame_list[i] = raw_results_frame_list[i][raw_results_frame_list[i]['name'].isin(self.classes_of_interest_list)].reset_index()

            # Insert unix timestamp from image name and convert to datetime
            raw_results_frame_list[i]['timestamp'] = pd.to_datetime(float(frame_name[:-4]), unit='s')

        # Pop empty frames from results frame list
        if len(empty_frames_list) != 0:
            empty_frames_list.reverse()
            for i in empty_frames_list:
                raw_results_frame_list.pop(i)

        # Return cleaned list
        return raw_results_frame_list

if __name__ == "__main__":
    # Assemble path to sequence of images
    image_sequence_path = os.path.join('2022-04-28-track3', 'camera_0')

    # Run detector
    car_detector = CarDetector(image_sequence_path)
    car_detector.manage_detection()