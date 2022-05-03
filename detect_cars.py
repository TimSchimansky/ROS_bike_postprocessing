import os

import torch
import numpy as np
import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib

# List of classes of interest
classes_of_interest_list = ['car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'person']

# Assemble path to sequence of images
image_sequence_path = os.path.join('2022-04-28-track3', 'camera_0')

# Start model and receive results
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# List of paths to images
image_path_list = [os.path.join(image_sequence_path, image_name) for image_name in sorted(os.listdir(image_sequence_path))[:100]]# batch of images

# Inference
results = model(image_path_list)

# Results
#results.print()
#results.save()  # or .show()

width_list = []

for i, (frame_results, frame_name) in enumerate(zip(results.pandas().xywh, results.files)):
    frame_results = frame_results[frame_results['name'].isin(['car'])].reset_index()

    #print(i, frame_results.width.max())
    #width_list.append(frame_results.width.max())

    frame_results.to_feather(os.path.join(image_sequence_path, frame_name[:-4] + '.feather'))

    print(0)