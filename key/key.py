# Key Detection Testing Tool
#   For COMP3330 Machine Intelligence Homework Assignment 2
#  By Matthew Amos

# Modified from Tensorflow Object Detection API testing tutorial at
#   https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/blob/master/Object_detection_image.py

# Imports
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

PATH_TO_OBJ_DET_DIR = os.path.abspath(
    '/home/matt/Downloads/tensorflow/models/research/object_detection')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append(PATH_TO_OBJ_DET_DIR)

# Import tensorflow object detection utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Define which model we want to test
MODEL_NAME = 'm1'

# Name of the directory containing the object detection module we're using
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(CURR_DIR, 'models')
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)

# Define testing image directory
TEST_DIR = os.path.abspath('/home/matt/UON/COMP3330/keyDS/test')

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CURR_DIR, 'labels', 'labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

files = os.listdir(TEST_DIR)

img_width = 1280
img_height = 720

for img_path in files:
    if os.path.isfile(os.path.join(TEST_DIR, img_path)):
        cv2.namedWindow('{0}'.format(img_path), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('{0}'.format(img_path), img_width, img_height)
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(os.path.join(TEST_DIR, img_path))
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [
                detection_boxes, detection_scores, detection_classes,
                num_detections
            ],
            feed_dict={image_tensor: image_expanded})

        # Print results to terminal
        print(img_path)
        print('\tBoxes:')
        for box in boxes[0]:
            if box.sum() != 0:
                print('\t\t[{0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}]'.format(
                    box[0], box[1], box[2], box[3]))

        print('\tScores:')
        num_scores = 0
        for score in scores[0]:
            if score.all() != 0:
                num_scores += 1
                print('\t\t', score)

        # Draw the results of the detection (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.50)

        # All the results have been drawn on image. Now display the image.
        cv2.imshow('{0}'.format(img_path), image)

        # Press any key to close the image
        cv2.waitKey(0)

        # Clean up
        cv2.destroyAllWindows()
