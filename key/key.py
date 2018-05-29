#
# Car Key Detector
# ----------------
#   Part of solution for Homework Assignment 2 (HWA2)
#   as part of COMP3330 Machine Intelligence
#

## -------------------------------------
## Import required packages
# Keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam

# h5py
import h5py

# Additional imports
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

# Import YOLO V2 network
from yolov2 import YoloV2
## -------------------------------------

## -------------------------------------
## Define config directory
config_dir = os.path.join(
    os.path.join('..', os.path.dirname(os.path.realpath(__file__))), 'config')

## -------------------------------------


# Get configuration values
def read_config(cfg_file):
    try:
        with open(cfg_file, 'r') as yml:
            return yaml.load(yml)
    except FileNotFoundError:
        print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
        exit()


# Main function
def main():
    # Get config
    cfg = read_config(os.path.join(config_dir, 'key.yaml'))

    # Construct our model
    yolo = YoloV2()
    model = yolo.model()

    # Output model summary to ensure we have a properly formed model
    model.summary()

    return 1


# If main point is this file
if __name__ == '__main__':
    main()
