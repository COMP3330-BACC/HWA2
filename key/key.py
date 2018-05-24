#
# Car Key Detector
# ----------------
# 	Part of solution for Homework Assignment 2 (HWA2)
# 	as part of COMP3330 Machine Intelligence
#

## -------------------------------------
## Import required packages
# Keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import Adam

# Additional imports
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

## -------------------------------------

## -------------------------------------
## Define config directory
config_dir = os.path.join(
    os.path.join('..', os.path.realpath(__file__)), 'config')

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
    cfg = read_config(config_dir + 'key.yaml')
    return 1


# If main point is this file
if __name__ == '__main__':
    main()