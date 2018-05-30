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
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# h5py
import h5py

# Additional imports
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import cv2

# Import YOLO V2 network
from yolov2 import YoloV2
## -------------------------------------

## -------------------------------------
## Define current working directory
current_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
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

def train_model(yolo, model, train_batch, valid_batch):
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('weights_key.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

    tb_counter = len([log for log in os.listdir(os.path.expanduser('logs/')) if 'key_' in log]) + 1
    tensorboard = TensorBoard(log_dir=os.path.expanduser('logs/') + 'key_' + '_' + str(tb_counter), histogram_freq=0, write_graph=True, write_images=False)

    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss=yolo.custom_loss, optimizer=optimizer)

    model.fit_generator(generator           =   train_batch,
                        steps_per_epoch     =   len(train_batch),
                        epochs              =   100,
                        verbose             =   1,
                        validation_data     =   valid_batch,
                        validation_steps    =   len(valid_batch),
                        callbacks           =   [early_stop, checkpoint, tensorboard],
                        max_queue_size      =   3)

# Main function
def main():
    # Get config
    cfg = read_config(os.path.join(config_dir, 'key.yaml'))

    # Construct our model
    yolo = YoloV2()
    model = yolo.model()

    # Output model summary to ensure we have a properly formed model
    #model.summary()

    # Try loading custom weights
    yolo.load_weights(
        os.path.join(os.path.join(current_dir, 'data/weights'), 'yolo.weights'))

    # Load training data
    train_anno_dir = cfg['train_anno_dir']
    train_raw_dir = cfg['train_raw_dir']
    train_imgs, seen_train_labels = yolo.parse_anno(
        train_anno_dir, train_raw_dir, labels=['key'])

    # # Load validation data
    valid_anno_dir = cfg['valid_anno_dir']
    valid_raw_dir = cfg['valid_raw_dir']
    valid_imgs, seen_valid_labels = yolo.parse_anno(
        valid_anno_dir, valid_raw_dir, labels=['key'])

    train_batch, valid_batch = yolo.batch_setup(train_imgs, valid_imgs)

    train_model(yolo, model, train_batch, valid_batch)

    return 1


# If main point is this file
if __name__ == '__main__':
    main()
