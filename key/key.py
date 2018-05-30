#
# Car Key Detector
# ----------------
#   Part of solution for Homework Assignment 2 (HWA2)
#   as part of COMP3330 Machine Intelligence
#

## -------------------------------------
## Import required packages
# Keras
from keras.applications.mobilenet import MobileNet, _depthwise_conv_block
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import *
from keras.preprocessing.image import *
from keras.utils import Sequence


# Additional imports
import csv
import math
import glob
import os
import cv2
import numpy as np
import re
import xml.etree.ElementTree as ET
import imgaug as ia
from imgaug import augmenters as iaa

## -------------------------------------

## -------------------------------------
## Define current working directory
current_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
## Define config directory
config_dir = os.path.join(
    os.path.join('..', os.path.dirname(os.path.realpath(__file__))), 'config')

## -------------------------------------

class data_sequence(Sequence):
	def __init__(self, csv_file, image_size, batch_size=32, feature_scaling = False):
		self.csv_file = csv_file
		with open(self.csv_file, "r") as file:
			reader = csv.reader(file, delimiter=",")
			arr = list(reader)
		self.y = np.zeros((len(arr), 4))
		self.x = []
		self.image_size = image_size

		for index, (path, class_id, width, height, x0, y0, x1, y1) in enumerate(arr):
			width, height, x0, y0, x1, y1 = int(width), int(height), int(x0), int(y0), int(x1), int(y1)
			mid_x = x0 + (x1 - x0) / 2
			mid_y = y0 + (y1 - y0) / 2
			self.y[index][0] = (mid_x / width) * IMAGE_SIZE
			self.y[index][1] = (mid_y / height) * IMAGE_SIZE
			self.y[index][2] = ((x1 - x0) / width) * IMAGE_SIZE
			self.y[index][3] = ((y1 - y0) / height) * IMAGE_SIZE
			self.x.append(path)

		self.batch_size = batch_size
		self.feature_scaling = feature_scaling
		if self.feature_scaling:
			dataset = self.__load_images(self.x)
			broadcast_shape = [1, 1, 1]
			broadcast_shape[2] = dataset.shape[3]

			self.mean = np.mean(dataset, axis=(0, 1, 2))
			self.mean = np.reshape(self.mean, broadcast_shape)
			self.std = np.std(dataset, axis=(0, 1, 2))
			self.std = np.reshape(self.std, broadcast_shape) + K.epsilon()

	def __load_images(self, dataset):
		out = []
		for file_name in dataset:
			im = cv2.resize(cv2.imread(file_name), (self.image_size, self.image_size))
			out.append(im)
		return np.array(out)

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		images = self.__load_images(batch_x).astype('float32')
		if self.feature_scaling:
			images -= self.mean
			images /= self.std
		return images, batch_y

# Get configuration values
def read_config(cfg_file):
    try:
        with open(cfg_file, 'r') as yml:
            return yaml.load(yml)
    except FileNotFoundError:
        print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
        exit()

#no clue how o interpret the xml to get all the good shit out of it :(
def generate_sets(cfg):
	t_out = cfg['train_output']
	v_out = cfg['validation_output']
	d_out = cfg['dictionary_output']

	with open(t_out, "w") as train, open(v_out, "w") as valid, open(d_out, "w") as dic:
		writer_t = csv.writer(train, delimiter=",")
		writer_v = csv.writer(valid, delimiter=",")

		seen=[]
		class_id = 0
		for xml_file in sorted(glob.glob("{}/*xml".format(cfg['anno_folder']))):
			tree = ET.parse(xml_file)

			filename = os.path.basename(xml_file).replace(".xml", "")
			

def create_model(cfg):
	size = cfg['img_size']
	alpha = cfg['alpha']

	model_net = MobileNet(input_shape=(size[0], size[1], 3), include_top=False, alpha=alpha)
	x = _depthwise_conv_block(model_net.layers[-1].output, 1024, alpha, 1, block_id=14)
	x = MaxPooling2D(pool_size=(3, 3))(x)
	x = Conv2D(4, kernel_size=(1, 1), padding="same")(x)
	x = Reshape((4,))(x)

	return Model(inputs=model_net.input, outputs=x)

def train(cfg, model):
	epochs = cfg['epochs']
	img_size = cfg['img_size']
	#do the rest


# Main function
def main():
    # Get config
    cfg = read_config(os.path.join(config_dir, 'key.yaml'))
    generate_sets(cfg)

    model = create_model(cfg)

    return 1


# If main point is this file
if __name__ == '__main__':
    main()
