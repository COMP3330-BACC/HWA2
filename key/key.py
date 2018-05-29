#
# Car Key Detector
# ----------------
# 	Part of solution for Homework Assignment 2 (HWA2)
# 	as part of COMP3330 Machine Intelligence
#

## -------------------------------------
## Import required packages
# Keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate

#from tqdm import tqdm
#from imgaug import augmenters as iaa

#from preprocessing import parse_annotation, BatchGenerator
#from utils import WeightReader, decode_netout, draw_boxes

# Additional imports
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
#import imgaug as ia
import numpy as np
import pickle
import os, cv2
import yaml

## -------------------------------------

## -------------------------------------
## Define config directory
config_dir = os.path.join(
    os.path.join('..', os.path.dirname(os.path.realpath(__file__))), 'config')

## -------------------------------------

class WeightReader:
	def __init__(self, weight_file):
		self.offset = 4
		self.all_weights = np.fromfile(weight_file, dtype='float32')
    
	def read_bytes(self, size):
		self.offset = self.offset + size
		return self.all_weights[self.offset-size:self.offset]

	def reset(self):
		self.offset = 4

# Get configuration values
def read_config(cfg_file):
    try:
        with open(cfg_file, 'r') as yml:
            return yaml.load(yml)
    except FileNotFoundError:
        print('[ERR] Failed to load config file \'{0}\''.format(cfg_file))
        exit()

# Constantly reading in config vals is DUMB and GAY -> i dont give a funk if it saves space
def get_config_vals(cfg):
    img_dim = cfg['img_dim']
    grid_dim = cfg['grid_dim']
    box = cfg['box']
    n_class = cfg['n_class']
    obj_thresh = cfg['obj_thresh']
    nms_thresh = cfg['nms_thresh']
    anchors = cfg['anchors']
    no_obj_scale = cfg['no_obj_scale']
    obj_scale = cfg['obj_scale']
    coord_scale = cfg['coord_scale']
    class_scale = cfg['class_scale']

    batch_size = cfg['batch_size']
    warm_ups = cfg['warm_ups']
    true_box_buff = cfg['true_box_buff']

    weight_path = cfg['weight_path']
    train_path_img = cfg['train_path_img']
    train_path_ann = cfg['train_path_ann']
    valid_path_img = cfg['valid_path_img']
    valid_path_ann = cfg['valid_path_ann']

    return img_dim, grid_dim, box, n_class, obj_thresh, nms_thresh, anchors, no_obj_scale, obj_scale, coord_scale, class_scale, batch_size, warm_ups, true_box_buff, weight_path, train_path_img, train_path_ann, valid_path_img, valid_path_ann

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

def model_construction(img_dim, true_box_buff, box, n_class, grid_dim):
	input_image = Input(shape=(img_dim[0], img_dim[1], 3))
	true_boxes = Input(shape=(1, 1, 1, true_box_buff, 4))

	#L1
	x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
	x = BatchNormalization(name='norm_1')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	#L2
	x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
	x = BatchNormalization(name='norm_2')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	#L3
	x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
	x = BatchNormalization(name='norm_3')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L4
	x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
	x = BatchNormalization(name='norm_4')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L5
	x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
	x = BatchNormalization(name='norm_5')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	#L6
	x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
	x = BatchNormalization(name='norm_6')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L7
	x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
	x = BatchNormalization(name='norm_7')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L8
	x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
	x = BatchNormalization(name='norm_8')(x)
	x = LeakyReLU(alpha=0.1)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)

	#L9
	x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
	x = BatchNormalization(name='norm_9')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L10
	x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
	x = BatchNormalization(name='norm_10')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L11
	x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
	x = BatchNormalization(name='norm_11')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L12
	x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
	x = BatchNormalization(name='norm_12')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L13
	x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
	x = BatchNormalization(name='norm_13')(x)
	x = LeakyReLU(alpha=0.1)(x)

	skip_connection = x

	x = MaxPooling2D(pool_size=(2, 2))(x)

	#L14
	x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
	x = BatchNormalization(name='norm_14')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L15
	x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
	x = BatchNormalization(name='norm_15')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L16
	x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
	x = BatchNormalization(name='norm_16')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L17
	x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
	x = BatchNormalization(name='norm_17')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L18
	x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
	x = BatchNormalization(name='norm_18')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L19
	x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
	x = BatchNormalization(name='norm_19')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L20
	x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
	x = BatchNormalization(name='norm_20')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L21
	skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
	skip_connection = BatchNormalization(name='norm_21')(skip_connection)
	skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
	skip_connection = Lambda(space_to_depth_x2)(skip_connection)

	x = concatenate([skip_connection, x])

	#L22
	x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
	x = BatchNormalization(name='norm_22')(x)
	x = LeakyReLU(alpha=0.1)(x)

	#L23
	x = Conv2D(box*(4+1+n_class), (1, 1), padding = 'same', name = 'conv_23')(x)
	output = Reshape((grid_dim[0], grid_dim[1], box, 4+1+n_class))(x)

	output = Lambda(lambda args: args[0])([output, true_boxes])
	model = Model([input_image, true_boxes], output)

	return model

def load_pretraining(model, weight_path, grid_dim):
	weight_reader = WeightReader(weight_path)
	weight_reader.reset()
	nb_conv=23

	for i in range(1, nb_conv+1):
		conv_layer = model.get_layer('conv_' + str(i))

		if i<nb_conv:
			norm_layer = model.get_layer('norm_'+str(i))
			size = np.prod(norm_layer.get_weights()[0].shape)
			beta  = weight_reader.read_bytes(size)
			gamma = weight_reader.read_bytes(size)
			mean  = weight_reader.read_bytes(size)
			var   = weight_reader.read_bytes(size)
			weights = norm_layer.set_weights([gamma, beta, mean, var]) 

		if len(conv_layer.get_weights()) > 1:
			bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
			kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
			kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
			kernel = kernel.transpose([2,3,1,0])
			conv_layer.set_weights([kernel, bias])
		else:
			kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
			kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
			kernel = kernel.transpose([2,3,1,0])
			conv_layer.set_weights([kernel])

	layer = model.layers[-4]
	weights = layer.get_weights()

	new_kernel = np.random.normal(size=weights[0].shape)/(grid_dim[0]*grid_dim[1])
	new_bias = np.random.normal(size=weights[1].shape)/(grid_dim[0]*grid_dim[1])

	layer.set_weights([new_kernel, new_bias])

	return model

# Main function
def main():
    cfg = read_config(os.path.join(config_dir, 'key.yaml'))
    labels = ['key']
    img_dim, grid_dim, box, n_class, obj_thresh, nms_thresh, anchors, no_obj_scale, obj_scale, coord_scale, class_scale, batch_size, warm_ups, true_box_buff, weight_path, train_path_img, train_path_ann, valid_path_img, valid_path_ann = get_config_vals(cfg)
    
    model = model_construction(img_dim, true_box_buff, box, n_class, grid_dim)
    model = load_pretraining(model, weight_path, grid_dim)
    return 1


# If main point is this file
if __name__ == '__main__':
    main()
