from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import cv2


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import plot_model, to_categorical
from keras.applications import VGG16

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def initialise_model(input_shape, classes):
	model = Sequential()
	model.add(Dense(512, activation='relu', input_dim=1*1*512))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	return model

def main():
	train_data = 'ArtificialDigits/ArtificialDigits/Dataset_Digits/Test'
	valid_data = 'ArtificialDigits/ArtificialDigits/Dataset_Digits/Validate'
	tester_data = 'ArtificialDigits/ArtificialDigits/Dataset_Digits/Tester'
	height = 48
	width = 48
	t_samps = 700
	t_valid = 140
	epochs = 125
	batch_size = 20
	classes = 10

	if K.image_data_format() == 'channels_first':
		input_shape = (3, width, height)
	else:
		input_shape = (width, height, 3)

	vgg_conv = VGG16(weights='imagenet',
					  include_top=False,
					  input_shape=input_shape)

	train_dgen = ImageDataGenerator(
		rescale = 1./2,
		shear_range=0.2,
		zoom_range=0.2,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=False)
	
	test1_dgen = ImageDataGenerator(
		rescale = 1./2)
	
	test_dgen = ImageDataGenerator(
		rescale = 1./2,
		shear_range=0.2,
		zoom_range=0.2,
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		horizontal_flip=False)
	
	train_generator = train_dgen.flow_from_directory(
		train_data,
		target_size=(width, height),
		batch_size=batch_size,
		class_mode='categorical')

	valid_generator = test_dgen.flow_from_directory(
		valid_data,
		target_size=(width, height),
		batch_size=batch_size,
		class_mode='categorical')

	test_generator = test1_dgen.flow_from_directory(
		tester_data,
		target_size=(width, height),
		batch_size=batch_size,
		class_mode='categorical',
		shuffle=False)

	train_features = np.zeros(shape=(t_samps, 1, 1, 512))
	train_labels = np.zeros(shape=(t_samps, 10))

	i=0
	for inputs_batch, labels_batch in train_generator:
		features_batch = vgg_conv.predict(inputs_batch)
		train_features[i*batch_size:(i+1)*batch_size] = features_batch
		train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
		i += 1
		if i*batch_size >= t_samps:
			break

	train_features = np.reshape(train_features, (t_samps, 1*1*512))

	validation_features = np.zeros(shape=(t_valid, 1, 1, 512))
	validation_labels = np.zeros(shape=(t_valid, 10))

	i=0
	for inputs_batch, labels_batch in valid_generator:
		features_batch = vgg_conv.predict(inputs_batch)
		validation_features[i*batch_size:(i+1)*batch_size] = features_batch
		validation_labels[i*batch_size:(i+1)*batch_size] = labels_batch
		i += 1
		if i* batch_size >= t_valid:
			break

	validation_features = np.reshape(validation_features, (t_valid, 1*1*512))

	model = initialise_model(input_shape, classes)

	ADAM = Adam(lr=0.0001)
	model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['acc'])

	'''
	history = model.fit(train_features,
						train_labels,
						epochs=epochs,
						batch_size=batch_size,
						validation_data=(validation_features, validation_labels))

	model.save_weights('VGG_pretrained_aug.h5')

	# Accuracy and Validation Graphs
	plt.rcParams['figure.figsize'] = (6,5)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title( "Accuracy ")
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train', 'val'], loc='upper left')
	plt.show()
	plt.close()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title("Error")
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['train', 'val'], loc='upper right')
	plt.show()
	plt.close()
	'''
	
	model.load_weights('VGG_pretrained_aug.h5')
	plot_model(model, to_file='model_VGG.png')
	fnames = valid_generator.filenames
	ground_truth = valid_generator.classes
	label2index = valid_generator.class_indices

	idx2label = dict((v,k) for k,v in label2index.items())

	predictions = model.predict_classes(validation_features)
	prob = model.predict(validation_features)
	errors = np.where(predictions != ground_truth)[0]
	print("Valid no of errors = {}/{}".format(len(errors),t_valid))


if __name__ == '__main__':
	main()