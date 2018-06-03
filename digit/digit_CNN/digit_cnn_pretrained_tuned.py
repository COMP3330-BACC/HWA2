from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import keras
import cv2
import time

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

def initialise_model(input_shape, classes, base):
	model = Sequential()
	model.add(base)
	model.add(Flatten())
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
	epochs = 35
	batch_size = 20
	classes = 10

	if K.image_data_format() == 'channels_first':
		input_shape = (3, width, height)
	else:
		input_shape = (width, height, 3)

	vgg_conv = VGG16(weights='imagenet',
					  include_top=False,
					  input_shape=input_shape)

	for layer in vgg_conv.layers[:-4]:
		layer.trainable = False

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

	valid_generator = test1_dgen.flow_from_directory(
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

	model = initialise_model(input_shape, classes, vgg_conv)

	ADAM = Adam(lr=0.0001)
	model.compile(optimizer=ADAM, loss='categorical_crossentropy', metrics=['acc'])
	print(model.summary())
	for layer in vgg_conv.layers:
		print(layer, layer.trainable)
	'''
	t_start = time.time()
	history = model.fit_generator(train_generator,
						steps_per_epoch=t_samps//batch_size,
						epochs=epochs,
						validation_data=valid_generator,
						validation_steps=t_valid//batch_size)
	t_elapsed = time.time() - t_start

	print("Time: " + str(t_elapsed))

	model.save_weights('VGG_pretrained_unaug.h5')

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

	plot_model(model, to_file='model_basic.png')
	evaluation = model.evaluate_generator(test_generator)
	print(evaluation)

	predictions = model.predict_generator(test_generator)
	
	predictions = np.argmax(predictions, axis=-1)
	label_map = (train_generator.class_indices)
	label_map = dict((v,k) for k, v in label_map.items())
	predictions = [label_map[k] for k in predictions]

	incorrect = []
	inc_i = []
	count = 0
	for i in range(len(predictions)):
		if predictions[i] == test_generator.filenames[i][0]:
			count = count + 1
		else:
			incorrect.append(test_generator.filenames[i])
			inc_i.append(i)


	print("Correct prediction rate: "+str(100*count/len(predictions))+"%")
	
	print("Incorrect Predictions: ")
	for i in range(len(incorrect)):
		img = cv2.imread(tester_data+"/"+incorrect[i], 0)
		cv2.imshow("Incorrect", img)
		print("Correct Label: "+str(incorrect[i][0])+", Predicted Label: "+str(predictions[inc_i[i]]))
		cv2.waitKey(0)

if __name__ == '__main__':
	main()