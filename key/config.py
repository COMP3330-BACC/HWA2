from os import path

# Model name: selects model folder from models directory {'1', '2'}
MODEL_NAME = '1'

# Test directory: selects directory where the test images are held
TEST_DIR = path.abspath('/home/matt/UON/COMP3330/keyDS/test')

# Image display dimensions: defines how large the image display will be
#	(affects how large the prediction number displayed on the bounding boxes are)
IMG_WIDTH = 1280
IMG_HEIGHT = 720
