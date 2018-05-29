# Import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate

# Import tensorflow
import tensorflow as tf

# Import network config
import net_cfg as n_cfg


# Class for establishing a YOLO V2 network model
# Based on work from github.com/experiencor/keras-yolo2
class YoloV2:

    # Function for creating the organisation layer
    # Credit to github.com/allanzelener/YAD2K
    def space_to_depth_x2(self, x):
        """Creates organisational layer for Yolo V2 network"""
        return tf.space_to_depth(x, block_size=2)

    # Function to create Yolo V2 network, and return
    def model(self):
        """Return untrained model for Yolo V2 network"""

        # Set our input image size and boxes
        input_image = Input(shape=(n_cfg.IMAGE_H, n_cfg.IMAGE_W, 3))
        true_boxes = Input(shape=(1, 1, 1, n_cfg.TRUE_BOX_BUFFER))

        # Layer 1
        x = Conv2D(
            32, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_1',
            use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(
            64, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_2',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(
            128, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_3',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(
            64, (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_4',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(
            128, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_5',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(
            256, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_6',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(
            128, (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_7',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(
            256, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_8',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(
            512, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_9',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(
            256, (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_10',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(
            512, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_11',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(
            256, (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_12',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(
            512, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_13',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(
            1024, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_14',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(
            512, (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_15',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(
            1024, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_16',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(
            512, (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_17',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(
            1024, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_18',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(
            1024, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_19',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(
            1024, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_20',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(
            64, (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_21',
            use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(self.space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(
            1024, (3, 3),
            strides=(1, 1),
            padding='same',
            name='conv_22',
            use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 23
        x = Conv2D(
            n_cfg.BOX * (4 + 1 + n_cfg.CLASS), (1, 1),
            strides=(1, 1),
            padding='same',
            name='conv_23')(x)
        output = Reshape((n_cfg.GRID_H, n_cfg.GRID_W, n_cfg.BOX,
                          4 + 1 + n_cfg.CLASS))(x)

        output = Lambda(lambda args: args[0])([output, true_boxes])

        model = Model([input_image, true_boxes], output)

        self.model = model
        return model

    # Load pretrained weights at path
    def load_weights(self, path):
        """Load pretrained weights from designated path"""
        # Establish our weight reader
        wr = WeightReader(path)
        # TODO: check if this line is necessary
        wr.reset()

        # Number of convolutional layers
        num_conv = 23

        # Iterate through convolutional layers and designate weights
        for i in range(1, num_conv + 1):
            conv_layer = model.get_layer('conv_' + str(i))

            if i < num_conv:
                norm_layer = model.get_layer('norm_' + str(i))

                size = np.prod(norm_layer.get_weights()[0].shape)

                beta = wr.read(size)
                gamma = wr.read(size)
                mean = wr.read(size)
                var = wr.read(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            if len(conv_layer.get_weights()) > 1:
                bias = wr.read(np.prod(conv_layer.get_weights()[1].shape))
                kernel = wr.read(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(
                    list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = wr.read(np.prod(conv_Layer.get_weights()[0].shape))
                kernel = kernel.reshape(
                    list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])

        # Get last convolutional layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        # Randomize weights of the last convolutional layer
        new_kernel = np.random.normal(size=weights[0].shape) / (
            GRID_H * GRID_W)
        new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)

        layer.set_weights([new_kernel, new_bias])

    # Loss function for bounding boxes
    # def custom_loss(y_true, y_pred):


# Weight reader for allowing weight file processing of Yolo V2 pretrained weights
# Based on work from github.com/experiencor/keras-yolo2
class WeightReader:
    def __init__(self, w_file):
        self.offset = 4
        self.weights = np.fromfile(w_file, dtype='float32')

    def read(self, size):
        self.offset = self.offset + size
        return self.weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4