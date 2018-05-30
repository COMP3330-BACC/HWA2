# Import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
from keras.utils import Sequence
import imgaug as ia
from imgaug import augmenters as iaa

# Import tensorflow
import tensorflow as tf

# Import numpy
import numpy as np

# Import XML parser
import xml.etree.ElementTree as ET

# Import network config
import net_cfg as n_cfg

import os


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
            conv_layer = self.model.get_layer('conv_' + str(i))

            if i < num_conv:
                norm_layer = self.model.get_layer('norm_' + str(i))

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
                kernel = wr.read(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(
                    list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])

        # Get last convolutional layer
        layer = self.model.layers[-4]
        weights = layer.get_weights()

        # Randomize weights of the last convolutional layer
        new_kernel = np.random.normal(size=weights[0].shape) / (
            n_cfg.GRID_H * n_cfg.GRID_W)
        new_bias = np.random.normal(size=weights[1].shape) / (
            n_cfg.GRID_H * n_cfg.GRID_W)

        layer.set_weights([new_kernel, new_bias])

    # Loss function for bounding boxes
    def custom_loss(self, y_true, y_pred):
        input_image = Input(shape=(n_cfg.IMAGE_H, n_cfg.IMAGE_W, 3))
        true_boxes = Input(shape=(1, 1, 1, n_cfg.TRUE_BOX_BUFFER))
        mask_shape = tf.shape(y_true)[:4]

        # Create x and y grid cells
        cell_x = tf.to_float(
            tf.reshape(
                tf.tile(tf.range(n_cfg.GRID_W), [n_cfg.GRID_H]),
                (1, n_cfg.GRID_H, n_cfg.GRID_W, 1, 1),
            ))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        # Tile cells to make grid
        cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], -1), [n_cfg.BATCH_SIZE, 1, 1, 5, 1])

        # Create masks
        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        """Adjust the prediction"""
        # Adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        # Adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(
            n_cfg.ANCHORS, [1, 1, 1, n_cfg.BOX, 2])

        # Adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        # Adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        """Adjust ground truth"""
        # Adjust x and y
        true_box_xy = y_true[..., 0:2]

        # Adjust w and h
        true_box_wh = y_true[..., 2:4]

        # Adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxs = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxs = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxs = tf.minimum(pred_maxs, true_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        # Adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        """Determine masks"""
        ## Coordinate mask: position of ground truth boxes
        coord_mask = tf.expand_dims(
            y_true[..., 4], axis=-1) * n_cfg.COORD_SCALE

        ## Confidence mask: penalize predictors + penalize boxes with low IOU
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxs = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxs = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxs = tf.minimum(pred_maxs, true_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < n_cfg.IOU_THRESH) * (
            1 - y_true[..., 4]) * n_cfg.NO_OBJECT_SCALE

        # Penalize confidence of boxes responsible for ground truth box
        conf_mask = conf_mask + y_true[..., 4] * n_cfg.OBJECT_SCALE

        ## Class mask: position of ground truth boxes
        class_mask = y_true[..., 4] * tf.gather(
            n_cfg.CLASS_WEIGHTS, true_box_class) * n_cfg.CLASS_SCALE
        """Warmup Training"""
        no_boxes_mask = tf.to_float(coord_mask < n_cfg.COORD_SCALE / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(
            tf.less(seen, n_cfg.WARM_UP_BATCHES),
            lambda : [
                true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                true_box_wh + tf.ones_like(true_box_wh) * np.reshape(n_cfg.ANCHORS,
                    [1, 1, 1, n_cfg.BOX, 2]) * no_boxes_mask, tf.ones_like(coord_mask)
                ],
            lambda : [
                true_box_xy,
                true_box_wh,
                coord_mask,
                ],
        )
        """Finalise loss"""
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.))

        # Calculate xy loss
        loss_xy = tf.reduce_sum(
            tf.square(true_box_xy - pred_box_xy) * coord_mask) / (
                nb_coord_box + 1e-6) / 2.
        # Calculate wh loss
        loss_wh = tf.reduce_sum(
            tf.square(true_box_wh - pred_box_wh) * coord_mask) / (
                nb_coord_box + 1e-6) / 2.
        # Calculate conf loss
        loss_conf = tf.reduce_sum(
            tf.square(true_box_conf - pred_box_conf) * conf_mask) / (
                nb_conf_box + 1e-6) / 2.
        # Calculate class loss
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(
            loss_class * class_mask) / (nb_class_box + 1e-6)

        # Calculate total loss
        loss = loss_xy + loss_wh + loss_conf + loss_class

        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(
            tf.to_float(true_box_conf > 0.5) *
            tf.to_float(pred_box_conf > 0.3))
        """Debug"""
        current_recall = nb_pred_box / (nb_true_box + 1e-6)
        total_recall = tf.assign_add(total_recall, current_recall)

        loss = tf.Print(
            loss,
            [tf.zeros((1))],
            message='Dummy Line \t',
            summarize=1000,
        )
        loss = tf.Print(
            loss,
            [loss_xy],
            message='Loss XY \t',
            summarize=1000,
        )
        loss = tf.Print(
            loss,
            [loss_wh],
            message='Loss WH \t',
            summarize=1000,
        )
        loss = tf.Print(
            loss,
            [loss_conf],
            message='Loss Conf \t',
            summarize=1000,
        )
        loss = tf.Print(
            loss,
            [loss_class],
            message='Loss Class \t',
            summarize=1000,
        )
        loss = tf.Print(
            loss,
            [loss],
            message='Total Loss \t',
            summarize=1000,
        )
        loss = tf.Print(
            loss,
            [current_recall],
            message='Current Recall \t',
            summarize=1000,
        )
        loss = tf.Print(
            loss,
            [total_recall / seen],
            message='Average Recall \t',
            summarize=1000,
        )

        return loss

    # Parse annotations from
    def parse_anno(self, anno_dir, img_dir, labels=[]):
        imgs = []
        seen_labels = {}

        for ann in sorted(os.listdir(anno_dir)):
            img = {'object': []}

            tree = ET.parse(anno_dir + ann)

            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}

                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1

                            if len(labels) > 0 and obj['name'] not in labels:
                                print('Unlisted label {0} found'.format(
                                    obj['name']))
                                break
                            else:
                                img['object'] += [obj]

                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                imgs += [img]

        return imgs, seen_labels
    
    def normalize(self, image):
        return image/255

    def batch_setup(self, train_imgs, valid_imgs,):
    	train_batch = BatchGenerator(train_imgs, n_cfg.generator_config, norm=self.normalize)
    	valid_batch = BatchGenerator(valid_imgs, n_cfg.generator_config, norm=self.normalize, jitter=False)
    	return train_batch, valid_batch
	

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

class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        
                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou     = -1
                        
                        shifted_box = BoundBox(0, 
                                               0,
                                               center_w,                                                
                                               center_h)
                        
                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)
                            
                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1
                        
                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12), 
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1  

        #print(' new batch created', idx)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            
            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)
                
            image = self.aug_pipe.augment_image(image)            
            
        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin   
        return image, all_objs

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score