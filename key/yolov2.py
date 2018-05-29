# Import keras
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate

# Import tensorflow
import tensorflow as tf

# Import numpy
import numpy as np

# Import XML parser
import xml.etree.ElementTree as ET

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