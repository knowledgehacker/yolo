# -*- coding: utf-8 -*-

import numpy as np

import config
from net.small import Small
from utils.iou import cal_iou

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Input

H, W = config.H, config.W
B = config.B
C = config.C


class FastYolo(object):
    def __init__(self):
        #print("FastYolo")
        self.net = Small()

    def forward(self, image_batch, input_shape, data_format, dropout_keep_prob, trainable=True):
        input_image = Input(shape=input_shape, name="input_image")
        output, pretrained_model, _ = self.net.build(input_image, data_format, dropout_keep_prob, trainable)

        model = Model(input_image, output)
        #model.summary()

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out, pretrained_model

    """
    reference to yolo_loss in yad2k/models/keras_yolo.py
    TODO: get the detector mask, that is, find the best anchor for each ground truth box.
    https://github.com/allanzelener/YAD2K/blob/a42c760ef868bc115e596b56863dc25624d2e756/yad2k/models/keras_yolo.py#L66
    """
    def opt(self, net_out, nd_class_probs, nd_conf, nd_coords, box_mask):
        # parameters
        class_scale = config.class_scale
        obj_scale = config.object_scale
        noobj_scale = config.noobject_scale
        coord_scale = config.coord_scale
        print('scales  = {}'.format([class_scale, obj_scale, noobj_scale, coord_scale]))

        nd_net_out = tf.reshape(net_out, [-1, H, W, B, (C + 1 + 4)])

        # adjust class probability, objectness(confidence), coordinate parts of net_out for loss calculation
        adjusted_class_prob = tf.nn.softmax(nd_net_out[:, :, :, :, :C])
        adjusted_class_prob = tf.reshape(adjusted_class_prob, [-1, H*W, B, C])

        adjusted_conf = tf.sigmoid(nd_net_out[:, :, :, :, C])
        adjusted_conf = tf.reshape(adjusted_conf, [-1, H*W, B, 1])

        nd_coords_predict = nd_net_out[:, :, :, :, C+1:]
        nd_coords_predict = tf.reshape(nd_coords_predict, [-1, H*W, B, 4])
        # sigmoid to make sure nd_coords_predict[:, :, :, 0:2] is positive
        adjusted_coords_xy = tf.sigmoid(nd_coords_predict[:, :, :, 0:2])
        adjusted_coords_wh = nd_coords_predict[:, :, :, 2:4]
        adjusted_coords_predict = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_net_out = tf.concat([adjusted_class_prob, adjusted_conf, adjusted_coords_predict], 3)

        """
        confidence weight, positive and negative samples overlap, and some samples are neither positive nor negative ones
        (the bounding boxes with highest iou >= threshold, but not responsible for any ground truth object detection).
        """
        # positive samples(bounding boxes with the highest iou)' ground truth confidence is iou
        positive = box_mask

        # negative samples(bounding boxes with highest iou < threshold)' ground truth confidence is 0
        iou = cal_iou(adjusted_coords_predict, nd_coords)
        best_box = tf.equal(iou, tf.reduce_max(iou, axis=2, keepdims=True))
        best_box_ge_thres = tf.equal(best_box, tf.greater_equal(iou, config.IOU_THRESHOLD))
        negative = (1.0 - tf.to_float(best_box_ge_thres)) * (1.0 - box_mask)

        conf_weight = obj_scale * positive + noobj_scale * negative

        """
        class weight, multiply positive to get the positive samples
        """
        box_class = tf.concat(C * [tf.expand_dims(positive, -1)], 3)
        #class_weight = class_scale * class_mask * box_class
        class_weight = class_scale * box_class

        """
        coordinate weight, multiply positive to get the positive samples.
        use (b_x - c_x, b_y - c_y), (b_w / p_w, b_h / p_h) in coordinate loss, is it correct???
        I think so, (width, height) loss should be irrelevant to anchor size (width, height).
        """
        box_coord = tf.concat(4 * [tf.expand_dims(positive, -1)], 3)

        # the bigger the box is, the smaller its weight is. the trick to improve detection on small objects???
        true_wh = tf.exp(nd_coords[:, :, :, 2:4]) * np.reshape(config.anchors, [1, 1, config.B, 2])
        coord_ratio = 2 - (true_wh[:, :, :, 0] / config.W) * (true_wh[:, :, :, 1] / config.H)
        coord_ratio = tf.concat(4 * [tf.expand_dims(coord_ratio, -1)], 3)

        coord_weight = coord_scale * box_coord * coord_ratio

        # do not normalize with anchors in data.py, so do not adjust coordinate width and height here
        true = tf.concat([nd_class_probs, tf.expand_dims(nd_conf, 3), nd_coords], 3)
        weights = tf.concat([class_weight, tf.expand_dims(conf_weight, 3), coord_weight], 3)
        weighted_square_error = weights * ((adjusted_net_out - true) ** 2)
        weighted_square_error = tf.reshape(weighted_square_error, [-1, H*W * B * (C + 1 + 4)])
        loss_op = 0.5 * tf.reduce_mean(tf.reduce_sum(weighted_square_error, 1), name="loss")

        return loss_op


def flat(x):
    return tf.layers.flatten(x)
