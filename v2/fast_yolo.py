# -*- coding: utf-8 -*-

import numpy as np

import config
from v2.small import Small
from utils.iou import find_best_box

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Input

H, W = config.H, config.W
B = config.B
C = config.C


"""
# TODO: train gets nan values after several steps. 
checklist
1) wrong input images?
No, YOLO v1 runs well.
2) wrong learning rate, too large?
Yes, starts with 1e-5 gets nan after ~18 steps. Start with 1e-6 works, can switch to 1e-5 after 1 epoch.
3) parameter gradient explosion?
Probably, output in some layer of DarkNet too big?
Try to use weights pretrained darknet19 on ImageNet works with learning rate starting with 1e-5. 
4) wrong loss implementation?
Possible.
"""
class FastYolo(object):
    def __init__(self):
        #print("FastYolo")
        self.net = Small()

    def forward(self, image_batch, input_shape, data_format, dropout_keep_prob, trainable=True):
        input_image = Input(shape=input_shape, name="input_image")
        output, conv13_weights = self.net.build(input_image, data_format, dropout_keep_prob, trainable)

        model = Model(input_image, output)
        #model.summary()

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out, conv13_weights

    #def opt(self, net_out, class_probs, class_proids, object_proids, coords):
    def opt(self, net_out, nd_class_probs, nd_class_proids, nd_object_proids, nd_coords):
        # parameters
        class_scale = config.class_scale
        obj_scale = config.object_scale
        noobj_scale = config.noobject_scale
        coord_scale = config.coord_scale
        print('scales  = {}'.format([class_scale, obj_scale, noobj_scale, coord_scale]))

        """
        The following code calculate weight vector of three parts: class, coordinate, confidence,
        initial weight vector is passed from input, we adjust it by scale parameters and best box with highest iou.
        take care of the weight terms, construct indicator matrix(which grids the objects in, which ones not in).
        """

        nd_net_out = tf.reshape(net_out, [-1, H, W, B, (C + 1 + 4)])

        # TODO: tf.nn.softmax may cause nan?
        adjusted_class_prob = tf.nn.softmax(nd_net_out[:, :, :, :, :C])
        adjusted_class_prob = tf.reshape(adjusted_class_prob, [-1, H*W, B, C])

        adjusted_object = tf.sigmoid(nd_net_out[:, :, :, :, C])
        adjusted_object = tf.reshape(adjusted_object, [-1, H*W, B, 1])

        nd_coords_predict = nd_net_out[:, :, :, :, C+1:]
        nd_coords_predict = tf.reshape(nd_coords_predict, [-1, H*W, B, 4])
        # sigmoid to make sure nd_coords_predict[:, :, :, 0:2] is positive
        adjusted_coords_xy = tf.sigmoid(nd_coords_predict[:, :, :, 0:2])
        # nd_coords_predict[:, :, :, 2:4] predicts (t_w, t_h), adjusted_coords_wh predicts (sqrt(w/image_w), sqrt(h/image_h))
        normalized_anchors = np.reshape(config.anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2])
        adjusted_coords_wh = tf.sqrt(tf.exp(nd_coords_predict[:, :, :, 2:4]) * normalized_anchors)
        adjusted_nd_coords_predict = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        #adjusted_net_out = tf.concat([adjusted_class_prob, adjusted_object, adjusted_coords_xy, adjusted_coords_wh], 3)
        adjusted_net_out = tf.concat([adjusted_class_prob, adjusted_object, adjusted_nd_coords_predict], 3)

        """
        confidence weight, nd_confids get the bounding box that has the highest iou.
        """
        best_box = find_best_box(adjusted_nd_coords_predict, nd_coords)
        nd_confids = best_box * nd_object_proids

        nd_confid_weight = noobj_scale * (1.0 - nd_confids) + obj_scale * nd_confids

        """
        class weight, multiply nd_confids to only penalizes the bounding box has the highest iou.
        """
        bounding_box_class = tf.concat(C * [tf.expand_dims(nd_confids, -1)], 3)
        # TODO: the following two statements are equivalent???
        #nd_class_weight = class_scale * nd_class_proids * bounding_box_class
        nd_class_weight = class_scale * bounding_box_class

        """
        coordinate weight, multiply nd_confids to only penalizes the bounding box has the highest iou.
        """
        bounding_box_coord = tf.concat(4 * [tf.expand_dims(nd_confids, -1)], 3)
        nd_coord_weight = coord_scale * bounding_box_coord

        true = tf.concat([nd_class_probs, tf.expand_dims(nd_confids, 3), nd_coords], 3)
        weights = tf.concat([nd_class_weight, tf.expand_dims(nd_confid_weight, 3), nd_coord_weight], 3)
        weighted_square_error = weights * ((adjusted_net_out - true) ** 2)
        weighted_square_error = tf.reshape(weighted_square_error, [-1, H*W * B * (C + 1 + 4)])
        loss_op = 0.5 * tf.reduce_mean(tf.reduce_sum(weighted_square_error, 1), name="loss")

        return loss_op


def flat(x):
    return tf.layers.flatten(x)
