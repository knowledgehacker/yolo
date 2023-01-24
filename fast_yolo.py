# -*- coding: utf-8 -*-

import numpy as np

import config
from base_network import BaseNetwork
from utils.iou import find_best_box

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

H, W = config.H, config.W
C, B = config.C, config.B


class FastYolo(object):
    def __init__(self):
        #print("FastYolo")
        self.net = BaseNetwork()

    def forward(self, image_batch, data_format, input_shape, dropout_keep_prob=tf.constant(0.0, dtype=tf.float32)):
        net_out = self.net.forward(image_batch, data_format, input_shape, dropout_keep_prob)

        return net_out

    #def opt(self, net_out, class_probs, class_proids, object_proids, coords):
    def opt(self, net_out, nd_class_probs, nd_class_proids, nd_object_proids, nd_coords):
        # parameters
        coord_scale = config.coord_scale
        #conf_scale = config.object_scale # object scale set to default value 1.0
        noobj_scale = config.noobject_scale
        class_scale = config.class_scale
        print('scales  = {}'.format([coord_scale, noobj_scale, class_scale]))

        """
        The following code calculate weight vector of three parts: class, coordinate, confidence,
        initial weight vector is passed from input, we adjust it by scale parameters and best box with highest iou.
        take care of the weight terms, construct indicator matrix(which grids the objects in, which ones not in).
        """

        nd_net_out = tf.reshape(net_out, [-1, H, W, B, (C + 1 + 4)])

        nd_coords_predict = nd_net_out[:, :, :, :, C + 1:C + 5]
        nd_coords_predict = tf.reshape(nd_coords_predict, [-1, H * W, B, 4])
        adjusted_coords_xy = tf.sigmoid(nd_coords_predict[:, :, :, 0:2])
        adjusted_coords_wh = tf.sqrt(
            tf.exp(nd_coords_predict[:, :, :, 2:4]) * np.reshape(config.anchors, [1, 1, B, 2]) / np.reshape([W, H], [1, 1, 1, 2]))
        nd_coords_predict = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_c = tf.sigmoid(nd_net_out[:, :, :, :, C])
        adjusted_c = tf.reshape(adjusted_c, [-1, H * W, B, 1])

        adjusted_prob = tf.nn.softmax(nd_net_out[:, :, :, :, :C])
        adjusted_prob = tf.reshape(adjusted_prob, [-1, H * W, B, C])

        adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

        """
        confidence weight, nd_confids get the bounding box that has the highest iou.
        """
        best_box = find_best_box(nd_coords_predict, nd_coords)
        nd_confids = best_box * nd_object_proids

        nd_confid_weight = noobj_scale * (1.0 - nd_confids) + nd_confids

        """
        class weight, multiply nd_confids to only penalizes the bounding box has the highest iou.
        """
        bounding_box_class = tf.concat(C * [tf.expand_dims(nd_confids, -1)], 3)
        nd_class_weight = class_scale * nd_class_proids * bounding_box_class

        """
        coordinate weight, multiply nd_confids to only penalizes the bounding box has the highest iou.
        """
        bounding_box_coord = tf.concat(4 * [tf.expand_dims(nd_confids, -1)], 3)
        nd_coord_weight = coord_scale * bounding_box_coord

        # reconstruct label with adjusted confs. Q: nd_object_proids or nd_confids in true???
        true = tf.concat([nd_class_probs, tf.expand_dims(nd_confids, 3), nd_coords], 3)
        weights = tf.concat([nd_class_weight, tf.expand_dims(nd_confid_weight, 3), nd_coord_weight], 3)
        weighted_square_error = weights * ((adjusted_net_out - true) ** 2)
        weighted_square_error = tf.reshape(weighted_square_error, [-1, H * W * B * (C + 1 + 4)])
        loss_op = 0.5 * tf.reduce_mean(tf.reduce_sum(weighted_square_error, 1), name="loss")

        return loss_op


def flat(x):
    return tf.layers.flatten(x)
