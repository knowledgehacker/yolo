# -*- coding: utf-8 -*-

import config
from v1.small import Small
from utils.iou import find_best_box

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
        output, layer_weights = self.net.build(input_image, data_format, dropout_keep_prob, trainable)

        model = Model(input_image, output)
        # model.summary()

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out, layer_weights

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

        """
        class weight, class is in grid unit instead of bounding box unit,
        so we don't need to multiply nd_confids here.
        """
        nd_class_weight = class_scale * nd_class_proids

        """
        confidence weight, nd_confids get the bounding box that has the highest iou.

        TODO: adjust nd_object_proids during train on the fly??? yes, mask the boxes except the highest iou one with 0.0.
        as to B=2 bounding boxes, the last dimension of best_box looks like [0.0, 1.0] or [1.0, 0.0].

        If (box) confidence score reflects how likely the box contains an object(objectness) and how accurate is the bounding box,
        that is, P((box) confidence) = P(object) * IOU, shouldn't we multiply the confidence part of net_out with IOU? No, in loss,
        the confidence part refers to objectness (P(object)), the class part refers to class probability (P(class | object)).
        Similarly, (class) confidence score is defined as P((class) confidence) = P(class) * IOU.
        I think, both (box) and (class) confidence scores serve as easy illustration purpose in the article, we can safely ignore them.

        When prediction, by multiply P(class | object) * P(object), we get class probability (P(class)),
        filter out predictions P(class) < config.THRESHOLD, then get the final bounding box for each remaining predictions in NMS by IOU.
        """
        nd_coords_predict = tf.reshape(net_out[:, H*W * (C + B):], shape=[-1, H*W, B, 4])
        best_box = find_best_box(nd_coords_predict, nd_coords)
        nd_confids = best_box * nd_object_proids

        nd_confid_weight = noobj_scale * (1.0 - nd_confids) + obj_scale * nd_confids

        """
        coordinate weight, we need to multiply nd_confids here,
        since we only penalizes the bounding box has the highest iou.
        """
        bounding_box_coord = tf.concat(4 * [tf.expand_dims(nd_confids, -1)], 3)
        nd_coord_weight = coord_scale * bounding_box_coord

        true = tf.concat([flat(nd_class_probs), flat(nd_confids), flat(nd_coords)], 1)
        weights = tf.concat([flat(nd_class_weight), flat(nd_confid_weight), flat(nd_coord_weight)], 1)
        weighted_square_error = weights * ((net_out - true) ** 2)
        loss_op = 0.5 * tf.reduce_mean(tf.reduce_sum(weighted_square_error, 1), name="loss")

        return loss_op


def flat(x):
    return tf.layers.flatten(x)
