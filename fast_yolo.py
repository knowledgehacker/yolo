# -*- coding: utf-8 -*-
import config
from base_network import BaseNetwork
from utils.iou import find_best_box

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


SS, C, B = config.S * config.S, config.C, config.B


class FastYolo(object):
    def __init__(self):
        #print("FastYolo")
        self.net = BaseNetwork()

    def forward(self, image_batch, data_format, input_shape, dropout_keep_prob=tf.constant(0.0, dtype=tf.float32)):
        net_out = self.net.forward(image_batch, data_format, input_shape, dropout_keep_prob)

        return net_out

    """
    TODO: Learning rate Tuning
    Throughout training we use a batch size of 64, a momentum of 0.9 and a decay of 0.0005.
    Our learning rate schedule is as follows: For the first epochs we slowly raise the learning rate from 10^-3 to 10^-2.
    If we start at a high learning rate our model often diverges due to unstable gradients.
    We continue training with 10^-2 for 75 epochs, then 10^-3 for 30 epochs, and finally 10^-4 for 30 epochs.
    """
    #def opt(self, net_out, class_probs, class_proids, object_proids, coords):
    def opt(self, net_out, nd_class_probs, nd_class_proids, nd_object_proids, nd_coords):
        # parameters
        coord_scale = config.coord_scale
        #conf_scale = config.object_scale # object scale set to default value 1.0
        noobj_scale = config.noobject_scale
        class_scale = config.class_scale
        print('scales  = {}'.format([coord_scale, noobj_scale, class_scale]))
        """
        # only object_proids will be adjusted
        nd_class_probs = tf.reshape(class_probs, shape=[-1, SS, C])
        nd_class_proids = tf.reshape(class_proids, shape=[-1, SS, C])
        nd_object_proids = tf.reshape(object_proids, shape=[-1, SS, B])
        nd_coords = tf.reshape(coords, shape=[-1, SS, B, 4])
        """

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

        TODO: adjust object_proids during train on the fly??? yes, mask the boxes except the highest iou one with 0.0.
        as to B=2 bounding boxes, the last dimension of best_box looks like [0.0, 1.0] or [1.0, 0.0].
        """
        nd_coords_predict = tf.reshape(net_out[:, SS * (C + B):], shape=[-1, SS, B, 4])
        best_box = find_best_box(nd_coords_predict, nd_coords)
        nd_confids = best_box * nd_object_proids
        """
        print("--- nd_object_proids.shape")
        print(nd_object_proids.shape)
        print("--- best_box shape")
        print(best_box.shape)
        print("--- nd_confids.shape")
        print(nd_confids.shape)
        """
        nd_confid_weight = noobj_scale * (1.0 - nd_confids) + nd_confids

        """
        coordinate weight, we need to multiply nd_confids here,
        since we only penalizes the bounding box has the highest iou.
        """
        bounding_box_coord = tf.concat(4 * [tf.expand_dims(nd_confids, -1)], 3)
        """
        print("--- bounding_box_coord")
        print(bounding_box_coord.shape)
        """
        nd_coord_weight = coord_scale * bounding_box_coord

        # reconstruct label with adjusted confs. Q: nd_object_proids or nd_confids in true???
        true = tf.concat([flat(nd_class_probs), flat(nd_confids), flat(nd_coords)], 1)
        weights = tf.concat([flat(nd_class_weight), flat(nd_confid_weight), flat(nd_coord_weight)], 1)
        weighted_square_error = weights * ((net_out - true) ** 2)
        loss_op = 0.5 * tf.reduce_mean(tf.reduce_sum(weighted_square_error, 1), name="loss")

        return loss_op


def flat(x):
    return tf.layers.flatten(x)
