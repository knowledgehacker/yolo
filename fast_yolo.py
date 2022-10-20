# -*- coding: utf-8 -*-
import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import keras
from keras import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers.core import Flatten, Dense, Dropout

from utils.iou import calc_best_box_iou


SS, C, B = config.S * config.S, config.C, config.B


class FastYolo(object):
    def __init__(self):
        print("FastYolo")

    def forward(self, image_batch, data_format, input_shape, dropout_keep_prob=tf.constant(0.0, dtype=tf.float32)):
        # 9 conv layers + 3 fc layers
        model = Sequential()
        model.add(Convolution2D(16, kernel_size=(3, 3), input_shape=input_shape, padding='same', data_format=data_format))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format))

        for i in range(5):
            model.add(Convolution2D(2 ** (5 + i), kernel_size=(3, 3), padding='same', data_format=data_format))
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format))

        for i in range(3):
            model.add(Convolution2D(1024, kernel_size=(3, 3), padding='same', data_format=data_format))
            model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dropout(rate=1 - dropout_keep_prob))

        model.add(Dense(SS * (C + B * 5)))

        # model.summary()

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out

    def opt(self, net_out, probs_true, proids, confs_true, coords_true):
        # parameters
        prob_scale = config.class_scale
        conf_scale = config.object_scale
        noobj_scale = config.noobject_scale
        coord_scale = config.coord_scale
        print('scales  = {}'.format([prob_scale, conf_scale, noobj_scale, coord_scale]))

        # only confs will be adjusted
        nd_probs_true = tf.reshape(probs_true, shape=[-1, SS, C])
        nd_proids = tf.reshape(proids, shape=[-1, SS, C])
        nd_confs_true = tf.reshape(confs_true, shape=[-1, SS, B])
        nd_coords_true = tf.reshape(coords_true, shape=[-1, SS, B, 4])

        # calculate iou of predicted and true bounding boxes, get the best predicted one, adjust confs_true
        # TODO: adjust confs_true during train on the fly???
        nd_coords_predict = tf.reshape(net_out[:, SS * (C + B):], shape=[-1, SS, B, 4])
        best_boxes_iou = calc_best_box_iou(nd_coords_predict, nd_coords_true)
        print("--- best_boxes_iou shape")
        print(best_boxes_iou.shape)
        # * is equivalent to tf.multiply
        nd_confs = best_boxes_iou * nd_confs_true

        # calculate loss between predicted tensor 'net_out' and ground truth tensor 'true'
        # take care of the weight terms, construct indicator matrix(which grids the objects in, which ones not in).
        # TODO: why object confidence and coordinate terms in loss formula need to multiply with confs???
        nd_confid = noobj_scale * (1.0 - nd_confs) + conf_scale * nd_confs
        # TODO: weight_coo???
        weight_coo = tf.concat(4 * [tf.expand_dims(nd_confs, -1)], 2)
        nd_coordid = coord_scale * weight_coo
        nd_proid = prob_scale * nd_proids

        # reconstruct label with adjusted confs
        true = tf.concat([tf.layers.flatten(nd_probs_true), tf.layers.flatten(nd_confs), tf.layers.flatten(nd_coords_true)], 1)
        weights = tf.concat([tf.layers.flatten(nd_proid), tf.layers.flatten(nd_confid), tf.layers.flatten(nd_coordid)], 1)
        print("--- weights shape")
        print(weights.shape)

        #loss = tf.pow(net_out - true, 2)
        loss = (net_out - true) ** 2
        loss = loss * weights
        loss = tf.reduce_sum(loss, 1)
        loss_op = 0.5 * tf.reduce_mean(loss, name="loss")

        # construct train_op
        train_op = tf.train.RMSPropOptimizer(learning_rate=config.LEARNING_RATE).minimize(loss_op)

        return loss_op, train_op
