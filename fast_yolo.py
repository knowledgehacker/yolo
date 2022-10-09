# -*- coding: utf-8 -*-
import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import keras
from keras import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Dropout

from utils.iou import calc_best_box_iou


class FastYolo(object):
    def __init__(self):
        print("FastYolo")

    def forward(self, image_batch, data_format, dropout_keep_prob=tf.constant(0.0, dtype=tf.float32)):
        if data_format == "channels_last":
            keras.backend.set_image_data_format(data_format)

            input_shape = (config.IMG_H, config.IMG_W, config.IMG_C)
        elif data_format == "channels_first":
            keras.backend.set_image_data_format(data_format)

            input_shape = (config.IMG_C, config.IMG_H, config.IMG_W)
        else:
            print("Unsupported data format: %s" % data_format)
            exit(-1)

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

        model.add(Dense(config.S * config.S * (config.C + config.B * 5)))

        # model.summary()

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out

    """
    # TODO: calculate mAP
    def calc_mAP(self, net_out, probs_true, confs_true, coords_true):
        with tf.name_scope("output"):
            # prediction
            SS = config.S * config.S
            C = config.C
            B = config.B

            probs_predict = tf.reshape(net_out[:, : SS * C], shape=([-1, SS, C]))
            confs_predict = tf.reshape(net_out[:, SS * C:SS * (C + B)], shape=([-1, SS, B]))
            coords_predict = tf.reshape(net_out[:, SS * (C + B):], shape=([-1, SS, B, 4]))

            # confidence(object) * iou(bounding boxes) * p(class)
            probs_predict_max = tf.reduce_max(probs_predict, [2], True) # [batch_size, SS, 1]
            # TODO: confs_predict takes iou into account already???
            final_probs_predict = confs_predict * probs_predict_max
            filtered_confs_predict = tf.greater(final_probs_predict, tf.constant(config.THRESHOLD, dtype=tf.float32))
            filtered_coords_predict = tf.equal(coords_predict, filtered_confs_predict)

            # accuracy
            correct_preds = tf.equal(tf.argmax(filtered_confs_predict, 1), confs_true)
            acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

        return preds, acc
    """

    def opt(self, net_out, probs_true, proids, confs_true, coords_true):
        # parameters
        prob_scale = config.class_scale
        conf_scale = config.object_scale
        noobj_scale = config.noobject_scale
        coord_scale = config.coord_scale
        print('scales  = {}'.format([prob_scale, conf_scale, noobj_scale, coord_scale]))

        SS, C, B = config.S * config.S, config.C, config.B

        # calculate iou of predicted and true bounding boxes, get the best predicted one, adjust confs_true
        # TODO: adjust confs_true during train on the fly???
        coords_predict = tf.reshape(net_out[:, SS * (C + B):], [-1, SS, B, 4])
        best_boxes_iou = calc_best_box_iou(coords_predict, coords_true)
        print("--- best_boxes_iou shape")
        print(best_boxes_iou.shape)
        # * is equivalent to tf.multiply
        confs = best_boxes_iou * confs_true

        # calculate loss between predicted tensor 'net_out' and ground truth tensor 'true'
        # take care of the weight terms, construct indicator matrix(which grids the objects in, which ones not in).
        # TODO: why object confidence and coordinate terms in loss formula need to multiply with confs???
        confid = noobj_scale * (1.0 - confs) + conf_scale * confs
        # TODO: weight_coo???
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 2)
        coordid = coord_scale * weight_coo
        proid = prob_scale * proids

        # reconstruct label with adjusted confs
        true = tf.concat([tf.layers.flatten(probs_true), tf.layers.flatten(confs), tf.layers.flatten(coords_true)], 1)
        weights = tf.concat([tf.layers.flatten(proid), tf.layers.flatten(confid), tf.layers.flatten(coordid)], 1)
        print("--- weights shape")
        print(weights.shape)

        loss = tf.pow(net_out - true, 2)
        loss = loss * weights
        loss = tf.reduce_sum(loss, 1)
        loss_op = 0.5 * tf.reduce_mean(loss, name="loss")

        # construct train_op
        train_op = tf.train.RMSPropOptimizer(learning_rate=config.LEARNING_RATE).minimize(loss_op)

        return loss_op, train_op
