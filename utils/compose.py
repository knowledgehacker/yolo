# -*- coding: utf-8 -*-

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, LeakyReLU, concatenate
from keras.regularizers import l2


def ConvBatchLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable):
    x = Conv(x, filter, size, stride, padding_mode, data_format, index, use_bias=False, trainable=trainable)
    """
    x = Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
               padding=padding_mode, data_format=data_format, name='conv_{}'.format(index),
               use_bias=False, trainable=trainable)(x)
    """
    bn_axis = -1
    if data_format == "channels_first":
        bn_axis = 1
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-5, name='norm_{}'.format(index), trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1, name="relu_{}".format(index))(x)

    return x


def ConvBatchLReLu_loop(x, convs, padding_mode, data_format, index, trainable):
    for (filter, size, stride) in convs:
        x = ConvBatchLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable)
        index += 1

    return x


def ConvLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable):
    x = Conv(x, filter, size, stride, padding_mode, data_format, index, use_bias=True, trainable=trainable)
    """
    x = Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
               padding=padding_mode, data_format=data_format, name='conv_{}'.format(index),
               use_bias=True, trainable=trainable)(x)
    """
    x = LeakyReLU(alpha=0.1, name="relu_{}".format(index))(x)

    return x


def Conv(x, filter, size, stride, padding_mode, data_format, index, use_bias=True, trainable=True):
    return Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
                  padding=padding_mode, data_format=data_format, name='conv_{}'.format(index),
                  use_bias=use_bias, kernel_regularizer=l2(5e-4), trainable=trainable)(x)


def MaxPool(x, size, padding_mode, data_format, name):
    return MaxPool2D(pool_size=(size, size), padding=padding_mode, data_format=data_format, name=name)(x)


def FullyConnRelu(x, out_size):
    x = Dense(out_size, name="fc_%s" % out_size)(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x, data_format, name):
    if data_format == "channels_first":
        x = tf.transpose(x, [0, 2, 3, 1])

    x = tf.space_to_depth(x, block_size=2, name=name)

    if data_format == "channels_first":
        x = tf.transpose(x, [0, 3, 1, 2])

    return x
    #return tf.space_to_depth(x, block_size=2, data_format=df, name=name)


def depth_concat(x, data_format):
    axis = -1
    if data_format == "channels_first":
        axis = 1

    return concatenate(x, axis=axis)
    #return tf.concat(x, axis=axis)
