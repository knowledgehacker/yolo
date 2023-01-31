# -*- coding: utf-8 -*-

from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, LeakyReLU


def ConvBatchLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable):
    x = Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
               padding=padding_mode, data_format=data_format, name='conv_{}'.format(index),
               use_bias=False, trainable=trainable)(x)
    bn_axis = -1
    if data_format == "channels_first":
        bn_axis = 1
    x = BatchNormalization(axis=bn_axis, name='norm_{}'.format(index), trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1, name="relu_{}".format(index))(x)

    return x


def ConvBatchLReLu_loop(x, convs, padding_mode, data_format, index, trainable):
    for (filter, size, stride) in convs:
        x = ConvBatchLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable)
        index += 1

    return x


def ConvLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable):
    x = Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
               padding=padding_mode, data_format=data_format, name='conv_{}'.format(index),
               use_bias=True, trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1, name="relu_{}".format(index))(x)

    return x


def Conv(x, filter, size, stride, padding_mode, data_format, index, trainable):
    return Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
                  padding=padding_mode, data_format=data_format, name='conv_{}'.format(index),
                  use_bias=True, trainable=trainable)(x)


def MaxPool(x, size, data_format, name):
    return MaxPool2D(pool_size=(size, size), data_format=data_format, name=name)(x)


def FullyConnRelu(x, out_size):
    x = Dense(out_size, name="fc_%s" % out_size)(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x

