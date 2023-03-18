# coding: utf-8

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, concatenate, Lambda
from keras.regularizers import l2

"""
def pad2d(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
    return padded_inputs
"""

def conv2d(inputs, filter, size, stride, data_format, use_bias=False, trainable=True):
    """
    if stride > 1:
        # Darknet uses left and top padding instead of 'same' mode
        #inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        inputs = pad2d(inputs, size)
    """

    #padding = ('SAME' if stride == 1 else 'VALID')
    padding = 'SAME'

    x = Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
               padding=padding, data_format=data_format, use_bias=use_bias, kernel_regularizer=l2(5e-4),
               trainable=trainable)(inputs)

    return x


def dbl(inputs, filter, size, stride, data_format, trainable=True):
    x = conv2d(inputs, filter, size, stride, data_format, trainable=trainable)

    bn_axis = -1
    if data_format == "channels_first":
        bn_axis = 1
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-5, trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def res_block(inputs, filter, data_format, trainable):
    shortcut = inputs
    net = dbl(inputs, filter * 1, 1, 1, data_format, trainable=trainable)
    net = dbl(net, filter * 2, 3, 1, data_format, trainable=trainable)

    net = net + shortcut

    return net


def yolo_block(inputs, filter, data_format, trainable):
    net = dbl(inputs, filter * 1, 1, 1, data_format, trainable=trainable)
    net = dbl(net, filter * 2, 3, 1, data_format, trainable=trainable)
    net = dbl(net, filter * 1, 1, 1, data_format, trainable=trainable)
    net = dbl(net, filter * 2, 3, 1, data_format, trainable=trainable)
    net = dbl(net, filter * 1, 1, 1, data_format, trainable=trainable)
    route = net
    net = dbl(net, filter * 2, 3, 1, data_format, trainable=trainable)
    return route, net

"""
def upsample(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')

    return inputs
"""

def upsample(inputs, stride, data_format):
    assert stride == 2, 'Only stride=2 supported.'
    upsampled = UpSampling2D(stride, data_format)(inputs)

    return upsampled


def concat(xs, data_format):
    axis = -1
    if data_format == "channels_first":
        axis = 1

    return concatenate(xs, axis=axis)


def identity(inputs, name, trainable):
    #output = Identity(name=name, trainable=trainable)(inputs)
    # workaround for tf.identity functionality in keras
    output = Lambda(lambda x: x, name=name, trainable=trainable)(inputs)

    return output


def naming(x, data_format, name):
    if data_format == 'channels_first':
        x = tf.transpose(x, [0, 2, 3, 1])
    x = tf.identity(x, name=name)

    return x
