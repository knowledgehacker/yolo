# coding: utf-8

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D


def pad2d(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
    return padded_inputs


def conv2d(inputs, filter, size, stride, data_format, trainable=True):
    """
    if stride > 1:
        # Darknet uses left and top padding instead of 'same' mode
        #inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        inputs = pad2d(inputs, size)
    """

    #padding = ('SAME' if stride == 1 else 'VALID')
    padding = 'SAME'

    x = Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
               padding=padding, data_format=data_format, use_bias=False,
               trainable=trainable)(inputs)

    return x


def dbl(inputs, filter, size, stride, data_format, trainable=True):
    x = conv2d(inputs, filter, size, stride, data_format, trainable)

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
def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')

    return inputs
"""

def upsample_layer(inputs, stride, data_format):
    assert stride == 2, 'Only stride=2 supported.'
    upsampled = UpSampling2D(stride, data_format)(inputs)

    return upsampled


def concatenate(x, y, data_format):
    axis = -1
    if data_format == 'channels_first':
        axis = 1

    return tf.concat([x, y], axis=axis)

