# -*- coding: utf-8 -*-

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, LeakyReLU, concatenate

H, W = config.H, config.W
B = config.B
C = config.C


# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x, data_format):
    df = "NHWC"
    if data_format == "channels_first":
        df = "NCHW"

    return tf.space_to_depth(x, block_size=2, data_format=df)


def ConvBatchLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable):
    x = Conv2D(filter, kernel_size=(size, size), strides=(stride, stride),
               padding=padding_mode, data_format=data_format, name='conv_{}'.format(index),
               use_bias=False, trainable=trainable)(x)
    x = BatchNormalization(name='norm_{}'.format(index), trainable=trainable)(x)
    x = LeakyReLU(alpha=0.1)(x)

    #return (x)
    return x


def ConvBatchLReLu_loop(x, convs, padding_mode, data_format, index, trainable):
    for (filter, size, stride) in convs:
        x = ConvBatchLReLu(x, filter, size, stride, padding_mode, data_format, index, trainable)
        index += 1

    #return (x)
    return x


class DarkNet(object):
    def __init__(self):
        print("small")

    def forward(self, image_batch, data_format, input_shape, dropout_keep_prob, trainable=True):
        padding_mode = 'same'

        input_image = Input(shape=input_shape, name="input_image")
        # Layer 1
        x = ConvBatchLReLu(input_image, 32, 3, 1, padding_mode, data_format, 1, trainable)
        x = MaxPool2D(pool_size=(2, 2), name="maxpool1_416to208")(x)
        tf.nn.conv2d

        # Layer 2
        x = ConvBatchLReLu(x, 64, 3, 1, padding_mode, data_format, 2, trainable)
        x = MaxPool2D(pool_size=(2, 2), name="maxpool1_208to104")(x)

        # Layer 3 - 5
        convs = [(128, 3, 1), (64, 1, 1), (128, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 3, trainable)
        x = MaxPool2D(pool_size=(2, 2), name="maxpool1_104to52")(x)

        # Layer 6 - 8
        convs = [(256, 3, 1), (128, 1, 1), (256, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 6, trainable)
        x = MaxPool2D(pool_size=(2, 2), name="maxpool1_52to26")(x)

        # Layer 9 - 13
        convs = [(512, 3, 1), (256, 1, 1), (512, 3, 1), (256, 1, 1), (512, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 9, trainable)

        skip_connection = x
        x = MaxPool2D(pool_size=(2, 2), name="maxpool1_26to13")(x)

        # Layer 14 - 20
        convs = [(1024, 3, 1), (512, 1, 1), (1024, 3, 1), (512, 1, 1), (1024, 3, 1), (1024, 3, 1), (1024, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 14, trainable)

        # TODO: work???
        # Layer 21, reorg
        """
        reorg_layer reorganize the output from layer 17 as the shape of the layer 25's output.
        Then the second route layer will concat the two output together. Its function is mostly shape transform.
        """
        skip_connection = ConvBatchLReLu(skip_connection, 64, 1, 1, padding_mode, data_format, 21, trainable)
        skip_connection = space_to_depth_x2(skip_connection, data_format)

        # TODO: work???
        x = concatenate([skip_connection, x], axis=-1)

        # Layer 22
        x = ConvBatchLReLu(x, 1024, 3, 1, padding_mode, data_format, 22, trainable)

        # Layer 23
        x = Conv2D(filters=B * (C + 1 + 4), kernel_size=(1, 1), strides=(1, 1), padding=padding_mode, name='conv_23')(x)

        model = Model(input_image, x)

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out
