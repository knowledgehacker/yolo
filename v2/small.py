# -*- coding: utf-8 -*-

import numpy as np

import config
from utils.compose import ConvBatchLReLu, ConvBatchLReLu_loop
from v2.darknet import DarkNet

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Conv2D, Lambda, concatenate


B = config.B
C = config.C


# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
def space_to_depth_x2(x, data_format, name):
    df = "NHWC"
    if data_format == "channels_first":
        df = "NCHW"

    return Lambda(tf.space_to_depth, arguments={"block_size": 2, "data_format": df, "name": name})(x)
    #return tf.space_to_depth(x, block_size=2, data_format=df, name=name)


def depth_concat(x, data_format):
    axis = -1
    if data_format == "channels_first":
        axis = 1

    return concatenate(x, axis=axis)
    #return tf.concat(x, axis=axis)


# Use pretrained weights from classification model with DarkNet pretrained on ImageNet dataset.
class Small(object):
    def __init__(self):
        print("small")
        self.net = DarkNet()

    def build(self, input_image, data_format, dropout_keep_prob, trainable=True):

        darknet_output = self.net.build(input_image, data_format, dropout_keep_prob, trainable)
        pretrained_model = Model(input_image, darknet_output)
        #pretrained_model.load_weights("data/weights/darknet19.h5", by_name=True)
        pretrained_model.load_weights("data/weights/darknet19.h5")
        pretrained_model.summary()
        """
        darknet19_model = load_model("data/weights/darknet19.h5")
        darknet19_topless = Model(darknet19_model.inputs, darknet19_model.layers[-1].output)
        darknet19_topless.save_weights("data/weights/darknet19_topless.h5")
        
        darknet_body = self.net.build(input_image, data_format, dropout_keep_prob, trainable)
        pretrained_model = Model(darknet_body.input, darknet_body.layers[-1].output)
        pretrained_model.load_weights("data/weights/darknet19_topless.h5")
        """
        conv13_layer = pretrained_model.get_layer("conv_13")
        conv13_weights = conv13_layer.weights
        print(conv13_weights)
        conv13_output = conv13_layer.output

        conv18_layer = pretrained_model.get_layer("conv_18")
        conv18_output = conv18_layer.output

        padding_mode = 'same'

        # Layer 19 - 20
        convs = [(1024, 3, 1), (1024, 3, 1)]
        x = ConvBatchLReLu_loop(conv18_output, convs, padding_mode, data_format, 19, trainable) # (?, 1024, 13, 13) in NCHW?

        # reorg layer
        """
        reorg_layer reorganizes the output from conv13 as the shape of conv20.
        Then the second route layer will concat the two output together. Its function is mostly shape transform.

        tf.space_to_depth converts conv13(?, 512, 26, 26) to (?, 512 * block_size * block_size, 26 / block_size, 26 / block_size).
        to be able to concatenate with conv20(?, 1024, 13, 13), block_size = 2, that is what space_to_depth_x2 does.
        """
        passthrough = space_to_depth_x2(conv13_output, data_format, "reorg")  # (?, 2048, 13, 13) in NCHW?

        x = depth_concat([passthrough, x], data_format)

        # Layer 21
        x = ConvBatchLReLu(x, 1024, 3, 1, padding_mode, data_format, 21, trainable)

        # layer 22
        output = Conv2D(filters=B * (C + 1 + 4), kernel_size=(1, 1), strides=(1, 1), padding=padding_mode, data_format=data_format, name='conv_22')(x)

        return output, conv13_weights
