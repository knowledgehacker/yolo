# -*- coding: utf-8 -*-

import config
from utils.compose import ConvBatchLReLu, ConvBatchLReLu_loop, Conv, space_to_depth_x2
from net.darknet19 import DarkNet19

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Lambda, concatenate


B = config.B
C = config.C


def depth_concat(x, data_format):
    axis = -1
    if data_format == "channels_first":
        axis = 1

    return concatenate(x, axis=axis)
    #return tf.concat(x, axis=axis)


# Use pretrained weights from classification model with DarkNet19 pretrained on ImageNet dataset.
class Small(object):
    def __init__(self):
        print("small")
        self.net = DarkNet19()

    def build(self, input_image, data_format, dropout_keep_prob, trainable=True):
        net_output = self.net.build(input_image, data_format, dropout_keep_prob, trainable)
        pretrained_model = Model(inputs=input_image, outputs=net_output)
        #pretrained_model.summary()
        # use bias if conv layer not followed by batch normalization
        #pretrained_model.load_weights("data/weights/%s.h5" % config.pt_net, by_name=True, skip_mismatch=True)
        #pretrained_model.load_weights("data/weights/%s.h5" % config.pt_net)

        layer_13 = pretrained_model.get_layer("relu_13")
        layer_13_output = layer_13.output

        layer_18 = pretrained_model.get_layer("relu_18")
        layer_18_output = layer_18.output

        padding_mode = 'same'

        # Layer 19 - 20
        convs = [(1024, 3, 1), (1024, 3, 1)]
        x = ConvBatchLReLu_loop(layer_18_output, convs, padding_mode, data_format, 19, trainable) # (?, 1024, 13, 13) in NCHW?

        # reorg layer
        """
        reorg_layer reorganizes the output from conv13 as the shape of conv20.
        Then the second route layer will concat the two output together. Its function is mostly shape transform.

        tf.space_to_depth converts conv13(?, 512, 26, 26) to (?, 512 * block_size * block_size, 26 / block_size, 26 / block_size).
        to be able to concatenate with conv20(?, 1024, 13, 13), block_size = 2, that is what space_to_depth_x2 does.
        """
        passthrough = Lambda(space_to_depth_x2, arguments={"data_format": data_format, "name": "reorg"})(layer_13_output)
        #passthrough = space_to_depth_x2(layer_13_output, data_format, "reorg")  # (?, 2048, 13, 13) in NCHW?

        x = depth_concat([passthrough, x], data_format)

        # Layer 21
        x = ConvBatchLReLu(x, 1024, 3, 1, padding_mode, data_format, 21, trainable)

        # layer 22, use bias
        output = Conv(x, B * (C + 1 + 4), 1, 1, padding_mode, data_format, 22, trainable)

        return output, pretrained_model, pretrained_model.get_layer("conv_13").weights
