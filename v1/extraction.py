# -*- coding: utf-8 -*-

import config
from utils.compose import ConvBatchLReLu, ConvBatchLReLu_loop, ConvLReLu, MaxPool

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

H, W = config.H, config.W
B = config.B
C = config.C


class Extraction(object):
    """
    def __init__(self):
        print("extraction")
    """

    def build(self, input_image, data_format, dropout_keep_prob, trainable=True):
        # 24 conv + 3 fully connected layers
        padding_mode = 'same'

        # layer 1
        x = ConvBatchLReLu(input_image, 64, 7, 2, padding_mode, data_format, 1, trainable)
        x = MaxPool(x, 2, data_format, name="maxpool1_448to224")

        # layer 2
        x = ConvBatchLReLu(x, 192, 3, 1, padding_mode, data_format, 2, trainable)
        x = MaxPool(x, 2, data_format, name="maxpool1_224to112")

        # layer 3 - 6
        convs = [(128, 1, 1), (256, 3, 1), (256, 1, 1), (512, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 3, trainable)

        x = MaxPool(x, 2, data_format, name="maxpool1_112to56")

        # layer 7 - 14
        convs = [(256, 1, 1), (512, 3, 1)]
        for i in range(4):
            x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 7 + 2 * i, trainable)

        # layer 15 - 16
        convs = [(512, 1, 1), (1024, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 15, trainable)

        x = MaxPool(x, 2, data_format, name="maxpool1_56to28")

        # layer 17 - 20
        convs = [(512, 1, 1), (1024, 3, 1)]
        for i in range(2):
            x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 17 + 2 * i, trainable)

        # layer 21, use bias
        output = ConvLReLu(x, 1000, 1, 1, padding_mode, data_format, 21, trainable)

        return output
