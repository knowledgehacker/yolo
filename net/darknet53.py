# -*- coding: utf-8 -*-

import config
from utils.compose import conv2d, dbl, res_block, identity

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

B = config.B
C = config.C


class DarkNet53(object):
    def __init__(self):
        print("darknet53")

    def build(self, input_image, data_format, trainable=True):
        # first conv2d layers
        net = dbl(input_image, 32, 3, 1, data_format, trainable=trainable)

        """ route_1 """
        # res1 = dbl + res_block * 1
        net = dbl(net, 64, 3, 2, data_format, trainable=trainable)
        net = res_block(net, 32, data_format, trainable)

        # res2 = dbl + res_block * 2
        net = dbl(net, 128, 3, 2, data_format, trainable=trainable)
        for i in range(2):
            net = res_block(net, 64, data_format, trainable)

        # res8 = dbl + res_block * 8
        net = dbl(net, 256, 3, 2, data_format, trainable=trainable)
        for i in range(8):
            net = res_block(net, 128, data_format, trainable)

        route_1 = identity(net, name='route_1', trainable=trainable)

        """ route_2 """
        # res8 = res_block * 8
        net = dbl(route_1, 512, 3, 2, data_format, trainable=trainable)
        for i in range(8):
            net = res_block(net, 256, data_format, trainable)

        route_2 = identity(net, name='route_2', trainable=trainable)

        """ route_3 """
        # res4 = dbl + res_block * 4
        net = dbl(route_2, 1024, 3, 2, data_format, trainable=trainable)
        for i in range(4):
            net = res_block(net, 512, data_format, trainable)

        route_3 = identity(net, name='route_3', trainable=trainable)

        output = conv2d(route_3, 1000, 1, 1, data_format, use_bias=True, trainable=trainable)

        return output
