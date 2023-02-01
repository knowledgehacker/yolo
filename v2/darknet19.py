# -*- coding: utf-8 -*-

import config
from utils.compose import ConvBatchLReLu, ConvBatchLReLu_loop, Conv, MaxPool

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

H, W = config.H, config.W
B = config.B
C = config.C


"""
DarNet for yolo v2 are as follows.
Note that the 1st link takes pooling layer into account(index starts with 0), thus the 1st and 2nd links are equivalent.
https://github.com/AlexeyAB/darknet/issues/279#issuecomment-397248821
https://ethereon.github.io/netscope/#/gist/d08a41711e48cf111e330827b1279c31
"""
class DarkNet19(object):
    def __init__(self):
        print("darknet19")

    def build(self, input_image, data_format, dropout_keep_prob, trainable=True):
        padding_mode = 'same'

        # Layer 1
        x = ConvBatchLReLu(input_image, 32, 3, 1, padding_mode, data_format, 1, trainable)
        x = MaxPool(x, 2, padding_mode, data_format, name="maxpool1_416to208")

        # Layer 2
        x = ConvBatchLReLu(x, 64, 3, 1, padding_mode, data_format, 2, trainable)
        x = MaxPool(x, 2, padding_mode, data_format, name="maxpool1_208to104")

        # Layer 3 - 5
        convs = [(128, 3, 1), (64, 1, 1), (128, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 3, trainable)
        x = MaxPool(x, 2, padding_mode, data_format, name="maxpool1_104to52")

        # Layer 6 - 8
        convs = [(256, 3, 1), (128, 1, 1), (256, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 6, trainable)
        x = MaxPool(x, 2, padding_mode, data_format, name="maxpool1_52to26")

        # Layer 9 - 13
        convs = [(512, 3, 1), (256, 1, 1), (512, 3, 1), (256, 1, 1), (512, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 9, trainable)

        #passthrough = x

        x = MaxPool(x, 2, padding_mode, data_format, name="maxpool1_26to13")

        # Layer 14 - 18
        convs = [(1024, 3, 1), (512, 1, 1), (1024, 3, 1), (512, 1, 1), (1024, 3, 1)]
        x = ConvBatchLReLu_loop(x, convs, padding_mode, data_format, 14, trainable) # (?, 1024, 13, 13) in NCHW?

        output = Conv(x, 1000, 1, 1, padding_mode, data_format, 19, trainable)

        #return passthrough, x
        return output


