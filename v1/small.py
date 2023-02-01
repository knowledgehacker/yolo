# -*- coding: utf-8 -*-

import config
from utils.compose import ConvBatchLReLu_loop, FullyConnRelu
from v1.extraction import Extraction

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout

H, W = config.H, config.W
B = config.B
C = config.C


# Use pretrained weights from classification model with Extraction pretrained on ImageNet dataset.
class Small(object):
    def __init__(self):
        print("small")
        self.net = Extraction()

    def build(self, input_image, data_format, dropout_keep_prob, trainable=True):

        darknet_output = self.net.build(input_image, data_format, dropout_keep_prob, trainable)
        pretrained_model = Model(input_image, darknet_output)
        #pretrained_model.summary()
        # use bias if conv layer not followed by batch normalization
        #pretrained_model.load_weights("data/weights/%s.h5" % config.pt_net, by_name=True, skip_mismatch=True)
        pretrained_model.load_weights("data/weights/%s.h5" % config.pt_net)

        layer_20 = pretrained_model.get_layer("relu_20")
        layer_20_weights = layer_20.weights
        print(layer_20_weights)
        layer_20_output = layer_20.output

        padding_mode = 'same'

        # layer 21 - 24
        convs = [(1024, 3, 1), (1024, 3, 2), (1024, 3, 1), (1024, 3, 1)]
        x = ConvBatchLReLu_loop(layer_20_output, convs, padding_mode, data_format, 21, trainable)

        # 3 fully connected layers
        x = Flatten()(x)
        x = FullyConnRelu(x, 512)
        x = FullyConnRelu(x, 4096)

        x = Dropout(rate=1 - dropout_keep_prob)(x)

        output = Dense(H*W * (C + B * 5), name="output")(x)

        return output, layer_20_weights
