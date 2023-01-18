import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers.core import Flatten, Dense, Dropout


SS, C, B = config.S * config.S, config.C, config.B


class BaseNetwork(object):
    """
    def __init__(self):
        print("FastYolo")
    """

    def forward(self, image_batch, data_format, input_shape, dropout_keep_prob=tf.constant(0.0, dtype=tf.float32)):
        # 9 conv layers + 3 fc layers
        model = Sequential()
        model.add(Convolution2D(16, kernel_size=(3, 3), input_shape=input_shape, padding='same', data_format=data_format))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format))

        for i in range(5):
            model.add(Convolution2D(2 ** (5 + i), kernel_size=(3, 3), padding='same', data_format=data_format))
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format))

        for i in range(3):
            model.add(Convolution2D(1024, kernel_size=(3, 3), padding='same', data_format=data_format))
            model.add(LeakyReLU(alpha=0.1))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dropout(rate=1 - dropout_keep_prob))

        model.add(Dense(SS * (C + B * 5)))

        # model.summary()

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out