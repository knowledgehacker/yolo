import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.layers.core import Flatten, Dense, Dropout


SS, B, C = config.S * config.S, config.B, config.C


class DarkNet(object):
    """
    def __init__(self):
        print("small")
    """

    def forward(self, image_batch, data_format, input_shape, dropout_keep_prob):
        print("data_format: %s" % data_format)
        # 24 conv + 3 fully connected layers
        padding_mode = 'same'

        model = Sequential()
        # 2 conv layers
        convs = [(64, 7, 2), (192, 3, 1)]
        for (filter, size, stride) in convs:
            model.add(Conv2D(filter, kernel_size=(size, size), strides=(stride, stride), input_shape=input_shape, padding=padding_mode, data_format=data_format))
            #model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))
            model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format))

        # 3 conv layers
        convs = [(128, 1, 1), (256, 3, 1), (256, 1, 1)]
        for (filter, size, stride) in convs:
            model.add(Conv2D(filter, kernel_size=(size, size), strides=(stride, stride), input_shape=input_shape, padding=padding_mode, data_format=data_format))
            #model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))

        # 1 conv + pooling layer
        model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape, padding=padding_mode, data_format=data_format))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format))

        # 8 conv layers
        convs = [(256, 1, 1), (512, 3, 1)]
        for i in range(4):
            for (filter, size, stride) in convs:
                model.add(Conv2D(filter, kernel_size=(size, size), strides=(stride, stride), input_shape=input_shape, padding=padding_mode, data_format=data_format))
                #model.add(BatchNormalization())
                model.add(LeakyReLU(alpha=0.1))

        # 1 conv layer
        model.add(Conv2D(512, kernel_size=(1, 1), strides=(1, 1), input_shape=input_shape, padding=padding_mode, data_format=data_format))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        # 1 conv + pooling layer
        model.add(Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape, padding=padding_mode, data_format=data_format))
        #model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), data_format=data_format))

        # 4 conv layers
        convs = [(512, 1, 1), (1024, 3, 1)]
        for i in range(2):
            for (filter, size, stride) in convs:
                model.add(Conv2D(filter, kernel_size=(size, size), strides=(stride, stride), input_shape=input_shape, padding=padding_mode, data_format=data_format))
                #model.add(BatchNormalization())
                model.add(LeakyReLU(alpha=0.1))

        # 4 conv layers
        convs = [(1024, 3, 1), (1024, 3, 2), (1024, 3, 1), (1024, 3, 1)]
        for (filter, size, stride) in convs:
            model.add(Conv2D(filter, kernel_size=(size, size), strides=(stride, stride), padding=padding_mode, data_format=data_format))
            #model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=0.1))

        # 3 fully connected layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(4096))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dropout(rate=1 - dropout_keep_prob))

        model.add(Dense(SS * (C + B * 5)))

        # model.summary()

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out