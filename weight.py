# -*- coding: utf-8 -*-

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import h5py

import config
from v1.extraction import Extraction
from v2.darknet19 import DarkNet19

from keras.models import Model
from keras.layers import Input


def get_weight_by_index(model_weight_file, pretrained_model, index):
    pretrained_model.load_weights(model_weight_file, by_name=True)

    w = pretrained_model.get_weights()
    conv13 = w[index]
    print(np.shape(conv13))
    print(conv13)


def get_weight_by_layer(model_weight_file, pretrained_model):
    conv13_layer = pretrained_model.get_layer("conv_13")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        pretrained_model.load_weights(model_weight_file, by_name=True)
        print(sess.run(conv13_layer.weights))


"""
TODO: it's strange that get by index 40 gets the weights equivalent to "conv_9" in h5py file.
I don't know why, since by counting conv, batch, relu, maxpool layers, I think index of layer "conv_13" is 40.
by layer gets the weights equivalent to "conv_13" in h5py file, 
but we should call load_weights after tf.global_variables_initializer().run() avoid weights being re-initialized.
"""
def verify():
    input_image = Input(shape=config.input_shape, name="input_image")
    if config.VERSION == 'v1':
        net = Extraction()
        model_weight_file = "data/weights/extraction.h5"
    elif config.VERSION == 'v2':
        net = DarkNet19()
        model_weight_file = "data/weights/darknet19.h5"


    wf = h5py.File(model_weight_file)
    for key in wf.keys():
        print(key)
        if key.startswith("conv"):
            print(np.shape(wf[key][key]["kernel:0"]))
            print(wf[key][key]["kernel:0"][0][0][0][:5])
        """
        if key == "conv_13":
            conv13 = wf[key][key]["kernel:0"][:]
            print(np.shape(conv13))
            print(conv13)
        """
        """
        for subkey in wf[key].keys():
            print("\t", subkey)
            if subkey is not None:
                for subsubkey in wf[key][subkey].keys():
                    print("\t\t", subsubkey)
        """

    net_output = net.build(input_image, config.data_format, config.TRAIN_KEEP_PROB)
    pretrained_model = Model(inputs=input_image, outputs=net_output)
    pretrained_model.summary()

    #print("--- before load_weights")
    #get_weight_by_index(pretrained_model, 40)
    #get_weight_by_layer(pretrained_model)
    print("--- after load_weights")
    #get_weight_by_index(pretrained_model, 40)
    get_weight_by_layer(model_weight_file, pretrained_model)
    get_weight_by_layer(model_weight_file, pretrained_model)


verify()
