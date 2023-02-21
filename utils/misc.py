# -*- coding: utf-8 -*-

import os
import time

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import config


def current_time():
    return time.strftime('%H:%M:%S', time.localtime(time.time()))


def load_files(input):
    files = []

    # os.listdir lists files and directories, not only directories in the specified directory
    for sub_dir in os.listdir(input):
        path = os.path.join(input, sub_dir)
        if os.path.isdir(os.path.join(os.getcwd(), path)):
            #print("dir path: %s" % path)
            for file in os.listdir(path):
                if not (file.startswith("._") or file.startswith(".DS_Store")):
                    files.append(os.path.join(input, file))
        else:
            #print("file path: %s" % path)
            if not (sub_dir.startswith("._") or sub_dir.startswith(".DS_Store")):
                files.append(path)

    return files


def with_prefix(prefix, op):
    return "%s/%s" % (prefix, op)
    #return op


def load_model(model_dir, model_name, outputs):
    model_filepath = "%s/%s.pb" % (model_dir, model_name)

    print("Loading model %s ..." % model_filepath)

    with tf.gfile.FastGFile(model_filepath, 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())

    g = tf.Graph()
    with g.as_default():
        tf.import_graph_def(graph_def, return_elements=outputs, name=model_name)

    print("Model %s loaded!" % model_filepath)

    return g


def save_model(sess, model_dir, model_name, outputs):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        outputs)

    model_filepath = "%s/%s.pb" % (model_dir, model_name)
    with tf.gfile.GFile(model_filepath, "wb") as fout:
        fout.write(output_graph_def.SerializeToString())


def get_optimizer(lr):
    if config.OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=config.MOMENTUM)
    elif config.OPTIMIZER == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=1e-8)
    elif config.OPTIMIZER == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=config.DECAY, momentum=config.MOMENTUM)
    elif config.OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    else:
        print("Unsupported optimizer: %s" % config.OPTIMIZER)

    return optimizer


def get_boundary(epoch, boundaries):
    idx = -1

    boundary_num = len(boundaries)
    for i in range(1, boundary_num):
        if epoch < boundaries[i]:
            idx = i - 1
            break
    if idx == -1:
        idx = boundary_num - 1

    return idx
