# -*- coding: utf-8 -*-

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def create_dataset(image_names, images, probs, proids, confs, coords, test=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_names, images, probs, proids, confs, coords))
    """
    print("--- confs shape")
    print(confs)
    print(confs.shape)
    """
    with tf.device('/cpu:0'):
        if not test:
            dataset = dataset.shuffle(config.SHUFFLE_SIZE)
            dataset = dataset.batch(config.BATCH_SIZE)
        else:
            dataset = dataset.padded_batch(config.TEST_BATCH_SIZE)

    return dataset
