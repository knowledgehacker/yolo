# -*- coding: utf-8 -*-

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


SS, B, C = config.S * config.S, config.B, config.C


def create_dataset(input, test=False):
    dataset = tf.data.TFRecordDataset(input)
    dataset = dataset.map(parse)

    with tf.device('/cpu:0'):
        if not test:
            dataset = dataset.shuffle(config.SHUFFLE_SIZE)
            dataset = dataset.batch(config.BATCH_SIZE)
        else:
            dataset = dataset.padded_batch(config.TEST_BATCH_SIZE)

    return dataset


def parse(record):
    example = tf.io.parse_single_example(
        record,
        features={
            'image_idx': tf.FixedLenFeature([], tf.int64),
            'content': tf.FixedLenFeature([config.IMG_H * config.IMG_W * config.IMG_CH], tf.float32),
            'probs': tf.FixedLenFeature([SS * C], tf.float32),
            'proids': tf.FixedLenFeature([SS * C], tf.float32),
            'confs': tf.FixedLenFeature([SS * B], tf.float32),
            'coords': tf.FixedLenFeature([SS * B * 4], tf.float32)
        })

    return example['image_idx'], example['content'], \
           example['probs'], example['proids'], example['confs'], example['coords']
