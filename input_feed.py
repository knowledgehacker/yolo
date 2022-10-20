# -*- coding: utf-8 -*-

import config
from utils.misc import current_time, load_image_indexes

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


SS, B, C = config.S * config.S, config.B, config.C


def create_dataset(image_dir, tfrecord_file, test=False):
    # load and parse image files
    if not test:
        image_index2names = load_image_indexes("%s-%s.txt" % (config.IMAGE_INDEX_FILE, "train"))
    else:
        image_index2names = load_image_indexes("%s-%s.txt" % (config.IMAGE_INDEX_FILE, "test"))
    image_file_names = [file_name for index, file_name in image_index2names]
    image_files = ["%s/%s" % (image_dir, file_name) for file_name in image_file_names]
    image_dataset = tf.data.Dataset.from_tensor_slices(image_files)
    image_dataset = image_dataset.map(parse_image)

    # load and parse tfrecord files
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_file)

    dataset = tf.data.Dataset.zip((image_dataset, tfrecord_dataset))
    if not test:
        dataset = dataset.shuffle(config.SHUFFLE_SIZE)
        dataset = dataset.batch(config.BATCH_SIZE)
    else:
        dataset = dataset.batch(config.TEST_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(lambda image_batch, tfrecord_batch: (image_batch, parse_tfrecords(tfrecord_batch)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def parse_image(image_file):
    image_str = tf.read_file(image_file)
    image = tf.image.decode_jpeg(image_str, channels=3)
    image = tf.image.resize(image, [config.IMG_H, config.IMG_W])
    image = image / 255.0

    return image


def parse_tfrecords(record_batch):
    example = tf.io.parse_example(
        record_batch,
        features={
            'image_idx': tf.FixedLenFeature([], tf.int64),
            'probs': tf.FixedLenFeature([SS * C], tf.float32),
            'proids': tf.FixedLenFeature([SS * C], tf.float32),
            'confs': tf.FixedLenFeature([SS * B], tf.float32),
            'coords': tf.FixedLenFeature([SS * B * 4], tf.float32)
        })

    return example['image_idx'], example['probs'], example['proids'], example['confs'], example['coords']
