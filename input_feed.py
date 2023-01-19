# -*- coding: utf-8 -*-

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


SS, B, C = config.S * config.S, config.B, config.C


def create_dataset(image_dir, tfrecord_file, image_index2names, test=False):
    print("image_dir: %s, tfrecord_file: %s" % (image_dir, tfrecord_file))

    # load and parse image files
    image_file_names = [file_name for index, file_name in image_index2names]
    image_files = ["%s/%s" % (image_dir, file_name) for file_name in image_file_names]
    image_dataset = tf.data.Dataset.from_tensor_slices(image_files)

    # load and parse tfrecord files
    tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_file)

    dataset = tf.data.Dataset.zip((image_dataset, tfrecord_dataset))

    # parse image one by one
    dataset = dataset.map(lambda image, tfrecord: (parse_image(image), tfrecord))

    if not test:
        dataset = dataset.shuffle(config.SHUFFLE_SIZE)
        dataset = dataset.batch(config.BATCH_SIZE)
    else:
        dataset = dataset.batch(config.TEST_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # parse tfrecord in batch
    dataset = dataset.map(lambda image_batch, tfrecord_batch: (image_batch, parse_tfrecords(tfrecord_batch)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


"""
TODO: Data Augmentation.
we introduce random scaling and translations of up to 20% of the original image size.
We also randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.
"""
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
            'class_probs': tf.FixedLenFeature([SS * C], tf.float32),
            'class_proids': tf.FixedLenFeature([SS * C], tf.float32),
            'object_proids': tf.FixedLenFeature([SS * B], tf.float32),
            'coords': tf.FixedLenFeature([SS * B * 4], tf.float32)
        })

    return example['image_idx'], example['class_probs'], example['class_proids'], \
           example['object_proids'], example['coords']
