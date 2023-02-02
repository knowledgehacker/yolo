# -*- coding: utf-8 -*-

import os
"""
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"    # To use GPU, you must set the right slot
"""

import numpy as np
from numpy.random import permutation as perm

import config
from utils.voc_parser import parse
if config.VERSION == "v1":
    print("YOLO v1")
    from v1.fast_yolo import FastYolo
elif config.VERSION == "v2":
    print("YOLO v2")
    from v2.fast_yolo import FastYolo
else:
    print("Unsupported version: %s" % config.VERSION)
    exit(-1)
#from data import shuffle
from data import get_batch_num, batch
from utils.misc import current_time, get_optimizer, save_model

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


H, W = config.H, config.W
B = config.B
C = config.C


def train():
    print(current_time(), "Training starts ...")

    with tf.device("/cpu:0"):
        print(current_time(), "Parse starts ...")
        data = parse(config.ANNOTATION_TRAIN_DIR, config.CLASSES)
        print(current_time(), "Parse finished!")

    g = tf.Graph()
    with g.as_default():
        # create model network
        if config.MODEL_NAME == "fast_yolo":
            model = FastYolo()
        else:
            print("Unsupported model %s" % config.MODEL_NAME)
            exit(-1)

        #To be able to feed with batches of different size, the first dimension should be None
        image_ph = tf.placeholder(dtype=tf.float32, shape=config.placeholder_image_shape, name="image_ph")
        if config.VERSION == "v1":
            bounding_box_ph_dict = {
                "class_probs": tf.placeholder(dtype=tf.float32, shape=(None, H*W, C), name="class_probs_ph"),
                "class_proids": tf.placeholder(dtype=tf.float32, shape=(None, H*W, C), name="class_proids_ph"),
                "object_proids": tf.placeholder(dtype=tf.float32, shape=(None, H*W, B), name="object_proids_ph"),
                "coords": tf.placeholder(dtype=tf.float32, shape=(None, H*W, B, 4), name="coords_ph")
            }
        elif config.VERSION == "v2":
            bounding_box_ph_dict = {
                "class_probs": tf.placeholder(dtype=tf.float32, shape=(None, H*W, B, C), name="class_probs_ph"),
                "class_proids": tf.placeholder(dtype=tf.float32, shape=(None, H*W, B, C), name="class_proids_ph"),
                "object_proids": tf.placeholder(dtype=tf.float32, shape=(None, H*W, B), name="object_proids_ph"),
                "coords": tf.placeholder(dtype=tf.float32, shape=(None, H*W, B, 4), name="coords_ph")
            }
        else:
            print("Unsupported version: %s" % config.VERSION)
            exit(-1)
        dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob_ph")

        net_out_op, pretrained_model = model.forward(image_ph, config.input_shape, config.data_format,
                                                     dropout_keep_prob_ph, True)
        loss_op = model.opt(net_out_op, bounding_box_ph_dict["class_probs"], bounding_box_ph_dict["class_proids"],
                            bounding_box_ph_dict["object_proids"],
                            bounding_box_ph_dict["coords"])
        optimizer = get_optimizer()
        train_op = optimizer.minimize(loss_op)

    with tf.Session(graph=g, config=cfg) as sess:
        """
        # parameters for profiling
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        """

        # create saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        ckpt_path = '%s/%s' % (config.CKPT_DIR, config.MODEL_NAME)
        trained_epoch = cal_trained_epoch(config.CKPT_DIR)
        if trained_epoch == 0:
            # initialize global variables
            tf.global_variables_initializer().run()

            # load pretrained weights after initializing global variables to avoid weights being re-initialized
            pretrained_model.load_weights("data/weights/%s.h5" % config.pt_net)
        else:
            print("resume from last epoch %d ..." % trained_epoch)
            # restore will re-initialize global variables with the saved ones
            saver.restore(sess, "%s-%d" % (ckpt_path, trained_epoch))

        for i in range(trained_epoch, config.NUM_EPOCH):
            epoch = i + 1

            print(current_time(), "epoch: %d" % epoch)

            """
            with tf.device("/cpu:0"):
                batches = shuffle(config.IMAGE_TRAIN_DIR, data)

            for j, (images, bounding_box_dict) in enumerate(batches):
            """

            shuffle_idx = perm(len(data))

            batch_size = config.BATCH_SIZE
            batch_num = get_batch_num(data, batch_size)
            print("batch_num: %d" % batch_num)
            for b in range(batch_num):
                step = b + 1

                # get data
                print(current_time(), "batch %d get data starts ..." % step)
                chunks = [data[idx] for idx in shuffle_idx[b * batch_size: (b + 1) * batch_size]]
                images, bounding_box_dict = batch(config.IMAGE_TRAIN_DIR, chunks)
                if images is None:
                    print(current_time(), "batch %d skipped!" % step)
                    continue

                # convert input in NHWC to NCHW on gpu
                if config.DEVICE_TYPE == "gpu":
                    images = np.transpose(images, [0, 3, 1, 2])

                # train data
                print(current_time(), "batch %d train data starts ..." % step)

                feed_dict = dict()
                feed_dict[image_ph] = images
                for key in bounding_box_ph_dict:
                    feed_dict[bounding_box_ph_dict[key]] = bounding_box_dict[key]
                feed_dict[dropout_keep_prob_ph] = config.TRAIN_KEEP_PROB

                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

                # print train loss message
                if step % config.STEPS_PER_CKPT == 0:
                    print(current_time(), "step %d, loss: %.3f" % (step, loss))
                    # saver.save(sess, config.CKPT_PATH, global_step=step)

                    """
                    # profiling
                    from tensorflow.python.client import timeline
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open("%s/timeline_%d.json" % (config.PROF_DIR, step), 'w') as f:
                        f.write(chrome_trace)
                    """

            # checkpoint upon each epoch instead of some step during an epoch, for convenient restore
            saver.save(sess, ckpt_path, global_step=epoch)

            # save model each epoch
            outputs = ["net_out", "loss"]
            save_model(sess, config.MODLE_DIR, config.MODEL_NAME, outputs)

    print(current_time(), "Training finished!")


def cal_trained_epoch(ckpt_dir):
    trained_epoch = 0
    ckpt_file = os.path.join(ckpt_dir, "checkpoint")
    if os.path.exists(ckpt_file):
        print("load from checkpoint %s" % ckpt_file)
        with open(ckpt_file, 'r') as f:
            last = f.readlines()[-1].strip()
            load_point = last.split(' ')[1]
            load_point = load_point.split('"')[1]
            trained_epoch = int(load_point.split('-')[-1])

    return trained_epoch


if __name__ == "__main__":
    train()
