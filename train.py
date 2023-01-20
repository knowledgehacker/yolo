# -*- coding: utf-8 -*-
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"    # To use GPU, you must set the right slot
"""

import numpy as np

import config
from fast_yolo import FastYolo
from utils.voc_parser import parse
from data import shuffle
from utils.misc import current_time, get_optimizer, save_model

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


SS, B, C = config.S * config.S, config.B, config.C


def train():
    print(current_time(), "Training starts ...")

    with tf.device("/cpu:0"):
        print(current_time(), "Parse starts ...")
        # test using tmp directory
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
        bounding_box_ph_dict = {
            "class_probs": tf.placeholder(dtype=tf.float32, shape=(None, SS, C), name="class_probs_ph"),
            "class_proids": tf.placeholder(dtype=tf.float32, shape=(None, SS, C), name="class_proids_ph"),
            "object_proids": tf.placeholder(dtype=tf.float32, shape=(None, SS, B), name="object_proids_ph"),
            "coords": tf.placeholder(dtype=tf.float32, shape=(None, SS, B, 4), name="coords_ph")
        }
        dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")

        net_out_op = model.forward(image_ph, config.data_format, config.input_shape, dropout_keep_prob_ph)
        loss_op = model.opt(net_out_op, bounding_box_ph_dict["class_probs"], bounding_box_ph_dict["class_proids"],
                            bounding_box_ph_dict["object_proids"],
                            bounding_box_ph_dict["coords"])
        optimizer = get_optimizer()
        train_op = optimizer.minimize(loss_op)

        # create saver
        #saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(graph=g, config=cfg) as sess:
        tf.global_variables_initializer().run()

        """
        # parameters for profiling
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        """

        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))

            # test using tmp directory
            batches = shuffle(config.IMAGE_TRAIN_DIR, data)

            for step, (images, bounding_box_dict) in enumerate(batches):
                # convert input in NHWC to NCHW on gpu
                if config.DEVICE_TYPE == "gpu":
                    images = np.transpose(images, [0, 3, 1, 2])

                # train a batch
                feed_dict = dict()
                feed_dict[image_ph] = images
                for key in bounding_box_ph_dict:
                    feed_dict[bounding_box_ph_dict[key]] = bounding_box_dict[key]
                feed_dict[dropout_keep_prob_ph] = config.TRAIN_KEEP_PROB

                _, train_loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

                # print train loss message
                step += 1
                if step % config.STEPS_PER_CKPT == 0:
                    print(current_time(), "step %d, train_loss: %.3f" % (step, train_loss))
                    # saver.save(sess, config.CKPT_PATH, global_step=step)

                    """
                    # profiling
                    from tensorflow.python.client import timeline
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open("%s/timeline_%d.json" % (config.PROF_DIR, step), 'w') as f:
                        f.write(chrome_trace)
                    """

            # save model each epoch
            outputs = ["net_out", "loss"]
            save_model(sess, config.MODLE_DIR, config.MODEL_NAME, outputs)

    print(current_time(), "Training finished!")


if __name__ == "__main__":
    train()
