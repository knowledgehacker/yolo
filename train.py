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
from net.fast_yolo import FastYolo

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
        box_ph_dict = {
            "cls": tf.placeholder(dtype=tf.float32, shape=(None, H, W, B, C), name="cls_ph"),
            "conf": tf.placeholder(dtype=tf.float32, shape=(None, H, W, B, 1), name="conf_ph"),
            "coord": tf.placeholder(dtype=tf.float32, shape=(None, H, W, B, 4), name="coord_ph")
        }
        dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob_ph")

        net_out_op, pretrained_model = model.forward(image_ph, config.input_shape, config.data_format,
                                                     dropout_keep_prob_ph, True)
        loss_op = model.opt(net_out_op, box_ph_dict["cls"], box_ph_dict["conf"], box_ph_dict["coord"])

        batch_size = config.TRAIN_BATCH_SIZE
        batch_num, last_batch_size = get_batch_num(data, batch_size)
        print("batch_num: %d, batch_size: %d, last_batch_size: %d" % (batch_num, batch_size, last_batch_size))
        total_warmup_step = batch_num * config.WARMUP_EPOCH

        global_step_op = tf.Variable(1., trainable=False)
        if config.USE_WARMUP:
            lr_op = tf.cond(tf.less(global_step_op, total_warmup_step),
                            lambda: config.LR_INIT * (global_step_op / total_warmup_step),
                            lambda: config_lr((global_step_op - total_warmup_step) / batch_num))
        else:
            lr_op = config_lr(global_step_op / batch_num)
        optimizer = get_optimizer(lr_op)

        gvs = optimizer.compute_gradients(loss_op)
        # apply gradient clip to avoid gradient exploding
        clip_grad_vars = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_vars, global_step=global_step_op)

        # call tf.global_variables() here will include global variables defined above, including global_step_op, etc.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)

    with tf.Session(graph=g, config=cfg) as sess:
        ckpt_path = '%s/%s' % (config.CKPT_DIR, config.MODEL_NAME)
        trained_epoch = cal_trained_epoch(config.CKPT_DIR)
        if trained_epoch == -1:
            # initialize global variables
            tf.global_variables_initializer().run()

            # load pretrained weights after initializing global variables to avoid weights being re-initialized
            pretrained_model.load_weights("data/weights/%s.h5" % config.pt_net)
        else:
            print("resume from last epoch %d ..." % trained_epoch)
            # restore will initialize global variables with the saved ones
            saver.restore(sess, "%s-%d" % (ckpt_path, trained_epoch))

        tf.local_variables_initializer().run()

        # start train
        epoch = trained_epoch + 1
        for i in range(epoch, config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % i)

            # shuffle data
            shuffle_idx = perm(len(data))
            for b in range(batch_num):
                step = b + 1

                if (step == batch_num) and (last_batch_size > 0):
                    batch_size = last_batch_size

                # get data
                #print(current_time(), "batch %d get data starts ..." % step)
                chunks = [data[idx] for idx in shuffle_idx[b * batch_size: (b + 1) * batch_size]]
                images, box_dict = batch(config.IMAGE_TRAIN_DIR, chunks)
                if images is None:
                    print(current_time(), "batch %d skipped!" % step)
                    continue

                # convert input in NHWC to NCHW on gpu
                if config.DEVICE_TYPE == "gpu":
                    images = np.transpose(images, [0, 3, 1, 2])

                # train data
                #print(current_time(), "batch %d train data starts ..." % step)

                feed_dict = dict()
                feed_dict[image_ph] = images
                for key in box_ph_dict:
                    feed_dict[box_ph_dict[key]] = box_dict[key]
                feed_dict[dropout_keep_prob_ph] = config.TRAIN_KEEP_PROB

                _, loss, global_step, lr = sess.run([train_op, loss_op, global_step_op, lr_op], feed_dict=feed_dict)

                # print train loss message
                if step % config.STEPS_PER_CKPT == 0:
                    print(current_time(), "step %d, loss: %.3f, global_step: %d, lr: %.6f" % (step, loss, global_step, lr))
                    # saver.save(sess, config.CKPT_PATH, global_step=step)

            # checkpoint upon each epoch instead of some step during an epoch, for convenient restore
            saver.save(sess, ckpt_path, global_step=i)

            # save model each epoch
            outputs = ["net_out", "loss"]
            save_model(sess, config.MODLE_DIR, config.MODEL_NAME, outputs)

    print(current_time(), "Training finished!")


def config_lr(epoch):
    return tf.train.piecewise_constant(tf.cast(epoch, tf.int32),
                                       boundaries=config.BOUNDARIES,
                                       values=config.LRS,
                                       name='piecewise_learning_rate')


def cal_trained_epoch(ckpt_dir):
    trained_epoch = -1
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
