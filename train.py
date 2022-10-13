# -*- coding: utf-8 -*-

import config
from input_feed import create_dataset
from fast_yolo import FastYolo
from utils.misc import current_time

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


SS, B, C = config.S * config.S, config.B, config.C


def train():
    print(current_time(), "training starts...")

    g = tf.Graph()
    with g.as_default():
        # create a feed-able iterator, to be feed by train and test datasets
        handle_ph = tf.placeholder(dtype=tf.string, name="handle_ph")
        train_dataset = create_dataset(config.TF_IMAGE_TRAIN_DIR)
        train_iterator = tf.data.make_initializable_iterator(train_dataset)
        #train_iterator = train_dataset.make_initializable_iterator()

        iterator = tf.data.Iterator.from_string_handle(
            handle_ph,
            tf.data.get_output_types(train_dataset),
            tf.data.get_output_shapes(train_dataset),
            tf.data.get_output_classes(train_dataset))
        image_idx, content, probs, proids, confs, coords = iterator.get_next(name="next_batch")

        # create model network
        #To be able to feed with batches of different size, the first dimension should be None
        image_idx_ph = tf.placeholder(dtype=tf.int64, shape=(None, ), name="image_idx_ph")
        content_ph = tf.placeholder(dtype=tf.float32, shape=(None, config.IMG_H * config.IMG_W * config.IMG_CH), name="content_ph")
        probs_ph = tf.placeholder(dtype=tf.float32, shape=(None, SS * C), name="probs_ph")
        proids_ph = tf.placeholder(dtype=tf.float32, shape=(None, SS * C), name="proids_ph")
        confs_ph = tf.placeholder(dtype=tf.float32, shape=(None, SS * B), name="confs_ph")
        coords_ph = tf.placeholder(dtype=tf.float32, shape=(None, SS * B * 4), name="coords_ph")

        dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")

        if config.MODEL_NAME == "fast_yolo":
            model = FastYolo()
        else:
            print("Unsupported model %s" % config.MODEL_NAME)
            exit(-1)

        net_out_op = model.forward(content_ph, "channels_last", dropout_keep_prob_ph)
        loss_op, train_op = model.opt(net_out_op, probs_ph, proids_ph, confs_ph, coords_ph)
        #preds_op, acc_op = model.predict(logits, label_ph)

        # create saver
        saver = tf.train.Saver()

    with tf.Session(graph=g, config=cfg) as sess:
        tf.global_variables_initializer().run()

        # create handle to feed to iterator's string_handle placeholder
        train_handle = sess.run(train_iterator.string_handle())
        #test_handle = sess.run(test_iterator.string_handle())

        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(train_iterator.initializer)

            while True:
                try:
                    image_idx_ts, content_ts, probs_ts, proids_ts, confs_ts, coords_ts = sess.run(
                        [image_idx, content, probs, proids, confs, coords], feed_dict={handle_ph: train_handle})
                    _, train_loss = sess.run([train_op, loss_op],
                                             feed_dict={image_idx_ph: image_idx_ts,
                                                        content_ph: content_ts,
                                                        probs_ph: probs_ts,
                                                        proids_ph: proids_ts,
                                                        confs_ph: confs_ts,
                                                        coords_ph: coords_ts,
                                                        dropout_keep_prob_ph: config.TRAIN_KEEP_PROB})

                    if step % config.STEPS_PER_CKPT == 0:
                        print("step %d, train_loss: %.3f" % (step, train_loss))
                        saver.save(sess, config.CKPT_PATH, global_step=step)

                    step += 1
                except tf.errors.OutOfRangeError:
                    print("step %d, train_loss: %.3f" % (step, train_loss))
                    saver.save(sess, config.CKPT_PATH, global_step=step)
                    break

            # save model
            save_model(sess, config.MODLE_DIR, config.MODEL_NAME)

    print(current_time(), "training finishes...")


def save_model(sess, model_dir, filename):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        #["logits", "loss", "output/predictions", "output/accuracy"])
        ["net_out", "loss"])

    model_filepath = "%s/%s.pb" % (model_dir, filename)
    with tf.gfile.GFile(model_filepath, "wb") as fout:
        fout.write(output_graph_def.SerializeToString())


def main():
    # train
    train()


if __name__ == "__main__":
    main()
