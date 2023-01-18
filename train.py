# -*- coding: utf-8 -*-
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"    # To use GPU, you must set the right slot
"""

import config
from input_feed import create_dataset
from fast_yolo import FastYolo
from utils.misc import current_time, load_image_indexes, get_optimizer, save_model

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

    index_file = "%s-%s.txt" % (config.IMAGE_INDEX_FILE, "train")
    print("index_file: %s" % index_file)
    image_index2names = load_image_indexes(index_file)

    g = tf.Graph()
    with g.as_default():
        # create a feed-able iterator, to be feed by train and test datasets
        handle_ph = tf.placeholder(dtype=tf.string, name="handle_ph")
        train_dataset = create_dataset(config.IMAGE_TRAIN_DIR, config.TF_IMAGE_TRAIN_FILE, image_index2names)
        train_iterator = tf.data.make_initializable_iterator(train_dataset)

        iterator = tf.data.Iterator.from_string_handle(
            handle_ph,
            tf.data.get_output_types(train_dataset),
            tf.data.get_output_shapes(train_dataset),
            tf.data.get_output_classes(train_dataset))
        content, (image_idx, class_probs, class_proids, object_proids, coords) = iterator.get_next(name="next_batch")
        if config.DEVICE_TYPE == "gpu":
            content = tf.transpose(content, [0, 3, 1, 2])

        # create model network
        if config.MODEL_NAME == "fast_yolo":
            model = FastYolo()
        else:
            print("Unsupported model %s" % config.MODEL_NAME)
            exit(-1)

        #To be able to feed with batches of different size, the first dimension should be None
        content_ph = tf.placeholder(dtype=tf.float32, shape=config.placeholder_image_shape, name="content_ph")
        image_idx_ph, class_probs_ph, class_proids_ph, object_proids_ph, coords_ph = (
            tf.placeholder(dtype=tf.int64, shape=(None, ), name="image_idx_ph"),
            tf.placeholder(dtype=tf.float32, shape=(None, SS * C), name="class_probs_ph"),
            tf.placeholder(dtype=tf.float32, shape=(None, SS * C), name="class_proids_ph"),
            tf.placeholder(dtype=tf.float32, shape=(None, SS * B), name="object_proids_ph"),
            tf.placeholder(dtype=tf.float32, shape=(None, SS * B * 4), name="coords_ph")
        )
        dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")

        net_out_op = model.forward(content_ph, config.data_format, config.input_shape, dropout_keep_prob_ph)
        loss_op = model.opt(net_out_op, class_probs_ph, class_proids_ph, object_proids_ph, coords_ph)
        optimizer = get_optimizer()
        train_op = optimizer.minimize(loss_op)

        # create saver
        #saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(graph=g, config=cfg) as sess:
        tf.global_variables_initializer().run()

        # create handle to feed to iterator's string_handle placeholder
        train_handle = sess.run(train_iterator.string_handle())
        #test_handle = sess.run(test_iterator.string_handle())

        # parameters for profiling
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(train_iterator.initializer)

            while True:
                try:
                    #print(current_time(), "Read batch starts ...")
                    content_ts, image_idx_ts, class_probs_ts, class_proids_ts, object_proids_ts, coords_ts = \
                        sess.run([content, image_idx, class_probs, class_proids, object_proids, coords],
                                 feed_dict={handle_ph: train_handle})
                    #print(current_time(), "Read batch finished!")
                    """
                    print("--- content_ts")
                    print(content_ts)
                    """

                    #print(current_time(), "Train batch starts ...")
                    _, train_loss = sess.run([train_op, loss_op],
                                             feed_dict={content_ph: content_ts,
                                                        image_idx_ph: image_idx_ts,
                                                        class_probs_ph: class_probs_ts,
                                                        class_proids_ph: class_proids_ts,
                                                        object_proids_ph: object_proids_ts,
                                                        coords_ph: coords_ts,
                                                        dropout_keep_prob_ph: config.TRAIN_KEEP_PROB})
                    #print(current_time(), "Train batch finished!")

                    step += 1
                    if step % config.STEPS_PER_CKPT == 0:
                        print(current_time(), "step %d, train_loss: %.3f" % (step, train_loss))
                        #saver.save(sess, config.CKPT_PATH, global_step=step)

                        """
                        # profiling
                        from tensorflow.python.client import timeline
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open("%s/timeline_%d.json" % (config.PROF_DIR, step), 'w') as f:
                            f.write(chrome_trace)
                        """
                except tf.errors.OutOfRangeError:
                    if step % config.STEPS_PER_CKPT != 0:
                        print(current_time(), "step %d, train_loss: %.3f" % (step, train_loss))
                        #saver.save(sess, config.CKPT_PATH, global_step=step)
                    break

            # save model
            outputs = ["net_out", "loss"]
            save_model(sess, config.MODLE_DIR, config.MODEL_NAME, outputs)

    print(current_time(), "Training finished!")


def main():
    # train
    train()


if __name__ == "__main__":
    main()
