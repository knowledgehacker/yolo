# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"    # To use GPU, you must set the right slot

import config
from input_feed import create_dataset
from predict import postprocess
from utils.misc import current_time, with_prefix, load_image_indexes

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def test():
    print(current_time(), "Testing starts ...")

    image_index2names = dict(load_image_indexes("%s-%s.txt" % (config.IMAGE_INDEX_FILE, "test")))

    g = tf.Graph()
    with tf.Session(graph=g, config=cfg) as sess:
        # load trained model
        load_model(sess, config.MODLE_DIR, config.MODEL_NAME)
        #load_ckpt_model(sess, config.CKPT_DIR)

        # get prediction and other dependent tensors from the graph in the trained model for inference
        """
        for operation in g.get_operations():
            print(operation.name)
        """

        content_ph = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "content_ph:0"))
        #image_idx_ph, \
        probs_ph, proids_ph, confs_ph, coords_ph = (
            #g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "image_idx_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "probs_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "proids_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "confs_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "coords_ph:0"))
        )
        dropout_keep_prob_ph = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "dropout_keep_prob:0"))

        net_out_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "net_out:0"))

        # create iterator for test dataset
        #handle_ph = g.get_tensor_by_name("handle_ph:0")
        test_dataset = create_dataset(config.IMAGE_TEST_DIR, config.TF_IMAGE_TEST_FILE, test=True)
        test_iterator = test_dataset.make_initializable_iterator()
        content, (image_idx, probs, proids, confs, coords) = test_iterator.get_next("next_batch")
        if config.DEVICE_TYPE == "gpu":
            content = tf.transpose(content, [2, 0, 1])

        #test_handle = sess.run(test_iterator.string_handle())

        # important!!! Don't call 'tf.global_variables_initializer().run()' when doing inference using trained model
        #tf.global_variables_initializer().run()
        sess.run(test_iterator.initializer)

        try:
            while True:
                # we don't need to feed test_handle to handle_ph here
                content_ts, image_idx_ts, probs_ts, proids_ts, confs_ts, coords_ts = sess.run([content, image_idx, probs, proids, confs, coords])
                net_out_ts = sess.run([net_out_op], feed_dict={#image_idx_ph: image_idx_ts,
                                                               content_ph: content_ts,
                                                               probs_ph: probs_ts,
                                                               proids_ph: proids_ts,
                                                               confs_ph: confs_ts,
                                                               coords_ph: coords_ts,
                                                               dropout_keep_prob_ph: config.TEST_KEEP_PROB})
                print("--- net_out_ts")
                print(net_out_ts)
                write_preds(image_idx_ts, net_out_ts, image_index2names)
        except tf.errors.OutOfRangeError:
            write_preds(image_idx_ts, net_out_ts, image_index2names)
            pass

    print(current_time(), "Testing finished!")


def load_model(sess, model_dir, filename):
    model_filepath = "%s/%s.pb" % (model_dir, filename)

    print("Loading model %s ..." % model_filepath)

    with tf.gfile.GFile(model_filepath, 'rb') as fin:
        graph_def = sess.graph.as_graph_def()
        graph_def.ParseFromString(fin.read())

    tf.import_graph_def(graph_def, name=config.MODEL_NAME)

    print("Model %s loaded!" % model_filepath)


def load_ckpt_model(sess, ckpt_dir):
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    print("ckpt_file: %s" % ckpt_file)
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
    saver.restore(sess, ckpt_file)


def write_preds(image_idx_ts, net_out_ts, image_idx_to_name):
    for (image_idx, net_out) in zip(image_idx_ts, net_out_ts):
        image_file = "%s/%s" % (config.IMAGE_TEST_DIR, image_idx_to_name[image_idx])
        postprocess(image_file, net_out[0])


def main():
    # test
    test()


if __name__ == "__main__":
    main()
