#!/usr/bin/env python
# encoding=utf-8
# Created by andy on 2016-07-31 21:36.
from tensorflow.contrib.learn.python.learn.utils.inspect_checkpoint import print_tensors_in_checkpoint_file

import model
import tensorflow as tf

import utils
from utils import decode_sparse_tensor

__author__ = "andy"


def detect(test_inputs, test_targets, test_seq_len):
    logits, inputs, targets, seq_len, W, b = model.get_train_model()
    decoded, log_prob = tf.contrib.ctc.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "model/ocr.model.1035")
        print("Model restored.")
        #feed_dict = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
        feed_dict = {inputs: test_inputs, seq_len: test_seq_len}
        dd = sess.run(decoded[0], feed_dict=feed_dict)
        return decode_sparse_tensor(dd)


if __name__ == '__main__':
    test_inputs, test_targets, test_seq_len = utils.get_data_set('validate')
    print detect(test_inputs, test_targets, test_seq_len)
    # print_tensors_in_checkpoint_file("model/ocr.model.50", None)
