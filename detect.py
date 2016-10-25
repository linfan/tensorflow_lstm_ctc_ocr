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
        saver.restore(sess, "models/ocr.model-0.52-48999")
        print("Model restored.")
        #feed_dict = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
        feed_dict = {inputs: test_inputs, seq_len: test_seq_len}
        dd = sess.run(decoded[0], feed_dict=feed_dict)
        #return decode_sparse_tensor(dd)
    	original_list = decode_sparse_tensor(test_targets)
    	detected_list = decode_sparse_tensor(dd)
	true_numer = 0
	# print(detected_list)
	if len(original_list) != len(detected_list):
		print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
				  " test and detect length desn't match")
		return
	print("T/F: original(length) <-------> detectcted(length)")
	for idx, number in enumerate(original_list):
	    detect_number = detected_list[idx]
            print(number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
            if(len(number) == len(detect_number)):
		hit = True
		for idy, value in  enumerate(number):
		    detect_value = detect_number[idy]
		    if(value != detect_value):
		        hit = False
		        break
		if hit:
	            true_numer = true_numer + 1
	accuraccy = true_numer * 1.0 / len(original_list)
	print("Test Accuracy:", accuraccy)
	return accuraccy

if __name__ == '__main__':
    test_inputs, test_targets, test_seq_len = utils.get_data_set('small_test')
    print test_inputs[0].shape
    print detect(test_inputs, test_targets, test_seq_len)
   # print_tensors_in_checkpoint_file("model/ocr.model.50", None)
