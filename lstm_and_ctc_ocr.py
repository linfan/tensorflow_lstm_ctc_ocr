#!/usr/bin/env python
# encoding=utf-8
# Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import numpy as np

import common, model
from common import unzip, read_data_for_lstm_ctc

from utils import sparse_tuple_from as sparse_tuple_from, decode_sparse_tensor, get_data_set

# Some configs
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('9') - ord('0') + 1 + 1 + 1
print("num_classes", num_classes)
# Hyper-parameters
num_epochs = 10000
num_hidden = 64
num_layers = 1
print("num_hidden:", num_hidden, "num_layers:", num_layers)

# THE MAIN CODE!

train_inputs, train_targets, train_seq_len = get_data_set('train')
test_inputs, test_targets, test_seq_len = get_data_set('test')
print("Data loaded....")
graph = tf.Graph()


def train():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, targets, seq_len = model.get_train_model()

    loss = tf.contrib.ctc.ctc_loss(logits, targets, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=common.MOMENTUM).minimize(cost, global_step=global_step)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.contrib.ctc.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    # Accuracy: label error rate
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    # Initializate the weights and biases
    init = tf.initialize_all_variables()

    def do_report():
        test_feed = {inputs: test_inputs,
                     targets: test_targets,
                     seq_len: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        print(decode_sparse_tensor(dd))

    def do_batch():
        start = time.time()
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        batch_cost, steps, _ = session.run([cost, global_step, optimizer], feed)

    with tf.Session(graph=graph) as session:
        session.run(init)
        saver = tf.train.Saver()
        for curr_epoch in xrange(num_epochs):
            train_cost = train_ler = 0

            for batch in xrange(common.BATCHES):
                do_batch(batch)

            batch_cost, steps, _ = session.run([cost, global_step, optimizer], feed)
            train_cost += batch_cost * common.BATCH_SIZE
            if steps > 0 and steps % common.REPORT_STEPS == 0:
                do_report()
                save_path = saver.save(session, "ocr.model")

        train_cost /= common.TRAIN_SIZE
        train_ler /= common.TRAIN_SIZE

        val_feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}

        val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

        log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
        print(
            log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start,
                       lr))
    # Decoding
    d = session.run(decoded[0], feed_dict=feed)
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + common.FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print('Original:\n%s' % original)
    print('Decoded:\n%s' % str_decoded)
