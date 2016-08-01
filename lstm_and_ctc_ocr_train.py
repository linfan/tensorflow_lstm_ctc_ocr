#!/usr/bin/env python
# encoding=utf-8
# Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import common, model
import utils

from utils import decode_sparse_tensor

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

test_inputs, test_targets, test_seq_len = utils.get_data_set('test')
print("Data loaded....")


# graph = tf.Graph()
def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0
    # print(detected_list)
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(test_targets))


def train():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(common.INITIAL_LEARNING_RATE,
                                               global_step,
                                               common.DECAY_STEPS,
                                               common.LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
    logits, inputs, targets, seq_len, W, b = model.get_train_model()

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
        report_accuracy(dd, test_targets)
        # decoded_list = decode_sparse_tensor(dd)

    def do_batch():
        feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
        b_cost, steps, _ = session.run([cost, global_step, optimizer], feed)
        if steps > 0 and steps % common.REPORT_STEPS == 0:
            do_report()
            save_path = saver.save(session, "model/ocr.model", global_step=steps)
            # print(save_path)
        return b_cost

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)
        for curr_epoch in xrange(num_epochs):
            # variables = tf.all_variables()
            # for i in variables:
            #     print(i.name)

            print("Epoch.......", curr_epoch)
            train_cost = train_ler = 0
            for batch in xrange(common.BATCHES):
                start = time.time()
                train_inputs, train_targets, train_seq_len = utils.get_data_set('train', batch,
                                                                                (batch + 1) * common.BATCH_SIZE)
                print("get data time",time.time() - start )
                start = time.time()
                c = do_batch()
                train_cost += c * common.BATCH_SIZE
                seconds = time.time() - start
                print("Batch seconds:", seconds)

            train_cost /= common.TRAIN_SIZE
            # train_ler /= common.TRAIN_SIZE

            val_feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len}

            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler,
                             time.time() - start, lr))


if __name__ == '__main__':
    train()
