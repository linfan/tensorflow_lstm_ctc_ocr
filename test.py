#!/usr/bin/env python
# encoding=utf-8
# Created by andy on 2016-08-03 18:38.
import pickle

import common
import utils

__author__ = "andy"
for batch in xrange(common.BATCHES):
    train_inputs, train_targets, train_seq_len = utils.get_data_set('train', batch*common.BATCH_SIZE, (batch + 1) * common.BATCH_SIZE)
    print batch, train_inputs.shape
   # pickle_file = 'test/test.pickle' + str(batch)
   # f = open(pickle_file, 'wb')
   # pickle.dump(batch_data, f, pickle.HIGHEST_PROTOCOL)
