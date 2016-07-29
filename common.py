# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
Definitions that don't fit elsewhere.

"""
import glob

import cv2
import numpy as np

# Constants
SPACE_INDEX = 0
FIRST_INDEX = ord('0') - 1  # 0 is reserved to space

SPACE_TOKEN = '<space>'

__all__ = (
    'DIGITS',
    'sigmoid',
    'softmax',
)
OUTPUT_SHAPE = (64, 256)
import numpy

DIGITS = "0123456789"
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


CHARS = DIGITS
LENGTH = 16
TEST_SIZE = 200

LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
INITIAL_LEARNING_RATE = 0.0001
DECAY_STEPS = 50

# parameters for bdlstm ctc
MAX_LENGTH = 20  # max length of the sequence
MIN_LENGTH = 16  # min length of the sequence
BATCH_SIZE = 3
BATCHES = 1
TRAIN_SIZE = BATCH_SIZE * BATCHES
MOMENTUM = 0.9
REPORT_STEPS = 50


def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]


def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))


def read_data_for_lstm_ctc(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = list(fname.split("/")[1].split("_")[1])
        yield im, numpy.asarray([SPACE_INDEX if x == SPACE_TOKEN else (ord(x) - FIRST_INDEX) for x in list(code)])


def convert_original_code_train_code(code):
    return numpy.asarray([SPACE_INDEX if x == SPACE_TOKEN else (ord(x) - FIRST_INDEX) for x in code])


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


if __name__ == '__main__':
    train_inputs, train_codes = unzip(list(read_data_for_lstm_ctc("test/*.png"))[:2])
    print train_codes
    print("train_codes", train_codes)
    targets = np.asarray(train_codes).flat[:]
    print targets
