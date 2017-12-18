# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np


class Temp:
    def __init__(self):
        self.a = tf.placeholder(tf.int32, [2, 2])
        weights = tf.Variable(10)
        self.b = self.a * weights
        self.c = self.a * self.b

    def run1(self, sess):
        feed_dict = {self.b: np.array([[1, 1], [1, 1]])}
        print(sess.run(self.c, feed_dict))

    def run2(self, sess):
        feed_dict = {self.a: np.array([[2, 2], [2, 2]])}
        print(sess.run(self.c, feed_dict))
