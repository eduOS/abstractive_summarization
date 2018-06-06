# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf


indices = tf.constant([[[0, 0], [0, 0], [0, 1], [0, 1]], [[1, 0], [1, 1], [1, 2], [1, 2]]])

shape = tf.constant([2, 3, 4])

updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                        [7, 7, 7, 7], [8, 8, 8, 8]],
                       [[5, 5, 5, 5], [6, 6, 6, 6],
                        [7, 7, 7, 7], [8, 8, 8, 8]]])

scatter = tf.scatter_nd(indices, updates, shape)
scatter.eval()

# array([[[11, 11, 11, 11],
#         [15, 15, 15, 15],
#         [ 0,  0,  0,  0]],

#        [[ 5,  5,  5,  5],
#         [ 6,  6,  6,  6],
#         [15, 15, 15, 15]]], dtype=int32)
