# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from six.moves import xrange

### Building Blocks ###

def fully_connected(inputs,
                    num_outputs,
                    decay=0.999,
                    activation_fn=None,
                    dropout=None,
                    is_training=True,
                    reuse=None,
                    scope=None):
    """Adds a fully connected layer.

    """

    if not isinstance(num_outputs, int):
        raise ValueError('num_outputs should be integer, got %s.', num_outputs)

    with tf.variable_scope(scope,
                           'fully_connected',
                           [inputs],
                           reuse=reuse) as sc:
        dtype = inputs.dtype.base_dtype
        num_input_units = inputs.get_shape()[-1].value

        static_shape = inputs.get_shape().as_list()
        static_shape[-1] = num_outputs

        out_shape = tf.unstack(tf.shape(inputs))
        out_shape[-1] = num_outputs

        weights_shape = [num_input_units, num_outputs]
        weights = tf.contrib.framework.model_variable(
            'weights',
            shape=weights_shape,
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(),
            collections=tf.GraphKeys.WEIGHTS,
            trainable=True)
        biases = tf.contrib.framework.model_variable(
            'biases',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            collections=tf.GraphKeys.BIASES,
            trainable=True)

        if len(static_shape) > 2:
            # Reshape inputs
            inputs = tf.reshape(inputs, [-1, num_input_units])

        if dropout != None and is_training:
            inputs = tf.nn.dropout(inputs, dropout)

        outputs = tf.matmul(inputs, weights)
        moving_mean = tf.contrib.framework.model_variable(
            'moving_mean',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            trainable=False)

        if is_training:
            # Calculate the moments based on the individual batch.
            mean, _ = tf.nn.moments(outputs, [0], shift=moving_mean)
            # Update the moving_mean moments.
            update_moving_mean = tf.assign_sub(moving_mean, (moving_mean - mean) * (1.0 - decay))
            #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            outputs = outputs + biases
        else:
            outputs = outputs + biases

        if activation_fn:
            outputs = activation_fn(outputs)

        if len(static_shape) > 2:
            # Reshape back outputs
            outputs = tf.reshape(outputs, tf.stack(out_shape))
            outputs.set_shape(static_shape)
        return outputs

# convolutional layer
def convolution2d(inputs,
                  num_outputs,
                  kernel_size,
                  pool_size=None,
                  decay=0.999,
                  activation_fn=None,
                  dropout=None,
                  is_training=True,
                  reuse=None,
                  scope=None):
    """Adds a 2D convolution followed by a maxpool layer.

    """

    with tf.variable_scope(scope,
                           'Conv',
                           [inputs],
                           reuse=reuse) as sc:
        dtype = inputs.dtype.base_dtype
        num_filters_in = inputs.get_shape()[-1].value
        weights_shape = list(kernel_size) + [num_filters_in, num_outputs]
        weights = tf.contrib.framework.model_variable(
            name='weights',
            shape=weights_shape,
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(),
            collections=tf.GraphKeys.WEIGHTS,
            trainable=True)
        biases = tf.contrib.framework.model_variable(
            name='biases',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            collections=tf.GraphKeys.BIASES,
            trainable=True)

        if dropout != None and is_training:
            inputs = tf.nn.dropout(inputs, dropout)

        outputs = tf.nn.conv2d(inputs, weights, [1,1,1,1], padding='SAME')
        moving_mean = tf.contrib.framework.model_variable(
            'moving_mean',
            shape=[num_outputs,],
            dtype=dtype,
            initializer=tf.zeros_initializer(),
            trainable=False)

        if is_training:
            # Calculate the moments based on the individual batch.
            mean, _ = tf.nn.moments(outputs, [0, 1, 2], shift=moving_mean)
            # Update the moving_mean moments.
            update_moving_mean = tf.assign_sub(
                moving_mean,
                (moving_mean - mean) * (1.0 - decay))
            #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            outputs = outputs + biases
        else:
            outputs = outputs + biases

        if pool_size:
            pool_shape = [1] + list(pool_size) + [1]
            outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')

        if activation_fn:
            outputs = activation_fn(outputs)
        return outputs


### Regularization ###

def params_decay(decay):
    """ Add ops to decay weights and biases

    """

    params = tf.get_collection_ref(
        tf.GraphKeys.WEIGHTS) + tf.get_collection_ref(tf.GraphKeys.BIASES)

    while len(params) > 0:
        p = params.pop()
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            p.assign(decay*p + (1-decay)*tf.truncated_normal(
                p.get_shape(), stddev=0.01)))


### Nets ###

def ResDNN(inputs,
           num_layers,
           decay=0.99999,
           activation_fn=tf.nn.relu,
           dropout=None,
           is_training=True,
           reuse=None,
           scope=None):
    """ a deep neural net with fully connected layers

    """

    with tf.variable_scope(scope,
                           "ResDNN",
                           [inputs],
                           reuse=reuse) as sc:
        size = inputs.get_shape()[-1].value
        outputs = inputs

        # residual layers
        for i in xrange(num_layers):
            outputs -= fully_connected(activation_fn(outputs),
                                       size,
                                       decay=decay,
                                       activation_fn=activation_fn,
                                       dropout=dropout,
                                       is_training=is_training)
        return outputs

def ResCNN(inputs,
           num_layers,
           kernel_size,
           pool_size,
           pool_layers=1,
           decay=0.99999,
           activation_fn=tf.nn.relu,
           dropout=None,
           is_training=True,
           reuse=None,
           scope=None):
    """ a convolutaional neural net with conv2d and max_pool layers

    """

    with tf.variable_scope(scope,
                           "ResCNN",
                           [inputs],
                           reuse=reuse) as sc:
        size = inputs.get_shape()[-1].value
        if not pool_size:
            pool_layers = 0
        outputs = inputs

        # residual layers
        for j in xrange(pool_layers+1):
            if j > 0:
                pool_shape = [1] + list(pool_size) + [1]
                inputs = tf.nn.max_pool(outputs,
                                        pool_shape,
                                        pool_shape,
                                        padding='SAME')
                outputs = inputs
            with tf.variable_scope("layer{0}".format(j)) as sc:
                for i in xrange(num_layers):
                    outputs -= convolution2d(activation_fn(outputs),
                                             size,
                                             kernel_size,
                                             decay=decay,
                                             activation_fn=activation_fn,
                                             dropout=dropout,
                                             is_training=is_training)
        return outputs


### RNN ###
class GRUCell(tf.contrib.rnn.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.tanh, linear=fully_connected):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._linear = linear

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = tf.split(fully_connected(tf.concat([inputs, state], 1),
                                             2 * self._num_units), 2, 1)
        r, u = tf.sigmoid(r), tf.sigmoid(u)
      with tf.variable_scope("Candidate"):
        c = self._activation(self._linear(tf.concat([inputs, r * state], 1),
                                     self._num_units))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class RANCell(tf.contrib.rnn.RNNCell):
  """Recurrent Additive Unit cell."""

  def __init__(self, num_units, activation=tf.nn.relu, linear=fully_connected):
    self._num_units = num_units
    self._activation = activation
    self._linear = linear

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        i, f = tf.split(fully_connected(tf.concat([inputs, state], 1),
                                             2 * self._num_units), 2, 1)
        i, f = tf.sigmoid(i), tf.sigmoid(f)
      new_h = f * state + i * inputs
    return new_h, new_h


def attention(query,
              keys,
              values,
              masks,
              is_training=True):
    """ implements the attention mechanism

    query: [batch_size x dim]
    keys: [batch_size x length x dim]
    values: [batch_size x length x dim]
    """

    query = tf.expand_dims(query, 1)
    logits = fully_connected(
        tf.tanh(query+keys), 1, is_training=is_training, scope="attention")
    logits = tf.squeeze(logits, [2])
    probs = tf.where(masks, tf.nn.softmax(logits), tf.zeros(tf.shape(logits)))
    attn_dist = probs / tf.reduce_sum(probs, -1, keep_dims=True)
    attn_feat = tf.reduce_sum(tf.expand_dims(attn_dist, 2) * values, [1])
    return attn_feat, attn_dist

class AttentionCellWrapper(tf.contrib.rnn.RNNCell):
    """Wrapper for attention mechanism"""

    def __init__(self,
                 cell,
                 num_attention=2,
                 self_attention_idx=0,
                 use_copy=None,
                 attention_fn=attention,
                 is_training=True):
        self.cell = cell
        self.num_attention = num_attention
        self.self_attention_idx = self_attention_idx
        self.use_copy = use_copy if use_copy != None else [False]*num_attention
        assert(len(self.use_copy) == self.num_attention)
        self.attention_fn = attention_fn
        self.is_training=is_training

    def __call__(self,
                 inputs,
                 state,
                 scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            keys = []
            values = []
            masks = []
            attn_feats = []
            for _ in xrange(self.num_attention):
                keys.append(state[0])
                values.append(state[1])
                masks.append(state[2])
                attn_feats.append(state[3])
                state = state[4:]
            cell_state = state if len(state) > 1 else state[0]
            batch_size = tf.shape(inputs)[0]

            cell_outputs, cell_state = self.cell(inputs, cell_state)
            # update self attention
            if (self.self_attention_idx >= 0 and
                self.self_attention_idx < self.num_attention):
                value_size = values[self.self_attention_idx].get_shape()[-1].value
                key_size = keys[self.self_attention_idx].get_shape()[-1].value
                new_value = cell_outputs
                values[self.self_attention_idx] = tf.concat(
                    [values[self.self_attention_idx],
                     tf.reshape(new_value, [batch_size, 1, value_size])],
                    axis=1)
                new_key = fully_connected(new_value,
                                          key_size,
                                          is_training=self.is_training,
                                          scope="key")
                keys[self.self_attention_idx] = tf.concat(
                    [keys[self.self_attention_idx],
                     tf.reshape(new_key, [batch_size, 1, key_size])],
                    axis=1)
                new_mask = tf.ones([batch_size, 1], dtype=tf.bool)
                masks[self.self_attention_idx] = tf.concat(
                    [masks[self.self_attention_idx], new_mask],
                    axis=1)

            # attend
            key_sizes = []
            for i in xrange(self.num_attention):
                key_sizes.append(keys[i].get_shape()[-1].value)
            query_all = fully_connected(
                tf.concat([cell_outputs,] + attn_feats, axis=-1),
                sum(key_sizes),
                is_training=self.is_training,
                scope="query_proj")
            queries = tf.split(query_all, key_sizes, axis=-1)
            copy_logits = [tf.zeros([batch_size, 1])]
            copy_dists = []
            for i in xrange(self.num_attention):
                query = queries[i]
                with tf.variable_scope("attention"+str(i)):
                    attn_feats[i], attn_dist = self.attention_fn(
                        query,
                        keys[i],
                        values[i],
                        masks[i],
                        is_training=self.is_training)
                if self.use_copy[i]:
                    with tf.variable_scope("copy"+str(i)):
                        copy_logits.append(fully_connected(
                            tf.concat([cell_outputs, attn_feats[i]], axis=-1),
                            1,
                            is_training=self.is_training,
                            scope="copy_logit"))
                    copy_dists.append(attn_dist)
            outputs = [tf.concat([cell_outputs,] +  attn_feats, 1),]
            outputs = tuple(
                outputs + [tf.concat(copy_logits, axis=-1), tuple(copy_dists)]
                if len(copy_dists) > 0 else outputs[0])
            cell_state = cell_state if isinstance(cell_state, (tuple, list)) else (cell_state,)
            state = []
            for i in xrange(self.num_attention):
                state.extend([keys[i], values[i], masks[i], attn_feats[i]])
            state = tuple(state) + cell_state
            return outputs, state


def create_cell(size,
                num_layers,
                cell_type="GRU",
                decay=0.99999,
                activation_fn=tf.tanh,
                linear=None,
                is_training=True):
    """create various type of rnn cells"""

    def _linear(inputs, num_outputs):
        """fully connected layers inside the rnn cell"""

        return fully_connected(
            inputs, num_outputs, decay=decay, is_training=is_training)

    if not linear:
        linear=_linear

    # build single cell
    if cell_type == "GRU":
        single_cell = GRUCell(size, activation=activation_fn, linear=linear)
    elif cell_type == "RAN":
        single_cell = RANCell(size, activation=activation_fn, linear=linear)
    elif cell_type == "LSTM":
        single_cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=True, cell_clip=5.0, num_proj=size)
    else:
        raise ValueError('Incorrect cell type! (GRU|LSTM)')
    cell = single_cell
    # stack multiple cells
    if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)
    return cell


### Recurrent Decoders ###

def greedy_dec(length,
               initial_state,
               input_embedding,
               cell,
               logit_fn):
    """ A greedy decoder.

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))

    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    symbol = tf.argmax(logits, 1)
    seq = [symbol]
    mask = tf.not_equal(symbol, 0)
    tf.get_variable_scope().reuse_variables()
    for _ in xrange(length-1):

        inputs = tf.nn.embedding_lookup(input_embedding, symbol)

        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        symbol = tf.argmax(logits, 1)
        symbol = tf.where(mask, symbol, tf.zeros([batch_size], dtype=tf.int64))
        mask = tf.not_equal(symbol, 0)

        seq.append(symbol)

    return tf.expand_dims(tf.stack(seq, 1), 1)

def stochastic_dec(length,
                   initial_state,
                   input_embedding,
                   cell,
                   logit_fn,
                   num_candidates=1):
    """ A stochastic decoder.

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))

    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    symbol = tf.reshape(tf.multinomial(logits, num_candidates), [-1])
    mask = tf.equal(symbol, 0)
    seq = [symbol]
    tf.get_variable_scope().reuse_variables()

    beam_parents = tf.reshape(
        tf.tile(tf.expand_dims(tf.range(batch_size), 1),
                [1, num_candidates]), [-1])
    if isinstance(state, tuple):
        state = tuple([tf.gather(s, beam_parents) for s in state])
    else:
        state = tf.gather(state, beam_parents)

    for _ in xrange(length-1):

        inputs = tf.nn.embedding_lookup(input_embedding, symbol)

        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        symbol = tf.squeeze(tf.multinomial(logits, 1), [1])
        symbol = tf.where(mask,
                          tf.to_int64(tf.zeros([batch_size*num_candidates])),
                          symbol)
        mask = tf.equal(symbol, 0)
        seq.append(symbol)

    return tf.reshape(tf.stack(seq, 1), [batch_size, num_candidates, length])

# beam decoder
def beam_dec(length,
             initial_state,
             input_embedding,
             cell,
             logit_fn,
             num_candidates=1,
             beam_size=100,
             gamma=0.65):
    """ A basic beam decoder

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))
    vocab_size = tf.shape(input_embedding)[0]

    # iter
    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    prev = tf.nn.log_softmax(logits)
    probs = tf.slice(prev, [0, 1], [-1, -1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size + 1
    beam_parent = indices // (vocab_size - 1)
    beam_parent = tf.reshape(
        tf.expand_dims(tf.range(batch_size), 1) + beam_parent, [-1])
    paths = tf.reshape(symbols, [-1, 1])

    candidates = [tf.to_int32(tf.zeros([batch_size, 1, length]))]
    scores = [tf.slice(prev, [0, 0], [-1, 1])]

    tf.get_variable_scope().reuse_variables()

    for i in xrange(length-1):

        if isinstance(state, tuple):
            state = tuple([tf.gather(s, beam_parent) for s in state])
        else:
            state = tf.gather(state, beam_parent)

        inputs = tf.reshape(tf.nn.embedding_lookup(input_embedding, symbols),
                            [-1, inputs_size])

        # iter
        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        prev = tf.reshape(tf.nn.log_softmax(logits),
                          [batch_size, beam_size, vocab_size])

        # add the path and score of the candidates in the current beam to the lists
        fn = lambda seq: tf.size(tf.unique(seq)[0])
        uniq_len = tf.reshape(
            tf.to_float(tf.map_fn(fn,
                                  paths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000)),
            [batch_size, beam_size])
        close_score = best_probs / (uniq_len ** gamma) + tf.squeeze(
            tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2])
        candidates.append(tf.reshape(tf.pad(paths,
                                            [[0, 0], [0, length-1-i]],
                                            "CONSTANT"),
                                     [batch_size, beam_size, length]))
        scores.append(close_score)

        prev += tf.expand_dims(best_probs, 2)
        probs = tf.reshape(
            tf.slice(prev, [0, 0, 1], [-1, -1, -1]), [batch_size, -1])
        best_probs, indices = tf.nn.top_k(probs, beam_size)

        symbols = indices % (vocab_size - 1) + 1
        beam_parent = indices // (vocab_size - 1)
        beam_parent = tf.reshape(
            tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent,
            [-1])
        paths = tf.gather(paths, beam_parent)
        paths = tf.concat([paths, tf.reshape(symbols, [-1, 1])], 1)

    # pick the topk from the candidates in the lists
    candidates = tf.reshape(tf.concat(candidates, 1), [-1, length])
    scores = tf.concat(scores, 1)
    best_scores, indices = tf.nn.top_k(scores, num_candidates)
    indices = tf.reshape(
        tf.expand_dims(
            tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices,
        [-1])
    best_candidates = tf.reshape(tf.gather(candidates, indices),
                                 [batch_size, num_candidates, length])

    return best_candidates, best_scores

# beam decoder
def stochastic_beam_dec(length,
                        initial_state,
                        input_embedding,
                        cell,
                        logit_fn,
                        num_candidates=1,
                        beam_size=100,
                        gamma=0.65):
    """ A stochastic beam decoder

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))
    vocab_size = tf.shape(input_embedding)[0]

    # iter
    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    prev = tf.nn.log_softmax(logits)
    probs = tf.slice(prev, [0, 1], [-1, -1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size + 1
    beam_parent = indices // (vocab_size - 1)
    beam_parent = tf.reshape(
        tf.expand_dims(tf.range(batch_size), 1) + beam_parent,
        [-1])
    paths = tf.reshape(symbols, [-1, 1])

    mask = tf.expand_dims(
        tf.nn.in_top_k(prev, tf.zeros([batch_size], dtype=tf.int32),
                       beam_size),
        1)
    candidates = [tf.to_int32(tf.zeros([batch_size, 1, length]))]
    scores = [tf.slice(prev, [0, 0], [-1, 1])]

    tf.get_variable_scope().reuse_variables()

    for i in xrange(length-1):

        if isinstance(state, tuple):
            state = tuple([tf.gather(s, beam_parent) for s in state])
        else:
            state = tf.gather(state, beam_parent)

        inputs = tf.reshape(
            tf.nn.embedding_lookup(input_embedding, symbols),
            [-1, inputs_size])

        # iter
        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        prev = tf.reshape(
            tf.nn.log_softmax(logits),
            [batch_size, beam_size, vocab_size])

        # add the path and score of the candidates in the current beam to the lists
        mask = tf.concat(
            [mask,
             tf.reshape(
                 tf.nn.in_top_k(tf.reshape(
                     prev, [-1, vocab_size]),
                     tf.zeros([batch_size*beam_size], dtype=tf.int32),
                     beam_size),
                 [batch_size, beam_size])],
            1)
        fn = lambda seq: tf.size(tf.unique(seq)[0])
        uniq_len = tf.reshape(
            tf.to_float(tf.map_fn(fn,
                                  paths,
                                  dtype=tf.int32,
                                  parallel_iterations=100000,
                                  swap_memory=True)),
            [batch_size, beam_size])
        close_score = best_probs / (uniq_len ** gamma) + tf.squeeze(
            tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2])
        candidates.append(tf.reshape(tf.pad(paths,
                                            [[0, 0],[0, length-1-i]],
                                            "CONSTANT"),
                                     [batch_size, beam_size, length]))
        scores.append(close_score)

        prev += tf.expand_dims(best_probs, 2)
        probs = tf.reshape(tf.slice(prev, [0, 0, 1], [-1, -1, -1]),
                           [batch_size, -1])
        best_probs, indices = tf.nn.top_k(probs, beam_size)

        symbols = indices % (vocab_size - 1) + 1
        beam_parent = indices // (vocab_size - 1)
        beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent, [-1])
        paths = tf.gather(paths, beam_parent)
        paths = tf.concat([paths, tf.reshape(symbols, [-1, 1])], 1)

    # pick the topk from the candidates in the lists
    candidates = tf.reshape(tf.concat(candidates, 1), [-1, length])
    scores = tf.concat(scores, 1)
    fillers = tf.tile(tf.expand_dims(tf.reduce_min(scores, 1) - 20.0, 1), [1, tf.shape(scores)[1]])
    scores = tf.where(mask, scores, fillers)
    indices = tf.to_int32(tf.multinomial(scores * (7**gamma), num_candidates))
    indices = tf.reshape(tf.expand_dims(tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices, [-1])
    best_candidates = tf.reshape(tf.gather(candidates, indices), [batch_size, num_candidates, length])
    best_scores = tf.reshape(tf.gather(tf.reshape(scores, [-1]), indices), [batch_size, num_candidates])

    return best_candidates, best_scores


### Copy Mechanism ###

def make_logit_fn(vocab_embedding,
                  copy_ids=None,
                  is_training=True):
    """implements logit function with copy mechanism

    """

    if copy_ids == None:
        def logit_fn(outputs):
            size = vocab_embedding.get_shape()[-1].value
            outputs_proj = fully_connected(outputs,
                                           size,
                                           is_training=is_training,
                                           scope="proj")
            logits_vocab = tf.reshape(
                tf.matmul(tf.reshape(outputs_proj,
                                     [-1, size]),
                          tf.transpose(vocab_embedding/(tf.norm(
                              vocab_embedding,
                              axis=1,
                              keep_dims=True)+1e-20))),
                tf.concat([tf.shape(outputs)[:-1],
                           tf.constant(-1, shape=[1])], 0))
            return logits_vocab
    else:
        def logit_fn(outputs):
            batch_size = tf.shape(copy_ids[0])[0]
            size = vocab_embedding.get_shape()[-1].value
            vocab_size = vocab_embedding.get_shape()[0].value
            outputs, copy_logits, copy_dists = outputs
            copy_weights = tf.unstack(tf.nn.softmax(copy_logits), axis=-1)
            assert(len(copy_ids) == len(copy_dists) and
                   len(copy_ids) == len(copy_weights)-1)
            outputs_vocab = fully_connected(outputs,
                                           size,
                                           is_training=is_training,
                                           scope="proj")
            if outputs.get_shape().ndims == 3:
                beam_size = outputs.get_shape()[1].value
                logits_vocab = tf.reshape(
                    tf.matmul(tf.reshape(outputs_vocab,
                                         [-1, size]),
                              tf.transpose(vocab_embedding/(tf.norm(
                                  vocab_embedding,
                                  axis=1,
                                  keep_dims=True)+1e-20))),
                    [batch_size, beam_size, vocab_size])
            else:
                assert(outputs.get_shape().ndims == 2)
                logits_vocab = tf.reshape(
                    tf.matmul(
                        outputs_vocab,
                        tf.transpose(vocab_embedding/(tf.norm(
                            vocab_embedding,
                            axis=1,
                            keep_dims=True)+1e-20))),
                    [batch_size, -1, vocab_size])
            beam_size = tf.shape(logits_vocab)[1]
            probs = tf.nn.softmax(logits_vocab)*tf.reshape(
                copy_weights[0],
                [batch_size, beam_size, 1])
            for i in xrange(len(copy_ids)):
                length = copy_ids[i].get_shape()[1].value
                data = tf.reshape(copy_dists[i], [batch_size, beam_size, length])
                data *= tf.reshape(copy_weights[i+1], [batch_size, beam_size, 1])
                batch_idx = tf.tile(
                    tf.expand_dims(tf.expand_dims(tf.range(batch_size), 1), 2),
                    [1, beam_size, length])
                beam_idx = tf.tile(
                    tf.expand_dims(tf.expand_dims(tf.range(beam_size), 0), 2),
                    [batch_size, 1, length])
                vocab_idx = tf.tile(
                    tf.expand_dims(copy_ids[i], 1),
                    [1, beam_size, 1])
                indices = tf.stack([batch_idx, beam_idx, vocab_idx], axis=-1)
                probs += tf.scatter_nd(indices, data, tf.shape(probs))
            logits = tf.log(probs)
            if outputs.get_shape().ndims == 2:
                logits = tf.reshape(logits, [-1, vocab_size])
            return logits
    return logit_fn
