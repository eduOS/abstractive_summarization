from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline
from utils import model_utils


class Seq2ClassModel(object):
  """Sequence-to-class model with multiple buckets.

     implements multiple classifiers
  """

  def __init__(self, hps):
  # model, vocab_size, num_class, buckets, size, num_layers, max_gradient, batch_size, learning_rate, learning_rate_decay_factor, cell_type="GRU", is_decoding=False):
    """Create the model.

    Args:
      vocab_size: size of the vocabulary.
      num_class: num of output classes
      buckets: a list of size of the input sequence
      size: number of units in each layer of the model.
      num_layers: number of layers.
      max_gradient: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      cell_type: choose between LSTM cells and GRU cells.
      is_decoding: if set, we do decoding instead of training.
    """
    self.vocab_size = vocab_size
    self.num_class = num_class
    self.buckets = buckets
    with tf.variable_scope("OptimizeLoss"):
      self.learning_rate = tf.get_variable("learning_rate", [], trainable=False, initializer=tf.constant_initializer(learning_rate))
    self.cell_type = cell_type
    self.global_step = tf.Variable(0, trainable=False)
    self.is_decoding = is_decoding
    self.num_models = 32
    self.batch_size = batch_size * self.num_models

  def init_emb(sess, emb_dir):
    for ld in self.loaders:
      ld.restore(sess, tf.train.get_checkpoint_state(emb_dir).model_checkpoint_path)

  def build_graph(self):
    # Feeds for inputs.
    self.encoder_inputs = []
    for i in xrange(self.buckets[-1]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))

    # Feeds target
    self.targets = tf.placeholder(tf.int32, shape=[None], name="target")
    if not self.is_decoding:
      self.encoder_inputs_splitted = []
      self.targets_splitted = tf.split(self.targets, self.num_models)
      for x in self.encoder_inputs:
        self.encoder_inputs_splitted.append(tf.split(x, self.num_models))

    # build the buckets
    self.outputs, self.losses, self.updates, self.indicators = [], [], [], []
    for i in xrange(len(self.buckets)):
      inputs_len = self.buckets[i]
      if self.is_decoding:
        probs = []
        for m in xrange(self.num_models):
          with tf.variable_scope("model"+str(m)):
            probs.append(self.get_loss(self.encoder_inputs[:inputs_len], self.targets, self.is_decoding)[0])
        self.outputs.append(tf.argmax(sum(probs), axis=1))
      else:
        loss_train = []
        loss_cv = []
        self.loaders = []
        for m in xrange(self.num_models):
          with tf.variable_scope("model"+str(m)) as sc:
            loss_train.append(tf.expand_dims(
                self.get_loss(list(np.transpose(self.encoder_inputs_splitted)[m][:inputs_len]), self.targets_splitted[m], self.is_decoding)[0], 0))
            sc.reuse_variables()
            loss_cv.append(tf.expand_dims(
                self.get_loss(self.encoder_inputs[:inputs_len], self.targets, self.is_decoding)[0], 0))
            var_dict = {}
            var_dict["embed/char_embedding"] = tf.get_variable("embed/char_embedding")
            self.loaders.append(tf.train.Saver(var_dict))
        loss_train = tf.reduce_mean(tf.concat(loss_train, 0))
        loss_cv = tf.reduce_mean(tf.concat(loss_cv, 0))
        self.indicators.append(loss_cv)
        self.losses.append(loss_train)
        model_utils.params_decay(1.0 - self.learning_rate)
        self.updates.append(tf.contrib.layers.optimize_loss(self.losses[i], self.global_step,
                                                            tf.identity(self.learning_rate), 'Adam', gradient_noise_scale=None, clip_gradients=None,
                                                            name="OptimizeLoss"))
        del tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)[:]
      tf.get_variable_scope().reuse_variables()

  def get_loss(self, querys, targets, is_decoding):

    batch_size = tf.shape(querys[0])[0]

    # embedding params
    with tf.variable_scope("embed"):
      char_embedding = tf.contrib.framework.model_variable("char_embedding",
                                                           shape=[self.vocab_size, size],
                                                           dtype=tf.float32,
                                                           initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                           collections=tf.GraphKeys.WEIGHTS,
                                                           trainable=True)
      class_output_weights = tf.contrib.framework.model_variable("class_output_weights",
                                                            shape=[self.num_class, size],
                                                            dtype=tf.float32,
                                                            initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                            collections=tf.GraphKeys.WEIGHTS,
                                                            trainable=True)

    with tf.variable_scope("encoder"):
      encoder_inputs = tf.stack(querys, 1)
      query_masks = tf.greater(encoder_inputs, 0)
      inputs = tf.nn.embedding_lookup(char_embedding, encoder_inputs)
      inputs = tf.reshape(tf.where(tf.reshape(query_masks, [-1]),
                                   tf.reshape(inputs, [-1, size]),
                                   tf.zeros([batch_size*len(querys), size])), [batch_size, len(querys), size])
      cnn_inputs = tf.expand_dims(inputs, 1)
      cnn_outputs = model_utils.ResCNN(cnn_inputs, num_layers, [1, 3], [1, 2],
                                       pool_layers=2, activation_fn=tf.nn.relu, is_training=not is_decoding, scope="cnn")
      cnn_outputs = tf.squeeze(cnn_outputs, [1])
      projection_input = tf.reduce_max(cnn_outputs, axis=1)

    with tf.variable_scope("projection"):
      logits = tf.matmul(projection_input, tf.transpose(class_output_weights/(tf.norm(class_output_weights, axis=1, keep_dims=True)+1e-20)))
      if is_decoding:
        return tf.nn.softmax(logits), None
      else:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        return loss, loss

  def run_one_step(self, session, encoder_inputs, targets,
                   bucket_id, update=False, do_profiling=False):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      targets: target class of the samples
      bucket_id: which bucket of the model to use.
      update: whether to do the update or not.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket, %d != %d." % (len(encoder_inputs), encoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    if not self.is_decoding:
      input_feed[self.targets.name] = targets

    # Output feed.
    if self.is_decoding:
      output_feed = [self.outputs[bucket_id]]
    elif update:
      output_feed = [self.losses[bucket_id],  # Update Op that does SGD.
                     self.updates[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.indicators[bucket_id]]  # Loss for this batch.

    if do_profiling:
      self.run_metadata = tf.RunMetadata()
      outputs = session.run(output_feed, input_feed,
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
      trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
      trace_file = open('timeline.ctf.json', 'w')
      trace_file.write(trace.generate_chrome_trace_format())
      trace_file.close()
    else:
      outputs = session.run(output_feed, input_feed)

    return outputs[0]
