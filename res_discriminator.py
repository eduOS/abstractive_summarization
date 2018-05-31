from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline
from utils import lstm_encoder
import dis_utils


class Seq2ClassModel(object):
  """Sequence-to-class model with multiple buckets.

     implements multiple classifiers
  """

  def __init__(self, hps):
    # model, vocab_size, num_class, buckets, size, num_layers, max_gradient,
    # batch_size, learning_rate, learning_rate_decay_factor, cell_type="GRU",
    # is_decoding=False):
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
    # TODO: is_decoding affects the graph loading? since the graph is differenct
    # if this differs
    self.hps = hps
    self.is_decoding = ('gan' in hps.mode)
    # self.is_decoding = True
    self.vocab_size = hps.dis_vocab_size
    with tf.variable_scope("OptimizeLoss"):
      self.learning_rate = tf.get_variable("learning_rate", [], trainable=False, initializer=tf.constant_initializer(hps.dis_lr))
    self.cell_type = hps.cell_type
    self.global_step = tf.Variable(0, trainable=False)
    self.mode = hps.mode
    self.num_models = hps.num_models  # only the negative for the reward
    if hps.mode == "pretrain_dis":
        self.batch_size = hps.batch_size * self.num_models * 2
    else:
        self.batch_size = hps.batch_size * self.num_models
    self.max_enc_steps = hps.max_enc_steps
    self.max_dec_steps = hps.max_dec_steps
    self.layer_size = hps.layer_size
    self.conv_layers = hps.conv_layers
    self.kernel_size = hps.kernel_size
    self.pool_layers = hps.pool_layers
    self.pool_size = hps.pool_size
    self.num_class = hps.num_class

  def _add_placeholders(self):
    # all the inputs and conditions should not be vocabulary extened
    self.inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_dec_steps, self.layer_size], name="inputs")
    self.conditions = tf.placeholder(tf.float32, shape=[self.batch_size, None, self.layer_size], name="conditions")
    self.condition_lens = tf.placeholder(tf.int32, [self.batch_size], name='condition_lens')
    self.targets = tf.placeholder(tf.float32, shape=[self.batch_size], name="targets")

    self.inputs_splitted = tf.split(self.inputs, self.num_models)
    self.conditions_splitted = tf.split(self.conditions, self.num_models)
    self.targets_splitted = tf.split(self.targets, self.num_models)
    self.condition_lens_splitted = tf.split(self.condition_lens, self.num_models)
    self.rand_unif_init = tf.random_uniform_initializer(-self.hps.rand_unif_init_mag, self.hps.rand_unif_init_mag, seed=123)

  def build_graph(self):
    self._add_placeholders()
    # build the buckets
    # self.outputs, self.losses, self.updates, self.indicators = [], [], [], []
    probs = []
    #  ------------------ for evaluation ----------------------
    # return tf.nn.softmax(logits), loss, accuracy
    for m in xrange(self.num_models):
        with tf.variable_scope("model"+str(m)):
            prob, _, _ = self._seq2class_model(
                self.inputs, self.conditions, self.condition_lens, self.targets)
            probs.append(prob)
        # print(prob.get_shape())
    self.dis_ypred_for_auc = tf.reduce_mean(tf.cast(tf.stack(probs, 1), tf.float32), 1)
    # would this lead the value run out to be a list of only one two
    # dimensional numpy array?
    #  ------------------ for training ----------------------
    loss_train = []
    loss_cv = []
    self.loaders = []
    accuracy = []
    for m in xrange(self.num_models):
        with tf.variable_scope("model"+str(m), reuse=True):
            _, loss, acy = self._seq2class_model(
                self.inputs_splitted[m], self.conditions_splitted[m],
                self.condition_lens_splitted[m], self.targets_splitted[m])
            loss_train.append(loss)
            accuracy.append(acy)
            loss_cv.append(loss)
    loss_train = sum(loss_train) / len(loss_train)
    loss_cv = tf.reduce_mean(tf.concat(loss_cv, 0))
    self.accuracy = sum(accuracy) / len(accuracy)
    self.indicator = loss_cv
    self.loss = loss_train
    # dis_utils.params_decay(1.0 - self.learning_rate)
    self.update = tf.contrib.layers.optimize_loss(
        self.loss, self.global_step, tf.identity(self.learning_rate),
        'RMSProp', gradient_noise_scale=None, clip_gradients=None, name="OptimizeLoss")

  def _seq2class_model(self, emb_inputs, emb_conditions, condition_lens, targets):
      """
      conditional sequence to class model
      """

      # embedding params
      with tf.variable_scope("embed"):
          condition_weights = tf.contrib.framework.model_variable("condition_weights",
                                                                  shape=[2 * self.hps.hidden_dim, self.layer_size],
                                                                  dtype=tf.float32,
                                                                  initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                                  collections=[tf.GraphKeys.WEIGHTS],
                                                                  trainable=True)

      with tf.variable_scope("input_encoder"):
          cnn_emb_inputs = tf.expand_dims(emb_inputs, 1)
          cnn_outputs = dis_utils.ResCNN(
              cnn_emb_inputs, self.conv_layers, self.kernel_size, self.pool_size,
              pool_layers=self.pool_layers, activation_fn=tf.nn.relu, scope="cnn")
          cnn_outputs = tf.squeeze(cnn_outputs, [1])
          # would it be better if use reduce_sum ?
          input_emb = tf.reduce_max(cnn_outputs, axis=1)

      with tf.variable_scope("condition_encoder"):
          _, condition_emb = lstm_encoder(
              emb_conditions, condition_lens,
              hidden_dim=self.hps.hidden_dim, rand_unif_init=self.rand_unif_init)

          condition_emb = tf.concat(values=[condition_emb.c, condition_emb.h], axis=1)
          with tf.variable_scope("conduction_projection"):
              condition_emb = tf.matmul(condition_emb, condition_weights)
          # (batch_size, 2*hidden_dim)
          # self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
          #     .minimize(self.d_loss_reg, var_list=self.d_net.vars)
          # self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
          #     .minimize(self.g_loss_reg, var_list=self.g_net.vars)

      with tf.variable_scope("dis_loss"):
          dot_product = tf.reduce_sum(tf.multiply(input_emb, condition_emb), axis=1)
          # loss = tf.nn.weighted_cross_entropy_with_logits(logits=dot_product, labels=targets, pos_weight=2)
          loss = tf.reduce_mean(tf.boolean_mask(dot_product, targets)) - \
              tf.reduce_mean(tf.boolean_mask(dot_product, tf.logical_not(targets)))

          prob = tf.sigmoid(dot_product)
          pred = tf.where(tf.less(tf.fill(tf.shape(prob), 0.5), prob),
                          tf.fill(tf.shape(prob), 1.0), tf.fill(tf.shape(prob), 0.0))
          accuracy = tf.count_nonzero(tf.equal(pred, targets)) / tf.cast(tf.shape(pred)[0], tf.int64)
      return prob, loss, accuracy

  def run_one_batch(self, sess, inputs, conditions, condition_lens, targets, update=True, do_profiling=False):
      """Run a step of the model feeding the given inputs.

      Args:
        session: tensorflow session to use.
        inputs: list of numpy int vectors to feed as encoder inputs.
        conditions: the article
        targets: target class of the samples
        update: whether to do the update or not.
        do_profiling: if to profile the model

      Returns:
        A triple consisting of gradient norm (or None if we did not do backward),
        average perplexity, and the outputs.

      Raises:
        ValueError: if length of encoder_inputs, decoder_inputs, or
          target_weights disagrees with bucket size for the specified bucket_id.
      """
      # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
      input_feed = {}

      input_feed[self.inputs] = inputs
      input_feed[self.conditions] = conditions
      input_feed[self.condition_lens] = condition_lens

      input_feed[self.targets] = targets

      to_return = {
          "global_step": self.global_step,
          "learning_rate": self.learning_rate,
      }
      # Output feed.
      if update:
          to_return["loss"] = self.loss
          to_return["accuracy"] = self.accuracy
          to_return["update"] = self.update

      else:
          to_return["loss"] = self.indicator
          to_return["accuracy"] = self.accuracy

      if do_profiling:
          self.run_metadata = tf.RunMetadata()
          outputs = sess.run(to_return, input_feed,
                             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
          trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
          trace_file = open('timeline.ctf.json', 'w')
          trace_file.write(trace.generate_chrome_trace_format())
          trace_file.close()
      else:
          outputs = sess.run(to_return, input_feed)

      return outputs
