from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline
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
    self.pool_layers = hps.pool_layers
    self.kernel_size = hps.kernel_size
    self.pool_size = hps.pool_size
    self.num_class = hps.num_class

  def init_emb(self, sess, emb_dir):
    for ld in self.loaders:
      ld.restore(sess, tf.train.get_checkpoint_state(emb_dir).model_checkpoint_path)
    # why a loop here, isn't once enough?

  def _add_placeholders(self):
    self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_dec_steps], name="inputs")
    self.conditions = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_enc_steps], name="conditions")
    self.targets = tf.placeholder(tf.int32, shape=[self.batch_size, 2], name="targets")

    self.inputs_splitted = tf.split(self.inputs, self.num_models)
    self.conditions_splitted = tf.split(self.conditions, self.num_models)
    self.targets_splitted = tf.split(self.targets, self.num_models)

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
            self.inputs, self.conditions, self.targets)
        probs.append(prob)
        # print(prob.get_shape())
    self.dis_ypred_for_auc = tf.reduce_mean(tf.cast(tf.stack(probs, 2), tf.float32), 2)
    self.output = tf.argmax(sum(probs), axis=1)
    # self.output = probs[0]
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
            self.targets_splitted[m])
        loss_train.append(tf.expand_dims(loss, 0))
        accuracy.append(acy)
        loss_cv.append(tf.expand_dims(loss, 0))
        var_dict = {}
        var_dict["embed/char_embedding"] = tf.get_variable("embed/char_embedding")
        self.loaders.append(tf.train.Saver(var_dict))
    loss_train = tf.reduce_mean(tf.concat(loss_train, 0))
    loss_cv = tf.reduce_mean(tf.concat(loss_cv, 0))
    self.accuracy = sum(accuracy) / len(accuracy)
    self.indicator = loss_cv
    self.loss = loss_train
    dis_utils.params_decay(1.0 - self.learning_rate)
    self.update = tf.contrib.layers.optimize_loss(
        self.loss, self.global_step, tf.identity(self.learning_rate),
        'Adam', gradient_noise_scale=None, clip_gradients=None, name="OptimizeLoss")

  def _seq2class_model(self, inputs, conditions, targets):
    """
    conditional sequence to class model
    """

    batch_size = inputs.get_shape()[0].value
    input_length = inputs.get_shape()[1].value
    condition_length = conditions.get_shape()[1].value
    # embedding params
    with tf.variable_scope("embed"):
      char_embedding = tf.contrib.framework.model_variable("char_embedding",
                                                           shape=[self.vocab_size, self.layer_size],
                                                           dtype=tf.float32,
                                                           initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                           collections=[tf.GraphKeys.WEIGHTS],
                                                           trainable=True)
      class_output_weights = tf.contrib.framework.model_variable("class_output_weights",
                                                                 shape=[self.num_class, self.layer_size],
                                                                 dtype=tf.float32,
                                                                 initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                                 collections=[tf.GraphKeys.WEIGHTS],
                                                                 trainable=True)

    with tf.variable_scope("encoder"):
      input_masks = tf.greater(inputs, 0)
      condition_masks = tf.greater(conditions, 0)
      emb_inputs = tf.nn.embedding_lookup(char_embedding, inputs)
      emb_conditions = tf.nn.embedding_lookup(char_embedding, conditions)
      # substitute pads with zeros
      emb_inputs = tf.reshape(tf.where(tf.reshape(input_masks, [-1]),
                                       tf.reshape(emb_inputs, [-1, self.layer_size]),
                                       tf.zeros([batch_size*input_length, self.layer_size])),
                              [batch_size, input_length, self.layer_size])
      emb_conditions = tf.reshape(tf.where(tf.reshape(condition_masks, [-1]),
                                           tf.reshape(emb_conditions, [-1, self.layer_size]),
                                           tf.zeros([batch_size*condition_length, self.layer_size])),
                                  [batch_size, condition_length, self.layer_size])
      cnn_emb_inputs = tf.expand_dims(emb_inputs, 1)
      cnn_emb_conditions = tf.expand_dims(emb_conditions, 1)
      conditional_encoder_outputs = dis_utils.CResCNN(cnn_emb_inputs, cnn_emb_conditions, self.conv_layers, self.kernel_size, self.pool_size,
                                                      pool_layers=self.pool_layers, activation_fn=tf.nn.relu, scope="cnn")

    with tf.variable_scope("projection"):
      logits = tf.matmul(
          conditional_encoder_outputs,
          tf.transpose(class_output_weights/(tf.norm(class_output_weights, axis=1, keep_dims=True)+1e-20))
      )
      # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
      accuracy = tf.count_nonzero(
          tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
      ) / logits.get_shape().as_list()[0]
      # tf.nn.in_top_k() is a little better
      return tf.nn.softmax(logits), loss, accuracy

  def run_one_batch(self, sess, inputs, conditions, targets, update=True, do_profiling=False):
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
      # to_return["output"] = self.output
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
