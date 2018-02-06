from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from utils import ensure_exists
from os.path import join as join_path
import data
import tensorflow as tf
import numpy as np
import sys


# convolutional layer
def convolution2d(inputs,
                  kernel_size,
                  pool_size=None,
                  decay=0.999,
                  activation_fn=None,
                  reuse=None,
                  scope=None):
  """Adds a 2D convolution followed by a maxpool layer.

  """
  with tf.variable_scope(scope, 'conv_inputs', [inputs], reuse=reuse):
    dtype = inputs.dtype.base_dtype
    num_filters_in = inputs.get_shape()[-1].value
    num_outputs = num_filters_in
    weights_shape = [1] + [kernel_size] + [num_filters_in, num_outputs]
    # 1, 3, emb_dim, emb_dim
    weights = tf.get_variable(name='weights',
                              shape=weights_shape,
                              dtype=dtype,
                              initializer=tf.contrib.layers.xavier_initializer(),
                              collections=[tf.GraphKeys.WEIGHTS],
                              trainable=True)
    biases = tf.get_variable(name='biases',
                             shape=[num_outputs, ],
                             dtype=dtype,
                             initializer=tf.zeros_initializer(),
                             collections=[tf.GraphKeys.BIASES],
                             trainable=True)
    outputs = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
    outputs += biases
    if pool_size:
      pool_shape = [1, 1] + [pool_size] + [1]
      outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
    if activation_fn:
      outputs = activation_fn(outputs)
    return outputs


def convolution4con(inputs,
                    kernel_size,
                    pool_size=None,
                    decay=0.999,
                    activation_fn=None,
                    inner_conv_layers=2,
                    reuse=None,
                    scope=None):
  """Adds a 2D convolution followed by a maxpool layer.

  """
  with tf.variable_scope(scope, 'conv_con', [inputs], reuse=reuse):
    dtype = inputs.dtype.base_dtype
    num_filters_in = inputs.get_shape()[-1].value
    num_outputs = num_filters_in

    for conv_i in range(inner_conv_layers):
      weights_shape = [1] + [kernel_size * (conv_i+1)] + [num_filters_in, num_outputs]
      # 1, 3, emb_dim, emb_dim
      weights = tf.get_variable(name='weights%s' % conv_i,
                                shape=weights_shape,
                                dtype=dtype,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                collections=[tf.GraphKeys.WEIGHTS],
                                trainable=True)
      biases = tf.get_variable(name='biases%s' % conv_i,
                                    shape=[num_outputs, ],
                                    dtype=dtype,
                                    initializer=tf.zeros_initializer(),
                                    collections=[tf.GraphKeys.BIASES],
                                    trainable=True)
      outputs = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
      outputs += biases
      inputs = outputs

    if pool_size:
      pool_shape = [1, 1] + [pool_size] + [1]
      outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
    if activation_fn:
      outputs = activation_fn(outputs)
    return outputs


def params_decay(decay):
  """ Add ops to decay weights and biases

  """
  params = tf.get_collection_ref(tf.GraphKeys.WEIGHTS) + tf.get_collection_ref(tf.GraphKeys.BIASES)
  while len(params) > 0:
    p = params.pop()
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                         p.assign(decay*p + (1-decay)*tf.truncated_normal(p.get_shape(), stddev=0.01)))


# ResCNN
def CResCNN(inputs, condition_emb, conv_layers, kernel_size, pool_size, pool_layers=1,
            decay=0.99999, activation_fn=tf.nn.relu, reuse=None, scope=None):
    """ a convolutaional neural net with conv2d and max_pool layers

    """
    inputs = tf.concat(values=[inputs, condition_emb])

    with tf.variable_scope(scope, "CResCNN", [inputs], reuse=reuse):
        if not pool_size:
            pool_layers = 0
        outputs = inputs
        # residual layers
        for j in range(pool_layers+1):
           if j > 0:
               pool_shape = [1, 1] + [pool_size] + [1]
               inputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
               outputs = inputs
               # why not tf.identity()
           with tf.variable_scope("layer{0}".format(j)):
               for i in range(conv_layers):
                   outputs -= convolution2d(
                       activation_fn(outputs), kernel_size, decay=decay, activation_fn=activation_fn)
    return outputs


def dump_chpt(eval_batcher, hps, model, sess, saver, eval_loss_best, early_stop=False):
    dump_model = False
    # Run evals on development set and print their perplexity.
    previous_losses = [eval_loss_best]
    eval_losses = []
    eval_accuracies = []
    stop_flag = False
    while True:
        batch = eval_batcher.next_batch()
        if not batch[0]:
            eval_batcher.reset()
            break
        eval_inputs, eval_conditions, eval_targets = \
            data.prepare_dis_pretraining_batch(batch)
        eval_inputs = np.split(eval_inputs, 2)[0]
        eval_conditions = np.split(eval_conditions, 2)[0]
        eval_targets = np.split(eval_targets, 2)[0]
        eval_results = model.run_one_batch(
            sess, eval_inputs, eval_conditions, eval_targets, update=False)
        eval_losses.append(eval_results["loss"])
        eval_accuracies.append(eval_results["accuracy"])

    eval_loss = sum(eval_losses) / len(eval_losses)
    eval_accuracy = sum(eval_accuracies) / len(eval_accuracies)
    previous_losses.append(eval_loss)
    sys.stdout.flush()
    threshold = 10
    if eval_loss > 0.99 * previous_losses[-2]:
        sess.run(model.learning_rate.assign(
            tf.maximum(hps.learning_rate_decay_factor*model.learning_rate, 1e-4)))
    if len(previous_losses) > threshold and \
            eval_loss > max(previous_losses[-threshold-1:-1]) and \
            eval_loss_best < min(previous_losses[-threshold:]):
        if early_stop:
            stop_flag = True
        else:
            stop_flag = False
            print("Proper time to stop...")
    if eval_loss < eval_loss_best:
        dump_model = True
        eval_loss_best = eval_loss
    # Save checkpoint and zero timer and loss.
    if dump_model:
        checkpoint_path = ensure_exists(join_path(hps.model_dir, "discriminator")) + "/model.ckpt"
        saver.save(sess, checkpoint_path, global_step=model.global_step)
        print("Saving the checkpoint to %s" % checkpoint_path)
    return eval_accuracy, eval_loss, stop_flag, eval_loss_best


def print_dashboard(train_accuracies, eval_loss, eval_accuracy):
    train_accuracy = sum(train_accuracies) / len(train_accuracies)
    train_accuracies = []
    print("Eval loss %.4f, train accuracy is %.4f and eval accuracy is %.4f" % (eval_loss, train_accuracy, eval_accuracy))
