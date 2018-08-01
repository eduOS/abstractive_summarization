# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import math
import datetime
import utils
from os.path import join as join_path
from termcolor import colored
from tensorflow.python import pywrap_tensorflow
from utils import linear
from tensorflow.python.ops import variable_scope
from dis_utils import convolution2d


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for
    coverage, and save as new checkpoint"""
    print("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=utils.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = utils.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()
    # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def calc_running_avg_loss(loss, running_avg_loss, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
        loss: loss on the most recent eval step
        running_avg_loss: running_avg_loss so far
        step: training iteration step
        decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
        running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    return running_avg_loss


def get_best_loss_from_chpt(val_dir, key_name="least_val_loss"):
    ckpt = tf.train.get_checkpoint_state(val_dir)
    best_loss = None
    if ckpt:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        best_loss = reader.get_tensor(
            [key for key in var_to_shape_map if key_name in key][0]).item()
        print(colored("the stored best loss is %s" % best_loss, 'green'))
    return best_loss


def save_ckpt(sess, model, best_loss, model_dir, model_saver,
              val_batcher, val_dir, val_saver, global_step, gan_eval=True):
    """
    save model to model dir or evaluation directory
    """
    if not val_batcher:
        return None, best_loss

    saved = False
    val_save_path = join_path(val_dir, "best_model")
    model_save_path = join_path(model_dir, "model")

    losses = []
    while True:
        val_batch = val_batcher.next_batch()
        if not val_batch:
            break
        results_val = model.run_one_batch(sess, val_batch, update=False, gan_eval=gan_eval)
        loss_eval = results_val["loss"]
        # why there exists nan?
        if not (math.isnan(loss_eval) or math.isinf(loss_eval)):
            losses.append(loss_eval)
        else:
            print(val_batch)
            for at, ab in zip(val_batch.original_articles, val_batch.original_abstracts):
                print(at)
                print(ab)
            print(colored("Encountered a NAN or INF.", 'red'))
    eval_loss = sum(losses) / len(losses)
    if best_loss is None or eval_loss < best_loss:
        sess.run(model.least_val_loss.assign(eval_loss))
        print(
            'Found new best model with %.3f evaluation loss. Saving to %s %s' %
            (eval_loss, val_save_path,
                datetime.datetime.now().strftime("on %m-%d at %H:%M")))
        val_saver.save(sess, val_save_path, global_step=global_step)
        print("Model is saved to" + colored(" %s", 'green') % val_save_path)
        saved = True
        best_loss = eval_loss

    if not saved:
        model_saver.save(sess, model_save_path, global_step=global_step)
        print("Model is saved to" + colored(" %s", 'yellow') % model_save_path)

    return eval_loss, best_loss


def get_local_global_features(inputs, local_attention_layers, attention_vec_size, conv_layers=3, kernel_size=3, pool_size=3,
                              decay=0.99999, activation_fn=tf.nn.relu, reuse=None, scope=None):
    """ a convolutaional neural net with conv2d and max_pool layers

    """

    local_attentions = []
    with tf.variable_scope(scope, "ResCNN", [inputs], reuse=reuse):
        outputs = inputs
        # residual layers
        for j in range(local_attention_layers):
           if j > 0:
               pool_shape = [1, 1] + [pool_size] + [1]
               inputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
               outputs = inputs
               # why not tf.identity()
           with tf.variable_scope("layer{0}".format(j)):
               for i in range(conv_layers):
                   outputs = convolution2d(
                       activation_fn(outputs), kernel_size, decay=decay, activation_fn=activation_fn)
               attention_outputs = tf.reduce_max(outputs, axis=1)
               attention_outputs = tf.squeeze(attention_outputs, [1])
               # attention_outputs = linear(attention_outputs, attention_vec_size, True)
               local_attentions.append(attention_outputs)

        with tf.variable_scope("final_layer"):
            for i in range(conv_layers):
                outputs = convolution2d(
                    activation_fn(outputs), kernel_size, decay=decay, activation_fn=activation_fn)
            # global_attention = linear(attention_outputs, attention_vec_size, True, scope="global_attention")
            attention_outputs = tf.reduce_max(outputs, axis=1)
            global_attention = tf.squeeze(attention_outputs, [1])

    return local_attentions, global_attention


def global_selective_fn(encoder_outputs, global_feature):
    enc_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
    dynamic_enc_steps = tf.shape(enc_outputs)[0]
    output_dim = encoder_outputs.get_shape()[-1]
    sele_ar = tf.TensorArray(dtype=tf.float32, size=dynamic_enc_steps)

    with tf.variable_scope('selective'):

        def cond(_e, i, _m):
            return i < dynamic_enc_steps

        def mask_fn(inputs, i, sele_ar):
            sGate = tf.sigmoid(
                linear(inputs[i], output_dim, True, scope="w") +
                linear(global_feature, output_dim, True, scope="u"))
            _h = inputs[i] * sGate
            sele_ar = sele_ar.write(i, _h)
            if i == tf.constant(0, dtype=tf.int32):
                tf.get_variable_scope().reuse_variables()
            return inputs, i+1, sele_ar

        _, _, sele_ar = tf.while_loop(
            cond, mask_fn, (enc_outputs, tf.constant(0, dtype=tf.int32), sele_ar))
        new_enc_outputs = tf.transpose(sele_ar.stack(), perm=[1, 0, 2])
    return new_enc_outputs


def global_attention(encoder_states, local_attention_layers, attention_vec_size):
    # Get the global attention: https://openreview.net/pdf?id=HyzbhfWRW

    local_features, global_feature = get_local_global_features(
        encoder_states, local_attention_layers, attention_vec_size)
    hidden_dim = global_feature.get_shape().as_list()[-1]

    with variable_scope.variable_scope("combined_attention"):
        c = []
        for n, lf in enumerate(local_features):
            if n > 0:
                variable_scope.get_variable_scope().reuse_variables()
            c_i = linear(lf + global_feature, 1, True, scope="u")
            c.append(tf.squeeze(c_i))

    # (batch_size, 3, hidden_num) dot (batch_size, 3, hidden_dim)
    weighted_local_features = tf.multiply(
        tf.stack(local_features, axis=1),
        tf.tile(tf.expand_dims(tf.nn.softmax(tf.stack(c, axis=1)), 2), [1, 1, hidden_dim])
    )

    global_feature = linear(tf.reduce_sum(weighted_local_features, axis=1),
                            attention_vec_size, True, scope="global_feature")

    # add the global feature to the encoder state forming the new encoder
    # state
    encoder_states = global_selective_fn(tf.squeeze(encoder_states, [2]), global_feature)
    # global selective function should be improved
    encoder_states = tf.expand_dims(encoder_states, axis=2)
    return encoder_states
