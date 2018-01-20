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


def calc_running_avg_loss(loss, running_avg_loss, step, decay=0.9):
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
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def get_best_loss_from_chpt(val_dir):
    ckpt = tf.train.get_checkpoint_state(val_dir)
    best_loss = None
    if ckpt:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        best_loss = reader.get_tensor(
            [key for key in var_to_shape_map if "least_val_loss" in key][0]).item()
        print(colored("the stored best loss is %s" % best_loss, "green"))
    else:
        print(colored("check point not found in %s" % val_dir, "red"))
    return best_loss


def save_ckpt(sess, model, best_loss, model_dir, model_saver,
              val_batcher, val_dir, val_saver, global_step):
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
        results_val = model.run_one_batch(sess, val_batch, update=False, gan_eval=True)
        loss_eval = results_val["loss"]
        # why there exists nan?
        if not math.isnan(loss_eval):
            losses.append(loss_eval)
        else:
            print(colored("Encountered a NAN.", 'red'))
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
