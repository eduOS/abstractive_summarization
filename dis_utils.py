from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from utils import ensure_exists
from os.path import join as join_path
from termcolor import colored
import data
import tensorflow as tf
import numpy as np
import sys
from data import pad_equal_length
import beam_search
from data import PAD_TOKEN, STOP_DECODING
from utils import tran_idmat2txt
from utils import get_mixed_samples
from sklearn.utils.extmath import softmax

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
def ResCNN(inputs, conv_layers, kernel_size, pool_size, pool_layers=1,
           decay=0.99999, activation_fn=tf.nn.relu, reuse=None, scope=None):
    """ a convolutaional neural net with conv2d and max_pool layers

    """

    with tf.variable_scope(scope, "ResCNN", [inputs], reuse=reuse):
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


def eval_save_dis(sess, hps, generator, discriminator, batcher, enc_vocab, dec_vocab, dis_saver, best_f1, DEBUG):

    stop_id = dec_vocab.word2id(STOP_DECODING)
    pad_id = dec_vocab.word2id(PAD_TOKEN)

    TP = FP = FN = 0
    while True:
        batch = batcher.next_batch()
        if not batch:
            break

        # half batch of generated samples and half batch of randomed ground truth
        best_hyps = beam_search.run_beam_search(sess, generator, dec_vocab, batch)
        random_hyps = [np.random.choice(
            hyps,
            size=1,
            p=list(softmax(np.array([[hyp.avg_prob for hyp in hyps]]))[0]))
            for hyps in best_hyps]

        # # generated inputs
        outputs_ids = [hyp[0].tokens[1:] for hyp in random_hyps]
        gen_inputs = pad_equal_length(
            outputs_ids, stop_id,
            pad_id, hps.max_dec_steps)

        mixed_inputs, mixed_conditions, mixed_condition_lens, mixed_targets = get_mixed_samples(
            gen_inputs, batch)

        for i in range(2):
            inputs = sess.run(
                generator.dec_temp_embedded,
                feed_dict={generator.dec_temp_batch: mixed_inputs[i]})
            conditions = sess.run(
                generator.enc_temp_embedded,
                feed_dict={generator.enc_temp_batch: mixed_conditions[i]})

            if DEBUG:
                tran_idmat2txt(mixed_inputs[i], dec_vocab, 'debug_matrix.txt', mark="val_mixed_inputs %s" % i)
                tran_idmat2txt(mixed_conditions[i], enc_vocab, 'debug_matrix.txt', mark="val_mixed_condition %s" % i)
                tran_idmat2txt(mixed_condition_lens[i], None, 'debug_matrix.txt', mark="val_mixed_mixed_condition_lens %s" % i)
                tran_idmat2txt(mixed_targets[i], None, 'debug_matrix.txt', mark="val_mixed_targets %s" % i)

            results = discriminator.run_one_batch(sess, inputs, conditions, mixed_condition_lens[i], mixed_targets[i], update=False)
            TP += results["tp"].item()
            FP += results["fp"].item()
            FN += results["fn"].item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    if f1 > best_f1:
        best_f1 = f1
        checkpoint_path = ensure_exists(join_path(hps.model_dir, "discriminator")) + "/model.ckpt"
        dis_saver.save(sess, checkpoint_path, global_step=results["global_step"])
        print("Model with best f1 " + colored('%s', 'green') + "is saved to" + colored(" %s", 'green') % (str(f1), checkpoint_path))
        best_f1 = f1

    return f1, precision, recall, best_f1
