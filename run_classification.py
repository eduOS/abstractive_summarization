from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data
import seq2class_model
import batcher

# Model parameters
tf.app.flags.DEFINE_string("model", "cnn_classifier", "name of the model function.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Cell type")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("num_class", 50, "num of output classes.")
tf.app.flags.DEFINE_string("buckets", "9,12,20,40", "buckets of different lengths")

# Training parameters
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.5,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_boolean("early_stop", False,
                            "Set to True to turn on early stop.")
tf.app.flags.DEFINE_integer("max_steps", -1,
                            "max number of steps to train")

# Misc
tf.app.flags.DEFINE_string("data_dir", "./js_corpus", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 5000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("gpu_id", 0, "Select which gpu to use.")

# Mode
tf.app.flags.DEFINE_boolean("interactive_test", False,
                            "Set to True for interactive testing.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Run a test on the eval set.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = map(int, re.split(' |,|;', FLAGS.buckets))


def read_data(data_path, vocab, cmd_class, separate=False):
  """Read data from file and put into buckets.

  """
  if not separate:
    data_set = [[] for _ in _buckets]
  else:
    data_set = [[[] for _ in xrange(cmd_class.size())] for _ in _buckets]
  with open(data_path, 'r') as f:
    for line in f:
      # preprocess the sample
      text, cmd = line.strip().split('\t')
      seq = data.sentence2ids(text, vocab)
      target = cmd_class.key2idx(cmd)
      # insert to one of the buckets
      for bucket_id, bucket_size in enumerate(_buckets):
        if len(seq) <= bucket_size:
          seq = seq + [0] * (bucket_size-len(seq))
          if not separate:
            data_set[bucket_id].append([seq, target])
          else:
            data_set[bucket_id][target].append([seq, target])
          break

  return data_set


def create_model(session, is_decoding):
  """Create translation model and initialize or load parameters in session."""
  model = seq2class_model.Seq2ClassModel(
      FLAGS.model,
      FLAGS.vocab_size, FLAGS.num_class, _buckets,
      FLAGS.size, FLAGS.num_layers,
      FLAGS.max_gradient, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      FLAGS.cell_type, is_decoding=is_decoding)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+".meta"):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    for ld in model.loaders:
      ld.restore(session, tf.train.get_checkpoint_state("pretrained").model_checkpoint_path)
  return model


def train():
  """Train a text classifier."""

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  with tf.Session(config=config) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
      model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print("Reading development and training data.")
    train_path = os.path.join(FLAGS.data_dir, "train")
    dev_path = os.path.join(FLAGS.data_dir, "dev")
    vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
    cmd_class = data_utils.Vocab(os.path.join(FLAGS.data_dir, "cmd_class"))
    dev_set = read_data(dev_path, vocab, cmd_class)
    train_set = read_data(train_path, vocab, cmd_class, separate=True)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    eval_loss_best = sys.float_info.max
    previous_losses = [eval_loss_best]
    while True:
      start_time = time.time()
      if True:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        encoder_inputs, targets, _ = batcher.get_batch(train_set[bucket_id], balance=True)
      step_loss = model.step(sess, encoder_inputs, targets, bucket_id, update=True)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        print("global step %d learning rate %.4f step-time %.4f loss "
              "%.4f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
        dump_model = True
        if FLAGS.early_stop:
          dump_model = False
          # Run evals on development set and print their perplexity.
          eval_losses = []
          for bucket_id_eval in xrange(len(_buckets)):
            while True:
              encoder_inputs_eval, targets_eval, vflag = batcher.get_batch(dev_set[bucket_id_eval], put_back=False)
              if vflag:
                eval_losses.append(model.step(sess, encoder_inputs_eval, targets_eval, bucket_id_eval, update=False))
              else:
                break
          eval_loss = sum(eval_losses) / len(eval_losses)
          print("  eval loss %.4f" % eval_loss)
          previous_losses.append(eval_loss)
          dev_set = read_data(dev_path, vocab, cmd_class)
          sys.stdout.flush()
          threshold = 10
          if eval_loss > 0.99 * previous_losses[-2]:
            sess.run(model.learning_rate.assign(tf.maximum(FLAGS.learning_rate_decay_factor*model.learning_rate, 1e-4)))
          if len(previous_losses) > threshold and eval_loss > max(previous_losses[-threshold-1:-1]) and eval_loss_best < min(previous_losses[-threshold:]):
            break
          if eval_loss < eval_loss_best:
            dump_model = True
            eval_loss_best = eval_loss
        # Save checkpoint and zero timer and loss.
        if dump_model:
          checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        if current_step >= FLAGS.max_steps:
          break


def interactive_test():

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  with tf.Session(config=config) as sess:
    # Create model and load parameters.
    with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
      model = create_model(sess, True)
      model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    vocab = data.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
    cmd_class = data.Vocab(os.path.join(FLAGS.data_dir, "cmd_class"))

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data.sentence2ids(tf.compat.as_bytes(sentence), vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      token_ids = token_ids + [0] * (_buckets[bucket_id] - len(token_ids))
      encoder_inputs = [np.array([i]) for i in token_ids]
      # Get output logits for the sentence.
      outputs = model.step(sess, encoder_inputs, None, bucket_id)
      # Print output class.
      outputs = cmd_class.idx2key(int(outputs))
      print(outputs)
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def test():
  """Test the text classification model."""
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  with tf.Session(config=config) as sess:
    print("test for text classification model.")
    with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
      model = create_model(sess, True)

    # Read test set data
    eval_path = os.path.join(FLAGS.data_dir, "eval")
    vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
    cmd_class = data_utils.Vocab(os.path.join(FLAGS.data_dir, "cmd_class"))
    print("Reading eval data")
    eval_set = read_data(eval_path, vocab, cmd_class)

    # Loop through all the test sample
    total = 0
    error = 0
    stats = {}
    # class_table = ["CommandChat", "Not_CommandChat"]
    with open(os.path.join(FLAGS.data_dir, "test_error_samples"), 'w') as f:
      for bucket_id in xrange(len(eval_set)):
        while True:
          encoder_inputs, targets, vflag = batcher.get_batch(
              eval_set[bucket_id], put_back=False)
          if vflag:
            outputs = model.step(sess, encoder_inputs, targets, bucket_id)
            mask = outputs != targets
            error += sum(mask)
            total += len(outputs)
            # collect stats
            for i in xrange(len(outputs)):
              if outputs[i] == targets[i]:
                if not outputs[i] in stats:
                  stats[outputs[i]] = {"tp": 0, "fp": 0, "fn": 0}
                stats[outputs[i]]["tp"] += 1
              else:
                if not outputs[i] in stats:
                  stats[outputs[i]] = {"tp": 0, "fp": 0, "fn": 0}
                if not targets[i] in stats:
                  stats[targets[i]] = {"tp": 0, "fp": 0, "fn": 0}
                stats[outputs[i]]["fp"] += 1
                stats[targets[i]]["fn"] += 1
            # write the error samples
            if sum(mask) > 0:
              seqs = list(np.transpose(np.array(encoder_inputs))[mask])
              targets = list(targets[mask])
              outputs = list(outputs[mask])
              for i in xrange(len(seqs)):
                line = data_utils.token_ids_to_sentence(list(seqs[i]), vocab) + \
                    "\ttarget: " + cmd_class.idx2key(int(targets[i])) + \
                    "\toutput: " + cmd_class.idx2key(int(outputs[i])) + "\n"
                f.write(line)

          else:
            break

    # show stats
    for i in stats:
      precision = float(stats[i]["tp"]) / float(stats[i]["tp"] + stats[i]["fp"]) if \
          float(stats[i]["tp"] + stats[i]["fp"]) > 0 else 0.0
      recall = float(stats[i]["tp"]) / float(stats[i]["tp"] + stats[i]["fn"]) if \
          float(stats[i]["tp"] + stats[i]["fn"]) > 0 else 0.0
      f1 = 2.0*precision*recall / (precision+recall) if precision+recall > 0 else 0.0
      print(cmd_class.idx2key(i) + ": Precision: " + str(precision) + " Recall: " + str(recall) + " F1: " + str(f1))
      # print (class_table[i] + ": Precision: " + str(precision) + " Recall: " + str(recall) + " F1: " + str(f1))
    print("Test set error rate: " + str(float(error) / float(total) * 100.0) + "%")


def main(_):
  if FLAGS.test:
    test()
  elif FLAGS.interactive_test:
    interactive_test()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
