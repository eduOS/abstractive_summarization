from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from collections import namedtuple
import numpy as np
import sys
import os
import datetime
import gen_utils
import time
import re
import data
from batcher import GenBatcher, DisBatcher
from decode import BeamSearchDecoder
from pointer_generator import PointerGenerator
from rollout import Rollout
from data import gen_vocab2dis_vocab

from res_discriminator import Seq2ClassModel
from data import Vocab
PAD_TOKEN = "[PAD]"
STOP_DECODING = '[STOP]'

# tf.logging.set_verbosity(tf.logging.ERROR)
tf.app.flags.DEFINE_string(
    'mode', 'train',
    'must be one of pretrain_gen/pretrain_dis/train_gan/decode')
# ------------------------------------- common
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# ------------------------------------- discriminator

# Model parameters
tf.app.flags.DEFINE_integer("layer_size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("conv_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("pool_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("kernel_size", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("pool_size", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Cell type")
tf.app.flags.DEFINE_integer("dis_vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("num_class", 2, "num of output classes.")
tf.app.flags.DEFINE_string("buckets", "9,12,20,40", "buckets of different lengths")
tf.app.flags.DEFINE_integer("num_models", 16, "Size of each model layer.")

# Training parameters
tf.app.flags.DEFINE_float("dis_lr", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.5, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("dis_max_gradient", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_boolean("early_stop", False, "Set to True to turn on early stop.")
tf.app.flags.DEFINE_integer("max_steps", -1, "max number of steps to train")

# Misc
tf.app.flags.DEFINE_string("dis_data_dir", "./js_corpus", "Data directory")
tf.app.flags.DEFINE_string("model_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_string("val_dir", "./model/val/", "Training directory.")
tf.app.flags.DEFINE_integer("gpu_id", 0, "Select which gpu to use.")

# Mode
tf.app.flags.DEFINE_boolean("interactive_test", False, "Set to True for interactive testing.")
tf.app.flags.DEFINE_boolean("test", False, "Run a test on the eval set.")

# ------------------------------------- generator

# Where to find data
tf.app.flags.DEFINE_string(
    'data_path', './data/', 'Path expression to tf.Example datafiles and vocabulary \
    Can include wildcards to access multiple datafiles.')

#  data_path/gen_vocab: vocabulary for the generator
#  data_path/dis_vocab: vocabulary for the generator
#  data_path/[decode/eval]_[positive/negative/source]: the data for the discriminator
#  data_path/[train/val/test]_\d+.bin: the data for the generator

# Important settings
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a'
                            'fixed checkpoint, i.e. take the current checkpoint, and use it to'
                            'produce one summary for each example in the dataset, writethesummaries'
                            'to file and then get ROUGE scores for the whole dataset. If False'
                            '(default), run concurrent decoding, i.e. repeatedly load latest'
                            'checkpoint, use it to produce summaries forrandomly-chosenexamples and'
                            'log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', './log/', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in adirectory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
# if batch_size is one and beam size is not one in the decode mode then the beam
# search is the same as the original beam search
tf.app.flags.DEFINE_integer('max_enc_steps', 80, 'max timesteps of encoder (max source text tokens)')  # 400
tf.app.flags.DEFINE_integer('max_dec_steps', 15, 'max timesteps of decoder (max summary tokens)')  # 100
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 8, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('gen_vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in'
                            ' order. If the vocabulary file contains fewer words than this number,'
                            ' or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('gen_lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('gen_max_gradient', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.app.flags.DEFINE_boolean('segment', True, 'If True, the source text is segmented, then max_enc_steps and max_dec_steps should be much smaller')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL '
                            'paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards.'
                            'i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
# coverage can be only used while decoding either in the gan or in the pretraining
tf.app.flags.DEFINE_float('cov_loss_wt', 1, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', True, 'Convert a non-coverage model to a coverage model. '
                            'Turn this on and run in train mode. \ Your current model will be copied to a new version '
                            '(same name with _cov_init appended)\ that will be ready to run with coverage flag turned on,\ for the coverage training stage.')

FLAGS = tf.app.flags.FLAGS
_buckets = map(int, re.split(' |,|;', FLAGS.buckets))

assert FLAGS.mode in ["pretrain_gen", "pretrain_dis", "train_gan", "decode"]

if FLAGS.mode == "train_gan":
    FLAGS.single_pass = False

if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

if not os.path.exists(FLAGS.val_dir):
    os.makedirs(FLAGS.val_dir)


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
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def pretrain_generator(model, batcher, sess_context_manager, batcher_val, saver):
    """Repeatedly runs training iterations, logging loss to screen and writing
    summaries"""
    print("starting run_training")
    bestmodel_save_path = os.path.join(FLAGS.val_dir, 'bestmodel')
    coverage_loss = None
    hps = model.hps
    # this is where checkpoints of best models are saved
    running_avg_loss = 0
    # the eval job keeps a smoother, running average loss to tell it when to
    # implement early stopping
    best_loss = None  # will hold the best loss achieved so far
    with sess_context_manager as sess:
        print_gap = 1000
        while True:  # repeats until interrupted
            batch = batcher.next_batch()

            # print('running training step...')
            t0 = time.time()
            results = model.run_one_step(sess, batch)
            t1 = time.time()
            step = results['global_step']
            # print('seconds for training step: %.3f', t1-t0)

            loss = results['loss']
            # print('loss: %f', loss)  # print the loss to screen
            if hps.coverage:
                coverage_loss = results['coverage_loss']
                # print the coverage loss to screen
                # print("coverage_loss: %f", coverage_loss)

            running_avg_loss = calc_running_avg_loss(
                np.asscalar(loss), running_avg_loss, step)

            if step * 2 % print_gap == 0:
                model_path = os.path.join(hps.model_dir, "model")
                saver.save(sess, model_path, global_step=step)
                print(
                    'Saving model with %.3f running_avg_loss. Saving to %s %s' %
                    (running_avg_loss, model_path,
                        datetime.datetime.now().strftime("on %m-%d at %H:%M")))

            if step % print_gap == 0:
                results_val = model.run_one_step(
                    sess, batcher_val.next_batch(), update=False)
                loss_val = results_val["loss"]
                if best_loss is None or loss_val < best_loss:
                    print(
                        'Found new best model with %.3f running_avg_loss. Saving to %s %s' %
                        (loss_val, bestmodel_save_path,
                         datetime.datetime.now().strftime("on %m-%d at %H:%M")))
                    saver.save(sess, bestmodel_save_path, global_step=step)
                    best_loss = loss_val

                print(
                  "\nDashboard updated %s, finished steps:\t%s\n"
                  "\tBatch size:\t%s\n"
                  "\tVocabulary size:\t%s\n"
                  "\tArticles trained:\t%s\n"
                  "\tTotal training time approxiately:\t%.4f hours\n"
                  "\tCurrent speed:\t%.4f seconds/article\n"
                  "\tLoss:\t%.4f;"
                  "\tand coverage loss:\t%s\n" % (
                    datetime.datetime.now().strftime("on %m-%d at %H:%M"),
                    step,
                    hps.batch_size,
                    hps.gen_vocab_size,
                    hps.batch_size * step,
                    (t1-t0) * step / 3600,
                    (t1-t0) / hps.batch_size,
                    running_avg_loss,
                    coverage_loss if hps.coverage else "not set",
                  )
                )


def pretrain_discriminator(sess_context_manager, model, vocab, batcher):
  """Train a text classifier. the ratio of the positive data to negative data is 1:1"""
  hps = model.hps
  with sess_context_manager as sess:
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    eval_loss_best = sys.float_info.max
    previous_losses = [eval_loss_best]
    if hps.early_stop:
      eval_batcher = DisBatcher(hps.data_path, "eval", vocab, hps.batch_size, single_pass=hps.single_pass)
    while True:
      start_time = time.time()
      batch = batcher.next()
      inputs, conditions, targets = data.prepare_dis_pretraining_batch(batch)
      step_loss = model.run_one_step(sess_context_manager, inputs, conditions, targets, update=True)
      step_time += (time.time() - start_time) / hps.steps_per_checkpoint
      loss += step_loss / hps.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % hps.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        print("global step %d learning rate %.4f step-time %.4f loss "
              "%.4f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
        dump_model = True
        if hps.early_stop:
          dump_model = False
          # Run evals on development set and print their perplexity.
          eval_losses = []
          for bucket_id_eval in range(len(_buckets)):
            while True:
              batch = eval_batcher.next()
              eval_inputs, eval_conditions, eval_targets = data.prepare_dis_pretraining_batch(batch)
              step_loss = model.run_one_step(sess_context_manager, eval_inputs, eval_conditions, eval_targets, update=False)
              eval_losses.append(step_loss)
          eval_loss = sum(eval_losses) / len(eval_losses)
          print("  eval loss %.4f" % eval_loss)
          previous_losses.append(eval_loss)
          sys.stdout.flush()
          threshold = 10
          if eval_loss > 0.99 * previous_losses[-2]:
            sess.run(model.learning_rate.assign(tf.maximum(hps.learning_rate_decay_factor*model.learning_rate, 1e-4)))
          if len(previous_losses) > threshold and eval_loss > max(previous_losses[-threshold-1:-1]) and eval_loss_best < min(previous_losses[-threshold:]):
            break
          if eval_loss < eval_loss_best:
            dump_model = True
            eval_loss_best = eval_loss
        # Save checkpoint and zero timer and loss.
        if dump_model:
          checkpoint_path = os.path.join(hps.model_dir, "translate.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        if current_step >= hps.max_steps:
          break


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for
    coverage, and save as new checkpoint"""
    print("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=gen_utils.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = gen_utils.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()
    # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def main(argv):
    tf.set_random_seed(111)  # a seed value for randomness

    # Create a batcher object that will create minibatches of data
    # TODO change to pass number

    # --------------- build graph ---------------
    hparam_gen = [
        'mode',
        'model_dir',
        'adagrad_init_acc',
        'batch_size',
        'beam_size',
        'cov_loss_wt',
        'coverage',
        'emb_dim',
        'rand_unif_init_mag',
        'gen_vocab_size',
        'hidden_dim',
        'gen_lr',
        'gen_max_gradient',
        'max_dec_steps',
        'max_enc_steps',
        'min_dec_steps',
        'pointer_gen',
        'trunc_norm_init_std',
        'single_pass',
        'log_root',
        'data_path',
    ]

    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_gen:  # if it's in the list
            hps_dict[key] = val  # add it to the dict

    hps_gen = namedtuple("HParams4Gen", hps_dict.keys())(**hps_dict)
    print("Building vocabulary for generator ...")
    gen_vocab = Vocab(os.path.join(hps_gen.data_path, 'gen_vocab'), hps_gen.gen_vocab_size)
    gen_batcher_train = GenBatcher(hps_gen.data_path, "train", gen_vocab, hps_gen, single_pass=hps_gen.single_pass)
    gen_batcher_val = GenBatcher(hps_gen.data_path, "val", gen_vocab, hps_gen, single_pass=False)

    if FLAGS.segment is not True:
        hps_gen = hps_gen._replace(max_enc_steps=110)
        hps_gen = hps_gen._replace(max_dec_steps=25)
    elif FLAGS.mode != "decode":
        assert hps_gen.max_enc_steps == 80, "No segmentation, max_enc_steps wrong"
        assert hps_gen.max_dec_steps == 15, "No segmentation, max_dec_steps wrong"

    if FLAGS.mode in ["decode", "gan"]:
        hps_gen = hps_gen._replace(max_dec_steps=1)

    print("Building generator graph ...")
    with tf.variable_scope("generator"):
        generator = PointerGenerator(hps_gen, gen_vocab)
        generator.build_graph()

    hparam_dis = [
        'mode',
        'model_dir',
        'dis_vocab_size',
        'num_class',
        'buckets',
        'layer_size',
        'conv_layers',
        'kernel_size',
        'pool_size',
        'pool_layers',
        'dis_max_gradient',
        'batch_size',
        'dis_lr',
        'lr_decay_factor',
        'cell_type',
        'max_enc_steps',
        'max_dec_steps',
        'single_pass',
        'data_path',
        'num_models',
    ]
    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_dis:  # if it's in the list
            hps_dict[key] = val  # add it to the dict

    hps_dis = namedtuple("HParams4Dis", hps_dict.keys())(**hps_dict)
    print("Building vocabulary for discriminator ...")
    dis_vocab = Vocab(os.path.join(hps_dis.data_path, 'dis_vocab'), hps_dis.dis_vocab_size)
    # the decode mode is refering to the decoding process of the generator
    print("Building discriminator graph ...")
    with tf.variable_scope("discriminator"), tf.device("/gpu:0"):
        discriminator = Seq2ClassModel(hps_dis)
        discriminator.build_graph()

    hparam_gan = [
        'mode',
        'model_dir',
        'gan_iter',
        'gan_gen_iter',
        'rollout_num',
    ]
    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_gan:  # if it's in the list
            hps_dict[key] = val  # add it to the dict

    hps_gan = namedtuple("HParams4GAN", hps_dict.keys())(**hps_dict)
    hps_gan = hps_gan._replace(mode="gan")
    print("Preparing rollout...")
    # with tf.variable_scope("ROLLOUT"), tf.device("/gpu:0"):
    #     print("Creating rollout...")
    #     rollout = Rollout(generator, 0.8)
    # this is about the variable sharing conflicts

    saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
    print("Creating session..")
    sess = tf.Session(config=gen_utils.get_config())
    print("Restoring models...")
    if FLAGS.restore_best_model:
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.val_dir))
    else:
        saver.restore(sess, tf.train.latest_checkpoint(hps_gen.model_dir))

    print("Creating beam search...")
    with tf.variable_scope("beam_search"), tf.device("/gpu:0"):
        decoder = BeamSearchDecoder(sess, generator, gen_vocab)

    # initialize the embeddings at the begging of training
    ckpt = tf.train.get_checkpoint_state(hps_dis.model_dir)
    if ckpt and not tf.gfile.Exists(ckpt.model_checkpoint_path+".meta"):
        discriminator.init_emb(sess, hps_dis.model_dir)

    # --------------- train generator ---------------
    if FLAGS.mode == "pretrain_gen":
        print('Going to pretrain the generator')
        try:
            pretrain_generator(generator, gen_batcher_train, sess, gen_batcher_val, saver)
        except KeyboardInterrupt:
            tf.logging.info("Caught keyboard interrupt on worker....")

    # --------------- train discriminator -----------
    elif FLAGS.mode == "pretrain_dis":
        print('Going to pretrain the discriminator')
        dis_batcher = DisBatcher(hps_dis.data_path, "decode", dis_vocab, hps_dis.batch_size, single_pass=hps_dis.single_pass)
        try:
            pretrain_discriminator(sess, discriminator, dis_vocab, dis_batcher)
        except KeyboardInterrupt:
            tf.logging.info("Caught keyboard interrupt on worker....")

    # --------------- finetune the generator --------
    elif FLAGS.mode == "train_gan":
        print('Going to tune the two using Gan')
        # decode_model_hps = hps_gen
        # decode_model_hps = decode_model_hps._replace(mode="gan")
        # model = PointerGenerator(decode_model_hps, gen_vocab)
        # get_reward, update_params
        for i_gan in range(hps_gan.gan_iter):
            # Train the generator for one step
            for it in range(hps_gan.gan_gen_iter):
                # can this be self.batch in decoder?
                source_batch, enc_states, enc_padding_mask, dec_in_state, best_samples = decoder.generate(
                    gen_batcher_train, include_start_token=True)
                rewards = rollout.get_reward(
                    sess, gen_vocab, dis_vocab, source_batch, enc_states, source_batch.enc_padding_mask,
                    dec_in_state, best_samples, 16, discriminator)
                print('Get the rewards in %s' % it)
                # only updates parameters without the rollout scope
                feed_dict = {}
                feed_dict[generator.enc_batch] = source_batch.enc_batch
                feed_dict[generator.enc_lens] = source_batch.enc_lens
                # TODO: enc_lens should be added in rollout
                feed_dict[generator.g_predictions] = best_samples
                feed_dict[generator.rewards] = rewards

                if hps_gen.pointer_gen:
                    feed_dict[generator.enc_batch_extend_vocab] = source_batch.enc_batch_extend_vocab
                    # this is the source
                    feed_dict[generator.max_art_oovs] = source_batch.max_art_oovs
                _ = sess.run(generator.g_updates, feed_dict=feed_dict)

            # Test
            print('Going to test the generator.' % it)
            if hps_gan.i_gan % 5 == 0 or hps_gan.i_gan == hps_gan.gan_iter - 1:
                source_batch, enc_states, dec_in_state, best_samples = decoder.generate(gen_batcher_val)
                # the true abstract is source_batch.dec_batch
                summary, test_loss, step = generator.run_eval_step(sess, source_batch)
                buffer = 'step:\t' + str(step) + '\tloss:\t' + str(test_loss) + '\n'
                # training would terminate here if the test loss is sound
                print(buffer)

            # Train the discriminator
            print('Going to train the discriminator.' % it)
            for _ in range(hps_gan.gan_dis_iter):
                for _ in range(3):
                    source_batch, enc_states, dec_in_state, samples_words = decoder.generate(gen_batcher_train)
                    articles_oovs = source_batch.art_oovs if hps_gen.pointer_gen else None

                    samples_chars = gen_vocab2dis_vocab(
                        samples_words, gen_vocab, articles_oovs,
                        dis_vocab, 25, STOP_DECODING)
                    dec_batch_words = source_batch.abs_ids_extend_vocab \
                        if hps_gen.pointer_gen else source_batch.abs_ids
                    dec_batch_chars = gen_vocab2dis_vocab(
                        dec_batch_words, gen_vocab, articles_oovs, dis_vocab, 25, PAD_TOKEN)
                    conditions_words = source_batch.enc_batch_extend_vocab \
                        if hps_gen.pointer_gen else source_batch.enc_batch
                    conditions_chars = gen_vocab2dis_vocab(
                        conditions_words, gen_vocab, articles_oovs,
                        dis_vocab, 110, PAD_TOKEN)
                    inputs = np.concat([samples_chars, dec_batch_chars], 0)
                    conditions = np.concat([conditions_chars, conditions_chars], 0)
                    targets = np.concat(
                        [np.zeros([hps_gan.batch_size]), np.ones([hps_gan.batch_size])], 0)

                    discriminator.run_one_step(sess, inputs, conditions, targets)

    elif FLAGS.mode == "decode":
        print('Going to decode from the generator.')
        decoder.decode(gen_batcher_train, saver)
        # decode for generating corpus for discriminator


if __name__ == '__main__':
  tf.app.run()
