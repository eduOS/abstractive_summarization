from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from collections import namedtuple
import numpy as np
import sys
import datetime
import utils
import time
import data
from batcher import GenBatcher, DisBatcher
from decode import BeamSearchDecoder
from pointer_generator import PointerGenerator
from rollout import Rollout
from data import gen_vocab2dis_vocab
from os.path import join as join_path
from utils import ensure_exists
from gen_utils import calc_running_avg_loss

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
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 1000, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'Learning rate decay by this rate')

# ------------------------------------- discriminator

# Model parameters
tf.app.flags.DEFINE_integer("layer_size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("conv_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("pool_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("kernel_size", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("pool_size", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Cell type")
tf.app.flags.DEFINE_integer("dis_vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_string("dis_vocab", "dis_vocab", "vocabulary size.")
tf.app.flags.DEFINE_integer("num_class", 2, "num of output classes.")
tf.app.flags.DEFINE_integer("num_models", 8, "Size of each model layer. The actural size is doubled.")

# Training parameters
tf.app.flags.DEFINE_float("dis_lr", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.5, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("dis_max_gradient", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_boolean("early_stop", False, "Set to True to turn on early stop.")
tf.app.flags.DEFINE_integer("max_steps", -1, "max number of steps to train")

# Misc
tf.app.flags.DEFINE_string("model_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_string("val_dir", "val", "Training directory.")

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
tf.app.flags.DEFINE_string('dec_dir', '', 'Where to generate the decode results. If false the time stamp is toke.')
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

assert FLAGS.mode in ["pretrain_gen", "pretrain_dis", "train_gan", "decode", "test"]

if FLAGS.mode == "train_gan":
    FLAGS.single_pass = False

ensure_exists(FLAGS.model_dir)


def pretrain_generator(model, batcher, sess, val_batcher, saver, val_saver):
    """Repeatedly runs training iterations, logging loss to screen and writing
    summaries"""
    print("starting run_training")
    best_loss = None  # will hold the best loss achieved so far
    val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.val_dir))
    ckpt = tf.train.get_checkpoint_state(val_dir)
    if ckpt:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        best_loss = reader.get_tensor(
            [key for key in var_to_shape_map if "least_val_loss" in key][0])
        print("the stored best loss is %s" % best_loss)
    # get the val loss score
    bestmodel_save_path = join_path(val_dir, 'bestmodel')
    coverage_loss = None
    hps = model.hps
    # this is where checkpoints of best models are saved
    running_avg_loss = 0
    # the eval job keeps a smoother, running average loss to tell it when to
    # implement early stopping
    while True:  # repeats until interrupted
        batch = batcher.next_batch()
        if batch is None:
            return None

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

        if step % FLAGS.steps_per_checkpoint == 0:
            model_path = join_path(hps.model_dir, "generator", "model")
            saver.save(sess, model_path, global_step=step)
            print(
                'Saving model with %.3f running_avg_loss. Saving to %s %s' %
                (running_avg_loss, model_path,
                    datetime.datetime.now().strftime("on %m-%d at %H:%M")))

            # check if it is the best checkpoint so far
            losses = []
            while True:
                val_batch = val_batcher.next_batch()
                if not val_batch:
                    break
                results_val = model.run_one_step(
                    sess, val_batch, update=False)
                losses.append(results_val["loss"])
            eval_loss = sum(losses) / len(losses)
            if best_loss is None or eval_loss < best_loss:
                sess.run(model.least_val_loss.assign(eval_loss))
                print(
                    'Found new best model with %.3f running_avg_loss. Saving to %s %s' %
                    (eval_loss, bestmodel_save_path,
                        datetime.datetime.now().strftime("on %m-%d at %H:%M")))
                val_saver.save(sess, bestmodel_save_path, global_step=step, latest_filename="checkpoint_best")
                best_loss = eval_loss

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


def pretrain_discriminator(sess, model, gen_vocab, dis_vocab, batcher, saver):
    """Train a text classifier. the ratio of the positive data to negative data is 1:1"""
    # TODO: load two pretained model: the generator and the embedding
    hps = model.hps
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    eval_loss_best = sys.float_info.max
    previous_losses = [eval_loss_best]
    if hps.early_stop:
        eval_batcher = DisBatcher(
            hps.data_path, "eval", gen_vocab, dis_vocab, hps.batch_size * hps.num_models, single_pass=True)
    train_accuracies = []
    while True:
        start_time = time.time()
        batch = batcher.next_batch()
        inputs, conditions, targets = data.prepare_dis_pretraining_batch(batch)
        if inputs.shape[0] != hps.batch_size * hps.num_models * 2:
            print("The expected batch_size is %s but given %s, escape.." %
                  (hps.batch_size * hps.num_models * 2, inputs.shape[0]))
            continue
        results = model.run_one_step(sess, inputs, conditions, targets)
        train_accuracies.append(results["accuracy"])
        step_time += (time.time() - start_time) / hps.steps_per_checkpoint
        loss += results["loss"] / hps.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % hps.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            print("global step %d learning rate %.4f step-time %.4f loss %.4f"
                  % (results["global_step"], results['learning_rate'], step_time, loss))
            dump_model = True
            if hps.early_stop:
                dump_model = False
                # Run evals on development set and print their perplexity.
                eval_losses = []
                eval_accuracies = []
                while True:
                    batch = eval_batcher.next_batch()
                    if not batch[0]:
                        eval_batcher.reset()
                        break
                    eval_inputs, eval_conditions, eval_targets = \
                        data.prepare_dis_pretraining_batch(batch)
                    if eval_inputs.shape[0] != hps.batch_size * hps.num_models * 2:
                        print("The expected batch_size is %s but given %s, escape.." %
                              (hps.batch_size * hps.num_models * 2, eval_inputs.shape[0]))
                        continue
                    eval_results = model.run_one_step(
                        sess, eval_inputs, eval_conditions, eval_targets, update=False)
                    eval_losses.append(eval_results["loss"])
                    eval_accuracies.append(eval_results["accuracy"])
                eval_loss = sum(eval_losses) / len(eval_losses)
                eval_accuracy = sum(eval_accuracies) / len(eval_accuracies)
                train_accuracy = sum(train_accuracies) / len(train_accuracies)
                train_accuracies = []
                print("Eval loss %.4f, train accuracy is %.4f and eval accuracy is %.4f" % (eval_loss, train_accuracy, eval_accuracy))
                previous_losses.append(eval_loss)
                sys.stdout.flush()
                threshold = 10
                if eval_loss > 0.99 * previous_losses[-2]:
                    sess.run(model.learning_rate.assign(
                        tf.maximum(hps.learning_rate_decay_factor*model.learning_rate, 1e-4)))
                if len(previous_losses) > threshold and \
                        eval_loss > max(previous_losses[-threshold-1:-1]) and \
                        eval_loss_best < min(previous_losses[-threshold:]):
                    # print("Proper time to stop...")
                    break
                if eval_loss < eval_loss_best:
                    dump_model = True
                    eval_loss_best = eval_loss
            # Save checkpoint and zero timer and loss.
            if dump_model:
                checkpoint_path = ensure_exists(join_path(hps.model_dir, "discriminator")) + "/model.ckpt"
                saver.save(sess, checkpoint_path, global_step=model.global_step)
                print("Saving the checkpoint to %s" % checkpoint_path)
            step_time, loss = 0.0, 0.0
            if current_step >= hps.max_steps:
                break


def main(argv):
    tf.set_random_seed(111)  # a seed value for randomness

    # Create a batcher object that will create minibatches of data
    # TODO change to pass number

    # --------------- building graph ---------------
    hparam_gen = [
        'mode',
        'model_dir',
        'adagrad_init_acc',
        'steps_per_checkpoint',
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
    gen_vocab = Vocab(join_path(hps_gen.data_path, 'gen_vocab'), hps_gen.gen_vocab_size)

    if FLAGS.segment is not True:
        hps_gen = hps_gen._replace(max_enc_steps=110)
        hps_gen = hps_gen._replace(max_dec_steps=25)
    elif FLAGS.mode not in ["decode", "train_gan"]:
        assert hps_gen.max_enc_steps == 80, "No segmentation, max_enc_steps wrong"
        assert hps_gen.max_dec_steps == 15, "No segmentation, max_dec_steps wrong"

    if FLAGS.mode in ["decode", "train_gan"]:
        hps_gen = hps_gen._replace(max_dec_steps=1)
    if FLAGS.mode == "train_gan":
        hps_gen = hps_gen._replace(batch_size=FLAGS.batch_size * FLAGS.num_models)

    with tf.variable_scope("generator"):
        if FLAGS.mode != "pretrain_dis":
            generator = PointerGenerator(hps_gen, gen_vocab)
            print("Building generator graph ...")
            gen_decoder_scope = generator.build_graph()

    hparam_dis = [
        'mode',
        'model_dir',
        'dis_vocab_size',
        'steps_per_checkpoint',
        'learning_rate_decay_factor',
        'dis_vocab',
        'num_class',
        'layer_size',
        'conv_layers',
        'max_steps',
        'kernel_size',
        'early_stop',
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
    if FLAGS.mode in ['train_gan', 'pretrain_dis']:
        print("Building vocabulary for discriminator ...")
        dis_vocab = Vocab(join_path(hps_dis.data_path, hps_dis.dis_vocab), hps_dis.dis_vocab_size)
    # the decode mode is refering to the decoding process of the generator
    with tf.variable_scope("discriminator"), tf.device("/gpu:0"):
        discriminator = Seq2ClassModel(hps_dis)
        if FLAGS.mode in ['train_gan', 'pretrain_dis']:
            print("Building discriminator graph ...")
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
    with tf.variable_scope("rollout"), tf.device("/gpu:0"):
        if FLAGS.mode == 'train_gan':
            print("Creating rollout...")
            rollout = Rollout(generator, 0.8)
            rollout.build_graph(gen_decoder_scope)

    # --------------- initializing variables ---------------
    all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) + \
        tf.get_collection_ref(tf.GraphKeys.WEIGHTS) + \
        tf.get_collection_ref(tf.GraphKeys.BIASES)
    sess = tf.Session(config=utils.get_config())
    sess.run(tf.variables_initializer(all_variables))
    if FLAGS.mode == "pretrain_gen":
        print("Restoring the generator model from the latest checkpoint...")
        gen_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "generator" in v.name])
        gen_dir = ensure_exists(join_path(FLAGS.model_dir, "generator"))
        # gen_dir = ensure_exists(FLAGS.model_dir)
        utils.load_ckpt(gen_saver, sess, gen_dir)

    elif FLAGS.mode in ["decode", "train_gan"]:
        print("Restoring the generator model from the best checkpoint...")
        dec_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "generator" in v.name])
        val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.val_dir))
        utils.load_ckpt(dec_saver, sess, val_dir, (FLAGS.mode == "train_gan"))

    elif FLAGS.mode in ["pretrain_dis", "train_gan"]:
        dis_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "discriminator" in v.name])
        dis_dir = ensure_exists(join_path(FLAGS.model_dir, 'discriminator'))
        ckpt = utils.load_ckpt(dis_saver, sess, dis_dir)
        if not ckpt:
            discriminator.init_emb(sess, join_path(FLAGS.model_dir, "init_embed"))

    if FLAGS.mode == "train_gan":
        rollout_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "rollout" in v.name])
        rlt_dir = ensure_exists(join_path(FLAGS.model_dir, 'rollout'))
        utils.load_ckpt(rollout_saver, sess, rlt_dir)

    # --------------- train models ---------------
    if FLAGS.mode != "pretrain_dis":
        gen_batcher_train = GenBatcher("train", gen_vocab, hps_gen, single_pass=hps_gen.single_pass)
        decoder = BeamSearchDecoder(sess, generator, gen_vocab)
        gen_batcher_val = GenBatcher("val", gen_vocab, hps_gen, single_pass=True)
    if FLAGS.mode == "pretrain_gen":
        # get reload the
        val_saver = tf.train.Saver(max_to_keep=10,
                                   var_list=[v for v in all_variables if "generator" in v.name])

        print('Going to pretrain the generator')
        try:
            pretrain_generator(generator, gen_batcher_train, sess, gen_batcher_val, gen_saver, val_saver)
        except KeyboardInterrupt:
            tf.logging.info("Caught keyboard interrupt on worker....")

    elif FLAGS.mode == "pretrain_dis":
        print('Going to pretrain the discriminator')
        dis_batcher = DisBatcher(
            hps_dis.data_path, "decode", gen_vocab, dis_vocab, hps_dis.batch_size * hps_dis.num_models,
            single_pass=hps_dis.single_pass)
        try:
            pretrain_discriminator(sess, discriminator, gen_vocab, dis_vocab, dis_batcher, dis_saver)
        except KeyboardInterrupt:
            tf.logging.info("Caught keyboard interrupt on worker....")

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
                batch = gen_batcher_train.next_batch()
                source_batch, enc_states, enc_padding_mask, dec_in_state, best_samples = decoder.generate(
                    batch, include_start_token=True)
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
            print('Going to test the generator.')
            if hps_gan.i_gan % 5 == 0 or hps_gan.i_gan == hps_gan.gan_iter - 1:
                batch = gen_batcher_val.next_batch()
                if batch is None:
                    return
                source_batch, enc_states, dec_in_state, best_samples = decoder.generate(batch)
                # the true abstract is source_batch.dec_batch
                results = generator.run_one_step(sess, source_batch, update=False)
                buffer = 'step:\t' + str(results["global_step"]) + \
                    '\tloss:\t' + str(results['loss']) + '\n'
                # training would terminate here if the test loss is sound
                print(buffer)

            # Train the discriminator
            print('Going to train the discriminator.' % it)
            for _ in range(hps_gan.gan_dis_iter):
                for _ in range(3):
                    batch = gen_batcher_train.next_batch()
                    source_batch, enc_states, dec_in_state, samples_words = decoder.generate(batch)
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

    # --------------- decoding samples ---------------
    elif FLAGS.mode == "decode":
        print('Going to decode from the generator.')
        decoder.decode(gen_batcher_train)
        print("Finished decoding..")
        # decode for generating corpus for discriminator

    sess.close()


if __name__ == '__main__':
  tf.app.run()
