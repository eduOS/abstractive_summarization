from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from collections import namedtuple
import numpy as np
import datetime
import utils
import time
import sys
import data
from batcher import GenBatcher, DisBatcher
from decode import BeamSearchDecoder
from pointer_generator import PointerGenerator
from rollout import Rollout
from data import gen_vocab2dis_vocab
from os.path import join as join_path
from utils import ensure_exists
from gen_utils import calc_running_avg_loss
from gen_utils import get_best_loss_from_chpt
from gen_utils import save_best_ckpt
from utils import print_dashboard
from utils import pad_sample
from dis_utils import dump_chpt
import math
from data import PAD_TOKEN
from termcolor import colored

from res_discriminator import Seq2ClassModel
from data import Vocab
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
tf.app.flags.DEFINE_integer("conv_layers", 2, "Number of convolution layers in the model.")
tf.app.flags.DEFINE_integer("pool_layers", 2, "Number of pooling layers in the model.")
tf.app.flags.DEFINE_integer("kernel_size", 3, "The kernel size of the filters along the sentence length dimension.")
tf.app.flags.DEFINE_integer("pool_size", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Cell type")
tf.app.flags.DEFINE_integer("dis_vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_string("dis_vocab", "dis_vocab", "vocabulary size.")
tf.app.flags.DEFINE_integer("num_class", 2, "num of output classes.")
tf.app.flags.DEFINE_integer("num_models", 3, "Size of each model layer. The actural size is doubled.")

# Training parameters
tf.app.flags.DEFINE_float("dis_lr", 0.0005, "Learning rate.")
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
tf.app.flags.DEFINE_boolean('update_gen', True, 'to decide if to train generator.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
# if batch_size is one and beam size is not one in the decode mode then the beam
# search is the same as the original beam search
tf.app.flags.DEFINE_integer('max_enc_steps', 75, 'max timesteps of encoder (max source text tokens)')  # 400
tf.app.flags.DEFINE_integer('max_dec_steps', 12, 'max timesteps of decoder (max summary tokens)')  # 100
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 8, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('gen_vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in'
                            ' order. If the vocabulary file contains fewer words than this number,'
                            ' or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('gen_lr', 0.0005, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('gen_max_gradient', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
# tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL '
                            'paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards.'
                            'i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
# coverage can be only used while decoding either in the gan or in the pretraining
tf.app.flags.DEFINE_float('cov_loss_wt', 1, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', True, 'Convert a non-coverage model to a coverage model. '
                            'Turn this on and run in train mode. \ Your current model will be copied to a new version '
                            '(same name with _cov_init appended)\ that will be ready to run with coverage flag turned on,\ for the coverage training stage.')


# ------------------------------------- gan

tf.app.flags.DEFINE_integer('gan_iter', 200000, 'how many times to run the gan')
tf.app.flags.DEFINE_integer('gan_gen_iter', 0, 'in each gan step run how many times the generator')
tf.app.flags.DEFINE_integer('gan_dis_iter', 10, 'in each gan step run how many times the generator')
tf.app.flags.DEFINE_integer('rollout_num', 3, 'how many times to repeat the rollout process.')
tf.app.flags.DEFINE_string("gan_dir", "gan_dir", "Training directory.")
tf.app.flags.DEFINE_boolean("decode_from_gan", False, "Either decode from gan checkpoint or not")

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
    best_loss = get_best_loss_from_chpt(val_dir)
    # get the val loss score
    coverage_loss = None
    hps = model.hps
    # this is where checkpoints of best models are saved
    running_avg_loss = 0
    # the eval job keeps a smoother, running average loss to tell it when to
    # implement early stopping
    start_time = time.time()
    counter = 0
    while True:  # repeats until interrupted
        batch = batcher.next_batch()
        if batch is None:
            return None

        # print('running training step...')
        results = model.run_one_step(sess, batch)
        counter += 1
        step = results['global_step']
        # print('seconds for training step: %.3f', t1-t0)

        loss = results['loss']
        # print('loss: %f', loss)  # print the loss to screen
        if hps.coverage:
            coverage_loss = results['coverage_loss']

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
            eval_loss, best_loss = save_best_ckpt(
                sess, model, best_loss, val_batcher, val_dir, val_saver, step)

            # print the print the dashboard
            current_speed = (time.time() - start_time) / (counter * hps.batch_size)
            total_training_time = (time.time() - start_time) * step / (counter * 3600)
            print_dashboard("Generator", step, hps.batch_size, hps.gen_vocab_size,
                            running_avg_loss, eval_loss,
                            total_training_time, current_speed,
                            coverage_loss if coverage_loss else "not set")


def pretrain_discriminator(sess, model, eval_batcher, dis_vocab, batcher, saver):
    """Train a text classifier. the ratio of the positive data to negative data is 1:1"""
    # TODO: load two pretained model: the generator and the embedding
    eval_loss_best = sys.float_info.max
    hps = model.hps
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    train_accuracies = []
    while True:
        start_time = time.time()
        batch = batcher.next_batch()
        inputs, conditions, targets = data.prepare_dis_pretraining_batch(batch)
        if inputs.shape[0] != hps.batch_size * hps.num_models * 2:
            print("The expected batch_size is %s but given %s, escape.." %
                  (hps.batch_size * hps.num_models, inputs.shape[0]))
            continue
        results = model.run_one_step(sess, inputs, conditions, targets)
        train_accuracies.append(results["accuracy"])
        step_time += (time.time() - start_time) / hps.steps_per_checkpoint
        loss += results["loss"] / hps.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % hps.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            eval_accuracy, eval_loss, stop_flag, eval_loss_best = dump_chpt(
                eval_batcher, hps, model, sess, saver, eval_loss_best, hps.early_stop)
            if stop_flag:
                break
            print_dashboard("Discriminator", results["global_step"], hps.batch_size,
                            hps.dis_vocab_size, loss, eval_loss, 0.0, step_time)
            print(colored("training accuracy: %.4f; eval_accuracy: %.4f"
                  % (results['accuracy'], eval_accuracy), "green"))
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
    hps_dis = hps_dis._replace(max_enc_steps=hps_dis.max_enc_steps * 2)
    hps_dis = hps_dis._replace(max_dec_steps=hps_dis.max_dec_steps * 2)
    if FLAGS.mode == "train_gan":
        hps_gen = hps_gen._replace(batch_size=hps_gen.batch_size * hps_dis.num_models)

    if FLAGS.mode != "pretrain_dis":
        with tf.variable_scope("generator"):
            generator = PointerGenerator(hps_gen, gen_vocab)
            print("Building generator graph ...")
            gen_decoder_scope = generator.build_graph()

    if FLAGS.mode != "pretrain_gen":
        print("Building vocabulary for discriminator ...")
        dis_vocab = Vocab(join_path(hps_dis.data_path, hps_dis.dis_vocab), hps_dis.dis_vocab_size)
    # the decode mode is refering to the decoding process of the generator
    if FLAGS.mode in ['train_gan', 'pretrain_dis']:
        with tf.variable_scope("discriminator"), tf.device("/gpu:0"):
            discriminator = Seq2ClassModel(hps_dis)
            print("Building discriminator graph ...")
            discriminator.build_graph()

    hparam_gan = [
        'mode',
        'model_dir',
        'gan_iter',
        'gan_gen_iter',
        'gan_dis_iter',
        'rollout_num',
    ]
    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_gan:  # if it's in the list
            hps_dict[key] = val  # add it to the dict

    hps_gan = namedtuple("HParams4GAN", hps_dict.keys())(**hps_dict)
    hps_gan = hps_gan._replace(mode="train_gan")
    if FLAGS.mode == 'train_gan':
        with tf.device("/gpu:0"):
            print("Creating rollout...")
            rollout = Rollout(generator, 0.8, gen_decoder_scope)

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
        # temp_saver = tf.train.Saver(
        #     var_list=[v for v in all_variables if "generator" in v.name and "Adagrad" not in v.name])
        utils.load_ckpt(gen_saver, sess, gen_dir)

    elif FLAGS.mode in ["decode", "train_gan"]:
        print("Restoring the generator model from the best checkpoint...")
        dec_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "generator" in v.name])
        val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.val_dir))
        gan_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.gan_dir))
        if FLAGS.decode_from_gan:
            utils.load_ckpt(dec_saver, sess, gan_dir, (FLAGS.mode in ["train_gan", "decode"]), "checkpoint_gan")
        else:
            utils.load_ckpt(dec_saver, sess, val_dir, (FLAGS.mode in ["train_gan", "decode"]), "checkpoint_best")

    if FLAGS.mode in ["pretrain_dis", "train_gan"]:
        dis_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "discriminator" in v.name])
        dis_dir = ensure_exists(join_path(FLAGS.model_dir, 'discriminator'))
        ckpt = utils.load_ckpt(dis_saver, sess, dis_dir)
        if not ckpt:
            discriminator.init_emb(sess, join_path(FLAGS.model_dir, "init_embed"))

    # --------------- train models ---------------
    if FLAGS.mode != "pretrain_dis":
        gen_batcher_train = GenBatcher("train", gen_vocab, hps_gen, single_pass=hps_gen.single_pass)
        decoder = BeamSearchDecoder(sess, generator, gen_vocab)
        gen_batcher_val = GenBatcher("val", gen_vocab, hps_gen, single_pass=True)
        val_saver = tf.train.Saver(max_to_keep=10,
                                   var_list=[v for v in all_variables if "generator" in v.name])

    if FLAGS.mode != "pretrain_gen":
        dis_val_batch_size = hps_dis.batch_size * hps_dis.num_models \
            if hps_dis.mode == "train_gan" else hps_dis.batch_size * hps_dis.num_models * 2
        dis_batcher_val = DisBatcher(
            hps_dis.data_path, "eval", gen_vocab, dis_vocab,
            dis_val_batch_size, single_pass=True,
            max_art_steps=hps_dis.max_enc_steps, max_abs_steps=hps_dis.max_dec_steps,
        )

    if FLAGS.mode == "pretrain_gen":
        # get reload the
        print('Going to pretrain the generator')
        try:
            pretrain_generator(generator, gen_batcher_train, sess, gen_batcher_val, gen_saver, val_saver)
        except KeyboardInterrupt:
            tf.logging.info("Caught keyboard interrupt on worker....")

    elif FLAGS.mode == "pretrain_dis":
        print('Going to pretrain the discriminator')
        dis_batcher = DisBatcher(
            hps_dis.data_path, "decode", gen_vocab, dis_vocab,
            hps_dis.batch_size * hps_dis.num_models, single_pass=hps_dis.single_pass,
            max_art_steps=hps_dis.max_enc_steps, max_abs_steps=hps_dis.max_dec_steps,
        )
        try:
            pretrain_discriminator(sess, discriminator, dis_batcher_val, dis_vocab, dis_batcher, dis_saver)
        except KeyboardInterrupt:
            tf.logging.info("Caught keyboard interrupt on worker....")

    elif FLAGS.mode == "train_gan":
        gen_best_loss = get_best_loss_from_chpt(val_dir)
        gen_global_step = 0
        print('Going to tune the two using Gan')
        for i_gan in range(hps_gan.gan_iter):
            # Train the generator for one step
            g_losses = []
            current_speed = []
            for it in range(hps_gan.gan_gen_iter):
                start_time = time.time()
                batch = gen_batcher_train.next_batch()
                # print('batch enc_batch_extend_vocab')
                # print(batch.enc_batch_extend_vocab)
                # print("batch.target_batch")
                # print(batch.target_batch)

                # generate samples
                enc_states, dec_in_state, k_best_samples, k_targets_padding_mask = decoder.generate(
                    batch, include_start_token=True, top_k=FLAGS.beam_size)

                k_best_samples = [np.squeeze(i, 1) for i in np.split(k_best_samples, k_best_samples.shape[1], 1)]
                k_targets_padding_mask = [
                    np.squeeze(i, 1)
                    for i in np.split(k_targets_padding_mask, k_targets_padding_mask.shape[1], 1)]
                # get rewards for the samples
                k_rewards = rollout.get_reward(
                    sess, gen_vocab, dis_vocab, batch, enc_states,
                    dec_in_state, k_best_samples, hps_gan.rollout_num, discriminator)

                # fine tune the generator
                k_sample_targets = [best_samples[:, 1:] for best_samples in k_best_samples]
                k_targets_padding_mask = [padding_mask[:, 1:] for padding_mask in k_targets_padding_mask]
                k_samples = [best_samples[:, :-1] for best_samples in k_best_samples]
                # sample_target_padding_mask = pad_sample(sample_target, gen_vocab, hps_gen)
                k_samples = [np.where(
                    np.less(samples, hps_gen.gen_vocab_size),
                    samples, np.array(
                        [[gen_vocab.word2id(data.UNKNOWN_TOKEN)] * hps_gen.max_dec_steps] * hps_gen.batch_size))
                    for samples in k_samples]
                results = generator.run_gan_step(
                    sess, batch, k_rewards, k_samples, k_sample_targets, k_targets_padding_mask)
                # stl = sample_target.tolist()
                # for st in stl:
                #     st = [str(s) for s in st]
                #     print(colored('\t'.join(st), "red"))
                # print('sample_target_padding_mask')
                # print(sample_target_padding_mask)
                # print('loss_per_step')
                # print(results['loss_per_step'])
                print('sample rewards')
                rwl = k_rewards[0].tolist()
                for n, rw in enumerate(rwl):
                    rw = [str(r)[:7] for r in rw]
                    print(str(n) + ": " + colored('\t'.join(rw), "blue"))
                # print('g_loss_per_step')
                # print(results['g_loss_per_step'])
                print('-------------------------------------------------')

                gen_global_step = results["global_step"]

                # for visualization
                g_loss = results["g_loss"]
                if not math.isnan(g_loss):
                    g_losses.append(g_loss)
                current_speed.append(time.time() - start_time)

            # Test
            if FLAGS.gan_gen_iter and (i_gan % 100 == 0 or i_gan == hps_gan.gan_iter - 1):
                print('Going to test the generator.')
                current_speed = sum(current_speed) / (len(current_speed) * hps_gen.batch_size)
                everage_g_loss = sum(g_losses) / len(g_losses)
                # one more process hould be opened for the evaluation
                eval_loss, gen_best_loss = save_best_ckpt(
                    sess, generator, gen_best_loss, None, val_dir, val_saver, gen_global_step, gan_dir=gan_dir)

                if eval_loss:
                    print(
                        "\nDashboard for " + colored("GAN Generator", 'green') + " updated %s, "
                        "finished steps:\t%s\n"
                        "\tBatch size:\t%s\n"
                        "\tVocabulary size:\t%s\n"
                        "\tCurrent speed:\t%.4f seconds/article\n"
                        "\tTraining loss:\t%.4f; "
                        "eval loss:\t%.4f" % (
                            datetime.datetime.now().strftime("on %m-%d at %H:%M"),
                            gen_global_step,
                            FLAGS.batch_size,
                            hps_gen.gen_vocab_size,
                            current_speed,
                            everage_g_loss.item(),
                            eval_loss.item(),
                            )
                    )

            # Train the discriminator
            print('Going to train the discriminator.')
            dis_best_loss = 1000
            dis_losses = []
            dis_accuracies = []
            for d_gan in range(hps_gan.gan_dis_iter):
                batch = gen_batcher_train.next_batch()
                enc_states, dec_in_state, samples_words, _ = decoder.generate(batch)
                # shuould first tanslate to words to avoid unk
                articles_oovs = batch.art_oovs
                samples_chars = gen_vocab2dis_vocab(
                    samples_words, gen_vocab, articles_oovs,
                    dis_vocab, hps_dis.max_dec_steps, STOP_DECODING)
                dec_batch_words = [b[1:] for b in batch.target_batch]
                dec_batch_chars = gen_vocab2dis_vocab(
                    dec_batch_words, gen_vocab, articles_oovs, dis_vocab, hps_dis.max_dec_steps,  STOP_DECODING)
                conditions_words = batch.enc_batch_extend_vocab
                conditions_chars = gen_vocab2dis_vocab(
                    conditions_words, gen_vocab, articles_oovs,
                    dis_vocab, hps_dis.max_enc_steps, PAD_TOKEN)
                # the unknown in target

                inputs = np.concatenate([samples_chars, dec_batch_chars], 0)
                conditions = np.concatenate([conditions_chars, conditions_chars], 0)

                targets = [[1, 0] for _ in samples_chars] + [[0, 1] for _ in dec_batch_chars]
                targets = np.array(targets)
                # randomize the samples
                assert len(inputs) == len(conditions) == len(targets), "lengthes of the inputs, conditions and targests should be the same."
                indices = np.random.permutation(len(inputs))
                inputs = np.split(inputs[indices], 2)
                conditions = np.split(conditions[indices], 2)
                targets = np.split(targets[indices], 2)
                assert len(inputs) % 2 == 0, "the length should be mean"

                results = discriminator.run_one_step(sess, inputs[0], conditions[0], targets[0])
                dis_accuracies.append(results["accuracy"].item())
                dis_losses.append(results["loss"].item())

                results = discriminator.run_one_step(sess, inputs[1], conditions[1], targets[1])
                dis_accuracies.append(results["accuracy"].item())

                if d_gan == hps_gan.gan_dis_iter - 1:
                    if (sum(dis_losses) / len(dis_losses)) < dis_best_loss:
                        dis_best_loss = sum(dis_losses) / len(dis_losses)
                        checkpoint_path = ensure_exists(join_path(hps_dis.model_dir, "discriminator")) + "/model.ckpt"
                        dis_saver.save(sess, checkpoint_path, global_step=results["global_step"])
                    print_dashboard("GAN Discriminator", results["global_step"].item(), hps_dis.batch_size, hps_dis.dis_vocab_size,
                                    results["loss"].item(), 0.00, 0.00, 0.00)
                    print("Average training accuracy: \t%.4f" %
                          (sum(dis_accuracies) / len(dis_accuracies)))

    # --------------- decoding samples ---------------
    elif FLAGS.mode == "decode":
        print('Going to decode from the generator.')
        decoder.decode(gen_batcher_train)
        print("Finished decoding..")
        # decode for generating corpus for discriminator

    sess.close()


if __name__ == '__main__':
  tf.app.run()
