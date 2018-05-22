from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from collections import namedtuple, deque
import numpy as np
import datetime
import utils
import time
import sys
import data
from batcher import GenBatcher, DisBatcher
from decode import Decoder
from pointer_generator import PointerGenerator
from rollout import Rollout
from os.path import join as join_path
from utils import ensure_exists
from gen_utils import calc_running_avg_loss
from gen_utils import get_best_loss_from_chpt
from gen_utils import save_ckpt as gen_save_ckpt
from gan_utils import save_ckpt as gan_save_ckpt
from tensorflow.python import debug as tf_debug
from utils import sattolo_cycle
from utils import print_dashboard
from dis_utils import dump_chpt
import math
from termcolor import colored
from data import POSITIVE_LABEL, NEGATIVE_LABEL

from res_discriminator import Seq2ClassModel
from data import Vocab
STOP_DECODING = '[STOP]'
epsilon = sys.float_info.epsilon

# tf.logging.set_verbosity(tf.logging.ERROR)
tf.app.flags.DEFINE_string(
    'mode', 'train',
    'must be one of pretrain_gen/pretrain_dis/train_gan/decode')
# ------------------------------------- common
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 10000, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5, 'Learning rate decay by this rate')
tf.app.flags.DEFINE_float('sample_rate', 0.001, 'the sample rate, should be [0, 0.5]')
tf.app.flags.DEFINE_boolean('debug', True, 'If debug is needed.')

# ------------------------------------- discriminator

# Model parameters
tf.app.flags.DEFINE_integer("layer_size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("conv_layers", 2, "Number of convolution layers in the model.")
tf.app.flags.DEFINE_integer("pool_layers", 2, "Number of pooling layers in the model.")
tf.app.flags.DEFINE_integer("kernel_size", 3, "the kernel size of the conv")
tf.app.flags.DEFINE_integer("pool_size", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Cell type")
tf.app.flags.DEFINE_integer("dis_vocab_size", 5000, "vocabulary size.")
tf.app.flags.DEFINE_string("dis_vocab_file", "vocab", "the path of the discriminator vocabulary.")
tf.app.flags.DEFINE_string("vocab_type", "char", "the path of the discriminator vocabulary.")
tf.app.flags.DEFINE_integer("num_class", 2, "num of output classes.")
tf.app.flags.DEFINE_integer("num_models", 3, "Size of each model layer. The actural size is doubled.")

# Training parameters
tf.app.flags.DEFINE_float("dis_lr", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.5, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("dis_max_gradient", 2.0, "Clip gradients to this norm.")
# TODO: how much thould this be?
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
tf.app.flags.DEFINE_string("gen_vocab_file", "vocab", "the path of the generator vocabulary.")

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
tf.app.flags.DEFINE_integer('emb_dim', 300, 'dimension of word embeddings')
# if batch_size is one and beam size is not one in the decode mode then the beam
# search is the same as the original beam search
tf.app.flags.DEFINE_integer('max_enc_steps', 73, 'max timesteps of encoder (max source text tokens)')  # 400
tf.app.flags.DEFINE_integer('max_dec_steps', 15, 'max timesteps of decoder (max summary tokens)')  # 100
tf.app.flags.DEFINE_integer('beam_size', 40, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 5, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('gen_vocab_size', 5000, 'Size of vocabulary. These will be read from the vocabulary file in'
                            ' order. If the vocabulary file contains fewer words than this number,'
                            ' or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('gen_lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('gen_max_gradient', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_string('encoder', 'conv_encoder', 'Name for the encoder type. Support lstm_encoder and conv_encoder so far.')
tf.app.flags.DEFINE_string('decoder', 'conv_decoder', 'Name for the decoder type. Support lstm_decoder and conv_decoder so far.')

# Pointer-generator or baseline model
# tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL '
                            'paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards.'
                            'i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
# coverage can be only used while decoding either in the gan or in the pretraining
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', True, 'Convert a non-coverage model to a coverage model. '
                            'Turn this on and run in train mode. \ Your current model will be copied to a new version '
                            '(same name with _cov_init appended)\ that will be ready to run with coverage flag turned on,\ for the coverage training stage.')


# ------------------------------------- gan
tf.app.flags.DEFINE_integer('rollout_start', 0, 'how many times to run the gan')
tf.app.flags.DEFINE_integer('gan_iter', 200000, 'how many times to run the gan')
tf.app.flags.DEFINE_integer('gan_gen_iter', 5, 'in each gan step run how many times the generator')
tf.app.flags.DEFINE_integer('gan_dis_iter', 10**8, 'in each gan step run how many times the generator')
tf.app.flags.DEFINE_integer('rollout_num', 12, 'how many times to repeat the rollout process.')
tf.app.flags.DEFINE_string("gan_dir", "gan", "Training directory.")
tf.app.flags.DEFINE_integer('sample_num', 2, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_float('gan_lr', 0.0005, 'learning rate for the gen in GAN training')
tf.app.flags.DEFINE_float('rouge_reward_ratio', 0, 'The importance of rollout in calculating the reward.')
tf.app.flags.DEFINE_float('dis_reward_ratio', 0, 'The importance of rollout in calculating the reward.')

FLAGS = tf.app.flags.FLAGS

assert FLAGS.mode in ["pretrain_gen", "pretrain_dis", "train_gan", "decode", "test"]
assert FLAGS.sample_rate >= 0 and FLAGS.sample_rate <= 0.5, "sample rate should be [0, 0.5]"

if FLAGS.mode == "train_gan":
    FLAGS.single_pass = False
    FLAGS.beam_size = int(FLAGS.beam_size / 2) if FLAGS.beam_size > 3 else 2

if FLAGS.min_dec_steps > FLAGS.max_dec_steps / 2:
    FLAGS.min_dec_steps = int(FLAGS.max_dec_steps / 2)

ensure_exists(FLAGS.model_dir)


def pretrain_generator(model, batcher, sess, batcher_val, model_saver, val_saver):
    """Repeatedly runs training iterations, logging loss to screen and writing
    summaries"""
    print("starting pre_training")
    best_loss = None  # will hold the best loss achieved so far
    val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.val_dir))
    model_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator'))
    best_loss = get_best_loss_from_chpt(val_dir)
    # get the val loss score
    coverage_loss = None
    hps = model.hps
    # this is where checkpoints of best models are saved
    running_avg_loss = 0
    # the eval job keeps a smoother, running average loss to tell it when to
    # implement early stopping
    start_time = time.time()
    eval_save_steps = FLAGS.steps_per_checkpoint
    last_ten_eval_loss = deque(maxlen=10)
    counter = 0
    while True:  # repeats until interrupted
        batch = batcher.next_batch()
        if batch is None:
            return None

        results = model.run_one_batch(sess, batch)
        counter += 1
        global_step = results['global_step']
        # print('seconds for training step: %.3f', t1-t0)

        loss = results['loss']
        if global_step == 1:
            print("The training starts with loss %s." % loss)
            print("The parameters: \n")

            print(
                'mode: %s\n'
                'model_dir: %s\n'
                'decoder: %s\n'
                'steps_per_checkpoint: %s\n'
                'batch_size: %s\n'
                'beam_size: %s\n'
                'coverage: %s\n'
                'emb_dim: %s\n'
                'rand_unif_init_mag: %s\n'
                'gen_vocab_file: %s\n'
                'vocab_type: %s\n'
                'gen_vocab_size: %s\n'
                'hidden_dim: %s\n'
                'gen_lr: %s\n'
                'gen_max_gradient: %s\n'
                'max_dec_steps: %s\n'
                'max_enc_steps: %s\n'
                'min_dec_steps: %s\n'
                'trunc_norm_init_std: %s\n'
                'single_pass: %s\n'
                'log_root: %s\n'
                'data_path: %s\n' % (
                    hps.mode,
                    hps.model_dir,
                    hps.decoder,
                    hps.steps_per_checkpoint,
                    hps.batch_size,
                    hps.beam_size,
                    hps.coverage,
                    hps.emb_dim,
                    hps.rand_unif_init_mag,
                    hps.gen_vocab_file,
                    hps.vocab_type,
                    hps.gen_vocab_size,
                    hps.hidden_dim,
                    hps.gen_lr,
                    hps.gen_max_gradient,
                    hps.max_dec_steps,
                    hps.max_enc_steps,
                    hps.min_dec_steps,
                    hps.trunc_norm_init_std,
                    hps.single_pass,
                    hps.log_root,
                    hps.data_path)
            )

        if hps.coverage:
            coverage_loss = results['coverage_loss']

        running_avg_loss = calc_running_avg_loss(
            np.asscalar(loss), running_avg_loss, global_step)

        if global_step % eval_save_steps == 0:
            # check if it is the best checkpoint so far
            eval_loss, best_loss = gen_save_ckpt(
                sess, model, best_loss, model_dir, model_saver,
                batcher_val, val_dir, val_saver, global_step, gan_eval=False)
            last_ten_eval_loss.append(eval_loss)
            if len(last_ten_eval_loss) == 10 and min(last_ten_eval_loss) == last_ten_eval_loss[0] and eval_save_steps > 5000:
                last_ten_eval_loss = deque(maxlen=10)
                eval_save_steps -= 1000

            # print the print the dashboard
            current_speed = (time.time() - start_time + epsilon) / ((counter * hps.batch_size) + epsilon)
            total_training_time = (time.time() - start_time) * global_step / (counter * 3600)
            print_dashboard("Generator", global_step, hps.batch_size, hps.gen_vocab_size,
                            running_avg_loss, eval_loss,
                            total_training_time, current_speed,
                            coverage_loss if coverage_loss else "not set")


def pretrain_discriminator(sess, model, batcher_val, dis_vocab, batcher, saver):
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
        results = model.run_one_batch(sess, inputs, conditions, targets)
        train_accuracies.append(results["accuracy"])
        step_time += (time.time() - start_time) / hps.steps_per_checkpoint
        loss += results["loss"] / hps.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % hps.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            eval_accuracy, eval_loss, stop_flag, eval_loss_best = dump_chpt(
                batcher_val, hps, model, sess, saver, eval_loss_best, hps.early_stop)
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
        'decoder',
        'steps_per_checkpoint',
        'batch_size',
        'beam_size',
        'cov_loss_wt',
        'coverage',
        'emb_dim',
        'rand_unif_init_mag',
        'gen_vocab_file',
        'vocab_type',
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
    gen_vocab = Vocab(join_path(hps_gen.data_path, hps_gen.gen_vocab_file), hps_gen.gen_vocab_size)

    hparam_dis = [
        'mode',
        'vocab_type',
        'model_dir',
        'dis_vocab_size',
        'steps_per_checkpoint',
        'learning_rate_decay_factor',
        'dis_vocab_file',
        'num_class',
        'layer_size',
        'conv_layers',
        'max_steps',
        'kernel_size',
        'early_stop',
        'pool_size',
        'hidden_dim',
        'pool_layers',
        'dis_max_gradient',
        'batch_size',
        'dis_lr',
        'lr_decay_factor',
        'rand_unif_init_mag',
        'cell_type',
        'max_enc_steps',
        'max_dec_steps',
        'single_pass',
        'data_path',
        'num_models',
        'trunc_norm_init_std',
    ]

    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_dis:  # if it's in the list
            hps_dict[key] = val  # add it to the dict

    hps_dis = namedtuple("HParams4Dis", hps_dict.keys())(**hps_dict)
    if hps_gen.gen_vocab_file == hps_dis.dis_vocab_file:
        assert hps_dis.vocab_type == hps_gen.vocab_type, (
            "the vocab type of the generator and the discriminator should be the same")
        hps_dis = hps_dis._replace(layer_size=hps_gen.emb_dim)
        hps_dis = hps_dis._replace(dis_vocab_size=hps_gen.gen_vocab_size)

    if hps_gen.vocab_type == 'word' == hps_dis.vocab_type:
        hps_dis = hps_dis._replace(max_dec_steps=15)
        hps_dis = hps_dis._replace(max_enc_steps=73)
        hps_gen = hps_gen._replace(max_dec_steps=15)
        hps_gen = hps_gen._replace(max_enc_steps=73)
    elif hps_gen.vocab_type == 'char' == hps_dis.vocab_type:
        hps_dis = hps_dis._replace(max_dec_steps=25)
        hps_dis = hps_dis._replace(max_enc_steps=120)
        hps_gen = hps_gen._replace(max_dec_steps=25)
        hps_gen = hps_gen._replace(max_enc_steps=120)

    if FLAGS.mode == "train_gan":
        hps_gen = hps_gen._replace(batch_size=hps_gen.batch_size * hps_dis.num_models)

    if FLAGS.mode != "pretrain_dis":
        with tf.variable_scope("generator"), tf.device("/gpu:0"):
            generator = PointerGenerator(hps_gen, gen_vocab)
            print("Building generator graph ...")
            gen_decoder_scope = generator.build_graph()

    if FLAGS.mode not in ["pretrain_gen", 'decode']:
        print("Building vocabulary for discriminator ...")
        dis_vocab = Vocab(join_path(hps_dis.data_path, hps_dis.dis_vocab_file), hps_dis.dis_vocab_size)
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
        'gan_lr',
        'rollout_num',
        'sample_num',
        'rouge_reward_ratio',
        'dis_reward_ratio',
        "rollout_start",
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
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter(str("has_inf_or_nan"), tf_debug.has_inf_or_nan)
    if FLAGS.mode == "pretrain_gen":
        print("Restoring the generator model from the latest checkpoint...")
        var_list = [v for v in all_variables if "generator" in v.name]
        gen_newly_added = []
        # add the newly added variables here
        for vn in gen_newly_added:
            var_list = [v for v in var_list if vn not in v.name]
        gen_saver = tf.train.Saver(max_to_keep=3, var_list=var_list)
        gen_val_saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
        gen_dir = ensure_exists(join_path(FLAGS.model_dir, "generator"))
        # gen_dir = ensure_exists(FLAGS.model_dir)
        # temp_saver = tf.train.Saver(
        #     var_list=[v for v in all_variables if "generator" in v.name and "Adagrad" not in v.name])
        ckpt_path = utils.load_ckpt(gen_saver, sess, gen_dir, mode="train")
        print('going to restore embeddings from checkpoint')
        if not ckpt_path:
            emb_path = join_path(FLAGS.model_dir, "generator", "init_embed")
            ckpt_state = tf.train.get_checkpoint_state(emb_path)
            if ckpt_state:
                ckpt = ckpt_state.model_checkpoint_path
                generator.saver.restore(sess, ckpt)
                print(colored("successfully restored embeddings form %s" % emb_path, 'green'))
            else:
                print(colored("failed to restore embeddings form %s" % emb_path, 'red'))

    elif FLAGS.mode in ["decode", "train_gan"]:
        print("Restoring the generator model from the best checkpoint...")
        dec_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "generator" in v.name])
        val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.val_dir))
        model_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator'))
        gan_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.gan_dir))
        gan_val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.gan_dir, "val"))
        gan_newly_added = []
        # add the newly added variables here
        var_list = [v for v in all_variables if "generator" in v.name]
        for vn in gan_newly_added:
            var_list = [v for v in var_list if vn not in v.name]
        gan_saver = tf.train.Saver(max_to_keep=3, var_list=var_list)
        # for the rouge or dis test
        gan_val_saver = tf.train.Saver(max_to_keep=3, var_list=var_list)
        # for the loss test
        gen_val_saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
        utils.load_ckpt(dec_saver, sess, model_dir, mode="val", force=True)
        decoder = Decoder(sess, generator, gen_vocab)

    if FLAGS.mode == "pretrain_dis" or (FLAGS.mode == "train_gan" and FLAGS.rouge_reward_ratio != 1):
        dis_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "discriminator" in v.name])
        dis_dir = ensure_exists(join_path(FLAGS.model_dir, 'discriminator'))
        mode = "train" if FLAGS.mode == "pretrain_dis" else "val"
        # ckpt = utils.load_ckpt(dis_saver, sess, dis_dir, mode=mode, force=(FLAGS.mode == "train_gan"))
        ckpt = utils.load_ckpt(dis_saver, sess, dis_dir, mode=mode, force=False)
        del mode

    # --------------- train models ---------------
    if FLAGS.mode not in ["pretrain_dis", "decode"]:
        gen_batcher_train = GenBatcher("train", "train", gen_vocab, hps_gen)
        gen_batcher_val = GenBatcher("val", "val", gen_vocab, hps_gen)

    if FLAGS.mode == "decode":
        decoder_batcher = GenBatcher("val", "test", gen_vocab, hps_gen)

    if FLAGS.mode == "train_gan":
        gan_batcher_val = GenBatcher("mini_val", "val", gen_vocab, hps_gen)

    if FLAGS.mode == "pretrain_dis":
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
            with tf.device("/gpu:0"):
                pretrain_generator(generator, gen_batcher_train, sess, gen_batcher_val, gen_saver, gen_val_saver)
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
        best_rouge = 0.2
        gen_best_loss = get_best_loss_from_chpt(val_dir)
        gen_global_step = 0
        print('Going to tune the two using Gan')
        for i_gan in range(hps_gan.gan_iter):
            # Train the generator for one step
            g_losses = []
            current_speed = []
            # for it in range(0):
            # print('Going to train the generator.')
            for it in range(hps_gan.gan_gen_iter):
                start_time = time.time()
                batch = gen_batcher_train.next_batch()

                # generate samples
                enc_states, dec_in_state, n_samples_extend, n_targets_padding_mask = decoder.mc_generate(
                    batch, s_num=hps_gan.sample_num)
                assert np.array(n_samples_extend).shape == (hps_gan.sample_num,
                                                            hps_gen.batch_size,
                                                            hps_gen.max_dec_steps + 1)
                assert np.array(n_targets_padding_mask).shape == (hps_gan.sample_num,
                                                                  hps_gen.batch_size,
                                                                  hps_gen.max_dec_steps)
                # get rewards for the samples
                # strip the start token and stop, the stop is masked out in
                # calculatting the loss
                # n_samples = [np.where(
                #     np.less(samples, hps_gen.gen_vocab_size),
                #     samples, np.array(
                #         [[gen_vocab.word2id(data.UNKNOWN_TOKEN)] * hps_gen.max_dec_steps] * hps_gen.batch_size))
                #     for samples in n_samples_extend]
                n_samples_extend_no_start = np.array(n_samples_extend)[:, :, 1:]
                # for the rouge
                # n_samples_no_start = np.array(n_samples)[:, :, 1:]
                # for the discriminator
                try:
                    n_rewards = rollout.get_reward(
                        hps_gan, sess, gen_vocab, dis_vocab, batch, enc_states,
                        dec_in_state, n_samples_extend_no_start, discriminator)
                except:
                    print('enc_states')
                    print(enc_states)
                    print('enc_states.shape')
                    print(enc_states.shape)
                    print('dec_in_state')
                    print(dec_in_state)
                    for st in enc_states:
                        print(st.shape)
                    raise

                # fine tune the generator
                n_sample_targets = np.array(n_samples_extend)[:, :, 1:]
                n_samples_extend = np.array(n_samples_extend)[:, :, :-1]
                # sample_target_padding_mask = pad_sample(sample_target, gen_vocab, hps_gen)
                n_samples = [np.where(
                    np.less(samples, hps_gen.gen_vocab_size),
                    samples, np.array(
                        [[gen_vocab.word2id(data.UNKNOWN_TOKEN)] * hps_gen.max_dec_steps] * hps_gen.batch_size))
                    for samples in n_samples_extend]
                results = generator.run_gan_batch(
                    sess, batch, n_samples, n_sample_targets, n_targets_padding_mask, n_rewards)

                gen_global_step = results["global_step"]

                # for visualization
                g_loss = results["loss"]
                if not math.isnan(g_loss):
                    g_losses.append(g_loss)
                else:
                    print(colored('a nan in gan loss', 'red'))
                current_speed.append(time.time() - start_time)

            # Test
            # if FLAGS.gan_gen_iter and (i_gan % 100 == 0 or i_gan == hps_gan.gan_iter - 1):
            if hps_gan.gan_gen_iter and (i_gan % 50 == 0 or i_gan == hps_gan.gan_iter - 1):
                print('Going to test the loss of the generator.')
                current_speed = (float(sum(current_speed)) + epsilon) / (int(len(current_speed)) * hps_gen.batch_size + epsilon)
                everage_g_loss = (float(sum(g_losses)) + epsilon) / float(len(g_losses) + epsilon)
                # one more process hould be opened for the evaluation
                gen_eval_loss, gen_best_loss, eval_rouge, best_rouge = gan_save_ckpt(
                    sess, generator, decoder, gen_best_loss, best_rouge, gan_dir, gan_saver,
                    gen_batcher_val, gan_batcher_val, gan_val_dir, gan_val_saver,
                    gen_global_step, FLAGS.sample_rate)

                if gen_eval_loss:
                    print(
                        "\nDashboard for " + colored("GAN Generator", 'green') + " updated %s, "
                        "finished steps:\t%s\n"
                        "\tBatch size:\t%s\n"
                        "\tVocabulary size:\t%s\n"
                        "\tCurrent speed:\t%.4f seconds/article\n"
                        "\tAverage GAN training loss:\t%.4f; "
                        "eval loss:\t%.4f\n"
                        "Average rouge %s; and the best rouge %s." % (
                            datetime.datetime.now().strftime("on %m-%d at %H:%M"),
                            gen_global_step,
                            hps_gen.batch_size,
                            hps_gen.gen_vocab_size,
                            current_speed,
                            everage_g_loss,
                            gen_eval_loss.item(),
                            eval_rouge, best_rouge
                            ))

            # Train the discriminator
            dis_best_loss = 1000
            dis_losses = []
            dis_accuracies = []
            gan_dis_iter = hps_gan.gan_dis_iter if hps_gan.rouge_reward_ratio != 1 else 0
            if gan_dis_iter:
                print('Going to train the discriminator.')
            for d_gan in range(gan_dis_iter):
                batch = gen_batcher_train.next_batch()
                _, _, n_samples_extend, _ = decoder.mc_generate(
                    batch, s_num=hps_gan.sample_num)
                assert np.array(n_samples_extend).shape == (hps_gan.sample_num,
                                                            hps_gen.batch_size,
                                                            hps_gen.max_dec_steps + 1)
                n_samples_extend_no_start = np.array(n_samples_extend)[:, :, 1:]
                # shuould first tanslate to words to avoid unk
                n_samples = [np.where(
                    np.less(samples, hps_gen.gen_vocab_size),
                    samples, np.array(
                        [[gen_vocab.word2id(data.UNKNOWN_TOKEN)] * (hps_gen.max_dec_steps)] * hps_gen.batch_size))
                    for samples in n_samples_extend_no_start]
                for samples in n_samples:
                    emb_dec_batch = sess.run(
                        generator.temp_embedded_seq,
                        feed_dict={generator.temp_batch: batch.padded_abs_ids})
                    emb_conditions = sess.run(
                        generator.temp_embedded_seq,
                        feed_dict={generator.temp_batch: batch.enc_batch})
                    # feed_dict={generator.temp_batch: batch.padded_enc_batch})
                    emb_samples = sess.run(
                        generator.temp_embedded_seq,
                        feed_dict={generator.temp_batch: samples})

                    _range = range(len(emb_dec_batch))
                    sattolo_cycle(_range)
                    indices = np.array(_range)

                    inputs = np.concatenate([emb_samples, emb_dec_batch, emb_dec_batch[indices]], 0)
                    conditions = np.concatenate([emb_conditions, emb_conditions, emb_conditions], 0)
                    condition_lens = np.concatenate([batch.enc_lens, batch.enc_lens, batch.enc_lens], 0)
                    targets = [NEGATIVE_LABEL for _ in samples] + [POSITIVE_LABEL for _ in emb_dec_batch] + [NEGATIVE_LABEL for _ in emb_dec_batch]
                    targets = np.array(targets)
                    assert len(inputs) == len(conditions) == len(condition_lens) == len(targets)

                    # randomize the samples
                    _range = range(len(inputs))
                    sattolo_cycle(_range)
                    indices = np.array(_range)

                    parts = 3
                    inputs = np.split(inputs[indices], parts)
                    conditions = np.split(conditions[indices], parts)
                    condition_lens = np.split(condition_lens[indices], parts)
                    targets = np.split(targets[indices], parts)

                    for p in range(parts):
                        results = discriminator.run_one_batch(sess, inputs[p], conditions[p], condition_lens[p], targets[p])
                        d_loss = results["loss"]
                        if not math.isnan(d_loss):
                            dis_losses.append(float(d_loss))
                            dis_accuracies.append(results["accuracy"].item())
                        else:
                            print(colored('a nan in dis loss', 'red'))
                            print('inputs[p]')
                            print(inputs[p])
                            print('conditions[p]')
                            print(conditions[p])
                            print('condition_lens[p]')
                            print(condition_lens[p])
                            print('targets[p]')
                            print(targets[p])
                            break

                ave_dis_acc = sum(dis_accuracies) / len(dis_accuracies)
                if d_gan % 50 == 0 or d_gan == hps_gan.gan_dis_iter - 1:
                    if (sum(dis_losses) / len(dis_losses)) < dis_best_loss:
                        dis_best_loss = sum(dis_losses) / len(dis_losses)
                        checkpoint_path = ensure_exists(join_path(hps_dis.model_dir, "discriminator")) + "/model.ckpt"
                        dis_saver.save(sess, checkpoint_path, global_step=results["global_step"])
                    print_dashboard("GAN Discriminator", results["global_step"].item(), hps_dis.batch_size, hps_dis.dis_vocab_size,
                                    results["loss"].item(), 0.00, 0.00, 0.00)
                    print("Average training accuracy: \t%.4f" % ave_dis_acc)

                if ave_dis_acc > 0.8:
                    break

    # --------------- decoding samples ---------------
    elif FLAGS.mode == "decode":
        print('Going to decode from the generator.')

        decoder.beam_search(decoder_batcher)
        print("Finished decoding..")
        # decode for generating corpus for discriminator

    sess.close()


if __name__ == '__main__':
  tf.app.run()
