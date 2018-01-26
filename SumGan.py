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
from data import gen_vocab2dis_vocab
from os.path import join as join_path
from utils import ensure_exists
from gen_utils import calc_running_avg_loss
from gen_utils import check_rouge
from gen_utils import get_best_loss_from_chpt
from gen_utils import save_ckpt
from utils import print_dashboard
from dis_utils import dump_chpt
import math
from data import PAD_TOKEN
from termcolor import colored

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
tf.app.flags.DEFINE_float('sample_rate', 0.004, 'the sample rate, should be [0, 0.5]')

# ------------------------------------- discriminator

# Model parameters
tf.app.flags.DEFINE_integer("layer_size", 300, "Size of each model layer.")
tf.app.flags.DEFINE_integer("conv_layers", 2, "Number of convolution layers in the model.")
tf.app.flags.DEFINE_integer("pool_layers", 2, "Number of pooling layers in the model.")
tf.app.flags.DEFINE_integer("kernel_size", 3, "The kernel size of the filters along the sentence length dimension.")
tf.app.flags.DEFINE_integer("pool_size", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Cell type")
tf.app.flags.DEFINE_integer("dis_vocab_size", 100000, "vocabulary size.")
tf.app.flags.DEFINE_string("dis_vocab_file", "dis_vocab", "the path of the discriminator vocabulary.")
tf.app.flags.DEFINE_string("vocab_type", "char", "the path of the discriminator vocabulary.")
tf.app.flags.DEFINE_integer("num_class", 2, "num of output classes.")
tf.app.flags.DEFINE_integer("num_models", 3, "Size of each model layer. The actural size is doubled.")

# Training parameters
tf.app.flags.DEFINE_float("dis_lr", 0.0005, "Learning rate.")
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
tf.app.flags.DEFINE_boolean('update_gen', True, 'to decide if to train generator.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 300, 'dimension of word embeddings')
# if batch_size is one and beam size is not one in the decode mode then the beam
# search is the same as the original beam search
tf.app.flags.DEFINE_integer('max_enc_steps', 75, 'max timesteps of encoder (max source text tokens)')  # 400
tf.app.flags.DEFINE_integer('max_dec_steps', 12, 'max timesteps of decoder (max summary tokens)')  # 100
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 5, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('gen_vocab_size', 100000, 'Size of vocabulary. These will be read from the vocabulary file in'
                            ' order. If the vocabulary file contains fewer words than this number,'
                            ' or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('gen_lr', 0.0009, 'learning rate')
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
tf.app.flags.DEFINE_integer('gan_gen_iter', 5, 'in each gan step run how many times the generator')
tf.app.flags.DEFINE_integer('gan_dis_iter', 10, 'in each gan step run how many times the generator')
tf.app.flags.DEFINE_integer('rollout_num', 12, 'how many times to repeat the rollout process.')
tf.app.flags.DEFINE_string("gan_dir", "gan_dir", "Training directory.")
tf.app.flags.DEFINE_integer('sample_num', 2, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_float('gan_lr', 0.0005, 'learning rate for the gen in GAN training')
tf.app.flags.DEFINE_float('rouge_reward_ratio', 0.0, 'The importance of rollout in calculating the reward.')

FLAGS = tf.app.flags.FLAGS

assert FLAGS.mode in ["pretrain_gen", "pretrain_dis", "train_gan", "decode", "test"]
assert FLAGS.sample_rate >= 0 and FLAGS.sample_rate <= 0.5, "sample rate should be [0, 0.5]"

if FLAGS.mode == "train_gan":
    FLAGS.single_pass = False

if FLAGS.min_dec_steps > FLAGS.max_dec_steps / 2:
    FLAGS.min_dec_steps = int(FLAGS.max_dec_steps / 2)

ensure_exists(FLAGS.model_dir)


def pretrain_generator(model, batcher, sess, batcher_val, model_saver, val_saver):
    """Repeatedly runs training iterations, logging loss to screen and writing
    summaries"""
    print("starting run_training")
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
        # print('loss: %f', loss)  # print the loss to screen
        if hps.coverage:
            coverage_loss = results['coverage_loss']

        running_avg_loss = calc_running_avg_loss(
            np.asscalar(loss), running_avg_loss, global_step)

        if global_step % eval_save_steps == 0:
            # check if it is the best checkpoint so far
            eval_loss, best_loss = save_ckpt(
                sess, model, best_loss, model_dir, model_saver,
                batcher_val, val_dir, val_saver, global_step)
            last_ten_eval_loss.append(eval_loss)
            if len(last_ten_eval_loss) == 10 and min(last_ten_eval_loss) == last_ten_eval_loss[0] and eval_save_steps > 5000:
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
        'adagrad_init_acc',
        'steps_per_checkpoint',
        'batch_size',
        'beam_size',
        'cov_loss_wt',
        'coverage',
        'emb_dim',
        'rand_unif_init_mag',
        'gen_vocab_file',
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
    if hps_gen.gen_vocab_file == hps_dis.dis_vocab_file:
        hps_dis = hps_dis._replace(vocab_type="word")
        hps_dis = hps_dis._replace(layer_size=hps_gen.emb_dim)
        hps_dis = hps_dis._replace(dis_vocab_size=hps_gen.gen_vocab_size)
    if FLAGS.mode == "train_gan":
        hps_gen = hps_gen._replace(batch_size=hps_gen.batch_size * hps_dis.num_models)

    if FLAGS.mode != "pretrain_dis":
        with tf.variable_scope("generator"):
            generator = PointerGenerator(hps_gen, gen_vocab)
            print("Building generator graph ...")
            gen_decoder_scope = generator.build_graph()

    if FLAGS.mode != "pretrain_gen":
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
        ckpt_path = utils.load_ckpt(gen_saver, sess, gen_dir)
        print('going to restore embeddings from checkpoint')
        if not ckpt_path:
            emb_path = join_path(FLAGS.model_dir, "generator", "init_embed")
            if emb_path:
                generator.saver.restore(
                    sess,
                    tf.train.get_checkpoint_state(emb_path).model_checkpoint_path
                )
                print(colored("successfully restored embeddings form %s" % emb_path, 'green'))
            else:
                print(colored("failed to restore embeddings form %s" % emb_path, 'red'))

    elif FLAGS.mode in ["decode", "train_gan"]:
        print("Restoring the generator model from the best checkpoint...")
        dec_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "generator" in v.name])
        val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.val_dir))
        gan_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.gan_dir))
        gan_val_dir = ensure_exists(join_path(FLAGS.model_dir, 'generator', FLAGS.gan_dir, "val"))
        gan_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "generator" in v.name])
        gan_val_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "generator" in v.name])
        utils.load_ckpt(dec_saver, sess, val_dir, (FLAGS.mode in ["train_gan", "decode"]))
        decoder = Decoder(sess, generator, gen_vocab)

    if FLAGS.mode in ["pretrain_dis", "train_gan"]:
        dis_saver = tf.train.Saver(
            max_to_keep=3, var_list=[v for v in all_variables if "discriminator" in v.name])
        dis_dir = ensure_exists(join_path(FLAGS.model_dir, 'discriminator'))
        ckpt = utils.load_ckpt(dis_saver, sess, dis_dir)
        if not ckpt:
            if hps_dis.vocab_type == "word":
                discriminator.init_emb(sess, join_path(FLAGS.model_dir, "generator", "init_embed"))
            else:
                discriminator.init_emb(sess, join_path(FLAGS.model_dir, "discriminator", "init_embed"))

    # --------------- train models ---------------
    if FLAGS.mode not in ["pretrain_dis", "decode"]:
        gen_batcher_train = GenBatcher("train", gen_vocab, hps_gen, single_pass=hps_gen.single_pass)
        gen_batcher_val = GenBatcher("val", gen_vocab, hps_gen, single_pass=True)
        val_saver = tf.train.Saver(max_to_keep=10,
                                   var_list=[v for v in all_variables if "generator" in v.name])

    if FLAGS.mode == "decode":
        decoder_batcher = GenBatcher("test", gen_vocab, hps_gen, single_pass=True)

    if FLAGS.mode not in ["pretrain_gen", "decode"]:
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
                enc_states, dec_in_state, n_samples, n_targets_padding_mask = decoder.mc_generate(
                    batch, include_start_token=True, s_num=hps_gan.sample_num)
                # get rewards for the samples
                n_rewards = rollout.get_reward(
                    hps_gan, sess, gen_vocab, dis_vocab, batch, enc_states,
                    dec_in_state, n_samples, discriminator)

                # fine tune the generator
                n_sample_targets = [samples[:, 1:] for samples in n_samples]
                n_targets_padding_mask = [padding_mask[:, 1:] for padding_mask in n_targets_padding_mask]
                n_samples = [samples[:, :-1] for samples in n_samples]
                # sample_target_padding_mask = pad_sample(sample_target, gen_vocab, hps_gen)
                n_samples = [np.where(
                    np.less(samples, hps_gen.gen_vocab_size),
                    samples, np.array(
                        [[gen_vocab.word2id(data.UNKNOWN_TOKEN)] * hps_gen.max_dec_steps] * hps_gen.batch_size))
                    for samples in n_samples]
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
            if i_gan % 100 == 0 or i_gan == hps_gan.gan_iter - 1:
                print('Going to test the loss of the generator.')
                current_speed = (float(sum(current_speed)) + epsilon) / (int(len(current_speed)) * hps_gen.batch_size + epsilon)
                everage_g_loss = (float(sum(g_losses)) + epsilon) / int((len(g_losses)) + epsilon)
                # one more process hould be opened for the evaluation
                eval_loss, gen_best_loss = save_ckpt(
                    sess, generator, gen_best_loss, gan_dir, gan_saver,
                    gen_batcher_val, gan_val_dir, val_saver, gen_global_step)

                if eval_loss:
                    print(
                        "\nDashboard for " + colored("GAN Generator", 'green') + " updated %s, "
                        "finished steps:\t%s\n"
                        "\tBatch size:\t%s\n"
                        "\tVocabulary size:\t%s\n"
                        "\tCurrent speed:\t%.4f seconds/article\n"
                        "\tAverage GAN training loss:\t%.4f; "
                        "eval loss:\t%.4f" % (
                            datetime.datetime.now().strftime("on %m-%d at %H:%M"),
                            gen_global_step,
                            FLAGS.batch_size,
                            hps_gen.gen_vocab_size,
                            current_speed,
                            everage_g_loss,
                            eval_loss.item(),
                            )
                    )

            if i_gan % 200 == 0 or i_gan == hps_gan.gan_iter - 1:
                print("Going to evaluate rouge.")
                ave_rouge, best_rouge = check_rouge(sess, decoder, best_rouge, gen_batcher_val,
                                                    gan_val_dir, gan_val_saver, gen_global_step, sample_rate=FLAGS.sample_rate)
                print("The best rouge is %s and the average rouge is %s %s" % (
                    best_rouge, ave_rouge,
                    datetime.datetime.now().strftime("on %m-%d at %H:%M")
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
                enc_states, dec_in_state, k_samples_words, _ = decoder.mc_generate(
                    batch, s_num=hps_gan.sample_num, include_start_token=True)
                # shuould first tanslate to words to avoid unk
                articles_oovs = batch.art_oovs
                for samples_words in k_samples_words:
                    dec_batch_words = batch.target_batch
                    conditions_words = batch.enc_batch_extend_vocab
                    zeros = np.zeros((conditions_words.shape[0], hps_gen.max_enc_steps))
                    zeros[:, :conditions_words.shape[1]] = conditions_words
                    conditions_words = zeros
                    if hps_dis.vocab_type == "char":
                        samples = gen_vocab2dis_vocab(
                            samples_words, gen_vocab, articles_oovs,
                            dis_vocab, hps_dis.max_dec_steps, STOP_DECODING)
                        dec_batch = gen_vocab2dis_vocab(
                            dec_batch_words, gen_vocab, articles_oovs, dis_vocab, hps_dis.max_dec_steps, STOP_DECODING)
                        conditions = gen_vocab2dis_vocab(
                            conditions_words, gen_vocab, articles_oovs,
                            dis_vocab, hps_dis.max_enc_steps, PAD_TOKEN)
                    else:
                        samples = samples_words
                        dec_batch = dec_batch_words
                        conditions = conditions_words
                        # the unknown in target

                    inputs = np.concatenate([samples, dec_batch], 0)
                    conditions = np.concatenate([conditions, conditions], 0)

                    targets = [[1, 0] for _ in samples] + [[0, 1] for _ in dec_batch]
                    targets = np.array(targets)
                    # randomize the samples
                    assert len(inputs) == len(conditions) == len(targets), "lengthes of the inputs, conditions and targests should be the same."
                    indices = np.random.permutation(len(inputs))
                    inputs = np.split(inputs[indices], 2)
                    conditions = np.split(conditions[indices], 2)
                    targets = np.split(targets[indices], 2)
                    assert len(inputs) % 2 == 0, "the length should be mean"

                    results = discriminator.run_one_batch(sess, inputs[0], conditions[0], targets[0])
                    dis_accuracies.append(results["accuracy"].item())
                    dis_losses.append(results["loss"].item())

                    results = discriminator.run_one_batch(sess, inputs[1], conditions[1], targets[1])
                    dis_accuracies.append(results["accuracy"].item())

                ave_dis_acc = sum(dis_accuracies) / len(dis_accuracies)
                if d_gan == hps_gan.gan_dis_iter - 1:
                    if (sum(dis_losses) / len(dis_losses)) < dis_best_loss:
                        dis_best_loss = sum(dis_losses) / len(dis_losses)
                        checkpoint_path = ensure_exists(join_path(hps_dis.model_dir, "discriminator")) + "/model.ckpt"
                        dis_saver.save(sess, checkpoint_path, global_step=results["global_step"])
                    print_dashboard("GAN Discriminator", results["global_step"].item(), hps_dis.batch_size, hps_dis.dis_vocab_size,
                                    results["loss"].item(), 0.00, 0.00, 0.00)
                    print("Average training accuracy: \t%.4f" % ave_dis_acc)

                if ave_dis_acc > 0.9:
                    break

    # --------------- decoding samples ---------------
    elif FLAGS.mode == "decode":
        print('Going to decode from the generator.')
        decoder.bs_decode(decoder_batcher)
        print("Finished decoding..")
        # decode for generating corpus for discriminator

    sess.close()


if __name__ == '__main__':
  tf.app.run()
