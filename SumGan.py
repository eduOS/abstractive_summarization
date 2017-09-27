import tensorflow as tf
from collections import namedtuple
import numpy
import sys
import os
import util
import time
from batcher import Batcher
from decode import BeamSearchDecoder
from cnn_discriminator import DisCNN
from pointer_generator import PointerGenerator
from share_function import prepare_data
from share_function import gen_force_train_iter
from share_function import FlushFile
from share_function import prepare_gan_dis_data
from share_function import prepare_sentence_to_maxlen
from share_function import deal_generated_y_sentence
from attention_decoder import attention_decoder
from decode import BeamSearchDecoder
from rollout import ROLLOUT

from res_discriminator import Seq2ClassModel
from data import VocabD, VocabG


tf.app.flags.DEFINE_string('mode', 'train', 'must be one of pretrain_gen/pretrain_dis/train_gan/decode')
# ------------------------------------- discriminator

# Model parameters
tf.app.flags.DEFINE_integer("dis_hidden_dim", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "Cell type")
tf.app.flags.DEFINE_integer("dis_vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("num_class", 50, "num of output classes.")
tf.app.flags.DEFINE_string("buckets", "9,12,20,40", "buckets of different lengths")

# Training parameters
tf.app.flags.DEFINE_float("dis_lr", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("lr_decay_factor", 0.5, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("dis_max_gradient", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("dis_batch_size", 64, "Batch size to use during training.")
tf.app.flags.DEFINE_boolean("early_stop", False, "Set to True to turn on early stop.")
tf.app.flags.DEFINE_integer("max_steps", -1, "max number of steps to train")

# Misc
tf.app.flags.DEFINE_string("dis_data_dir", "./js_corpus", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_integer("gpu_id", 0, "Select which gpu to use.")

# Mode
tf.app.flags.DEFINE_boolean("interactive_test", False, "Set to True for interactive testing.")
tf.app.flags.DEFINE_boolean("test", False, "Run a test on the eval set.")

# ------------------------------------- generator

# the GAN training setting
tf.app.flags.DEFINE_boolean('teacher_forcing', False, 'whether to do use teacher forcing for training the generator')

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a'
                            'fixed checkpoint, i.e. take the current checkpoint, and use it to'
                            'produce one summary for each example in the dataset, writethesummaries'
                            'to file and then get ROUGE scores for the whole dataset. If False'
                            '(default), run concurrent decoding, i.e. repeatedly load latest'
                            'checkpoint, use it to produce summaries forrandomly-chosenexamples and'
                            'log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in adirectory with this name, under log_root.')

# Hyperparameters
tf.app.flags.DEFINE_integer('gen_hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('gen_batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 80, 'max timesteps of encoder (max source text tokens)')  # 400
tf.app.flags.DEFINE_integer('max_dec_steps', 15, 'max timesteps of decoder (max summary tokens)')  # 100
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
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
tf.app.flags.DEFINE_boolean('coverage', True, 'Use coverage mechanism. Note, the experiments reported in the ACL '
                            'paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards.'
                            'i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', True, 'Convert a non-coverage model to a coverage model. '
                            'Turn this on and run in train mode. \ Your current model will be copied to a new version '
                            '(same name with _cov_init appended)\ that will be ready to run with coverage flag turned on,\ for the coverage training stage.')

FLAGS = tf.app.flags.FLAGS


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    """Repeatedly runs training iterations, logging loss to screen and writing
    summaries"""
    tf.logging.info("starting run_training")
    coverage_loss = None
    with sess_context_manager as sess:
        step = 0
        print_gap = 1000
        start_time = time.time()
        while True:  # repeats until interrupted
            batch = batcher.next_batch()

            # tf.logging.info('running training step...')
            t0 = time.time()
            results = model.run_train_step(sess, batch)
            t1 = time.time()
            # tf.logging.info('seconds for training step: %.3f', t1-t0)

            loss = results['loss']
            # tf.logging.info('loss: %f', loss)  # print the loss to screen
            step += 1
            if FLAGS.coverage:
                coverage_loss = results['coverage_loss']
                # print the coverage loss to screen
                # tf.logging.info("coverage_loss: %f", coverage_loss)

            if step % print_gap == 0:
                current_time = time.time()
                print(
                  "\nDashboard until the step:\t%s\n"
                  "\tBatch size:\t%s\n"
                  "\tRunning average loss:\t%s per article \n"
                  "\tArticles trained:\t%s\n"
                  "\tTotal training time:\t%ss(%s hours)\n"
                  "\tCurrent speed:\t%s seconds/article\n"
                  "\tCoverage loss\t%s\n" % (
                    step,
                    FLAGS.batch_size,
                    loss,
                    FLAGS.batch_size * step,
                    current_time - start_time,
                    (current_time - start_time) / 3600,
                    (t1-t0) / FLAGS.batch_size,
                    coverage_loss,
                  )
                )
            # get the summaries and iteration number so we can write summaries
            # to tensorboard
            # we will write these summaries to tensorboard using summary_writer
            summaries = results['summaries']
            # we need this to update our running average loss
            train_step = results['global_step']

            summary_writer.add_summary(summaries, train_step)  # write the summaries
            if train_step % 100 == 0:  # flush the summary writer every so often
                summary_writer.flush()


def train_discriminator():
    pass


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for
    coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver()
    # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def setup_training(max_to_keep):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    default_device = tf.device('/gpu:0')
    with default_device:
        if FLAGS.convert_to_coverage_model:
            assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
            convert_to_coverage_model()
        saver = tf.train.Saver(max_to_keep=max_to_keep)

    sv = tf.train.Supervisor(logdir=train_dir, is_chief=True, saver=saver, summary_op=None, save_summaries_secs=60, save_model_secs=60, global_step=generator.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info("Created session.")

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+".meta"):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        for ld in model.loaders:
            ld.restore(session, tf.train.get_checkpoint_state("pretrained").model_checkpoint_path)
    return generator, discriminator


def main(argv):
    tf.set_random_seed(111)  # a seed value for randomness

    # Create a batcher object that will create minibatches of data
    vocab = VocabG(FLAGS.vocab_path, FLAGS.vocab_size)
    batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
    # TODO change to pass number

    if "train" in FLAGS.mode:
        # --------------- build graph ---------------

        hparam_gen = [
            'mode',
            'adagrad_init_acc',
            'gen_batch_size',
            'cov_loss_wt',
            'coverage',
            'emb_dim',
            'gen_hidden_dim',
            'gen_lr',
            'max_dec_steps',
            'gen_max_gradient',
            'pointer_gen',
            'rand_unif_init_mag',
            'trunc_norm_init_std',
            'max_enc_steps',
        ]

        hps_dict = {}
        for key, val in FLAGS.__flags.iteritems():  # for each flag
            if key in hparam_gen:  # if it's in the list
                hps_dict[key] = val  # add it to the dict

        hps = namedtuple("HParams4Gen", hps_dict.keys())(**hps_dict)
        if FLAGS.segment is not True:
            hps = hps._replace(max_enc_steps=110)
            hps = hps._replace(max_dec_steps=25)
        else:
            assert hps.max_enc_steps == 80, "No segmentation, max_enc_steps wrong"
            assert hps.max_dec_steps == 15, "No segmentation, max_dec_steps wrong"
        with tf.variable_scope("generator"):
            generator = PointerGenerator(hps, vocab)
            generator.build_graph()

        hparam_dis = [
            'mode',
            'train_dir',
            'dis_vocab_size',
            'num_class',
            'buckets',
            'dis_hidden_dim',
            'num_layers',
            'dis_max_gradient',
            'dis_batch_size',
            'dis_lr',
            'lr_decay_factor',
            'cell_type',
        ]
        hps_dict = {}
        for key, val in FLAGS.__flags.iteritems():  # for each flag
            if key in hparam_dis:  # if it's in the list
                hps_dict[key] = val  # add it to the dict

        hps = namedtuple("HParams4Dis", hps_dict.keys())(**hps_dict)
        with tf.variable_scope("discriminator"):
            discriminator = Seq2ClassModel(hps)
            discriminator.build_graph()

        saver = tf.train.Saver()

        sv = tf.train.Supervisor(logdir=hps.train_dir, is_chief=True, saver=saver, summary_op=None, save_summaries_secs=60, save_model_secs=60)
        summary_writer = sv.summary_writer
        tf.logging.info("Preparing or waiting for session...")
        sess = sv.prepare_or_wait_for_session(config=util.get_config())

        # initialize the embedding at the begging of training
        ckpt = tf.train.get_checkpoint_state(hps.train_dir)
        if ckpt and not tf.gfile.Exists(ckpt.model_checkpoint_path+".meta"):
            discriminator.init_emb(sess, hps.train_dir)

        # --------------- train generator ---------------
        if "gen" in mode:
            generator_training_model_hps = hps
            train_generator()

        # --------------- train discriminator ---------------
        elif "dis" in mode:
            train_discriminator()

        # --------------- finetune the generator ---------------
        else:
            rollout = ROLLOUT(generator, 0.8)
            for i_gan in range(gan_iter):
                # Train the generator for one step
                for it in range(gan_gen_iter):
                    samples = generator.generate(sess)
                    rewards = rollout.get_reward(sess, samples, 16, discriminator)
                    feed = {generator.x: samples, generator.rewards: rewards}
                    _ = sess.run(generator.g_updates, feed_dict=feed)

                # Test
                if i_gan % 5 == 0 or i_gan == gan_iter - 1:
                    generate_samples(sess, generator, gan_iter, generated_num, eval_file)
                    likelihood_data_loader.create_batches(eval_file)
                    test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                    buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                    print('total_batch: ', total_batch, 'test_loss: ', test_loss)
                    log.write(buffer)

                # Update roll-out parameters
                rollout.update_params()

                # Train the discriminator
                for _ in range(gan_dis_iter):
                    generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                    dis_data_loader.load_train_data(positive_file, negative_file)

                    for _ in range(3):
                        dis_data_loader.reset_pointer()
                        for it in xrange(dis_data_loader.num_batch):
                            x_batch, y_batch = dis_data_loader.next_batch()
                            feed = {
                                discriminator.input_x: x_batch,
                                discriminator.input_y: y_batch,
                                discriminator.dropout_keep_prob: dis_dropout_keep_prob
                            }
                            _ = sess.run(discriminator.train_op, feed)

    elif FLAGS.mode == "decode":
        decode_model_hps = hps
        decode_model_hps = decode_model_hps._replace(mode="decode")
        decode_model_hps = decode_model_hps._replace(max_dec_steps=1)
        generator = PointerGenerator(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(generator, batcher, vocab)
        decoder.decode()
