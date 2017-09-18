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
from pointer_generator import GenSum
from share_function import prepare_data
from share_function import gen_force_train_iter
from share_function import FlushFile
from share_function import prepare_gan_dis_data
from share_function import prepare_sentence_to_maxlen
from share_function import deal_generated_y_sentence

from tensorflow.python.platform import tf_logging as logging
from data import Vocab

# ------------------------------------- from tfGan
tf.app.flags.DEFINE_integer(
    'max_epoches', 1, 'the max epoches for training')
tf.app.flags.DEFINE_integer('dis_epoches', 2, 'the max epoches for training')

tf.app.flags.DEFINE_integer(
    'gan_total_iter_num', 1, 'the max epoches for training')
tf.app.flags.DEFINE_integer(
    'gan_gen_iter_num', 1, 'the max epoches for training')
tf.app.flags.DEFINE_integer(
    'gan_dis_iter_num', 1, 'the max epoches for training')

tf.app.flags.DEFINE_integer(
    'dispFreq', 50, 'train for this many minibatches for displaying')
tf.app.flags.DEFINE_integer(
    'dis_dispFreq', 1,
    'train for this many minibatches for displaying discriminator')
tf.app.flags.DEFINE_integer(
    'gan_dispFreq', 50,
    'train for this many minibatches for displaying the gan gen training')
tf.app.flags.DEFINE_integer(
    'dis_saveFreq', 100,
    'train for this many minibatches for displaying discriminator')
tf.app.flags.DEFINE_integer(
    'gan_saveFreq', 100,
    'train for this many minibatches for displaying discriminator')
tf.app.flags.DEFINE_integer(
    'dis_devFreq', 100,
    'train for this many minibatches for displaying discriminator')
tf.app.flags.DEFINE_integer(
    'vocab_size', 80000, 'the size of the target vocabulary')

tf.app.flags.DEFINE_integer(
    'validFreq', 1000, 'train for this many minibatches for validation')
tf.app.flags.DEFINE_integer(
    'saveFreq', 2000, 'train for this many minibatches for saving model')
tf.app.flags.DEFINE_integer(
    'sampleFreq', 10000000, 'train for this many minibatches for sampling')

tf.app.flags.DEFINE_float('l2_r', 0.0001, 'L2 regularization penalty')

tf.app.flags.DEFINE_integer(
    'batch_size', 60, 'the size of the minibatch for training')
tf.app.flags.DEFINE_integer(
    'dis_batch_size', 10,
    'the size of the minibatch for training discriminator ')
tf.app.flags.DEFINE_integer(
    'gen_batch_size', 1, 'the size of the minibatch for training generator ')
tf.app.flags.DEFINE_integer(
    'gan_gen_batch_size', 2,
    'the size of the minibatch for training generator ')
tf.app.flags.DEFINE_integer(
    'gan_dis_batch_size', 1,
    'the size of the minibatch for training generator ')

tf.app.flags.DEFINE_integer(
    'valid_batch_size', 10, 'the size of the minibatch for validation')

tf.app.flags.DEFINE_string('optimizer', 'adadelta', 'the optimizing method')

tf.app.flags.DEFINE_string(
    'saveto', './gen_model/lcsts', 'the file name used to store the model')
tf.app.flags.DEFINE_string(
    'dis_saveto', './dis_model/lcsts',
    'the file name used to store the model of the discriminator')

tf.app.flags.DEFINE_string(
    'train_data_source', './data_1000w_golden/source_u8.txt.shuf',
    'the train data set of the soruce side')
tf.app.flags.DEFINE_string(
    'train_data_target', './data_1000w_golden/target_u8.txt.shuf',
    'the train data set of the target side')

tf.app.flags.DEFINE_string(
    'dis_positive_data', './data_test1000/positive.txt.shuf',
    'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string(
    'dis_negative_data', './data_test1000/negative.txt.shuf',
    'the negative train data set for the discriminator')
tf.app.flags.DEFINE_string(
    'dis_source_data', './data_test1000/source.txt.shuf',
    'the negative train data set for the discriminator')

tf.app.flags.DEFINE_string(
    'dis_dev_positive_data', './data_gan_100w_fromZxw/dev_positive_u8.txt',
    'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string(
    'dis_dev_negative_data', './data_gan_100w_fromZxw/dev_negative_u8.txt',
    'the negative train data set for the discriminator')
tf.app.flags.DEFINE_string(
    'dis_dev_source_data', './data_gan_100w_fromZxw/dev_source_u8.txt',
    'the negative train data set for the discriminator')

tf.app.flags.DEFINE_string(
    'gan_gen_source_data', './data_test1000/gan_gen_source_u8.txt',
    'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string(
    'gan_dis_source_data', './data_gan_100w_fromZxw/gan_dis_source_u8.txt',
    'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string(
    'gan_dis_positive_data', './data_gan_100w_fromZxw/gan_dis_positive_u8.txt',
    'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string(
    'gan_dis_negative_data', './data_gan_100w_fromZxw/gan_dis_negative_u8.txt',
    'the negative train data set for the discriminator')

tf.app.flags.DEFINE_string(
    'valid_data_source', 'data/zhyang/dl4mt/source.txt',
    'the valid data set of the soruce size')
tf.app.flags.DEFINE_string(
    'valid_data_target', 'data/zhyang/dl4mt/target.txt',
    'the valid data set of the target side')

tf.app.flags.DEFINE_string(
    'vocab_path', './vocab', "the vocabulary")
# tf.app.flags.DEFINE_string(
#     'target_dict', './data_1000w_golden/target_u8.txt.shuf.pkl',
#     'the target vocabulary')

tf.app.flags.DEFINE_boolean(
    'use_dropout', False, 'whether to use dropout')
tf.app.flags.DEFINE_boolean(
    'gen_reload', False, 'whether to reload the generate model from model file')
tf.app.flags.DEFINE_boolean(
    'dis_reload', False,
    'whether to reload the discriminator model from model file')

tf.app.flags.DEFINE_boolean(
    'reshuffle', False, 'whether to reshuffle train data')
tf.app.flags.DEFINE_boolean(
    'dis_reshuffle', False,
    'whether to reshuffle train data of the discriminator')
tf.app.flags.DEFINE_boolean(
    'gen_reshuffle', False, 'whether to reshuffle train data of the generator')
tf.app.flags.DEFINE_boolean(
    'gan_gen_reshuffle', False,
    'whether to reshuffle train data of the generator')
tf.app.flags.DEFINE_boolean(
    'gan_dis_reshuffle', False,
    'whether to reshuffle train data of the generator')

tf.app.flags.DEFINE_boolean('DebugMode', False, 'whether to debug')

tf.app.flags.DEFINE_string(
    'adagrad_init_acc', '0.1',
    'the initial accuracy for the adagrad optimizer')

tf.app.flags.DEFINE_string(
    'cov_loss_wt', '1',
    'the coverage loss weight')

tf.app.flags.DEFINE_string(
    'cpu_device', 'cpu-0', 'this cpu used to train the model')
tf.app.flags.DEFINE_string(
    'init_device', '/cpu:0', 'this cpu used to train the model')

tf.app.flags.DEFINE_string('precision', 'float32', 'precision on GPU')

tf.app.flags.DEFINE_integer('rollnum', 16, 'the rollnum for rollout')
tf.app.flags.DEFINE_integer('generate_num', 200000, 'the rollnum for rollout')
tf.app.flags.DEFINE_float('bias_num', 0.5, 'the bias_num  for rewards')

tf.app.flags.DEFINE_boolean(
    'decode_is_print', False, 'whether to decode')

tf.app.flags.DEFINE_string(
    'decode_file', '/home/lerner/data/LCSTS/finished_files/test-art.txt',
    'the file to be decoded')
tf.app.flags.DEFINE_string(
    'decode_result_file', './data_test/negative.txt',
    'the file to save the decode results')

FLAGS = tf.app.flags.FLAGS

logging.set_verbosity(logging.INFO)

# ------------------------------------- from pointer generator

# the GAN training setting
tf.app.flags.DEFINE_boolean(
    'teacher_forcing', False,
    'whether to do use teacher forcing for training the generator')
tf.app.flags.DEFINE_boolean(
    'is_gan_train', False, 'whether to do generative adversarial train')
tf.app.flags.DEFINE_boolean(
    'is_generator_train', True, 'whether to do generative adversarial train')
tf.app.flags.DEFINE_boolean(
    'is_discriminator_train', False,
    'whether to do generative adversarial train')
tf.app.flags.DEFINE_boolean(
    'is_decode', False, 'whether to decode')

# Where to find data
tf.app.flags.DEFINE_string(
    'data_path', '',
    ('Path expression to tf.Example datafiles. Can include wildcards to access'
     'multiple datafiles.'))
tf.app.flags.DEFINE_string(
    'vocab_path',
    '',
    'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean(
    'single_pass', False,
    (
        'For decode mode only. If True, run eval on the full dataset using a'
        'fixed checkpoint, i.e. take the current checkpoint, and use it to'
        'produce one summary for each example in the dataset, writethesummaries'
        'to file and then get ROUGE scores for the whole dataset. If False'
        '(default), run concurrent decoding, i.e. repeatedly load latest'
        'checkpoint, use it to produce summaries forrandomly-chosenexamples and'
        'log the results to screen, indefinitely.'))

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string(
    'exp_name', '',
    (
        'Name for experiment. Logs will be saved in'
        'adirectory with this name, under log_root.'
    ))

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer(
    'max_enc_steps',
    80,
    'max timesteps of encoder (max source text tokens)')  # 400
tf.app.flags.DEFINE_integer(
    'max_dec_steps',
    15,
    'max timesteps of decoder (max summary tokens)')  # 100
tf.app.flags.DEFINE_integer(
    'beam_size',
    4,
    'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer(
    'min_dec_steps', 35,
    'Minimum sequence length of generated summary. \
    Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer(
    'vocab_size',
    50000,
    (
        'Size of vocabulary. These will be read from the vocabulary file in'
        ' order. If the vocabulary file contains fewer words than this number,'
        ' or if this number is set to 0, will take all words in the'
        ' vocabulary file.'
    )
)
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float(
    'adagrad_init_acc',
    0.1,
    'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float(
    'rand_unif_init_mag', 0.02,
    'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float(
    'trunc_norm_init_std', 1e-4,
    'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean(
    'pointer_gen', True,
    'If True, use pointer-generator model. If False, use baseline model.')
tf.app.flags.DEFINE_boolean(
    'segment', True,
    'If True, the source text is segmented, \
    then max_enc_steps and max_dec_steps should be much smaller')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean(
    'coverage',
    True,
    'Use coverage mechanism. Note, the experiments reported in the ACL\
    paper train WITHOUT coverage until converged, \
    and then train for a short phase WITH coverage afterwards. \
    i.e. to reproduce the results in the ACL paper, \
    turn this off for most of training then turn on \
    for a short phase at the end.')
tf.app.flags.DEFINE_float(
    'cov_loss_wt', 1,
    'Weight of coverage loss (lambda in the paper). \
    If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean(
    'convert_to_coverage_model',
    True,
    'Convert a non-coverage model to a coverage model. \
    Turn this on and run in train mode. \
    Your current model will be copied to a new version \
    (same name with _cov_init appended)\
    that will be ready to run with coverage flag turned on,\
    for the coverage training stage.')


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

            summary_writer.add_summary(
                summaries, train_step)  # write the summaries
            if train_step % 100 == 0:  # flush the summary writer every so often
                summary_writer.flush()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for
    coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver(
        [v for v in tf.global_variables()
         if "coverage" not in v.name and "Adagrad" not in v.name])
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


def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    default_device = tf.device('/gpu:0')
    with default_device:
        model.build_graph()  # build the graph
        if FLAGS.convert_to_coverage_model:
            assert FLAGS.coverage, (
                "To convert your non-coverage model to a coverage model, run"
                "with convert_to_coverage_model=True and coverage=True")
            convert_to_coverage_model()
        # only keep 1 checkpoint at a time
        saver = tf.train.Saver(max_to_keep=1)

    sv = tf.train.Supervisor(logdir=train_dir,
                             is_chief=True,
                             saver=saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             # save summaries for tensorboard every 60 secs
                             save_model_secs=60,
                             # checkpoint every 60 secs
                             global_step=model.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(
        config=util.get_config())
    tf.logging.info("Created session.")
    try:
        # this is an infinite loop until interrupted
        run_training(model, batcher, sess_context_manager, sv, summary_writer)
    except KeyboardInterrupt:
        tf.logging.info(
            "Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def main(argv):  # NOQA
    # -----------   create the session  -----------

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.allow_soft_placement = True

    is_generator_train = FLAGS.is_generator_train
    is_decode = FLAGS.is_decode
    is_discriminator_train = FLAGS.is_discriminator_train
    is_gan_train = FLAGS.is_gan_train

    # -----------  pretraining  the generator -----------
    train_data_source = FLAGS.train_data_source
    train_data_target = FLAGS.train_data_target

    gan_gen_batch_size = FLAGS.gan_gen_batch_size

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)

    hparam_gen = [
        'mode',
        'adagrad_init_acc',
        'batch_size',
        'cov_loss_wt',
        'coverage',
        'emb_dim',
        'hidden_dim',
        'lr',
        'max_dec_steps',
        'max_grad_norm',
        'pointer_gen',
        'rand_unif_init_mag',
        'trunc_norm_init_std',
        'max_enc_steps',
    ]

    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():  # for each flag
        if key in hparam_gen:  # if it's in the list
            hps_dict[key] = val  # add it to the dict

    hps = namedtuple("HParamsGen", hps_dict.keys())(**hps_dict)

    if FLAGS.segment is not True:
        hps = hps._replace(max_enc_steps=110)
        hps = hps._replace(max_dec_steps=25)
    else:
        assert hps.max_enc_steps == 80, "No segmentation, max_enc_steps wrong"
        assert hps.max_dec_steps == 15, "No segmentation, max_dec_steps wrong"

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(
        FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    tf.set_random_seed(111)  # a seed value for randomness

    sess = tf.Session(config=config)
    with tf.variable_scope('second_generate'):
        if is_generator_train:
            print('train the model and build the generate')
            training_model_hps = hps
            training_model_hps = training_model_hps._replace(mode="train")
            model = GenSum(training_model_hps, vocab)
            setup_training(model, batcher)
            print('done')

        elif is_decode:
            print('generator decoding')
            decode_model_hps = hps
            decode_model_hps = decode_model_hps._replace(mode="decode")
            decode_model_hps = decode_model_hps._replace(max_dec_steps=1)
            model = GenSum(decode_model_hps, vocab)
            decoder = BeamSearchDecoder(model, batcher, vocab)
            decoder.decode()
            print('done')
            return 0

        elif is_gan_train:
            # draw the graph of the generator
            print('build the generate without training')
            build_graph_hps = hps
            build_graph_hps = build_graph_hps._replace(mode="train")
            model = GenSum(build_graph_hps, vocab)
            model.build_graph()

# ----------- pretraining the discriminator -----------

    if is_discriminator_train or is_gan_train:

        dis_max_epoches = FLAGS.dis_epoches
        dis_dispFreq = FLAGS.dis_dispFreq
        dis_saveFreq = FLAGS.dis_saveFreq
        dis_devFreq = FLAGS.dis_devFreq
        dis_batch_size = FLAGS.dis_batch_size
        dis_saveto = FLAGS.dis_saveto
        dis_reshuffle = FLAGS.dis_reshuffle
        max_dec_steps = FLAGS.max_dec_steps
        max_enc_steps = FLAGS.max_enc_steps
        dis_positive_data = FLAGS.dis_positive_data
        dis_negative_data = FLAGS.dis_negative_data
        dis_source_data = FLAGS.dis_source_data
        dis_dev_positive_data = FLAGS.dis_dev_positive_data
        dis_dev_negative_data = FLAGS.dis_dev_negative_data
        dis_dev_source_data = FLAGS.dis_dev_source_data
        dis_reload = FLAGS.dis_reload

        filter_sizes_s = [i for i in range(1, max_enc_steps, 4)]
        num_filters_s = [(100 + i*10) for i in range(1, max_enc_steps, 4)]
        dis_filter_sizes = [i for i in range(1, max_dec_steps, 4)]
        dis_num_filters = [(100 + i*10) for i in range(1, max_dec_steps, 4)]

        discriminator = DisCNN(
            sess=sess,
            max_enc_steps=max_enc_steps,
            max_dec_steps=max_dec_steps,
            num_classes=2,
            vocab,
            batch_size=dis_batch_size,
            dim_word=FLAGS.emb_dim,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            filter_sizes_s=filter_sizes_s,
            num_filters_s=num_filters_s,
            positive_data=dis_positive_data,
            negative_data=dis_negative_data,
            source_data=dis_source_data,
            dev_positive_data=dis_dev_positive_data,
            dev_negative_data=dis_dev_negative_data,
            dev_source_data=dis_dev_source_data,
            max_epoches=dis_max_epoches,
            dispFreq=dis_dispFreq,
            saveFreq=dis_saveFreq,
            devFreq=dis_devFreq,
            saveto=dis_saveto,
            reload_mod=dis_reload,
            clip_c=FLAGS.max_grad_norm,
            optimizer='rmsprop',
            reshuffle=dis_reshuffle,
            scope='discnn')

        if is_discriminator_train:
            print('train the discriminator')
            discriminator.train()
            print('done')

        else:
            print('building the discriminator without training done')
            print('done')

    #   ----------- Start Reinforcement Training -----------
        if is_gan_train:

            gan_total_iter_num = FLAGS.gan_total_iter_num
            gan_gen_iter_num = FLAGS.gan_gen_iter_num
            gan_dis_iter_num = FLAGS.gan_dis_iter_num
            gan_gen_reshuffle = FLAGS.gan_gen_reshuffle
            gan_dis_source_data = FLAGS.gan_dis_source_data
            gan_dis_positive_data = FLAGS.gan_dis_positive_data
            gan_dis_negative_data = FLAGS.gan_dis_negative_data
            gan_dispFreq = FLAGS.gan_dispFreq
            gan_saveFreq = FLAGS.gan_saveFreq
            roll_num = FLAGS.rollnum
            generate_num = FLAGS.generate_num
            bias_num = FLAGS.bias_num
            teacher_forcing = FLAGS.teacher_forcing

            print('reinforcement training begin...')

            for gan_iter in range(gan_total_iter_num):

                print('reinforcement training for %d epoch' % gan_iter)
                gen_train_it = gen_force_train_iter(
                    gan_dis_source_data,
                    gan_dis_positive_data,
                    gan_gen_reshuffle,
                    generator.vocab,
                    vocab_size,
                    gan_gen_batch_size,
                    max_enc_steps,
                    max_dec_steps
                )

                print('finetune the generator ...')
                for gen_iter in range(gan_gen_iter_num):

                    x, y_ground, _ = next(gen_train_it)
                    x_to_maxlen = prepare_sentence_to_maxlen(x, max_enc_steps)

                    x, x_mask, y_ground, y_ground_mask = prepare_data(
                        x, y_ground, max_enc_steps=max_enc_steps,
                        max_dec_steps=max_dec_steps, vocab_size=vocab_size
                    )
                    y_sample_out = generator.generate_step(x, x_mask)

                    y_input, y_input_mask = deal_generated_y_sentence(
                        y_sample_out, generator.vocab, precision=precision)
                    rewards = generator.get_reward(
                        x, x_mask, x_to_maxlen, y_input, y_input_mask,
                        roll_num, discriminator, bias_num=bias_num)
                    print('the reward is ', rewards)
                    loss = generator.generate_step_and_update(
                        x, x_mask, y_input, rewards)
                    if gen_iter % gan_dispFreq == 0:
                        print(
                            'the %d iter, seen %d examples, loss is %f ' % (
                                gen_iter, (
                                    (gan_iter) * gan_gen_iter_num + gen_iter + 1
                                ), loss)
                        )
                    if gen_iter % gan_saveFreq == 0:
                        generator.saver.save(
                            generator.sess, generator.saveto)
                        print(
                            'save the parameters when seen %d examples ' % (
                                (gan_iter) * gan_gen_iter_num + gan_iter + 1
                            ))

                    # teacher force training
                    if teacher_forcing:
                        y_ground = prepare_sentence_to_maxlen(
                            numpy.transpose(y_ground), maxlen=max_dec_steps,
                            precision=precision)
                        y_ground_mask = prepare_sentence_to_maxlen(
                            numpy.transpose(y_ground_mask), maxlen=max_dec_steps,
                            precision=precision)
                        rewards_ground = numpy.ones_like(y_ground)
                        rewards_ground = rewards_ground * y_ground_mask
                        rewards_ground = numpy.transpose(rewards_ground)
                        print(
                            'the reward for ground in teacher forcing is ',
                            rewards_ground)
                        loss = generator.generate_step_and_update(
                            x, x_mask, y_ground, rewards_ground)
                        if gen_iter % gan_dispFreq == 0:
                            print(
                                'the %d iter, seen %d ground examples,\
                                loss is %f ' % (gen_iter, (
                                    (gan_iter) * gan_gen_iter_num +
                                    gen_iter + 1), loss))

                generator.saver.save(generator.sess, generator.saveto)
                print('finetune the generator done!')

                print('prepare the gan_dis_data begin ')
                data_num = prepare_gan_dis_data(
                    train_data_source, train_data_target,
                    gan_dis_source_data, gan_dis_positive_data,
                    num=generate_num, reshuf=True)
                print(
                    'prepare the gan_dis_data done, \
                    the num of the gan_dis_data is %d' % data_num)

                print(
                    'generator generate and save to %s'
                    % gan_dis_negative_data)
                generator.generate_and_save(
                    gan_dis_source_data, gan_dis_negative_data,
                    generate_batch=gan_gen_batch_size
                )
                print('done!')

                print('finetune the discriminator begin...')
                discriminator.train(
                    max_epoch=gan_dis_iter_num,
                    positive_data=gan_dis_positive_data,
                    negative_data=gan_dis_negative_data,
                    source_data=gan_dis_source_data)
                discriminator.saver.save(
                    discriminator.sess, discriminator.saveto)
                print('finetune the discriminator done!')

            print('reinforcement training done')


if __name__ == '__main__':
    sys.stdout = FlushFile(sys.stdout)
    tf.app.run()
