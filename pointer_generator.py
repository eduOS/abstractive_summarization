# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the
sequence-to-sequence model"""
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import time
import numpy as np
import tensorflow as tf
from termcolor import colored
from attention_decoder import lstm_attention_decoder
from attention_decoder import conv_attention_decoder
from utils import lstm_encoder
from codecs import open
from six.moves import xrange
import data

FLAGS = tf.app.flags.FLAGS


class PointerGenerator(object):
    """A class to represent a sequence-to-sequence model for text summarization.
    Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, enc_vocab, dec_vocab):
        self.hps = hps
        self._enc_vocab = enc_vocab
        self._dec_vocab = dec_vocab
        self._log_writer = open("./pg_log", "a", "utf-8")
        self.is_training = self.hps.mode in ["pretrain_gen", "train_gan"]

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input
        data."""
        hps = self.hps
        batch_size = None
        if hps.mode in ["decode", "train_gan"]:
            max_dec_steps = 1
        else:
            max_dec_steps = hps.max_dec_steps

        # -------- placeholders for training generatror and beam search decoding
        # encoder part
        self.enc_batch = tf.placeholder(tf.int32, [batch_size, None], name='enc_batch')
        self.temp_batch = tf.placeholder(tf.int32, [batch_size, None], name='temp_batch_for_embedding')
        self.enc_lens = tf.placeholder(tf.int32, [batch_size], name='enc_lens')
        self.enc_padding_mask = tf.placeholder(tf.float32, [batch_size, None], name='enc_padding_mask')
        self.enc_batch_extend_vocab = tf.placeholder(tf.int32, [batch_size, None], name='enc_batch_extend_vocab')
        self.max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # decoder part
        # when max_dec_steps is 1, this is for beam search decoding, it is for
        # getting the loss otherwise
        self._dec_batch = tf.placeholder(tf.int32, [batch_size, max_dec_steps], name='dec_batch')
        self.target_batch = tf.placeholder(tf.int32, [batch_size, hps.max_dec_steps], name='target_batch')
        self.dec_padding_mask = tf.placeholder(tf.float32, [batch_size, hps.max_dec_steps], name='decoder_padding_mask')

        # ------------------------ placeholders for training GAN
        # encoder part
        self.cell_c = tf.placeholder(
            tf.float32, shape=[batch_size, self.hps.hidden_dim])
        self.cell_h = tf.placeholder(
            tf.float32, shape=[batch_size, self.hps.hidden_dim])

        # decoder part
        # in gan training all three modes of the generator are used:
        # decoding for the generating process; training for the tuning and
        # evaluation for its evaluation
        self.k_sample_targets = tf.placeholder(tf.int32, [FLAGS.sample_num, batch_size, hps.max_dec_steps], name='k_sample_targets')
        self.k_sample_targets_mask = tf.placeholder(tf.float32, [FLAGS.sample_num, batch_size, hps.max_dec_steps], name='k_padding_mask_of_the_sample_targets')
        self.k_samples = tf.placeholder(tf.int32, [FLAGS.sample_num, batch_size, hps.max_dec_steps], name='k_samples')
        self.k_rewards = tf.placeholder(tf.float32, shape=[FLAGS.sample_num, batch_size, hps.max_dec_steps], name="k_rewards")

        # ------------------------ placeholders for evaluation
        # decoder part
        self._eval_dec_batch = tf.placeholder(tf.int32, [batch_size, hps.max_dec_steps], name='eval_dec_batch')

        if hps.mode in ["decode", 'train_gan'] and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [None, None], name='prev_coverage')
            # so this need not to be reloaded and taken gradient hps

    def _make_feed_dict(self, batch, just_enc=False, gan_eval=False, gan=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate
        placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the
          encoder.
          update: only for the evaluation and training of the generator in gan training
        """
        if gan_eval:
            gan = True
        feed_dict = {}
        feed_dict[self.enc_batch] = batch.enc_batch
        feed_dict[self.enc_lens] = batch.enc_lens
        feed_dict[self.enc_padding_mask] = batch.enc_padding_mask
        if not just_enc:
            feed_dict[self.target_batch] = batch.target_batch
            feed_dict[self.dec_padding_mask] = batch.dec_padding_mask
            if gan_eval:
                feed_dict[self._eval_dec_batch] = batch.dec_batch
            elif not gan:
                feed_dict[self._dec_batch] = batch.dec_batch
        return feed_dict

    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self.hps

        with tf.name_scope('seq2seq'):
            self.rand_unif_init = tf.random_uniform_initializer(
                -hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            k_samples_ls = tf.unstack(self.k_samples, axis=0)
            k_sample_targets_ls = tf.unstack(self.k_sample_targets, axis=0)
            k_sample_targets_mask_ls = tf.unstack(self.k_sample_targets_mask, axis=0)
            k_rewards_ls = tf.unstack(self.k_rewards, axis=0)

            with tf.variable_scope('embeddings'):
                self.enc_embeddings = tf.get_variable(
                    'enc_embeddings', [self._enc_vocab.size(), hps.char_emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                self.dec_embeddings = tf.get_variable(
                    'dec_embeddings', [self._dec_vocab.size(), hps.word_emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                self.dec_emb_saver = tf.train.Saver({"dec_embeddings": self.dec_embeddings})
                self.enc_emb_saver = tf.train.Saver({"enc_embeddings": self.enc_embeddings})
                self._emb_enc_inputs = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_batch)
                # for gen training(mode is pretrain_gen) and
                # beam searching(mode is decode or train_gan)
                emb_dec_inputs = [tf.nn.embedding_lookup(self.dec_embeddings, x)
                                  for x in tf.unstack(self._dec_batch, axis=1)]

                emb_eval_dec_inputs = [tf.nn.embedding_lookup(self.dec_embeddings, x)
                                       for x in tf.unstack(self._eval_dec_batch, axis=1)]

                k_emb_samples_ls = [
                    [tf.nn.embedding_lookup(self.dec_embeddings, x)
                     for x in tf.unstack(samples, axis=1)]
                    for samples in k_samples_ls
                ]

            attention_keys, dec_in_state = lstm_encoder(
                self._emb_enc_inputs, self.enc_lens, hps.hidden_dim, self.rand_unif_init, keep_prob=hps.keep_prob, is_training=self.is_training)

            self.attention_keys, self.dec_in_state = attention_keys, dec_in_state

            # selective encoding: http://arxiv.org/abs/1704.07073
            # self.attention_keys = selective_fn(self.attention_keys, self.dec_in_state)

            with tf.variable_scope('decoder') as decoder_scope:
                self.vocab_dists, self.attn_dists, _ = self._lstm_decoder(emb_dec_inputs)
                decoder_scope.reuse_variables()
                if self.hps.mode == "decode":
                    self.beam_search()

                eval_final_dists, eval_attn_dists, _ = self._lstm_decoder(emb_eval_dec_inputs)

                k_sample_final_dists_ls = []
                for emb_samples in k_emb_samples_ls:
                    sample_final_dists, _, _ = self._lstm_decoder(emb_samples)
                    k_sample_final_dists_ls.append(sample_final_dists)

            def get_loss(vocab_dists, target_batch, padding_mask, rewards=None):
                batch_nums = tf.range(0, limit=tf.shape(target_batch)[0])

                loss_per_step = []
                for dec_step, dist in enumerate(vocab_dists):
                    targets = target_batch[:, dec_step]
                    indices = tf.stack((batch_nums, targets), axis=1)
                    gold_probs = tf.gather_nd(dist, indices)
                    losses = -tf.log(gold_probs) * padding_mask[:, dec_step]
                    loss_per_step.append(losses * rewards[:, dec_step] if rewards is not None else losses)
                return loss_per_step

            # Calculate the loss
            with tf.variable_scope('generator_loss'):

                # for training of generator
                loss_per_step = get_loss(self.vocab_dists, self.target_batch, self.dec_padding_mask)
                # self.loss_per_step = loss_per_step
                eval_loss_per_step = get_loss(eval_final_dists, self.target_batch, self.dec_padding_mask)
                # Apply padding_mask mask and get loss
                self._loss = _avg(loss_per_step, self.dec_padding_mask)
                self._eval_loss = _avg(eval_loss_per_step, self.dec_padding_mask)

                # for training of GAN
                # Calculate coverage loss from the attention distributions
                if hps.coverage:
                    with tf.variable_scope('coverage_loss'):
                        self._coverage_loss = _coverage_loss(
                            self.attn_dists, self.dec_padding_mask)
                    self._total_loss = \
                        self._loss + hps.cov_loss_wt * self._coverage_loss

            with tf.variable_scope('gan_loss'):
                k_gan_losses = []
                for k in range(len(k_sample_targets_ls)):
                    gan_loss_per_step = get_loss(
                        k_sample_final_dists_ls[k], k_sample_targets_ls[k],
                        k_sample_targets_mask_ls[k], k_rewards_ls[k])
                    masked_average = _avg(gan_loss_per_step, k_sample_targets_mask_ls[k])
                    k_gan_losses.append(masked_average)

                self.gan_loss = tf.reduce_mean(tf.stack(k_gan_losses))

        if len(emb_dec_inputs) == 1:
            assert len(self.vocab_dists) == 1
            vocab_dist = self.vocab_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(
                vocab_dist, hps.beam_size * 2)
            self._topk_log_probs = tf.log(topk_probs)

            # for the monte carlo searching
            self._ran_id = tf.multinomial(tf.log(vocab_dist), 1)

        # for the loss
        loss_to_minimize = self._total_loss if self.hps.coverage else self._loss
        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(
            loss_to_minimize, trainable_variables,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(
                gradients, self.hps.gen_max_gradient)

        # set the adaptive learning rate
        self.learning_rate = tf.train.exponential_decay(
            self.hps.gen_lr,               # Base learning rate.
            self.global_step * self.hps.batch_size,  # Current index into the dataset.
            1000000,             # Decay step.
            0.95,                # Decay rate.
            staircase=True)

        # Apply adagrad optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=self.global_step)

        # set the adaptive learning rate
        learning_rate_gan = tf.train.exponential_decay(
            FLAGS.gan_lr,               # Base learning rate.
            self.global_step * self.hps.batch_size,  # Current index into the dataset.
            1000000,             # Decay step.
            0.95,                # Decay rate.
            staircase=True)

        # for the gan loss
        g_opt = self.g_optimizer(learning_rate_gan)
        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(self.gan_loss, trainable_variables,
                                 aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        self.g_grad, _ = tf.clip_by_global_norm(gradients, self.hps.gen_max_gradient)
        with tf.device("/gpu:0"):
            self.g_updates = g_opt.apply_gradients(zip(self.g_grad, trainable_variables), global_step=self.global_step)

        return decoder_scope

    def _get_encoder_oov_mask(self, vsize, batch_size):
        # to reduce the pressure of the copy mechanism, the probabilities of
        # words that can be generated by the decoder should not be only from the
        # decoder and not from the copy mechanism
        batch_nums = tf.range(0, limit=batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        attn_len = tf.shape(self.enc_batch_extend_vocab)[1]
        batch_nums = tf.tile(batch_nums, [1, attn_len])
        indices = tf.stack((batch_nums, self.enc_batch_extend_vocab), axis=2)
        indices = tf.where(tf.less(indices, vsize), indices, tf.zeros_like(indices))
        shape = [batch_size, vsize]
        _mask = tf.where(
            tf.equal(tf.scatter_nd(indices, tf.ones_like(self.enc_batch_extend_vocab), shape), 0),
            tf.ones_like(self.enc_batch_extend_vocab),
            tf.zeros_like(self.enc_batch_extend_vocab))
        _mask = tf.cast(_mask, tf.float32)
        return _mask

    def _lstm_decoder(self, emb_dec_inputs,
                      enc_padding_mask=None, attention_keys=None, dec_in_state=None):
        """
        input:
            emb_dec_inputs, the input of the cell
        to get:
            output log distribution
            new state
        """
        if not dec_in_state:
            enc_padding_mask = self.enc_padding_mask
            attention_keys = self.attention_keys
            dec_in_state = self.dec_in_state

        if type(emb_dec_inputs) is not list:
            emb_dec_inputs = [emb_dec_inputs]
        vsize = self._dec_vocab.size()  # size of the vocabulary
        cell = tf.contrib.rnn.LSTMCell(
            self.hps.hidden_dim,
            state_is_tuple=True,
            initializer=self.rand_unif_init)

        prev_coverage = self.prev_coverage if self.hps.coverage and self.hps.mode in ["train_gan", "decode"] else None

        outputs, out_state, attn_dists = lstm_attention_decoder(
            emb_dec_inputs, enc_padding_mask, attention_keys, dec_in_state, cell,
            initial_state_attention=(len(emb_dec_inputs) == 1),
            use_coverage=self.hps.coverage, prev_coverage=prev_coverage)

        with tf.variable_scope('output_projection'):
            w = tf.get_variable(
                'w', [self.hps.hidden_dim, vsize],
                dtype=tf.float32, initializer=self.trunc_norm_init)
            w_t = tf.transpose(w)  # NOQA
            v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
            vocab_scores = []
            for i, output in enumerate(outputs):

                vocab_scores.append(tf.nn.xw_plus_b(output, w, v))

            vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

        return vocab_dists, attn_dists, out_state

    def _conv_decoder(self, emb_dec_inputs):
        emb_dec_inputs = tf.stack(emb_dec_inputs, axis=1)
        vsize = self.hps.gen_vocab_size
        self.is_training = self.hps.mode in ["pretrain_gen", "train_gan"]
        logits, p_gens, attn_dists, _, _ = conv_attention_decoder(
            self._emb_enc_inputs, self.enc_padding_mask, emb_dec_inputs, self.attentions_keys, vsize, self.is_training)

        vocab_dists = tf.unstack(tf.nn.softmax(logits), axis=1)

        return vocab_dists, None

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to
        the graph"""
        t0 = time.time()
        self._add_placeholders()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.device("/gpu:0"):
            decoder_scope = self._add_seq2seq()
        self.least_val_loss = tf.Variable(1000.0, name='least_val_loss', trainable=False)
        t1 = time.time()
        print(colored('Time to build graph: %s seconds' % (t1 - t0), "yellow"))
        return decoder_scope

    def get_cur_lr(self, sess):
        return sess.run(self.learning_rate)

    def run_one_batch(self, sess, batch, update=True, gan_eval=False):
        """Runs one training iteration. Returns a dictionary containing train
        op, summaries, loss, global_step and (optionally) coverage loss."""
        if gan_eval:
            update = False

        feed_dict = self._make_feed_dict(batch, gan_eval=gan_eval)

        to_return = {
            'global_step': self.global_step,
        }

        to_return['attn_dists'] = self.attn_dists
        to_return['vocab_dists'] = self.vocab_dists

        to_return['loss'] = self._eval_loss
        if gan_eval:
            to_return['loss'] = self._eval_loss
        else:
            to_return['loss'] = self._loss
        if update:
            # if update is False it is for the generator evaluation
            to_return['train_op'] = self._train_op
            # to_return['loss_per_step'] = self.loss_per_step
        if self.hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        rsts = sess.run(to_return, feed_dict)
        print('attn_dists')
        print(rsts['attn_dists'])
        print('vocab_dists')
        print(rsts['vocab_dists'])
        time.sleep(100*100)

        return rsts

    def run_gan_batch(self, sess, batch, samples, sample_targets,
                      sample_padding_mask, rewards, update=True, gan_eval=False
                      ):
        feed_dict = self._make_feed_dict(batch, gan_eval=gan_eval, gan=True)

        feed_dict.update({
            # for the decoder
            self.k_samples: samples,
            self.k_sample_targets: sample_targets,
            self.k_sample_targets_mask: sample_padding_mask,
            self.k_rewards: rewards,
        })

        to_return = {
            'global_step': self.global_step,
            'loss': self.gan_loss,
            # 'loss_per': self.gan_loss_per_step,
        }
        if update:
            to_return['updates'] = self.g_updates
        results = sess.run(to_return, feed_dict)
        # print("results['loss_per']")
        # print(results['loss_per'])
        # print('----------------------\n\n')
        return results

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the
        encoder states and decoder initial state.

        Args:
          sess: Tensorflow session.
          batch: Batch object that is the same example repeated across the batch
          (for beam search)

        Returns:
          attention_keys: The encoder states. A tensor of shape [batch_size,
          <=max_enc_steps, 2*hidden_dim].
          dec_in_state: A LSTMStateTuple of shape
          ([batch_size, hidden_dim],[batch_size, hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        # feed the batch into the placeholders
        (attention_keys, dec_in_state) = sess.run(
            [
                self.attention_keys,
                self.dec_in_state,
            ],
            feed_dict
        )  # run the encoder
        # attention_keys: [batch_size * beam_size, <=max_enc_steps, 2*hidden_dim]
        # dec_in_state: [batch_size * beam_size, ]

        # dec_in_state is LSTMStateTuple shape
        # ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is
        # identical across the batch so we just take the top row.
        # dec_in_state = [tf.contrib.rnn.LSTMStateTuple(
        #     dec_in_state.c[i], dec_in_state.h[i] for i in xrange(len(dec_in_state.h))]
        #     # TODO: should this be changed to shape?
        return attention_keys, dec_in_state

    def decode_onestep(self, emb_dec_inputs, dec_in_state):
        """
        function: decode onestep for rollout
        inputs:
            the embedded input
        """
        # attn_dists, p_gens, coverage, vocab_scores, log_probs, new_states
        _, _, _, final_dists, new_states = getattr(self, "_" + 'lstm_decoder')(emb_dec_inputs, dec_in_state)
        # how can it be fed by a [batch_size * 1 * emb_dim] while decoding?
        # final_dists_sliced = tf.slice(final_dists[0], [0, 0], [-1, self._vocab.size()])
        final_dists = final_dists[0]
        # final_dists += tf.ones_like(final_dists) * sys.float_info.epsilon
        output_id = tf.squeeze(tf.cast(tf.reshape(tf.multinomial(tf.log(final_dists), 1), [self.hps.batch_size]), tf.int32))
        # next_input = tf.nn.embedding_lookup(self.embeddings, next_token)  # batch x emb_dim
        return output_id, new_states

    def beam_search(self):
        # state, attention
        beam_size = self.hps.beam_size
        batch_size = self.hps.batch_size
        vocab_size = self._dec_vocab.size()
        num_steps = self.hps.max_dec_steps

        log_beam_probs, beam_symbols = [], []
        output_projection = None
        _attention_keys = tf.tile(tf.expand_dims(self.attention_keys, axis=1), [1, beam_size, 1, 1])
        _attention_keys = tf.reshape(_attention_keys, [batch_size*beam_size, tf.shape(self.attention_keys)[1], self.attention_keys.get_shape().as_list()[-1]])
        _enc_padding_mask = tf.tile(tf.expand_dims(self.enc_padding_mask, axis=1), [1, beam_size, 1])
        _enc_padding_mask = tf.reshape(_enc_padding_mask, [batch_size*beam_size, tf.shape(self.enc_padding_mask)[1]])

        def beam_search(prev, i, log_fn):
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
                # (batch_size*beam_size, embedding_size) -> (batch_size*beam_size, vocab_size)

            log_probs = log_fn(prev)

            if i > 1:
                log_probs = tf.reshape(tf.expand_dims(tf.reduce_sum(tf.stack(log_beam_probs, axis=1), axis=1), axis=1) + log_probs,
                                       [-1, beam_size * vocab_size])
                # (batch_size*beam_size, vocab_size) -> (batch_size, beam_size*vocab_size)
            best_probs, indices = tf.nn.top_k(log_probs, beam_size)
            # (batch_size, beam_size)
            indices = tf.squeeze(tf.reshape(indices, [-1, 1]))
            best_probs = tf.reshape(best_probs, [-1])
            # (batch_size*beam_size)

            symbols = indices % vocab_size       # which word in vocabulary
            beam_parent = indices // vocab_size  # which hypothesis it came from

            beam_symbols.append(symbols)

            index_base = tf.reshape(
                tf.tile(tf.expand_dims(tf.range(batch_size) * beam_size, axis=1), [1, beam_size]), [-1])
            # (batch_size*beam_size, num_steps)
            # real_path = tf.reshape(tf.stack(beam_path, axis=1) + index_base, [beam_size*batch_size, i])
            real_path = beam_parent + index_base
            # adapt the previous symbols according to the current symbol
            if i > 1:
                pre_sum = tf.reduce_sum(tf.stack(log_beam_probs, axis=1), axis=1)
                pre_sum = tf.gather(pre_sum, real_path)
            else:
                pre_sum = 0
            log_beam_probs.append(best_probs-pre_sum)
            if i > 1:
                for j in range(i)[:0:-1]:
                    beam_symbols[j-1] = tf.gather(beam_symbols[j-1], real_path)
                    log_beam_probs[j-1] = tf.gather(log_beam_probs[j-1], real_path)

            return tf.nn.embedding_lookup(self.dec_embeddings, symbols)
            # (batch_size*beam_size, embedding_size)

        dec_input = tf.convert_to_tensor(batch_size * [self._dec_vocab.word2id(data.START_DECODING)])
        dec_input = tf.nn.embedding_lookup(self.dec_embeddings, dec_input)
        for i in range(num_steps):
            if i > 0:
                attention_keys = _attention_keys
                enc_padding_mask = _enc_padding_mask
            else:
                dec_in_state = self.dec_in_state
                attention_keys = self.attention_keys
                enc_padding_mask = self.enc_padding_mask
            if i == 1:
                new_c = tf.tile(tf.expand_dims(dec_in_state.c, axis=1), [1, beam_size, 1])
                new_c = tf.reshape(new_c, [batch_size*beam_size, -1])
                new_h = tf.tile(tf.expand_dims(dec_in_state.h, axis=1), [1, beam_size, 1])
                new_h = tf.reshape(new_h, [batch_size*beam_size, -1])
                dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            vocab_dists, _, dec_in_state = self._lstm_decoder([dec_input], enc_padding_mask, attention_keys, dec_in_state)
            dec_input = beam_search(vocab_dists[0], i+1, tf.log)

        best_seq = tf.stack(values=beam_symbols, axis=1)
        self.best_seq = tf.reshape(best_seq, [batch_size, beam_size, num_steps])
        # (batch_size*beam_size, num_steps)

    def run_beam_search(self, sess, batch):
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        best_seq = sess.run(self.best_seq, feed_dict)  # run the encoder
        return best_seq

    def run_decode_onestep(self, sess,
                           latest_tokens, attention_keys, enc_padding_mask,
                           dec_init_states, prev_coverage, method="bs"):
        """For beam search decoding. Run the decoder for one step.

        Args:
          sess: Tensorflow session.
          enc_batch_extend_vocab: the encode batch with extended vocabulary
          max_art_oovs: the max article out of vocabulary
          latest_tokens: Tokens to be fed as input into the decoder for this
          timestep
          attention_keys: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states
          from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the
          previous timestep. List of None if not using coverage.
          method: can be bs standing for beam search and mc representing monte carlo

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size
          containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length
          beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List
          of None if coverage is not turned on.
        """

        # Turn dec_init_states (a list of LSTMStateTuples) into a single
        # LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        batch_size = attention_keys.shape[0]

        feed = {
            self.attention_keys: attention_keys,
            self.enc_padding_mask: enc_padding_mask,
            self.dec_in_state: new_dec_in_state,
            self._dec_batch: latest_tokens,
        }

        to_return = {
          "states": self._dec_out_state,
          "attn_dists": self.attn_dists,
        }

        if method == "bs":
            to_return["ids"] = self._topk_ids
            to_return["probs"] = self._topk_log_probs
        elif method == "mc":
            to_return["ran_id"] = self._ran_id

        if self.hps.coverage:
            feed[self.prev_coverage] = prev_coverage
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of
        # LSTMStateTuple -- one for each hypothesis
        new_states = [
            tf.contrib.rnn.LSTMStateTuple(
                results['states'].c[i, :], results['states'].h[i, :])
            for i in xrange(batch_size)]

        if method == "bs":
            sample_ids = results["ids"]
            probs = results["probs"]

            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['attn_dists']) == 1
            attn_dists = results['attn_dists'][0].tolist()

            # Convert singleton list containing a tensor to a list of k arrays
        else:
            sample_ids = results["ran_id"]
            probs = None

            attn_dists = None

        # Convert the coverage tensor to a list length k containing the coverage
        # vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == batch_size
        else:
            new_coverage = [None for _ in xrange(attention_keys.shape[1])]

        return sample_ids, probs, new_states, attn_dists, new_coverage

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and
      0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = tf.reduce_sum(tf.stack(values_per_step, 1), 1)/dec_lens
    return tf.reduce_mean(values_per_ex)  # overall average


def _avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and
      0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    # values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = tf.reduce_sum(tf.stack(values, 1), 1)/dec_lens
    return tf.reduce_mean(values_per_ex)  # overall average


def _mask(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and
      0s.

    Returns:
      a scalar
    """

    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = sum(values_per_step)
    return tf.reduce_sum(values_per_ex)  # overall loss


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list
      length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(
        attn_dists[0])
    # shape (batch_size, attn_length). Initial coverage is zero.
    # Coverage loss per decoder timestep. Will be list length max_dec_steps
    # containing shape (batch_size).
    covlosses = []
    for a in attn_dists:
        # calculate the coverage loss for this step
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss
